# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Scripted backend: answers without a network or a model.

Only the transport is replaced. Wrapped in an ``LLMClient``, every call still runs
through retries, callbacks, the ``ChatResponse.request`` back-reference and
``SAIAAdapter``, so a smoke run or a unit test exercises the real client path.

The script is one of:

- a list, consumed one entry per call: a ``ChatResponse`` is the answer, an
  exception is raised (a retryable one is retried by ``LLMClient`` and takes the next
  entry), and ``FailAfter`` streams part of an answer and then raises;
- a callable taking the ``ChatRequest`` and returning the ``ChatResponse``;
- nothing: answers are built from the request (see ``ScriptedBackend``).
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from appinfra.log import Logger

from ....schemas.openai import FinishReason, FunctionCall, ToolCall
from ...types import ChatRequest, ChatResponse, ResponseHolder
from ..base import Backend
from ..context import BackendContext

FIXED_TEXT = "scripted"
"""Content of an unscripted answer that has no tools and no response format."""

_CHUNK_RE = re.compile(r"\S+\s*|\s+")


def stream_chunks(content: str) -> list[str]:
    """Split content into the chunks a scripted stream yields: words with their spacing.

    Joining the chunks gives the content back unchanged.
    """
    return _CHUNK_RE.findall(content)


@dataclass(frozen=True)
class FailAfter:
    """Script entry for a stream that fails part-way.

    A streaming call yields the first ``chunks`` chunks of ``content`` and then raises
    ``error``. A non-streaming call raises ``error`` at once. With ``chunks=0`` the
    failure comes before the first token, where ``LLMClient`` still retries.

    Attributes:
        error: Exception raised after the chunks.
        chunks: Number of chunks yielded before the error.
        content: Text the chunks are taken from.
    """

    error: Exception
    chunks: int = 1
    content: str = FIXED_TEXT

    def __post_init__(self) -> None:
        available = len(stream_chunks(self.content))
        if not 0 <= self.chunks <= available:
            raise ValueError(
                f"chunks must be between 0 and {available} for content {self.content!r}, "
                f"got {self.chunks}"
            )


ScriptEntry = ChatResponse | Exception | FailAfter
Responder = Callable[[ChatRequest], ChatResponse]
Script = Sequence[ScriptEntry] | Responder


class ScriptExhaustedError(RuntimeError):
    """A list script was asked for more answers than it holds."""


class ScriptedBackend(Backend):
    """Backend that answers from a script, or from the request when there is none.

    Without a script the answers follow the rules of llm-saia's ``SmokeBackend``, so
    smoke runs behave the same with or without llm-infer in between:

    - tools offered: the first turn calls the first tool, every later turn the last
      one. This assumes the caller lists its terminal tool last, as SAIA's examples
      do; callers whose tools are ordered otherwise should pass a script. A turn is an
      assistant message with tool calls in the request's history.
    - a JSON-schema response format: a JSON object built from the schema.
    - otherwise: ``FIXED_TEXT``.

    Values built from a schema take the first enum value, two array items, zero for
    numbers and false for booleans (so a verification step fails and the caller's
    critique/refine path runs too); any other type gets ``FIXED_TEXT``.

    Every request is recorded in ``requests``. Returned responses are copies of the
    script's entries, so the client setting ``request`` on them leaves the script
    unchanged. A callable script runs without the backend's lock held, so concurrent
    calls can run it at the same time; one with shared state must be thread-safe.
    """

    def __init__(
        self,
        lg: Logger,
        script: Script | None = None,
        name: str = "scripted",
        ctx: BackendContext | None = None,
        default_model: str | None = "scripted",
    ) -> None:
        """Initialize the backend.

        Args:
            lg: Logger instance.
            script: List of entries, a callable, or None to answer from the request.
            name: Backend name (used for discovery/routing).
            ctx: Backend context; its retry config drives ``LLMClient``'s retries.
            default_model: Model name reported when the request names none.
        """
        super().__init__(lg, name, ctx, default_model)
        self._entries: list[ScriptEntry] | None = None
        self._responder: Responder | None = None
        if callable(script):
            self._responder = script
        elif script is not None:
            self._entries = list(script)
        self._lock = threading.Lock()
        self._last_response: ChatResponse | None = None
        self.requests: list[ChatRequest] = []

    @property
    def provider(self) -> str:
        """Provider identifier."""
        return "scripted"

    @property
    def last_response(self) -> ChatResponse | None:
        """Last response returned or streamed."""
        return self._last_response

    @property
    def remaining(self) -> int | None:
        """Entries of a list script not yet used; None for a callable or no script."""
        with self._lock:
            return None if self._entries is None else len(self._entries)

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Answer with the next script entry."""
        entry = self._next(request)
        if isinstance(entry, FailAfter):
            raise entry.error
        return self._finish(entry, None)

    async def chat_async(self, request: ChatRequest) -> ChatResponse:
        """Answer with the next script entry."""
        return self.chat(request)

    def chat_stream(
        self, request: ChatRequest, holder: ResponseHolder | None = None
    ) -> Iterator[str]:
        """Stream the next script entry's content in word chunks."""
        entry = self._next(request)
        if isinstance(entry, FailAfter):
            yield from stream_chunks(entry.content)[: entry.chunks]
            raise entry.error
        yield from stream_chunks(entry.content)
        self._finish(entry, holder)

    async def chat_stream_async(
        self, request: ChatRequest, holder: ResponseHolder | None = None
    ) -> AsyncIterator[str]:
        """Stream the next script entry's content in word chunks."""
        for chunk in self.chat_stream(request, holder):
            yield chunk

    def _next(self, request: ChatRequest) -> ChatResponse | FailAfter:
        """Record the request and take its answer; raise an exception entry.

        A callable script runs after the lock is released, so it may read
        ``remaining`` or call the client again.
        """
        entry: ScriptEntry
        with self._lock:
            self.requests.append(request)
            if self._responder is None:
                entry = self._take(request)
        if self._responder is not None:
            entry = self._responder(request)
        if isinstance(entry, Exception):
            raise entry
        return entry

    def _take(self, request: ChatRequest) -> ScriptEntry:
        """The next list entry, or the answer built from the request without a script."""
        if self._entries is None:
            return _unscripted(request, request.model or self._default_model)
        if not self._entries:
            raise ScriptExhaustedError(
                f"request {len(self.requests)}: script exhausted"
            )
        return self._entries.pop(0)

    def _finish(
        self, entry: ChatResponse, holder: ResponseHolder | None
    ) -> ChatResponse:
        """Copy the entry, publish it as the last response and to the holder."""
        response = dataclasses.replace(entry)
        self._last_response = response
        if holder is not None:
            holder.value = response
        return response


def _unscripted(request: ChatRequest, model: str | None) -> ChatResponse:
    """Answer built from the request: a tool call, schema-shaped JSON, or fixed text."""
    if request.tools:
        return _tool_turn(request, model)
    schema = _response_schema(request)
    if schema is not None:
        content = json.dumps(_schema_value(schema))
        return ChatResponse(
            content=content, finish_reason=FinishReason.STOP, model=model
        )
    return ChatResponse(
        content=FIXED_TEXT, finish_reason=FinishReason.STOP, model=model
    )


def _tool_turn(request: ChatRequest, model: str | None) -> ChatResponse:
    """Call the first tool on the first turn, the last one after that."""
    tools = request.tools or []
    turn = sum(
        1
        for m in request.messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    )
    function = (tools[0] if turn == 0 and len(tools) > 1 else tools[-1])["function"]
    arguments = _schema_value(function.get("parameters") or {"type": "object"})
    call = ToolCall(
        id=f"scripted-{turn}",
        function=FunctionCall(name=function["name"], arguments=json.dumps(arguments)),
    )
    return ChatResponse(
        content="",
        tool_calls=[call],
        finish_reason=FinishReason.TOOL_CALLS,
        model=model,
    )


def _response_schema(request: ChatRequest) -> dict[str, Any] | None:
    """The JSON schema of the request's response format, if it asks for one."""
    response_format = (request.extra or {}).get("response_format")
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") == "json_object":
        return {"type": "object"}
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return None
    schema = json_schema.get("schema")
    return schema if isinstance(schema, dict) else None


def _schema_value(spec: dict[str, Any]) -> Any:
    """A value of the schema's type: objects and arrays filled in, scalars fixed."""
    if spec.get("enum"):
        return spec["enum"][0]
    match spec.get("type"):
        case "object":
            props = spec.get("properties", {})
            return {name: _schema_value(sub) for name, sub in props.items()}
        case "array":
            return [_schema_value(spec.get("items") or {}) for _ in range(2)]
        case "boolean":
            return False
        case "integer" | "number":
            return 0
        case _:
            return FIXED_TEXT
