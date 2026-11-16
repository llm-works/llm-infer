# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for the scripted backend and Factory.scripted()."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import (
    BackendRequestError,
    ChatRequest,
    ChatResponse,
    Factory,
    FailAfter,
    LLMCallbacks,
    LLMClient,
    ResponseHolder,
    RetryConfig,
    ScriptedBackend,
    ScriptExhaustedError,
)
from llm_infer.client import retry as retry_module
from llm_infer.client.backends.providers.scripted import FIXED_TEXT, stream_chunks
from llm_infer.schemas.openai import FinishReason

USER = [{"role": "user", "content": "Hi"}]
NO_WAIT = RetryConfig(base=0.01, max_delay=0.01)


@pytest.fixture
def lg() -> Logger:
    return MagicMock(spec=Logger)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries go ahead at once: the client's backoff sleeps return immediately."""

    async def _no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(retry_module.time, "sleep", lambda delay: None)
    monkeypatch.setattr(retry_module.asyncio, "sleep", _no_sleep)


def _backend(client: LLMClient) -> ScriptedBackend:
    backend = client.backend
    assert isinstance(backend, ScriptedBackend)
    return backend


def _tool(name: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    function: dict[str, Any] = {"name": name, "description": name}
    if parameters is not None:
        function["parameters"] = parameters
    return {"type": "function", "function": function}


def _schema_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": "r", "strict": True, "schema": schema},
    }


class TestListScript:
    """A list script answers one entry per call."""

    def test_answers_in_order(self, lg: Logger) -> None:
        client = Factory(lg).scripted(
            [ChatResponse(content="one"), ChatResponse(content="two")]
        )
        assert client.chat(USER).content == "one"
        assert client.chat(USER).content == "two"
        assert _backend(client).remaining == 0

    def test_exhausted_raises(self, lg: Logger) -> None:
        client = Factory(lg).scripted([ChatResponse(content="one")])
        client.chat(USER)
        with pytest.raises(ScriptExhaustedError, match="request 2"):
            client.chat(USER)

    def test_exhausted_is_not_retried(self, lg: Logger) -> None:
        client = Factory(lg).scripted([], retry=NO_WAIT)
        with pytest.raises(ScriptExhaustedError):
            client.chat(USER)
        assert len(_backend(client).requests) == 1

    def test_records_requests(self, lg: Logger) -> None:
        client = Factory(lg).scripted(
            [ChatResponse(content="a"), ChatResponse(content="b")]
        )
        client.chat(USER, model="m1")
        client.chat([{"role": "user", "content": "again"}])
        requests = _backend(client).requests
        assert [r.model for r in requests] == ["m1", "scripted"]
        assert requests[1].messages[0]["content"] == "again"

    def test_script_entries_stay_unchanged(self, lg: Logger) -> None:
        entry = ChatResponse(content="one")
        client = Factory(lg).scripted([entry])
        response = client.chat(USER)
        assert response.request is not None
        assert response is not entry
        assert entry.request is None

    def test_exception_entry_raises_without_retry_config(self, lg: Logger) -> None:
        error = BackendRequestError("rate limited", status_code=429)
        client = Factory(lg).scripted([error, ChatResponse(content="ok")])
        with pytest.raises(BackendRequestError):
            client.chat(USER)
        assert _backend(client).remaining == 1

    def test_exception_entry_is_retried(self, lg: Logger) -> None:
        retries: list[int] = []
        callbacks = LLMCallbacks(
            on_retry=lambda req, err, attempt, delay: retries.append(attempt)
        )
        script = [
            BackendRequestError("rate limited", status_code=429),
            ChatResponse(content="ok"),
        ]
        client = Factory(lg).scripted(script, retry=NO_WAIT, callbacks=callbacks)
        assert client.chat(USER).content == "ok"
        assert len(_backend(client).requests) == 2
        assert retries == [1]

    def test_non_transient_error_is_not_retried(self, lg: Logger) -> None:
        script = [
            BackendRequestError("bad request", status_code=400),
            ChatResponse(content="ok"),
        ]
        client = Factory(lg).scripted(script, retry=NO_WAIT)
        with pytest.raises(BackendRequestError):
            client.chat(USER)
        assert _backend(client).remaining == 1

    def test_fail_after_raises_on_non_streaming_call(self, lg: Logger) -> None:
        client = Factory(lg).scripted([FailAfter(ValueError("boom"))])
        with pytest.raises(ValueError, match="boom"):
            client.chat(USER)

    @pytest.mark.asyncio
    async def test_async_answers(self, lg: Logger) -> None:
        client = Factory(lg).scripted([ChatResponse(content="one")])
        response = await client.chat_async(USER)
        assert response.content == "one"
        assert response.request is not None

    @pytest.mark.asyncio
    async def test_async_exception_entry_is_retried(self, lg: Logger) -> None:
        script = [
            BackendRequestError("unavailable", status_code=503),
            ChatResponse(content="ok"),
        ]
        client = Factory(lg).scripted(script, retry=NO_WAIT)
        assert (await client.chat_async(USER)).content == "ok"


class TestCallableScript:
    """A callable script answers every call from the request."""

    def test_called_with_request(self, lg: Logger) -> None:
        def respond(request: ChatRequest) -> ChatResponse:
            return ChatResponse(content=request.messages[-1]["content"].upper())

        client = Factory(lg).scripted(respond)
        assert client.chat(USER).content == "HI"
        assert _backend(client).remaining is None
        assert len(_backend(client).requests) == 1

    def test_called_outside_the_lock(self, lg: Logger) -> None:
        # Under the lock, a responder reading `remaining` or calling the client
        # again would block forever; checking the lock fails instead of hanging.
        held: list[bool] = []

        def respond(request: ChatRequest) -> ChatResponse:
            held.append(backend._lock.locked())
            return ChatResponse(content="ok")

        client = Factory(lg).scripted(respond)
        backend = _backend(client)
        client.chat(USER)
        assert held == [False]


class TestStreaming:
    """Streaming yields word chunks and publishes the response at the end."""

    def test_chunks_rejoin_to_content(self) -> None:
        for text in ["", "one", "  lead and trail  ", "a\nb\tc  d"]:
            assert "".join(stream_chunks(text)) == text

    def test_stream_yields_chunks_and_fires_on_response(self, lg: Logger) -> None:
        seen: list[ChatResponse] = []
        callbacks = LLMCallbacks(on_response=lambda req, resp: seen.append(resp))
        client = Factory(lg).scripted(
            [ChatResponse(content="hello big world")], callbacks=callbacks
        )
        stream = client.chat_stream(USER)
        assert list(stream) == ["hello ", "big ", "world"]
        assert stream.response is not None
        assert stream.response.content == "hello big world"
        assert stream.response.request is not None
        assert seen == [stream.response]

    @pytest.mark.asyncio
    async def test_async_stream(self, lg: Logger) -> None:
        client = Factory(lg).scripted([ChatResponse(content="hello world")])
        stream = client.chat_stream_async(USER)
        assert [chunk async for chunk in stream] == ["hello ", "world"]
        assert stream.response is not None

    def test_backend_fills_holder(self, lg: Logger) -> None:
        backend = ScriptedBackend(lg, [ChatResponse(content="x y")])
        holder = ResponseHolder()
        request = ChatRequest(messages=USER)
        assert list(backend.chat_stream(request, holder)) == ["x ", "y"]
        assert holder.value is backend.last_response
        assert holder.value is not None

    def test_fail_after_streams_then_raises(self, lg: Logger) -> None:
        errors: list[Exception] = []
        callbacks = LLMCallbacks(on_error=lambda req, err: errors.append(err))
        entry = FailAfter(ValueError("cut"), chunks=2, content="one two three")
        client = Factory(lg).scripted([entry], callbacks=callbacks)
        received: list[str] = []
        with pytest.raises(ValueError, match="cut"):
            received.extend(client.chat_stream(USER))
        assert received == ["one ", "two "]
        assert len(errors) == 1

    def test_failure_before_first_token_is_retried(self, lg: Logger) -> None:
        error = BackendRequestError("overloaded", status_code=529)
        script = [FailAfter(error, chunks=0), ChatResponse(content="ok")]
        client = Factory(lg).scripted(script, retry=NO_WAIT)
        assert list(client.chat_stream(USER)) == ["ok"]

    def test_failure_after_first_token_is_not_retried(self, lg: Logger) -> None:
        error = BackendRequestError("overloaded", status_code=529)
        script = [FailAfter(error, chunks=1), ChatResponse(content="ok")]
        client = Factory(lg).scripted(script, retry=NO_WAIT)
        with pytest.raises(BackendRequestError):
            list(client.chat_stream(USER))
        assert _backend(client).remaining == 1

    def test_fail_after_rejects_more_chunks_than_content(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 2"):
            FailAfter(ValueError("x"), chunks=3, content="two words")


class TestUnscripted:
    """Without a script, answers are built from the request."""

    def test_fixed_text(self, lg: Logger) -> None:
        response = Factory(lg).scripted().chat(USER)
        assert response.content == FIXED_TEXT
        assert response.finish_reason == FinishReason.STOP
        assert response.model == "scripted"

    def test_request_model_is_reported(self, lg: Logger) -> None:
        assert Factory(lg).scripted().chat(USER, model="m").model == "m"

    def test_schema_shaped_json(self, lg: Logger) -> None:
        schema = {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "score": {"type": "number"},
                "count": {"type": "integer"},
                "label": {"enum": ["b", "a"]},
                "tags": {"type": "array", "items": {"type": "string"}},
                "inner": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        }
        response = (
            Factory(lg).scripted().chat(USER, response_format=_schema_format(schema))
        )
        assert json.loads(response.content) == {
            "ok": False,
            "score": 0,
            "count": 0,
            "label": "b",
            "tags": [FIXED_TEXT, FIXED_TEXT],
            "inner": {"name": FIXED_TEXT},
        }

    def test_json_object_format(self, lg: Logger) -> None:
        response = (
            Factory(lg).scripted().chat(USER, response_format={"type": "json_object"})
        )
        assert json.loads(response.content) == {}

    def test_first_turn_calls_first_tool(self, lg: Logger) -> None:
        tools = [
            _tool(
                "search", {"type": "object", "properties": {"q": {"type": "string"}}}
            ),
            _tool("done"),
        ]
        response = Factory(lg).scripted().chat(USER, tools=tools)
        assert response.finish_reason == FinishReason.TOOL_CALLS
        assert response.tool_calls is not None
        (call,) = response.tool_calls
        assert call.id == "scripted-0"
        assert call.function.name == "search"
        assert json.loads(call.function.arguments) == {"q": FIXED_TEXT}

    def test_later_turns_call_last_tool(self, lg: Logger) -> None:
        tools = [_tool("search"), _tool("done")]
        history = [
            *USER,
            {"role": "assistant", "content": "", "tool_calls": [{"id": "scripted-0"}]},
            {"role": "tool", "tool_call_id": "scripted-0", "content": "found"},
        ]
        response = Factory(lg).scripted().chat(history, tools=tools)
        assert response.tool_calls is not None
        (call,) = response.tool_calls
        assert call.id == "scripted-1"
        assert call.function.name == "done"
        assert json.loads(call.function.arguments) == {}

    def test_single_tool_is_called_on_first_turn(self, lg: Logger) -> None:
        response = Factory(lg).scripted().chat(USER, tools=[_tool("only")])
        assert response.tool_calls is not None
        assert response.tool_calls[0].function.name == "only"


class TestThroughSAIAAdapter:
    """The SAIA adapter runs for real on top of the scripted backend."""

    @pytest.fixture
    def saia(self) -> Any:
        return pytest.importorskip("llm_saia.core")

    @pytest.mark.asyncio
    async def test_tool_turns(self, lg: Logger, saia: Any) -> None:
        from llm_infer.client.saia import SAIAAdapter

        adapter = SAIAAdapter(Factory(lg).scripted(), streaming=False)
        tools = [
            saia.ToolDef(
                name="search",
                description="s",
                parameters={"type": "object", "properties": {}},
            ),
            saia.ToolDef(
                name="done",
                description="d",
                parameters={"type": "object", "properties": {}},
            ),
        ]
        messages = [saia.Message(role="user", content="go")]
        first = await adapter.chat(messages, tools=tools)
        assert [tc.name for tc in first.tool_calls] == ["search"]

        messages += [
            saia.Message(role="assistant", content="", tool_calls=first.tool_calls),
            saia.Message(
                role="tool", content="found", tool_call_id=first.tool_calls[0].id
            ),
        ]
        second = await adapter.chat(messages, tools=tools)
        assert [tc.name for tc in second.tool_calls] == ["done"]

    @pytest.mark.asyncio
    async def test_structured_answer(self, lg: Logger, saia: Any) -> None:
        from llm_infer.client.saia import SAIAAdapter

        adapter = SAIAAdapter(Factory(lg).scripted())
        schema = {"type": "object", "properties": {"verified": {"type": "boolean"}}}
        response = await adapter.chat(
            [saia.Message(role="user", content="check")],
            response_schema={"name": "v", "schema": schema},
        )
        assert json.loads(response.content) == {"verified": False}
