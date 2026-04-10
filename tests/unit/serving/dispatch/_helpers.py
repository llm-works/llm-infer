"""Shared test helpers for serving/dispatch unit tests.

Provides minimal builders and fakes used across handler, processor, loop,
and per-handler test files. Domain-specific fakes (e.g., warmup engine,
metrics engine_with_stats) live in their own test files.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from llm_infer.serving.dispatch.types import Request


class ResponseQueueFake:
    """Drop-in for mp.Queue used by handlers for streaming responses.

    Records all puts in `items` for assertions.
    """

    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def make_request(
    req_id: str = "r1",
    *,
    prompt: str = "hello",
    stream: bool = False,
    adapter: str | None = None,
    model: str | None = None,
    tools: list | None = None,
    tool_choice: Any = None,
    response_format: Any = None,
    chat_template_kwargs: Any = None,
) -> Request:
    """Build a minimal Request with sensible defaults."""
    return Request(
        id=req_id,
        prompt=prompt,
        stream=stream,
        adapter=adapter,
        model=model,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        chat_template_kwargs=chat_template_kwargs,
    )


def make_engine(*, generate_result: Any = "world", count_tokens: int = 5) -> MagicMock:
    """Build a basic engine MagicMock with generate() and count_tokens()."""
    e = MagicMock()
    e.generate.return_value = generate_result
    e.count_tokens.return_value = count_tokens
    return e
