# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Shared parts of the engines that proxy an inference server over HTTP.

OllamaEngine and VLLMServerEngine forward requests to a server process that
schedules, batches and manages GPU memory on its own.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from typing import Any, NoReturn, Self

import httpx
from appinfra.log import Logger

from ..schemas.openai import FinishReason


def truncate_embedding(embedding: list[float], dimensions: int | None) -> list[float]:
    """Truncate an embedding to dimensions and renormalize it (Matryoshka).

    An embedding no longer than dimensions is returned unchanged.
    """
    if dimensions is None or len(embedding) <= dimensions:
        return embedding
    truncated = embedding[:dimensions]
    norm = math.sqrt(sum(x * x for x in truncated))
    return [x / norm for x in truncated] if norm > 0 else truncated


class RemoteEngine:
    """InferenceEngineProtocol methods the server handles on its own."""

    def prefill_request(self, request: Any) -> None:
        """No-op: the server handles prefill internally."""

    def step_decode(self, requests: list[Any]) -> list[int | None]:
        """No-op: the server handles decoding internally."""
        return [None] * len(requests)

    def free_request(self, request: Any) -> None:
        """No-op: the server manages resources internally."""

    def memory_stats(self) -> dict[str, int | float]:
        """Zero memory statistics: the server manages GPU memory."""
        return {
            "allocated": 0,
            "reserved": 0,
            "peak": 0,
            "model_memory": 0,
            "kv_cache_bytes": 0,
            "kv_blocks_used": 0,
            "kv_blocks_total": 0,
            "kv_block_size": 0,
            "device_used": 0,
            "device_total": 0,
            "device_free": 0,
            "kv_cache_usage_perc": 0.0,
        }

    def reset_peak_memory(self) -> None:
        """No-op: the server manages memory tracking."""


class LineStream:
    """Iterator over a streaming HTTP response, read line by line.

    Subclasses implement _process_line: it returns a text chunk, None to read
    the next line, or raises StopIteration once the stream is complete. The
    server's name goes into the error raised for a line that is not JSON.
    """

    _server_name: str
    _malformed_message: str

    def __init__(
        self, lg: Logger, client: httpx.Client, url: str, payload: dict[str, Any]
    ) -> None:
        self._lg = lg
        self._client = client
        self._url = url
        self._payload = payload

        # Stream state; the context manager must stay referenced or the stream closes
        self._stream_context: Any = None
        self._response: httpx.Response | None = None
        self._line_iter: Iterator[str] | None = None
        self._finished = False

        # Final stats (populated when generation completes)
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.finish_reason: FinishReason | None = None
        # Tool calls (populated on completion if the model made tool calls)
        self.tool_calls: list[dict[str, Any]] | None = None

    def _start_stream(self) -> None:
        """Start the streaming request."""
        self._stream_context = self._client.stream(
            "POST", self._url, json=self._payload
        )
        self._response = self._stream_context.__enter__()
        self._line_iter = self._response.iter_lines()

    def _process_line(self, line: str) -> str | None:
        """Parse one line of the stream; see the class docstring."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._cleanup()

    def __del__(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        """Close the streaming response."""
        if self._stream_context is not None:
            try:
                self._stream_context.__exit__(None, None, None)
            except Exception as e:
                self._lg.debug("cleanup exception suppressed", extra={"exception": e})
            self._stream_context = None
            self._response = None
            self._on_closed()

    def _on_closed(self) -> None:
        """Called once the response is closed."""

    def __next__(self) -> str:
        if self._finished:
            raise StopIteration
        if self._line_iter is None:
            self._start_stream()
            assert self._line_iter is not None
        line = ""  # Initialize before try block for exception handler
        try:
            while True:
                line = next(self._line_iter)
                if result := self._process_line(line):
                    # Clean up if this was the final chunk (done with content)
                    if self._finished:
                        self._cleanup()
                    return result
        except StopIteration:
            self._finished = True
            self._cleanup()
            raise
        except json.JSONDecodeError as e:
            self._lg.warning(
                self._malformed_message, extra={"line": line[:100], "exception": e}
            )
            self._finished = True
            self._cleanup()
            raise RuntimeError(
                f"{self._server_name} returned malformed JSON: {e}"
            ) from e
        except httpx.HTTPError as e:
            self._fail_transport(e)

    def _fail_transport(self, e: httpx.HTTPError) -> NoReturn:
        """End the stream after a transport error and raise it."""
        self._finished = True
        self._cleanup()
        raise e
