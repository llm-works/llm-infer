# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for the shared parts of the server-backed engines."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from llm_infer.engines.remote import LineStream, RemoteEngine, truncate_embedding

pytestmark = pytest.mark.unit


class TestTruncateEmbedding:
    def test_no_dimensions_returns_input(self) -> None:
        embedding = [3.0, 4.0]
        assert truncate_embedding(embedding, None) is embedding

    def test_not_longer_than_dimensions_returns_input(self) -> None:
        embedding = [3.0, 4.0]
        assert truncate_embedding(embedding, 2) is embedding
        assert truncate_embedding(embedding, 5) is embedding

    def test_truncates_and_renormalizes(self) -> None:
        result = truncate_embedding([3.0, 4.0, 12.0], 2)
        assert result == [0.6, 0.8]
        assert math.isclose(math.hypot(*result), 1.0)

    def test_zero_vector_is_truncated_only(self) -> None:
        assert truncate_embedding([0.0, 0.0, 1.0], 2) == [0.0, 0.0]


class TestRemoteEngine:
    def test_batched_processing_is_a_no_op(self) -> None:
        engine = RemoteEngine()
        assert engine.prefill_request(object()) is None
        assert engine.step_decode([object(), object()]) == [None, None]
        assert engine.free_request(object()) is None
        assert engine.reset_peak_memory() is None

    def test_memory_stats_are_zero(self) -> None:
        stats = RemoteEngine().memory_stats()
        assert stats["kv_cache_usage_perc"] == 0.0
        assert all(value == 0 for value in stats.values())


class _EchoStream(LineStream):
    """Yields each line; an empty line ends the stream, "bad" fails to parse."""

    _server_name = "Echo"
    _malformed_message = "malformed JSON in echo stream"

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.closed = 0

    def _process_line(self, line: str) -> str | None:
        if not line:
            raise StopIteration
        if line == "bad":
            raise KeyError("choices")
        return line

    def _on_closed(self) -> None:
        self.closed += 1


def _stream(lines: Iterator[str]) -> tuple[_EchoStream, MagicMock]:
    """An _EchoStream over lines, and the mocked stream context it opens."""
    response = MagicMock()
    response.iter_lines.return_value = lines
    stream_ctx = MagicMock()
    stream_ctx.__enter__.return_value = response
    client = MagicMock()
    client.stream.return_value = stream_ctx
    return _EchoStream(MagicMock(), client, "/x", {}), stream_ctx


class TestLineStream:
    def test_yields_lines_until_the_end(self) -> None:
        stream, stream_ctx = _stream(iter(["a", "b", ""]))
        assert list(stream) == ["a", "b"]
        stream_ctx.__exit__.assert_called_once()

    def test_transport_error_closes_and_reraises(self) -> None:
        def lines() -> Iterator[str]:
            yield "a"
            raise httpx.ReadError("connection reset")

        stream, stream_ctx = _stream(lines())
        assert next(stream) == "a"
        with pytest.raises(httpx.ReadError):
            next(stream)
        stream_ctx.__exit__.assert_called_once()
        with pytest.raises(StopIteration):
            next(stream)

    def test_parse_error_closes_and_reraises(self) -> None:
        stream, stream_ctx = _stream(iter(["a", "bad", "b"]))
        assert next(stream) == "a"
        with pytest.raises(KeyError, match="choices"):
            next(stream)
        stream_ctx.__exit__.assert_called_once()
        assert stream.closed == 1
        with pytest.raises(StopIteration):
            next(stream)
