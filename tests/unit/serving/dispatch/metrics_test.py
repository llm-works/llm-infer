"""Unit tests for serving/dispatch/metrics.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_infer.serving.dispatch.metrics import (
    GPUStats,
    KVCacheStats,
    MetricsBuilder,
    build_metrics_response,
    format_metrics_for_api,
)
from llm_infer.serving.dispatch.types import MetricsResponse

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# GPUStats / KVCacheStats / SequenceStats - dataclass + properties
# ---------------------------------------------------------------------------


class TestGPUStats:
    def test_mb_conversions(self) -> None:
        s = GPUStats(
            allocated_bytes=1024 * 1024,
            reserved_bytes=2 * 1024 * 1024,
            peak_bytes=4 * 1024 * 1024,
            device_used_bytes=8 * 1024 * 1024,
            device_total_bytes=16 * 1024 * 1024,
            device_free_bytes=8 * 1024 * 1024,
            model_memory_bytes=512 * 1024 * 1024,
        )
        assert s.allocated_mb == 1.0
        assert s.reserved_mb == 2.0
        assert s.peak_mb == 4.0
        assert s.device_used_mb == 8.0
        assert s.device_total_mb == 16.0
        assert s.device_free_mb == 8.0
        assert s.model_memory_mb == 512.0


class TestKVCacheStats:
    def test_properties(self) -> None:
        s = KVCacheStats(
            bytes=10 * 1024 * 1024,
            blocks_used=20,
            blocks_total=100,
            block_size=16,
            usage_perc=0.2,
        )
        assert s.mb == 10.0
        assert s.blocks_free == 80
        assert s.capacity_tokens == 1600


# ---------------------------------------------------------------------------
# MetricsBuilder
# ---------------------------------------------------------------------------


def _engine_with_stats(**overrides: int) -> MagicMock:
    e = MagicMock()
    stats = {
        "allocated": 1024,
        "reserved": 2048,
        "peak": 4096,
        "device_used": 8192,
        "device_total": 16384,
        "device_free": 8192,
        "model_memory": 102400,
        "kv_cache_bytes": 1000,
        "kv_blocks_used": 5,
        "kv_blocks_total": 100,
        "kv_block_size": 16,
        "kv_cache_usage_perc": 0.05,
    }
    stats.update(overrides)
    e.memory_stats.return_value = stats
    return e


def _handler(active: int = 0, total_tokens: int = 0, pending: int = 0) -> MagicMock:
    h = MagicMock()
    h.sequence_stats.return_value = {"active": active, "total_tokens": total_tokens}
    h.pending_count = pending
    return h


class TestMetricsBuilder:
    def test_with_engine_stats_populates_gpu_and_kv(self) -> None:
        e = _engine_with_stats()
        b = MetricsBuilder().with_engine_stats(e)
        assert b._gpu is not None
        assert b._gpu.allocated_bytes == 1024
        assert b._kv_cache is not None
        assert b._kv_cache.blocks_used == 5

    def test_with_engine_stats_handles_missing_keys(self) -> None:
        e = MagicMock()
        e.memory_stats.return_value = {}  # all defaults
        b = MetricsBuilder().with_engine_stats(e)
        assert b._gpu is not None
        assert b._gpu.allocated_bytes == 0
        assert b._kv_cache is not None
        assert b._kv_cache.bytes == 0

    def test_with_handler_stats(self) -> None:
        h = _handler(active=3, total_tokens=150, pending=5)
        b = MetricsBuilder().with_handler_stats(h)
        assert b._sequences is not None
        assert b._sequences.active == 3
        assert b._sequences.total_tokens == 150
        assert b._pending_requests == 5

    def test_with_handler_stats_handles_missing_keys(self) -> None:
        h = MagicMock()
        h.sequence_stats.return_value = None  # falsy
        h.pending_count = 0
        b = MetricsBuilder().with_handler_stats(h)
        assert b._sequences is not None
        assert b._sequences.active == 0

    def test_build_response_uses_defaults_when_stats_missing(self) -> None:
        b = MetricsBuilder()
        resp = b.build_response("req1")
        assert isinstance(resp, MetricsResponse)
        assert resp.id == "req1"
        assert resp.gpu_allocated_bytes == 0
        assert resp.kv_cache_bytes == 0
        assert resp.active_sequences == 0

    def test_build_response_full(self) -> None:
        b = (
            MetricsBuilder()
            .with_engine_stats(_engine_with_stats())
            .with_handler_stats(_handler(active=2, total_tokens=100, pending=4))
        )
        resp = b.build_response("req1")
        assert resp.gpu_allocated_bytes == 1024
        assert resp.kv_cache_bytes == 1000
        assert resp.active_sequences == 2
        assert resp.pending_requests == 4

    def test_build_api_response_structure(self) -> None:
        b = (
            MetricsBuilder()
            .with_engine_stats(_engine_with_stats())
            .with_handler_stats(_handler(active=2, total_tokens=100, pending=4))
        )
        api = b.build_api_response()
        assert "gpu" in api
        assert "torch" in api["gpu"]
        assert "device" in api["gpu"]
        assert api["gpu"]["torch"]["allocated_bytes"] == 1024
        assert "kv_cache" in api
        assert api["kv_cache"]["blocks_total"] == 100
        assert api["kv_cache"]["blocks_free"] == 95
        assert api["kv_cache"]["capacity_tokens"] == 1600
        assert api["sequences"]["active"] == 2
        assert api["pending_requests"] == 4

    def test_build_api_response_with_defaults(self) -> None:
        api = MetricsBuilder().build_api_response()
        assert api["gpu"]["torch"]["allocated_bytes"] == 0
        assert api["pending_requests"] == 0


# ---------------------------------------------------------------------------
# build_metrics_response convenience function
# ---------------------------------------------------------------------------


class TestBuildMetricsResponse:
    def test_no_reset_peak(self) -> None:
        h = _handler(active=1, total_tokens=50, pending=2)
        h.engine = _engine_with_stats(allocated=999)
        resp = build_metrics_response("r1", h, reset_peak=False)
        assert resp.gpu_allocated_bytes == 999
        h.engine.reset_peak_memory.assert_not_called()

    def test_reset_peak_calls_engine(self) -> None:
        h = _handler()
        h.engine = _engine_with_stats()
        build_metrics_response("r1", h, reset_peak=True)
        h.engine.reset_peak_memory.assert_called_once()


# ---------------------------------------------------------------------------
# format_metrics_for_api - reverse path
# ---------------------------------------------------------------------------


class TestFormatMetricsForApi:
    def test_round_trip(self) -> None:
        h = _handler(active=3, total_tokens=200, pending=7)
        h.engine = _engine_with_stats(allocated=4096, kv_blocks_used=10)
        resp = build_metrics_response("r1", h)
        api = format_metrics_for_api(resp)
        assert api["gpu"]["torch"]["allocated_bytes"] == 4096
        assert api["kv_cache"]["blocks_used"] == 10
        assert api["sequences"]["active"] == 3
        assert api["pending_requests"] == 7
