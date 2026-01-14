"""Metrics collection and response building."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .types import MetricsResponse

if TYPE_CHECKING:
    from .handler import RequestHandler


@dataclass
class GPUStats:
    """GPU memory statistics."""

    allocated_bytes: int
    reserved_bytes: int
    peak_bytes: int

    @property
    def allocated_mb(self) -> float:
        return self.allocated_bytes / (1024 * 1024)

    @property
    def reserved_mb(self) -> float:
        return self.reserved_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)


@dataclass
class KVCacheStats:
    """KV cache statistics."""

    bytes: int
    blocks_used: int
    blocks_total: int
    block_size: int

    @property
    def mb(self) -> float:
        return self.bytes / (1024 * 1024)

    @property
    def blocks_free(self) -> int:
        return self.blocks_total - self.blocks_used

    @property
    def capacity_tokens(self) -> int:
        return self.blocks_total * self.block_size


@dataclass
class SequenceStats:
    """Active sequence statistics."""

    active: int
    total_tokens: int


class MetricsBuilder:
    """Builder for metrics collection and response formatting.

    Centralizes metrics gathering from engine and handler, eliminating
    duplication between loop.py and routes.py.
    """

    def __init__(self) -> None:
        self._gpu: GPUStats | None = None
        self._kv_cache: KVCacheStats | None = None
        self._sequences: SequenceStats | None = None
        self._pending_requests: int = 0

    def with_engine_stats(self, engine: Any) -> MetricsBuilder:
        """Add GPU and KV cache stats from engine."""
        stats = engine.memory_stats()
        self._gpu = GPUStats(
            allocated_bytes=stats.get("allocated", 0),
            reserved_bytes=stats.get("reserved", 0),
            peak_bytes=stats.get("peak", 0),
        )
        self._kv_cache = KVCacheStats(
            bytes=stats.get("kv_cache_bytes", 0),
            blocks_used=stats.get("kv_blocks_used", 0),
            blocks_total=stats.get("kv_blocks_total", 0),
            block_size=stats.get("kv_block_size", 0),
        )
        return self

    def with_handler_stats(self, handler: RequestHandler) -> MetricsBuilder:
        """Add sequence and pending stats from handler."""
        seq_stats = handler.sequence_stats()
        self._sequences = SequenceStats(
            active=seq_stats["active"],
            total_tokens=seq_stats["total_tokens"],
        )
        self._pending_requests = handler.pending_count
        return self

    def build_response(self, request_id: str) -> MetricsResponse:
        """Build MetricsResponse for IPC."""
        gpu = self._gpu or GPUStats(0, 0, 0)
        kv = self._kv_cache or KVCacheStats(0, 0, 0, 0)
        seq = self._sequences or SequenceStats(0, 0)

        return MetricsResponse(
            id=request_id,
            gpu_allocated_bytes=gpu.allocated_bytes,
            gpu_reserved_bytes=gpu.reserved_bytes,
            gpu_peak_bytes=gpu.peak_bytes,
            kv_cache_bytes=kv.bytes,
            kv_blocks_used=kv.blocks_used,
            kv_blocks_total=kv.blocks_total,
            kv_block_size=kv.block_size,
            active_sequences=seq.active,
            total_sequence_tokens=seq.total_tokens,
            pending_requests=self._pending_requests,
        )

    def build_api_response(self) -> dict[str, Any]:
        """Build API response dict with nested structure."""
        gpu = self._gpu or GPUStats(0, 0, 0)
        kv = self._kv_cache or KVCacheStats(0, 0, 0, 0)
        seq = self._sequences or SequenceStats(0, 0)

        return {
            "gpu": {
                "allocated_bytes": gpu.allocated_bytes,
                "reserved_bytes": gpu.reserved_bytes,
                "peak_bytes": gpu.peak_bytes,
                "allocated_mb": gpu.allocated_mb,
                "reserved_mb": gpu.reserved_mb,
                "peak_mb": gpu.peak_mb,
            },
            "kv_cache": {
                "bytes": kv.bytes,
                "mb": kv.mb,
                "blocks_used": kv.blocks_used,
                "blocks_total": kv.blocks_total,
                "blocks_free": kv.blocks_free,
                "capacity_tokens": kv.capacity_tokens,
                "block_size": kv.block_size,
            },
            "sequences": {
                "active": seq.active,
                "total_tokens": seq.total_tokens,
            },
            "pending_requests": self._pending_requests,
        }


def build_metrics_response(
    request_id: str, handler: RequestHandler, reset_peak: bool = False
) -> MetricsResponse:
    """Convenience function to build MetricsResponse from handler."""
    engine = handler.engine
    if reset_peak:
        engine.reset_peak_memory()

    return (
        MetricsBuilder()
        .with_engine_stats(engine)
        .with_handler_stats(handler)
        .build_response(request_id)
    )


def format_metrics_for_api(response: MetricsResponse) -> dict[str, Any]:
    """Format MetricsResponse into API response dict."""
    # Reconstruct stats from response for API formatting
    builder = MetricsBuilder()
    builder._gpu = GPUStats(
        allocated_bytes=response.gpu_allocated_bytes,
        reserved_bytes=response.gpu_reserved_bytes,
        peak_bytes=response.gpu_peak_bytes,
    )
    builder._kv_cache = KVCacheStats(
        bytes=response.kv_cache_bytes,
        blocks_used=response.kv_blocks_used,
        blocks_total=response.kv_blocks_total,
        block_size=response.kv_block_size,
    )
    builder._sequences = SequenceStats(
        active=response.active_sequences,
        total_tokens=response.total_sequence_tokens,
    )
    builder._pending_requests = response.pending_requests
    return builder.build_api_response()
