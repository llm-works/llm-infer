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
    device_used_bytes: int = 0
    device_total_bytes: int = 0
    device_free_bytes: int = 0
    model_memory_bytes: int = 0

    @property
    def allocated_mb(self) -> float:
        return self.allocated_bytes / (1024 * 1024)

    @property
    def reserved_mb(self) -> float:
        return self.reserved_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)

    @property
    def device_used_mb(self) -> float:
        return self.device_used_bytes / (1024 * 1024)

    @property
    def device_total_mb(self) -> float:
        return self.device_total_bytes / (1024 * 1024)

    @property
    def device_free_mb(self) -> float:
        return self.device_free_bytes / (1024 * 1024)

    @property
    def model_memory_mb(self) -> float:
        return self.model_memory_bytes / (1024 * 1024)


@dataclass
class KVCacheStats:
    """KV cache statistics."""

    bytes: int
    blocks_used: int
    blocks_total: int
    block_size: int
    usage_perc: float = 0.0  # vLLM reports usage as percentage (0-1)

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
            device_used_bytes=stats.get("device_used", 0),
            device_total_bytes=stats.get("device_total", 0),
            device_free_bytes=stats.get("device_free", 0),
            model_memory_bytes=stats.get("model_memory", 0),
        )
        self._kv_cache = KVCacheStats(
            bytes=stats.get("kv_cache_bytes", 0),
            blocks_used=stats.get("kv_blocks_used", 0),
            blocks_total=stats.get("kv_blocks_total", 0),
            block_size=stats.get("kv_block_size", 0),
            usage_perc=stats.get("kv_cache_usage_perc", 0.0),
        )
        return self

    def with_handler_stats(self, handler: RequestHandler) -> MetricsBuilder:
        """Add sequence and pending stats from handler."""
        seq_stats = handler.sequence_stats() or {}
        self._sequences = SequenceStats(
            active=seq_stats.get("active", 0),
            total_tokens=seq_stats.get("total_tokens", 0),
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
            gpu_device_used_bytes=gpu.device_used_bytes,
            gpu_device_total_bytes=gpu.device_total_bytes,
            gpu_device_free_bytes=gpu.device_free_bytes,
            gpu_model_memory_bytes=gpu.model_memory_bytes,
            kv_cache_bytes=kv.bytes,
            kv_cache_usage_perc=kv.usage_perc,
            kv_blocks_used=kv.blocks_used,
            kv_blocks_total=kv.blocks_total,
            kv_block_size=kv.block_size,
            active_sequences=seq.active,
            total_sequence_tokens=seq.total_tokens,
            pending_requests=self._pending_requests,
        )

    def build_api_response(self) -> dict[str, Any]:  # cq: max-lines=45
        """Build API response dict with nested structure."""
        gpu = self._gpu or GPUStats(0, 0, 0)
        kv = self._kv_cache or KVCacheStats(0, 0, 0, 0)
        seq = self._sequences or SequenceStats(0, 0)

        return {
            "gpu": {
                "torch": {
                    "allocated_bytes": gpu.allocated_bytes,
                    "reserved_bytes": gpu.reserved_bytes,
                    "peak_bytes": gpu.peak_bytes,
                    "allocated_mb": gpu.allocated_mb,
                    "reserved_mb": gpu.reserved_mb,
                    "peak_mb": gpu.peak_mb,
                },
                "device": {
                    "used_bytes": gpu.device_used_bytes,
                    "total_bytes": gpu.device_total_bytes,
                    "free_bytes": gpu.device_free_bytes,
                    "used_mb": gpu.device_used_mb,
                    "total_mb": gpu.device_total_mb,
                    "free_mb": gpu.device_free_mb,
                },
                "model_memory_bytes": gpu.model_memory_bytes,
                "model_memory_mb": gpu.model_memory_mb,
            },
            "kv_cache": {
                "allocated_bytes": kv.bytes,
                "allocated_mb": kv.mb,
                "usage_perc": kv.usage_perc,
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
        device_used_bytes=response.gpu_device_used_bytes,
        device_total_bytes=response.gpu_device_total_bytes,
        device_free_bytes=response.gpu_device_free_bytes,
        model_memory_bytes=response.gpu_model_memory_bytes,
    )
    builder._kv_cache = KVCacheStats(
        bytes=response.kv_cache_bytes,
        blocks_used=response.kv_blocks_used,
        blocks_total=response.kv_blocks_total,
        block_size=response.kv_block_size,
        usage_perc=response.kv_cache_usage_perc,
    )
    builder._sequences = SequenceStats(
        active=response.active_sequences,
        total_tokens=response.total_sequence_tokens,
    )
    builder._pending_requests = response.pending_requests
    return builder.build_api_response()
