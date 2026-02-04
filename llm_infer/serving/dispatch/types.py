"""Internal data types for request dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...context import RequestContext


class RequestStatus(Enum):
    """Status of an inference request."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class Request:
    """Internal inference request (queue message)."""

    id: str
    prompt: str
    context: RequestContext | None = None  # Shared context for logging/timing
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.1
    stream: bool = False
    use_chat_template: bool | None = None
    stop_sequences: list[str] | None = None
    messages: list[dict[str, Any]] | None = None  # Chat messages for multi-turn
    adapter_id: str | None = None  # LoRA adapter name for vLLM
    # Tool calling support
    tools: list[dict[str, Any]] | None = None  # OpenAI-format tool definitions
    tool_choice: str | dict[str, Any] | None = (
        None  # "auto", "none", "required", or object
    )
    # Structured output support
    response_format: dict[str, Any] | None = None  # {"type": "json_object"} or schema


@dataclass
class Response:
    """Internal inference response (queue message)."""

    id: str
    status: RequestStatus
    result: str | None = None
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Tool calling response (list of tool call dicts if model requested tool calls)
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class StreamChunk:
    """
    Streaming token chunk (queue message).

    Sent incrementally during streaming generation.
    The final chunk has is_final=True and includes token counts.
    """

    id: str  # Request ID this chunk belongs to
    token: str  # The token text
    is_final: bool = False  # True for the last chunk
    finish_reason: str | None = (
        None  # "stop", "length", "tool_calls", etc. (only on final)
    )
    prompt_tokens: int | None = None  # Only set on final chunk
    completion_tokens: int | None = None  # Only set on final chunk
    # Tool calling support for streaming
    tool_calls: list[dict[str, Any]] | None = (
        None  # Tool calls (streamed incrementally)
    )


@dataclass
class MetricsRequest:
    """Request for server metrics (sent from API subprocess)."""

    id: str
    reset_peak: bool = False  # If True, reset peak memory after reading


@dataclass
class MetricsResponse:
    """Response with server metrics."""

    id: str
    # GPU memory (torch - per-process)
    gpu_allocated_bytes: int
    gpu_reserved_bytes: int
    gpu_peak_bytes: int
    # GPU memory (pynvml - device-level, for vLLM)
    gpu_device_used_bytes: int = 0
    gpu_device_total_bytes: int = 0
    gpu_device_free_bytes: int = 0
    # Model memory (estimated from GPU delta during load)
    gpu_model_memory_bytes: int = 0
    # KV cache
    kv_cache_bytes: int = 0
    kv_cache_usage_perc: float = 0.0  # vLLM real-time usage (0-1)
    kv_blocks_used: int = 0
    kv_blocks_total: int = 0
    kv_block_size: int = 0
    # Sequences
    active_sequences: int = 0
    total_sequence_tokens: int = 0
    # Request queue
    pending_requests: int = 0


@dataclass
class EmbeddingRequest:
    """Internal embedding request (queue message)."""

    id: str
    inputs: list[str]  # Texts to embed
    dimensions: int | None = None  # Output dimensions (for Matryoshka embeddings)


@dataclass
class EmbeddingResponse:
    """Internal embedding response (queue message)."""

    id: str
    status: RequestStatus
    embeddings: list[list[float]] | None = None
    total_tokens: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Adapter control requests (for LoRA adapter management via IPC)
# ---------------------------------------------------------------------------


@dataclass
class AdapterInfo:
    """Serializable adapter information for IPC responses."""

    adapter_id: str
    description: str | None
    loaded_at: str  # ISO timestamp


@dataclass
class AdapterListRequest:
    """Request to list loaded adapters (sent from API subprocess)."""

    id: str


@dataclass
class AdapterListResponse:
    """Response containing adapter list."""

    id: str
    adapters: list[AdapterInfo]


@dataclass
class AdapterRefreshRequest:
    """Request to refresh adapters (sent from API subprocess)."""

    id: str
    adapter_id: str | None = None  # None = full rescan, else refresh single adapter


@dataclass
class AdapterRefreshResponse:
    """Response from adapter refresh operation."""

    id: str
    adapter_id: str | None  # Echo back which adapter (None if full scan)
    adapters_loaded: int  # Count of enabled adapters after refresh
    status: str  # 'loaded', 'unloaded', or 'scanned'
