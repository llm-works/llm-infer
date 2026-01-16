"""Internal data types for request dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

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
    messages: list[dict[str, str]] | None = None  # Chat messages for multi-turn
    adapter_id: str | None = None  # LoRA adapter name for vLLM


@dataclass
class Response:
    """Internal inference response (queue message)."""

    id: str
    status: RequestStatus
    result: str | None = None
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


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
    finish_reason: str | None = None  # "stop", "length", etc. (only on final)
    prompt_tokens: int | None = None  # Only set on final chunk
    completion_tokens: int | None = None  # Only set on final chunk


@dataclass
class MetricsRequest:
    """Request for server metrics (sent from API subprocess)."""

    id: str
    reset_peak: bool = False  # If True, reset peak memory after reading


@dataclass
class MetricsResponse:
    """Response with server metrics."""

    id: str
    # GPU memory
    gpu_allocated_bytes: int
    gpu_reserved_bytes: int
    gpu_peak_bytes: int
    # KV cache
    kv_cache_bytes: int
    kv_blocks_used: int
    kv_blocks_total: int
    kv_block_size: int
    # Sequences
    active_sequences: int
    total_sequence_tokens: int
    # Request queue
    pending_requests: int


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
