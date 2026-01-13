"""Protocol interfaces for inference primitives.

These protocols define the contracts between layers, enabling:
- Dependency injection for testability
- Swappable implementations
- Clear architectural boundaries
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from torch import Tensor

if TYPE_CHECKING:
    from ..context import RequestContext
    from ..pipelines.scheduler import Request
    from .kv_cache import SequenceKVCache

# ---------------------------------------------------------------------------
# KV Cache Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class BlockAllocator(Protocol):
    """Protocol for block-level memory allocation.

    Implementations manage a pool of fixed-size memory blocks that can be
    allocated to sequences for KV cache storage.
    """

    @property
    def num_free_blocks(self) -> int:
        """Number of available blocks."""
        ...

    @property
    def num_allocated_blocks(self) -> int:
        """Number of blocks in use."""
        ...

    def allocate(self) -> int:
        """Allocate a free block.

        Returns:
            Block index.

        Raises:
            RuntimeError: If no free blocks available.
        """
        ...

    def free(self, block_id: int) -> None:
        """Return a block to the pool."""
        ...

    def can_allocate(self, num_blocks: int) -> bool:
        """Check if we can allocate the requested number of blocks."""
        ...


@runtime_checkable
class KVCacheStorage(Protocol):
    """Protocol for KV cache tensor storage.

    Provides access to the underlying K/V tensors organized by layer and block.
    Shape: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    """

    block_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    k_cache: Tensor
    v_cache: Tensor


@runtime_checkable
class KVCache(Protocol):
    """Protocol for per-sequence KV cache management.

    Manages the block table for a single sequence, tracking which blocks
    hold the KV cache for each position.
    """

    block_ids: list[int]
    num_tokens: int

    def num_blocks(self) -> int:
        """Number of blocks allocated to this sequence."""
        ...

    def append_token(self, pool: BlockAllocator) -> tuple[int, int]:
        """Allocate space for next token.

        Returns:
            Tuple of (block_id, offset within block).
        """
        ...

    def allocate_for_prompt(self, pool: BlockAllocator, num_tokens: int) -> None:
        """Allocate blocks for a prompt of given length."""
        ...

    def free_all(self, pool: BlockAllocator) -> None:
        """Return all blocks to the pool."""
        ...

    def get_block_table(self) -> list[int]:
        """Get the block table (list of block IDs)."""
        ...


# ---------------------------------------------------------------------------
# Attention Backend Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AttentionBackend(Protocol):
    """Protocol for attention computation backends.

    Implementations can use different attention algorithms:
    - FlashInfer (optimized CUDA kernels)
    - FlashAttention
    - Naive PyTorch implementation
    """

    def forward(
        self,
        q: Tensor,
        layer_idx: int,
        kv_caches: list["SequenceKVCache"],
        kv_storage: KVCacheStorage,
        num_heads: int,
        num_kv_heads: int,
        sm_scale: float,
    ) -> Tensor:
        """Compute attention output.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            layer_idx: Current layer index
            kv_caches: Per-sequence KV cache handles
            kv_storage: Underlying KV tensor storage
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads (for GQA)
            sm_scale: Softmax scaling factor (typically 1/sqrt(head_dim))

        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]
        """
        ...

    @property
    def name(self) -> str:
        """Backend name (e.g., 'flashinfer', 'naive')."""
        ...


# ---------------------------------------------------------------------------
# Tokenizer Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for text tokenization."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to tokenize.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of token IDs.
        """
        ...

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.
            skip_special_tokens: Whether to remove special tokens from output.

        Returns:
            Decoded text.
        """
        ...

    @property
    def eos_token_id(self) -> int | None:
        """End-of-sequence token ID, or None if not defined."""
        ...

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID, or None if not defined."""
        ...

    @property
    def has_chat_template(self) -> bool:
        """Whether this tokenizer has a chat template."""
        ...

    def encode_chat(
        self,
        message: str | list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """Encode messages using the chat template.

        Args:
            message: Either a user message string, or a list of message dicts
                with 'role' and 'content' keys.
            add_generation_prompt: Whether to add the assistant prompt prefix.

        Returns:
            List of token IDs with chat formatting applied.
        """
        ...


# ---------------------------------------------------------------------------
# Execution Backend Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ExecutionBackend(Protocol):
    """Protocol for model execution.

    Implementations run the forward pass of a transformer model.
    Could be local, tensor-parallel, or disaggregated across workers.
    """

    def forward(
        self,
        token_ids: Tensor,
        positions: Tensor,
        kv_caches: list[KVCache],
        kv_storage: KVCacheStorage,
    ) -> Tensor:
        """Run forward pass.

        Args:
            token_ids: Input token IDs [batch, seq_len]
            positions: Position indices [batch, seq_len]
            kv_caches: Per-sequence KV cache handles
            kv_storage: Underlying KV tensor storage

        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        ...


# ---------------------------------------------------------------------------
# Scheduler Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SchedulerProtocol(Protocol):
    """Protocol for request scheduling.

    Implementations manage request queues and batch formation.
    Could be FIFO, priority-based, or distributed.
    """

    def add_request(self, request: "RequestProtocol") -> str:
        """Add request to queue. Returns request ID."""
        ...

    def get_batch(
        self,
    ) -> tuple[list["RequestProtocol"], list["RequestProtocol"]]:
        """Get next batch: (prefill_requests, decode_requests)."""
        ...

    def update_after_step(self, finished_ids: list[str]) -> None:
        """Update state after generation step."""
        ...

    def get_request(self, request_id: str) -> "RequestProtocol | None":
        """Get request by ID."""
        ...

    @property
    def is_empty(self) -> bool:
        """Check if scheduler has no requests."""
        ...


@runtime_checkable
class RequestProtocol(Protocol):
    """Protocol for inference requests."""

    id: str
    prompt_tokens: list[int]
    output_tokens: list[int]
    kv_cache: KVCache
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    stop_token_ids: set[int]

    @property
    def total_tokens(self) -> int:
        """Total tokens processed so far."""
        ...

    @property
    def is_finished(self) -> bool:
        """Check if request is complete."""
        ...

    def mark_finished(self) -> None:
        """Mark the request as finished."""
        ...


# ---------------------------------------------------------------------------
# Streaming Result Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class StreamingResultProtocol(Protocol):
    """Protocol for streaming generation results.

    Implementations yield tokens one at a time and track generation metadata.
    """

    @property
    def prompt_tokens(self) -> int:
        """Number of prompt tokens."""
        ...

    @property
    def completion_tokens(self) -> int:
        """Number of completion tokens generated so far."""
        ...

    @property
    def finish_reason(self) -> str:
        """Reason generation stopped ('stop' or 'length')."""
        ...

    def __iter__(self) -> Iterator[str]:
        """Iterate over generated tokens."""
        ...

    def __next__(self) -> str:
        """Get next generated token."""
        ...


# ---------------------------------------------------------------------------
# Inference Engine Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class InferenceEngineProtocol(Protocol):
    """Protocol for inference engine abstraction.

    This protocol defines the contract between request handlers and the engine,
    enabling decoupling and testability. Handlers should depend on this protocol
    rather than the concrete InferenceEngine class.
    """

    @property
    def eos_token_id(self) -> int | None:
        """End-of-sequence token ID."""
        ...

    def should_use_chat_template(self) -> bool:
        """Check if chat template should be used based on model name."""
        ...

    def supports_embeddings(self) -> bool:
        """Check if engine supports embeddings."""
        ...

    def embed(
        self,
        inputs: list[str],
        dimensions: int | None = None,
    ) -> tuple[list[list[float]], int]:
        """Generate embeddings for input texts.

        Args:
            inputs: List of texts to embed.
            dimensions: Optional output dimensions (for Matryoshka embeddings).

        Returns:
            Tuple of (embeddings list, total tokens).
        """
        ...

    def count_tokens(self, text: str, use_chat_template: bool | None = None) -> int:
        """Count tokens in text.

        Args:
            text: Input text.
            use_chat_template: Whether to apply chat template. None = auto-detect.

        Returns:
            Number of tokens.
        """
        ...

    def tokenize(self, text: str, use_chat_template: bool | None = None) -> list[int]:
        """Tokenize text.

        Args:
            text: Input text.
            use_chat_template: Whether to apply chat template. None = auto-detect.

        Returns:
            List of token IDs.
        """
        ...

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text.
        """
        ...

    def build_stop_token_ids(self, stop_sequences: list[str] | None) -> set[int]:
        """Build set of stop token IDs from EOS and stop sequences.

        Args:
            stop_sequences: Optional list of stop strings.

        Returns:
            Set of token IDs that should stop generation.
        """
        ...

    def prefill_request(self, request: "Request") -> None:
        """Run prefill phase for a request.

        Allocates KV cache and runs the prefill forward pass.

        Args:
            request: Request with prompt_tokens set.
        """
        ...

    def step_decode(self, requests: "list[Request]") -> list[int | None]:
        """Run one batched decode step.

        Args:
            requests: List of requests (must have completed prefill).

        Returns:
            List of generated token IDs (None for finished/skipped).
        """
        ...

    def free_request(self, request: "Request") -> None:
        """Free resources (KV cache) for a completed request.

        Args:
            request: The request to free.
        """
        ...

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        use_chat_template: bool | None = None,
        stop_sequences: list[str] | None = None,
        context: "RequestContext | None" = None,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate text from a prompt (blocking).

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling.
            repetition_penalty: Penalty for repeating tokens.
            use_chat_template: Whether to apply chat template. None = auto-detect.
            stop_sequences: Optional list of strings that stop generation.
            context: Optional request context for logging.
            messages: Optional list of chat messages (for multi-turn/system).

        Returns:
            Generated text.
        """
        ...

    def generate_stream_sync(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        use_chat_template: bool | None = None,
        stop_sequences: list[str] | None = None,
        context: "RequestContext | None" = None,
        messages: list[dict[str, str]] | None = None,
    ) -> "StreamingResultProtocol":
        """Stream generated tokens (sync version).

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling.
            repetition_penalty: Penalty for repeating tokens.
            use_chat_template: Whether to apply chat template. None = auto-detect.
            stop_sequences: Optional list of strings that stop generation.
            context: Optional request context for logging.
            messages: Optional list of chat messages (for multi-turn/system).

        Returns:
            StreamingResult that yields tokens and tracks metadata.
        """
        ...

    def memory_stats(self) -> dict[str, int]:
        """Return GPU and KV cache memory statistics.

        Returns:
            Dict with GPU memory (bytes) and KV cache block stats.
        """
        ...

    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking (useful between benchmark runs)."""
        ...

    def shutdown(self) -> None:
        """Shutdown the engine and release resources.

        Called during graceful shutdown. Implementations should:
        - Stop any background workers
        - Release GPU memory
        - Clean up distributed process groups
        """
        ...
