"""Protocol interfaces for inference engines.

These protocols define the contracts for inference engines, enabling:
- Engine-agnostic handlers and serving layer
- Swappable engine implementations (native, vLLM, Ollama)
- Clear API boundaries
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..context import RequestContext
    from .native.scheduler import Request


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

    def memory_stats(self) -> dict[str, int | float]:
        """Return GPU and KV cache memory statistics.

        Returns:
            Dict with GPU memory (bytes) and KV cache block stats.
            Some fields (e.g., usage percentages) may be floats.
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
