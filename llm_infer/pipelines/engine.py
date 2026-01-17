"""Inference engine core."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch

from ..primitives.attention import get_attention_backend
from ..primitives.guards import GenerationGuard, RepetitionGuard
from ..primitives.kv_cache import BlockPool
from ..primitives.tokenizer import HuggingFaceTokenizer
from .config import EngineConfig
from .generation import run_decode, run_decode_batch, run_prefill
from .model import TransformerModel, get_architecture
from .scheduler import Request

if TYPE_CHECKING:
    from ..context import RequestContext


class InferenceEngine:
    """Main inference engine. Orchestrates model, scheduler, and KV cache."""

    def should_use_chat_template(self) -> bool:
        """Auto-detect if chat template should be used based on model name.

        Returns True if:
        - Model has a chat template AND
        - Model name suggests it's an instruct/chat model (contains 'instruct' or 'chat')

        Base models often ship with chat templates but shouldn't use them.
        """
        if not self.tokenizer.has_chat_template:
            return False

        model_name = self.model_name.lower()
        return "instruct" in model_name or "chat" in model_name

    def supports_embeddings(self) -> bool:
        """Check if engine supports embeddings. Native engine does not."""
        return False

    def embed(
        self,
        inputs: list[str],
        dimensions: int | None = None,
    ) -> tuple[list[list[float]], int]:
        """Generate embeddings - not supported by native engine."""
        raise NotImplementedError("Native engine does not support embeddings")

    def _init_block_pool(
        self, config: EngineConfig, on_progress: Callable | None
    ) -> BlockPool:
        """Initialize KV cache block pool."""
        if on_progress:
            on_progress("kv_cache", 0, 1)
        pool = BlockPool(
            num_blocks=config.num_blocks,
            block_size=config.block_size,
            num_layers=config.model.num_layers,
            num_kv_heads=config.model.num_kv_heads,
            head_dim=config.model.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        if on_progress:
            on_progress("kv_cache", 1, 1)
        return pool

    def _init_decode_buffers(self) -> dict[str, torch.Tensor]:
        """Create pre-allocated buffers for decode step."""
        return {
            "token_ids": torch.zeros((1, 1), dtype=torch.long, device=self.device),
            "positions": torch.zeros((1, 1), dtype=torch.long, device=self.device),
        }

    def _init_tokenizer(
        self, arch: Any, on_progress: Callable[[str, int, int], None] | None
    ) -> HuggingFaceTokenizer:
        """Load tokenizer with progress reporting."""
        if on_progress:
            on_progress("tokenizer", 0, 1)
        tokenizer = HuggingFaceTokenizer(
            self.config.model_path, arch.tokenizer_config()
        )
        if on_progress:
            on_progress("tokenizer", 1, 1)
        return tokenizer

    def _init_model(
        self,
        arch: Any,
        on_progress: Callable[[str, int, int], None] | None,
    ) -> TransformerModel:
        """Initialize transformer model with progress reporting."""

        def weights_progress(phase: str, current: int, total: int) -> None:
            if on_progress:
                on_progress(f"weights:{phase}", current, total)

        model = TransformerModel(
            self.config.model,
            arch,
            self.config.model_path,
            device=self.device,
            dtype=self.dtype,
            on_progress=weights_progress,
            attention_backend=self.attention_backend,
            linear_backend=self.config.linear_backend,
        )
        if self.config.torch_compile:
            model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
        return model

    def __init__(
        self,
        lg: Any,
        config: EngineConfig,
        on_progress: Callable[[str, int, int], None] | None = None,
        guards: Sequence[GenerationGuard] | None = None,
    ):
        """Initialize the inference engine."""
        self.lg = lg
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self._decode_buffers = self._init_decode_buffers()

        arch = get_architecture(lg, config.model)
        self.tokenizer = self._init_tokenizer(arch, on_progress)
        self.attention_backend = get_attention_backend(config.attention_backend)
        self.model = self._init_model(arch, on_progress)
        self.block_pool = self._init_block_pool(config, on_progress)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.guards = guards if guards is not None else [RepetitionGuard()]

    def _run_generation_loop(self, request: Request) -> None:
        """Run the prefill and decode loop for a request."""
        run_prefill(request, self.model, self.block_pool, self.device, self.guards)
        while not request.is_finished:
            run_decode(
                request,
                self.model,
                self.block_pool,
                self.device,
                self.guards,
                buffers=self._decode_buffers,
            )

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
        context: RequestContext | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate text from a prompt (blocking)."""
        if use_chat_template is None:
            use_chat_template = self.should_use_chat_template()

        request = self._create_request(
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            use_chat_template,
            stop_sequences,
            context,
            messages,
        )
        request.kv_cache.allocate_for_prompt(
            self.block_pool, len(request.prompt_tokens)
        )

        try:
            self._run_generation_loop(request)
            return self.tokenizer.decode(
                request.output_tokens, skip_special_tokens=True
            )
        finally:
            request.kv_cache.free_all(self.block_pool)

    def _decode_last_token(self, request: Request) -> str:
        """Decode the last generated token."""
        return self.tokenizer.decode(
            [request.output_tokens[-1]], skip_special_tokens=True
        )

    async def generate_stream(  # cq: max-lines=40
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        use_chat_template: bool | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream generated tokens (async version)."""
        if use_chat_template is None:
            use_chat_template = self.should_use_chat_template()

        request = self._create_request(
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            use_chat_template,
            stop_sequences,
        )
        request.kv_cache.allocate_for_prompt(
            self.block_pool, len(request.prompt_tokens)
        )

        try:
            run_prefill(request, self.model, self.block_pool, self.device, self.guards)
            if request.output_tokens:
                yield self._decode_last_token(request)

            while not request.is_finished:
                run_decode(
                    request,
                    self.model,
                    self.block_pool,
                    self.device,
                    self.guards,
                    buffers=self._decode_buffers,
                )
                if request.output_tokens:
                    yield self._decode_last_token(request)
                await asyncio.sleep(0)
        finally:
            request.kv_cache.free_all(self.block_pool)

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
        context: RequestContext | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> StreamingResult:
        """
        Stream generated tokens (sync version).

        Returns a StreamingResult that can be iterated for tokens and
        queried for metadata after iteration completes.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling.
            repetition_penalty: Penalty for repeating tokens.
            use_chat_template: Whether to apply chat template formatting.
                If None (default), auto-detects based on model name.
            stop_sequences: Optional list of strings that stop generation.
            context: Optional request context for logging.
            messages: Optional list of chat messages (for multi-turn/system).

        Returns:
            StreamingResult object that yields tokens and tracks metadata.
        """
        if use_chat_template is None:
            use_chat_template = self.should_use_chat_template()

        request = self._create_request(
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            use_chat_template,
            stop_sequences,
            context,
            messages,
        )

        return StreamingResult(
            request=request,
            model=self.model,
            block_pool=self.block_pool,
            tokenizer=self.tokenizer,
            device=self.device,
            guards=self.guards,
            decode_buffers=self._decode_buffers,
        )

    def _build_stop_tokens(self, stop_sequences: list[str] | None) -> set[int]:
        """Build stop token IDs from EOS and stop sequences."""
        stop_token_ids: set[int] = {self.eos_token_id} if self.eos_token_id else set()
        if stop_sequences:
            for seq in stop_sequences:
                if seq_tokens := self.tokenizer.encode(seq, add_special_tokens=False):
                    stop_token_ids.add(seq_tokens[0])
        return stop_token_ids

    def _create_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        use_chat_template: bool,
        stop_sequences: list[str] | None = None,
        context: RequestContext | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> Request:
        """Create a request from prompt and parameters."""
        if use_chat_template and self.tokenizer.has_chat_template:
            tokens = self.tokenizer.encode_chat(messages if messages else prompt)
        else:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=True)

        if context:
            from ..context import Event

            context.mark(Event.TOKENIZED, tokens=len(tokens))

        return Request.create(
            prompt_tokens=tokens,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_token_ids=self._build_stop_tokens(stop_sequences),
        )

    @property
    def model_name(self) -> str:
        """Get the model name from path."""
        return self.config.model_path.split("/")[-1]

    # -------------------------------------------------------------------------
    # Protocol methods for handler abstraction
    # -------------------------------------------------------------------------

    def count_tokens(self, text: str, use_chat_template: bool | None = None) -> int:
        """Count tokens in text.

        Args:
            text: Input text.
            use_chat_template: Whether to apply chat template. None = auto-detect.

        Returns:
            Number of tokens.
        """
        return len(self.tokenize(text, use_chat_template))

    def tokenize(self, text: str, use_chat_template: bool | None = None) -> list[int]:
        """Tokenize text.

        Args:
            text: Input text.
            use_chat_template: Whether to apply chat template. None = auto-detect.

        Returns:
            List of token IDs.
        """
        if use_chat_template is None:
            use_chat_template = self.should_use_chat_template()

        if use_chat_template and self.tokenizer.has_chat_template:
            return self.tokenizer.encode_chat(text)
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text.
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def build_stop_token_ids(self, stop_sequences: list[str] | None) -> set[int]:
        """Build set of stop token IDs from EOS and stop sequences.

        Args:
            stop_sequences: Optional list of stop strings.

        Returns:
            Set of token IDs that should stop generation.
        """
        stop_token_ids: set[int] = set()
        if self.eos_token_id:
            stop_token_ids.add(self.eos_token_id)

        if stop_sequences:
            for seq in stop_sequences:
                seq_tokens = self.tokenizer.encode(seq, add_special_tokens=False)
                if seq_tokens:
                    stop_token_ids.add(seq_tokens[0])

        return stop_token_ids

    def prefill_request(self, request: Request) -> None:
        """Run prefill phase for a request.

        Allocates KV cache and runs the prefill forward pass.

        Args:
            request: Request with prompt_tokens set.
        """
        request.kv_cache.allocate_for_prompt(
            self.block_pool, len(request.prompt_tokens)
        )
        run_prefill(request, self.model, self.block_pool, self.device, self.guards)

    def free_request(self, request: Request) -> None:
        """Free resources (KV cache) for a completed request.

        Args:
            request: The request to free.
        """
        request.kv_cache.free_all(self.block_pool)

    def step_decode(self, requests: list[Request]) -> list[int | None]:
        """Run one batched decode step for active requests.

        Processes all active (non-finished) requests in a single forward pass.
        Each request maintains its own sampling parameters.

        Args:
            requests: List of Request objects (must have completed prefill).

        Returns:
            List of generated token IDs (None for finished/skipped requests).
        """
        return run_decode_batch(
            requests, self.model, self.block_pool, self.device, self.guards
        )

    def memory_usage(self) -> dict[str, int]:
        """Get memory usage statistics (deprecated, use memory_stats())."""
        return {
            "kv_cache_bytes": self.block_pool.memory_usage_bytes(),
            "allocated_blocks": self.block_pool.num_allocated_blocks,
            "free_blocks": self.block_pool.num_free_blocks,
        }

    def memory_stats(self) -> dict[str, int | float]:
        """Return GPU and KV cache memory statistics.

        Returns:
            Dict with GPU memory (bytes) and KV cache block stats.
            Some fields (e.g., usage percentages) may be floats.
        """
        return {
            "allocated": torch.cuda.memory_allocated(self.device),
            "reserved": torch.cuda.memory_reserved(self.device),
            "peak": torch.cuda.max_memory_allocated(self.device),
            "kv_cache_bytes": self.block_pool.memory_usage_bytes(),
            "kv_blocks_used": self.block_pool.num_allocated_blocks,
            "kv_blocks_total": self.block_pool.num_blocks,
            "kv_block_size": self.block_pool.block_size,
        }

    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking (useful between benchmark runs)."""
        torch.cuda.reset_peak_memory_stats(self.device)

    def shutdown(self) -> None:
        """Shutdown the engine and release resources."""
        import gc

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()


class StreamingResult:
    """
    Iterator for streaming token generation.

    Yields tokens one at a time and tracks generation metadata.
    After iteration completes, prompt_tokens, completion_tokens,
    and finish_reason are available.
    """

    def __init__(
        self,
        request: Request,
        model: TransformerModel,
        block_pool: BlockPool,
        tokenizer: HuggingFaceTokenizer,
        device: str,
        guards: Sequence[GenerationGuard],
        decode_buffers: dict | None = None,
    ):
        self._request = request
        self._model = model
        self._block_pool = block_pool
        self._tokenizer = tokenizer
        self._device = device
        self._guards = guards
        self._decode_buffers = decode_buffers
        self._started = False
        self._finished = False
        self._completion_tokens = 0
        self._last_yielded_len = 0  # Characters already yielded

    @property
    def prompt_tokens(self) -> int:
        """Number of prompt tokens."""
        return len(self._request.prompt_tokens)

    @property
    def completion_tokens(self) -> int:
        """Number of completion tokens generated so far."""
        return self._completion_tokens

    @property
    def finish_reason(self) -> str:
        """Reason generation stopped ('stop' or 'length')."""
        if not self._finished:
            return "stop"  # Default if still running
        # Check if we hit max tokens
        if self._completion_tokens >= self._request.max_tokens:
            return "length"
        return "stop"

    def __iter__(self) -> StreamingResult:
        return self

    def _decode_incremental(self) -> str:
        """Decode all tokens and return only new characters.

        This preserves spaces correctly for SentencePiece tokenizers that
        encode spaces as part of tokens (e.g., '▁world').

        For byte-level tokenizers (like Qwen), incomplete UTF-8 sequences
        appear as replacement characters (U+FFFD). We filter these out and
        don't advance our position past them, so when the complete character
        is decoded on the next token, we'll yield it correctly.
        """
        self._completion_tokens += 1
        full_text = self._tokenizer.decode(
            self._request.output_tokens, skip_special_tokens=True
        )
        new_text = full_text[self._last_yielded_len :]

        # Filter out replacement characters (incomplete UTF-8 from byte-level tokenizers)
        # Only advance position by clean text length so we don't skip the complete char
        clean_text = new_text.replace("\ufffd", "")
        self._last_yielded_len += len(clean_text)

        return clean_text

    def _run_prefill_phase(self) -> str | None:
        """Run prefill and return first token if available."""
        self._request.kv_cache.allocate_for_prompt(
            self._block_pool, len(self._request.prompt_tokens)
        )
        run_prefill(
            self._request,
            self._model,
            self._block_pool,
            self._device,
            self._guards,
        )
        self._started = True
        if self._request.output_tokens:
            return self._decode_incremental()
        return None

    def _run_decode_step(self) -> str:
        """Run one decode step and return generated token."""
        run_decode(
            self._request,
            self._model,
            self._block_pool,
            self._device,
            self._guards,
            buffers=self._decode_buffers,
        )
        if self._request.output_tokens:
            return self._decode_incremental()
        return ""

    def _handle_finished(self, token: str = "") -> str:
        """Handle finished state, raise StopIteration or return final token."""
        self._cleanup()
        if token:
            return token
        raise StopIteration

    def __next__(self) -> str:
        if self._finished:
            raise StopIteration
        try:
            if not self._started:
                if (token := self._run_prefill_phase()) is not None:
                    return token
            if self._request.is_finished:
                return self._handle_finished()
            token = self._run_decode_step()
            if self._request.is_finished:
                return self._handle_finished(token)
            return token
        except StopIteration:
            raise
        except Exception:
            self._cleanup()
            raise

    def _cleanup(self) -> None:
        """Free resources."""
        if not self._finished:
            self._finished = True
            self._request.kv_cache.free_all(self._block_pool)
