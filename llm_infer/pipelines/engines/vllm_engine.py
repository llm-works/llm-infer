"""vLLM-backed inference engine.

This module provides a production-grade inference engine using vLLM's
LLM class for high-performance batched inference with continuous
batching, PagedAttention, and other optimizations.

vLLM is an optional dependency. If not installed, importing this module
will raise ImportError with a helpful message.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Check for vLLM availability
_VLLM_AVAILABLE = False
_VLLM_ERROR: str | None = None

try:
    from vllm import LLM, SamplingParams

    _VLLM_AVAILABLE = True
except ImportError as e:
    _VLLM_ERROR = str(e)

if TYPE_CHECKING:
    from ...context import RequestContext


def _check_vllm_available() -> None:
    """Raise ImportError if vLLM is not available."""
    if not _VLLM_AVAILABLE:
        raise ImportError(
            f"vLLM is not installed. Install it with: pip install vllm\n"
            f"Original error: {_VLLM_ERROR}"
        )


@dataclass
class VLLMConfig:
    """Configuration for vLLM engine.

    Comprehensive exposure of vLLM's AsyncEngineArgs for power users.
    All fields have sensible defaults matching vLLM's defaults.
    """

    # Model (required, set from inference config)
    model_path: str = ""

    # Task mode: "generate" for LLM, "embed" for embedding models
    task: str = "generate"

    # Memory management
    gpu_memory_utilization: float = 0.9
    cpu_offload_gb: float = 0.0
    swap_space: int = 4  # GB
    max_model_len: int | None = None  # Max context length (None = use model default)

    # Parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Scheduling
    max_num_seqs: int = 256
    max_num_batched_tokens: int | None = None  # Auto-calculated if None
    scheduling_policy: str = "fcfs"  # fcfs, priority

    # Caching
    enable_prefix_caching: bool = True
    kv_cache_dtype: str = "auto"

    # Performance tuning
    enforce_eager: bool = False  # Disable CUDA graph for debugging
    disable_custom_all_reduce: bool = False

    # Quantization (auto-detected from model, but can override)
    quantization: str | None = None  # awq, gptq, fp8, etc.

    # Speculative decoding (advanced)
    speculative_model: str | None = None
    num_speculative_tokens: int | None = None

    # Dtype
    dtype: str = "auto"  # auto, float16, bfloat16, float32

    # Trust remote code (for custom models)
    trust_remote_code: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any], model_path: str) -> VLLMConfig:
        """Create config from dictionary (vllm section of llm-infer.yaml)."""
        return cls(
            model_path=model_path,
            task=data.get("task", "generate"),
            gpu_memory_utilization=data.get("gpu_memory_utilization", 0.9),
            cpu_offload_gb=data.get("cpu_offload_gb", 0.0),
            swap_space=data.get("swap_space", 4),
            max_model_len=data.get("max_model_len"),
            tensor_parallel_size=data.get("tensor_parallel_size", 1),
            pipeline_parallel_size=data.get("pipeline_parallel_size", 1),
            max_num_seqs=data.get("max_num_seqs", 256),
            max_num_batched_tokens=data.get("max_num_batched_tokens"),
            scheduling_policy=data.get("scheduling_policy", "fcfs"),
            enable_prefix_caching=data.get("enable_prefix_caching", True),
            kv_cache_dtype=data.get("kv_cache_dtype", "auto"),
            enforce_eager=data.get("enforce_eager", False),
            disable_custom_all_reduce=data.get("disable_custom_all_reduce", False),
            quantization=data.get("quantization"),
            speculative_model=data.get("speculative_model"),
            num_speculative_tokens=data.get("num_speculative_tokens"),
            dtype=data.get("dtype", "auto"),
            trust_remote_code=data.get("trust_remote_code", True),
        )

    def to_llm_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for vLLM LLM constructor."""
        _check_vllm_available()

        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "swap_space": self.swap_space,
            "tensor_parallel_size": self.tensor_parallel_size,
            "enforce_eager": self.enforce_eager,
            "disable_custom_all_reduce": self.disable_custom_all_reduce,
            "trust_remote_code": self.trust_remote_code,
        }

        # Task mode for embedding models
        if self.task == "embed":
            kwargs["task"] = "embed"

        # Max model length (context window)
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len

        # Optional fields
        if self.quantization is not None:
            kwargs["quantization"] = self.quantization

        return kwargs


@dataclass
class VLLMStreamingResult:
    """Streaming result wrapper for vLLM output.

    Implements the StreamingResultProtocol expected by handlers.
    """

    _tokens: list[str] = field(default_factory=list)
    _current_idx: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        if self._current_idx >= len(self._tokens):
            raise StopIteration
        token = self._tokens[self._current_idx]
        self._current_idx += 1
        return token


class VLLMEngine:
    """vLLM-backed inference engine implementing InferenceEngineProtocol.

    This engine wraps vLLM's AsyncLLMEngine to provide high-performance
    inference with continuous batching, PagedAttention, and other
    optimizations.

    The engine handles the async/sync bridge internally, providing a
    synchronous interface compatible with existing handlers.
    """

    def __init__(self, config: VLLMConfig, lg: Any = None):
        """Initialize vLLM engine.

        Args:
            config: vLLM configuration
            lg: Optional logger
        """
        _check_vllm_available()

        self._config = config
        self._lg = lg

        # Get kwargs for LLM constructor
        llm_kwargs = config.to_llm_kwargs()

        # Create the sync LLM engine
        self._engine = LLM(**llm_kwargs)

        # Get tokenizer for protocol methods
        self._tokenizer = self._engine.get_tokenizer()

        # Determine EOS token
        self._eos_token_id: int | None = None
        if hasattr(self._tokenizer, "eos_token_id"):
            self._eos_token_id = self._tokenizer.eos_token_id

    @property
    def model_name(self) -> str:
        """Return model name extracted from path."""
        from pathlib import Path

        return Path(self._config.model_path).name

    @classmethod
    def from_config(cls, config: dict[str, Any], lg: Any = None) -> VLLMEngine:
        """Create engine from inference config dictionary.

        Args:
            config: Full inference config dict with 'model' and 'vllm' sections
            lg: Optional logger

        Returns:
            Initialized VLLMEngine
        """
        model_path = config.get("model", {}).get("path", "")
        vllm_config_data = config.get("vllm", {}) or {}

        vllm_config = VLLMConfig.from_dict(vllm_config_data, model_path)
        return cls(vllm_config, lg)

    @property
    def eos_token_id(self) -> int | None:
        """End-of-sequence token ID."""
        return self._eos_token_id

    # -------------------------------------------------------------------------
    # InferenceEngineProtocol: Tokenization methods
    # -------------------------------------------------------------------------

    def count_tokens(self, text: str, use_chat_template: bool | None = None) -> int:
        """Count tokens in text."""
        tokens = self.tokenize(text, use_chat_template)
        return len(tokens)

    def tokenize(self, text: str, use_chat_template: bool | None = None) -> list[int]:
        """Tokenize text to token IDs."""
        if use_chat_template is None:
            use_chat_template = self.should_use_chat_template()

        if use_chat_template:
            # Apply chat template - tokenize=False returns str
            messages: list[Any] = [{"role": "user", "content": text}]
            text = str(
                self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        return list(self._tokenizer.encode(text))

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        return str(self._tokenizer.decode(tokens, skip_special_tokens=True))

    def build_stop_token_ids(self, stop_sequences: list[str] | None) -> set[int]:
        """Build set of stop token IDs from EOS and stop sequences."""
        stop_ids: set[int] = set()

        if self._eos_token_id is not None:
            stop_ids.add(self._eos_token_id)

        if stop_sequences:
            for seq in stop_sequences:
                # Encode each stop sequence and add first token
                # (vLLM handles multi-token stop sequences differently)
                tokens = self._tokenizer.encode(seq, add_special_tokens=False)
                if tokens:
                    stop_ids.add(tokens[0])

        return stop_ids

    def should_use_chat_template(self) -> bool:
        """Check if chat template should be used based on model."""
        if not hasattr(self._tokenizer, "chat_template"):
            return False
        if self._tokenizer.chat_template is None:
            return False

        # Check model name for instruct/chat indicators
        model_name = self._config.model_path.lower()
        return "instruct" in model_name or "chat" in model_name

    # -------------------------------------------------------------------------
    # InferenceEngineProtocol: Generation methods
    # -------------------------------------------------------------------------

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
        """Generate text completion (blocking).

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            use_chat_template: Whether to apply chat template
            stop_sequences: Sequences that stop generation
            context: Request context for logging
            messages: Chat messages (alternative to prompt)

        Returns:
            Generated text
        """
        # Prepare prompt
        final_prompt = self._prepare_prompt(prompt, messages, use_chat_template)

        # Create sampling params
        sampling_params = self._create_sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
        )

        # Generate using sync API
        outputs = self._engine.generate(
            prompts=[final_prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # Extract text from output
        if outputs and outputs[0].outputs:
            return str(outputs[0].outputs[0].text)
        return ""

    def _build_streaming_result(self, outputs: Any, prompt: str) -> VLLMStreamingResult:
        """Build VLLMStreamingResult from generation outputs."""
        result = VLLMStreamingResult()
        result.prompt_tokens = len(self._tokenizer.encode(prompt))
        if outputs and outputs[0].outputs:
            output = outputs[0].outputs[0]
            if text := output.text:
                # Stream char-by-char for simulated streaming (avoids multi-byte char issues)
                result._tokens.extend(text)
                result.completion_tokens = len(
                    self._tokenizer.encode(text, add_special_tokens=False)
                )
            result.finish_reason = output.finish_reason
        return result

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
    ) -> VLLMStreamingResult:
        """Generate text with streaming (sync wrapper)."""
        final_prompt = self._prepare_prompt(prompt, messages, use_chat_template)
        sampling_params = self._create_sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
        )
        outputs = self._engine.generate(
            prompts=[final_prompt], sampling_params=sampling_params, use_tqdm=False
        )
        return self._build_streaming_result(outputs, final_prompt)

    def _prepare_prompt(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None,
        use_chat_template: bool | None,
    ) -> str:
        """Prepare final prompt from input."""
        if use_chat_template is None:
            use_chat_template = self.should_use_chat_template()

        if messages:
            # Use messages directly with chat template - tokenize=False returns str
            chat_messages: list[Any] = list(messages)
            return str(
                self._tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True
                )
            )
        elif use_chat_template:
            # Wrap prompt in user message - tokenize=False returns str
            chat_messages = [{"role": "user", "content": prompt}]
            return str(
                self._tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True
                )
            )
        else:
            return prompt

    def _create_sampling_params(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop_sequences: list[str] | None,
    ) -> SamplingParams:
        """Create vLLM SamplingParams."""
        _check_vllm_available()

        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }

        # top_k: vLLM uses -1 for disabled, we use 0
        if top_k > 0:
            kwargs["top_k"] = top_k

        # Stop sequences
        if stop_sequences:
            kwargs["stop"] = stop_sequences

        return SamplingParams(**kwargs)

    # -------------------------------------------------------------------------
    # Embedding methods (for task="embed" mode)
    # -------------------------------------------------------------------------

    def supports_embeddings(self) -> bool:
        """Check if engine is configured for embeddings."""
        return self._config.task == "embed"

    def embed(
        self,
        inputs: list[str],
        dimensions: int | None = None,
    ) -> tuple[list[list[float]], int]:
        """Generate embeddings for input texts.

        Args:
            inputs: List of texts to embed
            dimensions: Optional output dimensions (for Matryoshka embeddings)

        Returns:
            Tuple of (embeddings list, total tokens)

        Raises:
            RuntimeError: If engine not in embed mode
        """
        if self._config.task != "embed":
            raise RuntimeError(
                "Engine not configured for embeddings. Set engines.vllm.task='embed'"
            )

        # Generate embeddings using vLLM's embed API
        outputs = self._engine.embed(inputs)

        # Extract embeddings and count tokens
        embeddings: list[list[float]] = []
        total_tokens = 0

        for output in outputs:
            # Get embedding from output
            embedding = output.outputs.embedding
            if dimensions is not None:
                # Truncate for Matryoshka embeddings
                embedding = embedding[:dimensions]
            embeddings.append(list(embedding))

            # Count prompt tokens (prompt_token_ids can be None)
            total_tokens += (
                len(output.prompt_token_ids) if output.prompt_token_ids else 0
            )

        return embeddings, total_tokens

    # -------------------------------------------------------------------------
    # InferenceEngineProtocol: Batched processing methods
    # -------------------------------------------------------------------------
    # Note: vLLM handles batching internally. These methods are simplified
    # implementations that work with the existing handler interface but
    # delegate actual batching to vLLM.

    def prefill_request(self, request: Any) -> None:
        """Submit request to vLLM for processing.

        With vLLM, prefill is handled automatically during generate().
        This method is a no-op placeholder for protocol compatibility.
        """
        # vLLM handles prefill internally during generation
        pass

    def step_decode(self, requests: list[Any]) -> list[int | None]:
        """Batched decode step.

        With vLLM, decoding is handled by the engine's internal scheduler.
        This method is a placeholder for protocol compatibility.

        Returns:
            List of None (tokens are handled by generate())
        """
        return [None] * len(requests)

    def free_request(self, request: Any) -> None:
        """Free resources for completed request.

        With vLLM, resources are managed automatically.
        This method can abort an in-flight request if needed.
        """
        # vLLM manages cleanup automatically
        pass

    # -------------------------------------------------------------------------
    # InferenceEngineProtocol: Utility methods
    # -------------------------------------------------------------------------

    def memory_stats(self) -> dict[str, int]:
        """Return memory statistics.

        Note: vLLM manages memory internally, these are approximate.
        """
        import torch

        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "peak": torch.cuda.max_memory_allocated(),
                "kv_cache_bytes": 0,  # vLLM manages internally
                "kv_blocks_used": 0,
                "kv_blocks_total": 0,
                "kv_block_size": 0,
            }
        return {
            "allocated": 0,
            "reserved": 0,
            "peak": 0,
            "kv_cache_bytes": 0,
            "kv_blocks_used": 0,
            "kv_blocks_total": 0,
            "kv_block_size": 0,
        }

    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def shutdown(self) -> None:
        """Shutdown the vLLM engine and release resources.

        vLLM's cleanup is notoriously incomplete. We do best-effort:
        1. Destroy distributed process groups
        2. Clear CUDA cache
        3. Run garbage collection
        """
        import gc

        import torch

        # Try to destroy distributed process groups (fixes NCCL warning)
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass  # May already be destroyed

        # Try vLLM's model parallel cleanup
        try:
            from vllm.distributed import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            pass  # May not be initialized

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()
