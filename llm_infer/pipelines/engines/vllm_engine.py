"""vLLM-backed inference engine.

This module provides a production-grade inference engine using vLLM's
LLM class for high-performance batched inference with continuous
batching, PagedAttention, and other optimizations.

vLLM is an optional dependency. If not installed, importing this module
will raise ImportError with a helpful message.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from ...serving.dispatch.config import VLLMConfig

# Check for vLLM availability
_VLLM_AVAILABLE = False
_VLLM_ERROR: str | None = None

try:
    from vllm import LLM, SamplingParams

    _VLLM_AVAILABLE = True
except ImportError as e:
    _VLLM_ERROR = str(e)

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest

    from ...context import RequestContext


def _check_vllm_available() -> None:
    """Raise ImportError if vLLM is not available."""
    if not _VLLM_AVAILABLE:
        raise ImportError(
            f"vLLM is not installed. Install it with: pip install vllm\n"
            f"Original error: {_VLLM_ERROR}"
        )


@dataclass
class VLLMStreamingResult:
    """Streaming result wrapper for pre-generated vLLM output.

    Used for non-streaming generation that needs to conform to the
    StreamingResultProtocol interface.
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


class VLLMStreamingIterator:
    """True streaming iterator using vLLM's step-by-step generation.

    Uses the underlying LLMEngine's add_request/step API to yield tokens
    as they are generated, enabling real-time streaming output.

    Supports context manager protocol for reliable cleanup:
        with engine.generate_stream(...) as stream:
            for token in stream:
                print(token)
        # Request is automatically aborted if not fully consumed

    Note: Also implements __del__ for cleanup when not used as context manager,
    but context manager usage is preferred for deterministic cleanup.
    """

    def __init__(
        self,
        llm_engine: Any,
        request_id: str,
        prompt: str,
        sampling_params: Any,
        tokenizer: Any,
        lora_request: Any = None,
    ) -> None:
        self._llm_engine = llm_engine
        self._request_id = request_id
        self._prompt = prompt
        self._sampling_params = sampling_params
        self._tokenizer = tokenizer
        self._lora_request = lora_request

        # State tracking
        self._started = False
        self._finished = False
        self._prev_text = ""

        # Final stats (populated when generation completes)
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.finish_reason: str | None = None

    def _start_generation(self) -> None:
        """Add request to the engine's scheduler."""
        self._llm_engine.add_request(
            request_id=self._request_id,
            prompt=self._prompt,
            params=self._sampling_params,
            lora_request=self._lora_request,
        )
        self.prompt_tokens = len(self._tokenizer.encode(self._prompt))
        self._started = True

    def _process_output(self, output: Any) -> str | None:
        """Process a RequestOutput and return new text delta, or None if done."""
        if not output.outputs:
            return None

        completion = output.outputs[0]
        current_text = completion.text or ""

        # Calculate delta (new text since last step)
        delta = current_text[len(self._prev_text) :]
        self._prev_text = current_text

        # Check if finished
        if output.finished:
            self._finished = True
            self.finish_reason = completion.finish_reason
            self.completion_tokens = len(
                self._tokenizer.encode(current_text, add_special_tokens=False)
            )

        return delta if delta else None

    def __iter__(self) -> Iterator[str]:
        return self

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, ensuring cleanup."""
        self._abort_request()

    def _abort_request(self) -> None:
        """Abort the request in the engine's scheduler (best-effort cleanup)."""
        if self._started and not self._finished:
            try:
                self._llm_engine.abort_request(self._request_id)
            except Exception:
                pass  # Best-effort cleanup

    def __del__(self) -> None:
        """Clean up request on garbage collection."""
        self._abort_request()

    def __next__(self) -> str:
        if self._finished:
            raise StopIteration

        try:
            if not self._started:
                self._start_generation()

            # Step until we get output for our request or generation completes
            while not self._finished:
                outputs = self._llm_engine.step()

                for output in outputs:
                    if output.request_id == self._request_id:
                        delta = self._process_output(output)
                        if delta:
                            return delta
                        if self._finished:
                            raise StopIteration

            raise StopIteration
        except StopIteration:
            raise
        except Exception:
            self._abort_request()
            raise


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

        # Initialize pynvml for device-level GPU stats
        self._nvml_handle = self._init_nvml()

        # Track GPU memory before loading to estimate model size
        mem_before = self._get_device_memory_used()

        # Create the sync LLM engine
        self._engine = LLM(**config.to_llm_kwargs())

        # Estimate model and KV cache memory
        self._init_memory_estimation(mem_before)

        # Get tokenizer for protocol methods
        self._tokenizer = self._engine.get_tokenizer()

        # Determine EOS token
        self._eos_token_id: int | None = None
        if hasattr(self._tokenizer, "eos_token_id"):
            self._eos_token_id = self._tokenizer.eos_token_id

    def _get_physical_device_index(self) -> int:
        """Map torch logical device to pynvml physical device index.

        Handles CUDA_VISIBLE_DEVICES remapping. Falls back to logical index
        if env var uses non-integer format (UUIDs, MIG device IDs).
        """
        import os

        import torch

        if not torch.cuda.is_available():
            return 0
        logical_idx: int = torch.cuda.current_device()
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cuda_visible:
            return logical_idx
        try:
            visible_devices = [int(d.strip()) for d in cuda_visible.split(",")]
            if logical_idx < len(visible_devices):
                return visible_devices[logical_idx]
        except ValueError:
            pass  # Non-integer format (UUID, MIG), fall back to logical index
        return logical_idx

    def _init_nvml(self) -> Any | None:
        """Initialize pynvml and return device handle, or None if unavailable."""
        try:
            import pynvml

            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetHandleByIndex(self._get_physical_device_index())
        except ImportError:
            if self._lg:
                self._lg.warning(
                    "pynvml not available, device-level GPU stats disabled"
                )
        except Exception as e:
            if self._lg:
                self._lg.warning(
                    "pynvml init failed, device-level GPU stats disabled",
                    extra={"exception": e},
                )
        return None

    def _init_memory_estimation(self, mem_before: int | None) -> None:
        """Estimate model and KV cache memory from GPU usage delta."""
        mem_after = self._get_device_memory_used()
        if mem_before is not None and mem_after is not None:
            self._model_memory_bytes = max(0, mem_after - mem_before)
        else:
            self._model_memory_bytes = 0

        # Get KV cache size and block config from vLLM
        self._kv_cache_bytes, self._kv_blocks_total, self._kv_block_size = (
            self._get_kv_cache_info()
        )

        # The delta includes model + KV cache + overhead; subtract KV cache
        if self._kv_cache_bytes > 0 and self._model_memory_bytes > self._kv_cache_bytes:
            self._model_memory_bytes -= self._kv_cache_bytes

        if self._lg:
            self._lg.debug(
                "memory estimation complete",
                extra={
                    "mem_before": mem_before,
                    "mem_after": mem_after,
                    "model_memory": self._model_memory_bytes,
                    "kv_cache": self._kv_cache_bytes,
                },
            )

    def _get_device_memory_used(self) -> int | None:
        """Get current GPU memory usage via pynvml."""
        if self._nvml_handle is None:
            return None
        try:
            import pynvml

            info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return int(info.used)
        except Exception:
            return None

    def _get_configs_from_engine(self) -> tuple[Any, Any] | None:
        """Extract cache_config and model_config from vLLM engine (V0/V1)."""
        if not hasattr(self._engine, "llm_engine"):
            return None
        llm_engine = self._engine.llm_engine

        cache_config = getattr(llm_engine, "cache_config", None)
        model_config = getattr(llm_engine, "model_config", None)

        # Fallback to vllm_config if direct attributes not found
        if not cache_config or not model_config:
            vllm_config = getattr(llm_engine, "vllm_config", None)
            if vllm_config:
                cache_config = cache_config or getattr(
                    vllm_config, "cache_config", None
                )
                model_config = model_config or getattr(
                    vllm_config, "model_config", None
                )

        if cache_config and model_config:
            return cache_config, model_config
        return None

    def _get_kv_cache_info(self) -> tuple[int, int, int]:
        """Extract KV cache size and block config from vLLM engine."""
        try:
            configs = self._get_configs_from_engine()
            if configs:
                cache_config, model_config = configs
                if (
                    hasattr(cache_config, "num_gpu_blocks")
                    and cache_config.num_gpu_blocks
                ):
                    block_size = getattr(cache_config, "block_size", 16)
                    blocks_total = cache_config.num_gpu_blocks
                    kv_bytes = self._calculate_kv_cache_bytes(
                        cache_config, model_config
                    )
                    return kv_bytes, blocks_total, block_size
        except Exception as e:
            if self._lg:
                self._lg.warning("Failed to get KV cache info", extra={"exception": e})
        return 0, 0, 0

    def _get_arch_from_hf_config(self, hf_config: Any) -> tuple[int, int, int] | None:
        """Extract model architecture from HuggingFace config.

        Returns:
            Tuple of (num_layers, num_heads, head_dim) or None if extraction fails.
        """
        num_layers = getattr(hf_config, "num_hidden_layers", None)
        num_heads = getattr(hf_config, "num_key_value_heads", None)
        if num_heads is None:
            num_heads = getattr(hf_config, "num_attention_heads", None)
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None:
            hidden = getattr(hf_config, "hidden_size", None)
            n_heads = getattr(hf_config, "num_attention_heads", None)
            if hidden and n_heads:
                head_dim = hidden // n_heads

        if num_layers and num_heads and head_dim:
            return num_layers, num_heads, head_dim
        return None

    def _get_arch_from_vllm_config(
        self, model_config: Any
    ) -> tuple[int, int, int] | None:
        """Extract model architecture from vLLM config methods (V0/V1)."""
        # V1: direct methods
        try:
            return (
                model_config.get_num_layers(),
                model_config.get_num_kv_heads(),
                model_config.get_head_size(),
            )
        except (TypeError, AttributeError):
            pass

        # V0: methods with parallel_config
        if hasattr(model_config, "parallel_config"):
            pc = model_config.parallel_config
            return (
                model_config.get_num_layers(pc),
                model_config.get_num_kv_heads(pc),
                model_config.get_head_size(),
            )
        return None

    def _get_kv_cache_dtype_size(self, cache_config: Any) -> int:
        """Get KV cache element size in bytes from cache config."""
        dtype_sizes = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "fp8": 1,
            "fp8_e4m3": 1,
            "fp8_e5m2": 1,
        }
        # vLLM uses cache_dtype attribute
        cache_dtype = getattr(cache_config, "cache_dtype", None)
        if cache_dtype and cache_dtype != "auto":
            return dtype_sizes.get(str(cache_dtype).lower(), 2)
        return 2  # Default to float16

    def _calculate_kv_cache_bytes(self, cache_config: Any, model_config: Any) -> int:
        """Calculate KV cache size in bytes from vLLM configs."""
        try:
            block_size = cache_config.block_size
            num_blocks = cache_config.num_gpu_blocks
            dtype_size = self._get_kv_cache_dtype_size(cache_config)

            # Try HuggingFace config first (most reliable)
            hf_config = getattr(model_config, "hf_config", None)
            if hf_config:
                arch = self._get_arch_from_hf_config(hf_config)
                if arch:
                    return self._compute_kv_cache_total(
                        *arch, block_size, num_blocks, dtype_size
                    )

            # Fallback to vLLM model config methods
            arch = self._get_arch_from_vllm_config(model_config)
            if arch:
                return self._compute_kv_cache_total(
                    *arch, block_size, num_blocks, dtype_size
                )

        except Exception as e:
            if self._lg:
                self._lg.warning("KV cache calculation failed", extra={"exception": e})
        return 0

    def _compute_kv_cache_total(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        dtype_size: int = 2,
    ) -> int:
        """Compute KV cache total bytes from architecture params."""
        # KV cache per block = 2 * num_layers * num_heads * head_dim * block_size * dtype_size
        kv_per_block = 2 * num_layers * num_heads * head_dim * block_size * dtype_size
        total = num_blocks * kv_per_block

        if self._lg:
            self._lg.debug(
                "KV cache calculated",
                extra={
                    "layers": num_layers,
                    "heads": num_heads,
                    "head_dim": head_dim,
                    "blocks": num_blocks,
                    "block_size": block_size,
                    "dtype_size": dtype_size,
                    "total_gb": round(total / 1e9, 2),
                },
            )
        return total

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
        vllm_data = config.get("vllm", {}) or {}
        vllm_config = VLLMConfig.from_dict(vllm_data, model_path)
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
        lora_request: LoRARequest | None = None,
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
            lora_request: Optional vLLM LoRARequest for adapter inference

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
            lora_request=lora_request,
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
        lora_request: LoRARequest | None = None,
    ) -> VLLMStreamingIterator:
        """Generate text with true token-by-token streaming.

        Uses vLLM's underlying LLMEngine add_request/step API to yield
        tokens as they are generated, enabling real-time streaming.

        Returns:
            VLLMStreamingIterator that yields token strings and has
            prompt_tokens, completion_tokens, finish_reason attributes
            after iteration completes.
        """
        final_prompt = self._prepare_prompt(prompt, messages, use_chat_template)
        sampling_params = self._create_sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
        )

        # Generate unique request ID for this streaming request
        request_id = f"stream-{uuid.uuid4().hex[:16]}"

        return VLLMStreamingIterator(
            llm_engine=self._engine.llm_engine,
            request_id=request_id,
            prompt=final_prompt,
            sampling_params=sampling_params,
            tokenizer=self._tokenizer,
            lora_request=lora_request,
        )

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
                # Truncate for Matryoshka embeddings and renormalize
                embedding = embedding[:dimensions]
                norm = math.sqrt(sum(x * x for x in embedding))
                if norm > 0:
                    embedding = [x / norm for x in embedding]
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

    def _fetch_device_memory_stats(self) -> tuple[int, int, int] | None:
        """Fetch device memory stats via pynvml. Returns (used, total, free)."""
        if self._nvml_handle is None:
            return None
        try:
            import pynvml

            info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return int(info.used), int(info.total), int(info.free)
        except Exception:
            return None

    def _fetch_kv_cache_usage(self) -> tuple[float, int] | None:
        """Fetch KV cache usage from vLLM metrics. Returns (usage_perc, blocks_used)."""
        try:
            for metric in self._engine.get_metrics():
                if not hasattr(metric, "value"):
                    continue
                if metric.name == "vllm:kv_cache_usage_perc":
                    blocks_used = 0
                    if self._kv_blocks_total > 0:
                        blocks_used = int(metric.value * self._kv_blocks_total)
                    return metric.value, blocks_used
        except Exception as e:
            if self._lg:
                self._lg.debug("Failed to fetch KV cache usage", extra={"exception": e})
        return None

    def _fetch_torch_memory_stats(self) -> tuple[int, int, int]:
        """Fetch torch CUDA memory stats. Returns (allocated, reserved, peak)."""
        import torch

        if torch.cuda.is_available():
            return (
                torch.cuda.memory_allocated(),
                torch.cuda.memory_reserved(),
                torch.cuda.max_memory_allocated(),
            )
        return (0, 0, 0)

    def memory_stats(self) -> dict[str, int | float]:
        """Return memory statistics."""
        allocated, reserved, peak = self._fetch_torch_memory_stats()
        device_stats = self._fetch_device_memory_stats()
        kv_usage = self._fetch_kv_cache_usage()

        stats: dict[str, int | float] = {
            "allocated": allocated,
            "reserved": reserved,
            "peak": peak,
            "model_memory": self._model_memory_bytes,
            "kv_cache_bytes": self._kv_cache_bytes,
            "kv_blocks_used": kv_usage[1] if kv_usage else 0,
            "kv_blocks_total": self._kv_blocks_total,
            "kv_block_size": self._kv_block_size,
            "device_used": device_stats[0] if device_stats else 0,
            "device_total": device_stats[1] if device_stats else 0,
            "device_free": device_stats[2] if device_stats else 0,
            "kv_cache_usage_perc": kv_usage[0] if kv_usage else 0.0,
        }
        return stats

    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _shutdown_nvml(self) -> None:
        """Shutdown pynvml if initialized."""
        if self._nvml_handle is not None:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_handle = None

    def shutdown(self) -> None:
        """Shutdown the vLLM engine and release resources."""
        import gc

        import torch

        # Destroy distributed process groups (fixes NCCL warning)
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass

        # vLLM model parallel cleanup
        try:
            from vllm.distributed import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self._shutdown_nvml()
