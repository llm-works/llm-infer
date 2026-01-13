"""Configuration loading for inference server."""

import os
from dataclasses import dataclass, field
from typing import Any

from appinfra.app.fastapi import ApiConfig, UvicornConfig


@dataclass
class SelectionConfig:
    """Model selection configuration."""

    path: str | None = None  # Path to selection file (written by ops tools)
    default: str | None = None  # Fallback model name if file missing/empty


@dataclass
class ModelConfig:
    """Model configuration."""

    selection: SelectionConfig = field(default_factory=SelectionConfig)
    models_dir: str = ".models"
    path: str | None = None  # Resolved model path (set by serve tool after resolution)


@dataclass
class NativeEngineConfig:
    """Native engine configuration."""

    num_blocks: int = 1024
    block_size: int = 16
    max_batch_size: int = 1
    device: str = "cuda"
    dtype: str = "float16"
    attention_backend: str = "auto"  # auto, flashinfer, naive
    torch_compile: bool = False  # Use torch.compile for reduced CPU overhead
    warmup: bool = False  # Run warmup query on startup


@dataclass
class DispatchConfig:
    """Dispatch configuration."""

    handler: str = "bounded"
    max_pending: int = 10
    poll_timeout: float = 0.01


@dataclass
class BackendsConfig:
    """Backend selection configuration.

    Multi-level backend selection for flexibility in inference optimization:
    - engine: Full inference engine (native | vllm)
    - model: Model implementation (native | gptqmodel) - only if engine=native
    - linear: Kernel implementation (pytorch | marlin) - only if model=native

    Higher levels override lower levels.
    """

    engine: str = "native"  # native | vllm
    model: str = "native"  # native | gptqmodel
    linear: str = "pytorch"  # pytorch | marlin


@dataclass
class VLLMConfig:
    """vLLM engine configuration.

    Comprehensive exposure of vLLM's AsyncEngineArgs for power users.
    Only used when backends.engine = "vllm".
    """

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

    # Trust remote code
    trust_remote_code: bool = True

    # Warmup
    warmup: bool = True  # Run warmup query on startup to eliminate cold-start latency


@dataclass
class EnginesConfig:
    """Engine configurations container.

    Contains configuration for all available engine backends.
    The active engine is selected via backends.engine.
    """

    native: NativeEngineConfig = field(default_factory=NativeEngineConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)


@dataclass
class ThirdPartyLoggingConfig:
    """Third-party library logging levels.

    Controls log verbosity for external libraries:
    - torch: PyTorch (C++ core, Python bindings) - inductor, dynamo, distributed
    - transformers: HuggingFace transformers (Python)

    Note: HuggingFace tokenizers (Rust) prints to stdout and cannot be controlled.
    """

    torch: str = "warning"  # Suppresses inductor/dynamo noise
    transformers: str = "error"  # Suppresses "Special tokens" warning


@dataclass
class InferenceConfig:
    """Complete inference server configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    backends: BackendsConfig = field(default_factory=BackendsConfig)
    engines: EnginesConfig = field(default_factory=EnginesConfig)
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    logging: ThirdPartyLoggingConfig = field(default_factory=ThirdPartyLoggingConfig)

    @classmethod
    def _parse_model_config(cls, data: dict[str, Any]) -> ModelConfig:
        """Parse model configuration from models section."""
        sel = data.get("selection", {}) or {}
        return ModelConfig(
            selection=SelectionConfig(path=sel.get("path"), default=sel.get("default")),
            models_dir=str(data.get("location", ".models")),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceConfig":
        """Create config from dictionary."""
        backends = data.get("backends", {}) or {}
        dispatch = data.get("dispatch", {}) or {}
        logging = data.get("logging", {}) or {}

        return cls(
            model=cls._parse_model_config(data.get("models", {}) or {}),
            backends=BackendsConfig(
                engine=backends.get("engine", "native"),
                model=backends.get("model", "native"),
                linear=backends.get("linear", "pytorch"),
            ),
            engines=cls._parse_engines_config(data.get("engines", {}) or {}),
            dispatch=DispatchConfig(
                handler=dispatch.get("handler", "bounded"),
                max_pending=dispatch.get("max_pending", 10),
                poll_timeout=dispatch.get("poll_timeout", 0.01),
            ),
            api=cls._parse_api_config(data.get("api", {}) or {}),
            logging=ThirdPartyLoggingConfig(
                torch=logging.get("torch", "warning"),
                transformers=logging.get("transformers", "error"),
            ),
        )

    @classmethod
    def _parse_api_config(cls, api_data: dict[str, Any]) -> ApiConfig:
        """Parse API config including nested uvicorn config."""
        uvicorn_data = api_data.get("uvicorn", {}) or {}

        uvicorn_config = UvicornConfig(
            workers=uvicorn_data.get("workers", 1),
            timeout_keep_alive=uvicorn_data.get("timeout_keep_alive", 5),
            limit_concurrency=uvicorn_data.get("limit_concurrency"),
            limit_max_requests=uvicorn_data.get("limit_max_requests"),
            backlog=uvicorn_data.get("backlog", 2048),
            log_level=uvicorn_data.get("log_level", "warning"),
            access_log=uvicorn_data.get("access_log", False),
            ssl_keyfile=uvicorn_data.get("ssl_keyfile"),
            ssl_certfile=uvicorn_data.get("ssl_certfile"),
        )

        return ApiConfig(
            host=api_data.get("host", "0.0.0.0"),
            port=api_data.get("port", 8000),
            title="Inference Server",
            description="LLM inference with process isolation and OpenAI API compatibility",
            version="0.1.0",
            response_timeout=api_data.get("response_timeout", 60.0),
            log_file=api_data.get("log_file"),
            uvicorn=uvicorn_config,
        )

    @classmethod
    def _parse_native_config(cls, data: dict[str, Any]) -> NativeEngineConfig:
        """Parse native engine configuration."""
        return NativeEngineConfig(
            num_blocks=data.get("num_blocks", 1024),
            block_size=data.get("block_size", 16),
            max_batch_size=data.get("max_batch_size", 1),
            device=data.get("device", "cuda"),
            dtype=data.get("dtype", "float16"),
            attention_backend=data.get("attention_backend", "auto"),
            torch_compile=data.get("torch_compile", False),
            warmup=data.get("warmup", False),
        )

    @classmethod
    def _parse_vllm_config(cls, data: dict[str, Any]) -> VLLMConfig:
        """Parse vLLM engine configuration."""
        return VLLMConfig(
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
            warmup=data.get("warmup", True),
        )

    @classmethod
    def _parse_engines_config(cls, engines_data: dict[str, Any]) -> EnginesConfig:
        """Parse engines configuration section."""
        return EnginesConfig(
            native=cls._parse_native_config(engines_data.get("native", {}) or {}),
            vllm=cls._parse_vllm_config(engines_data.get("vllm", {}) or {}),
        )

    def apply_env_overrides(self) -> "InferenceConfig":
        """Apply environment variable overrides."""
        # Native engine overrides
        if env_val := os.environ.get("NUM_BLOCKS"):
            self.engines.native.num_blocks = int(env_val)
        if env_val := os.environ.get("BLOCK_SIZE"):
            self.engines.native.block_size = int(env_val)
        if env_val := os.environ.get("MAX_BATCH_SIZE"):
            self.engines.native.max_batch_size = int(env_val)

        # Dispatch overrides
        if env_val := os.environ.get("MAX_PENDING"):
            self.dispatch.max_pending = int(env_val)
        if env_val := os.environ.get("HANDLER"):
            self.dispatch.handler = env_val

        # API overrides
        if env_val := os.environ.get("HOST"):
            self.api.host = env_val
        if env_val := os.environ.get("PORT"):
            self.api.port = int(env_val)

        return self

    def apply_cli_overrides(
        self,
        host: str | None = None,
        port: int | None = None,
        handler: str | None = None,
        log_file: str | None = None,
        model_path: str | None = None,
        engine: str | None = None,
    ) -> "InferenceConfig":
        """Apply CLI argument overrides."""
        if host is not None:
            self.api.host = host
        if port is not None:
            self.api.port = port
        if handler is not None:
            self.dispatch.handler = handler
        if log_file is not None:
            self.api.log_file = log_file
        if model_path is not None:
            self.model.path = model_path
        if engine is not None:
            self.backends.engine = engine
        return self
