"""Configuration loading for inference server."""

from dataclasses import dataclass, field
from typing import Any

from appinfra.app.fastapi import ApiConfig, UvicornConfig

from ...models import ModelsConfig
from .config_overrides import CliOverrides, apply_standard_overrides


@dataclass
class LoRAConfig:
    """LoRA adapter configuration for vLLM engine."""

    enabled: bool = False  # Enable LoRA/QLoRA adapter support
    max_loras: int = 4  # Maximum concurrent adapters in GPU memory
    max_lora_rank: int = 64  # Maximum LoRA rank supported
    base_path: str | None = None  # Base directory for adapter weights


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
    batch_streaming: bool = False  # Allow streaming requests in batched decode


@dataclass
class BackendsConfig:
    """Backend selection configuration.

    Multi-level backend selection for flexibility in inference optimization:
    - engine: Full inference engine (native | vllm | ollama)
    - model: Model implementation (native | gptqmodel) - only if engine=native
    - linear: Kernel implementation (pytorch | marlin) - only if model=native

    Higher levels override lower levels.
    """

    engine: str = "native"  # native | vllm | ollama
    model: str = "native"  # native | gptqmodel
    linear: str = "pytorch"  # pytorch | marlin


@dataclass
class OllamaConfig:
    """Ollama engine configuration.

    Configuration for the Ollama backend. llm-infer will automatically
    start/stop the Ollama server with the configured models_path.
    """

    # Model name (as known to Ollama, e.g., "llama3.2", "qwen2.5:7b")
    model: str = ""

    # Task mode: "generate" for LLM, "embed" for embedding models
    task: str = "generate"

    # Server connection
    host: str = "http://localhost:11434"
    timeout: int = 300  # Request timeout in seconds

    # Model storage path (sets OLLAMA_MODELS when starting server)
    models_path: str | None = None

    # Model lifecycle
    keep_alive: str | None = (
        "5m"  # How long to keep model loaded (e.g., "5m", "1h", "0" to unload immediately)
    )

    # Generation options
    num_ctx: int | None = None  # Context window size (None = model default)
    num_gpu: int | None = None  # Number of GPU layers (None = auto, 0 = CPU only)

    # Warmup
    warmup: bool = True  # Run warmup query on startup

    # Server management
    auto_start: bool = True  # Automatically start Ollama server if not running
    binary_path: str = "ollama"  # Path to ollama binary

    @classmethod
    def from_dict(cls, data: dict[str, Any], model: str = "") -> "OllamaConfig":
        """Create config from dictionary (ollama section of config file).

        Args:
            data: Dictionary with Ollama config values.
            model: Model name (from CLI or config).

        Returns:
            OllamaConfig instance with values from dict, defaults for missing keys.
        """
        return cls(
            model=model or data.get("model", ""),
            task=data.get("task", "generate"),
            host=data.get("host", "http://localhost:11434"),
            timeout=data.get("timeout", 300),
            models_path=data.get("models_path"),
            keep_alive=data.get("keep_alive", "5m"),
            num_ctx=data.get("num_ctx"),
            num_gpu=data.get("num_gpu"),
            warmup=data.get("warmup", True),
            auto_start=data.get("auto_start", True),
            binary_path=data.get("binary_path", "ollama"),
        )


@dataclass
class VLLMConfig:
    """vLLM engine configuration.

    Comprehensive exposure of vLLM's AsyncEngineArgs for power users.
    Used both for configuration storage (engines.vllm in YAML) and
    runtime engine initialization.
    """

    # Model path (set at runtime, not in YAML config)
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
    disable_log_stats: bool = False  # False=stats enabled (for get_metrics() API)
    max_cudagraph_capture_size: int | None = (
        None  # Limit batch sizes for CUDA graph capture (lower = faster startup)
    )

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

    # LoRA adapter support (nested config)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any], model_path: str = "") -> "VLLMConfig":
        """Create config from dictionary (vllm section of config file).

        Args:
            data: Dictionary with vLLM config values.
            model_path: Path to model weights.

        Returns:
            VLLMConfig instance with values from dict, defaults for missing keys.
        """
        from dataclasses import fields

        kwargs: dict[str, Any] = {"model_path": model_path}
        for f in fields(cls):
            if f.name != "model_path" and f.name in data:
                kwargs[f.name] = data[f.name]
        return cls(**kwargs)

    def to_llm_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for vLLM LLM constructor.

        Returns a dict suitable for passing to vLLM's LLM(**kwargs).
        Note: vLLM availability is checked by VLLMEngineFactory before this is called.
        """
        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "cpu_offload_gb": self.cpu_offload_gb,
            "swap_space": self.swap_space,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "max_num_seqs": self.max_num_seqs,
            "scheduling_policy": self.scheduling_policy,
            "enable_prefix_caching": self.enable_prefix_caching,
            "kv_cache_dtype": self.kv_cache_dtype,
            "enforce_eager": self.enforce_eager,
            "disable_custom_all_reduce": self.disable_custom_all_reduce,
            "disable_log_stats": self.disable_log_stats,
            "trust_remote_code": self.trust_remote_code,
        }
        self._add_optional_kwargs(kwargs)
        return kwargs

    def _add_optional_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Add optional and conditional vLLM kwargs.

        Handles fields that are only set when:
        - Not None (vLLM auto-calculates these when omitted)
        - Conditionally enabled (task mode, LoRA)
        """
        # Fields that vLLM auto-calculates when None
        optional_fields = [
            ("max_cudagraph_capture_size", self.max_cudagraph_capture_size),
            ("max_model_len", self.max_model_len),
            ("max_num_batched_tokens", self.max_num_batched_tokens),
            ("quantization", self.quantization),
            ("speculative_model", self.speculative_model),
            ("num_speculative_tokens", self.num_speculative_tokens),
        ]
        for key, value in optional_fields:
            if value is not None:
                kwargs[key] = value

        # Task mode for embedding models (vLLM <0.14 only, now auto-detected)
        # if self.task == "embed":
        #     kwargs["task"] = "embed"

        # LoRA settings (not supported for embedding models)
        if self.lora.enabled and self.task != "embed":
            kwargs["enable_lora"] = True
            kwargs["max_loras"] = self.lora.max_loras
            kwargs["max_lora_rank"] = self.lora.max_lora_rank


@dataclass
class EnginesConfig:
    """Engine configurations container.

    Contains configuration for all available engine backends.
    The active engine is selected via backends.engine.
    """

    native: NativeEngineConfig = field(default_factory=NativeEngineConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


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

    models: ModelsConfig = field(default_factory=ModelsConfig)
    backends: BackendsConfig = field(default_factory=BackendsConfig)
    engines: EnginesConfig = field(default_factory=EnginesConfig)
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    logging: ThirdPartyLoggingConfig = field(default_factory=ThirdPartyLoggingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceConfig":
        """Create config from dictionary."""
        backends = data.get("backends", {}) or {}
        dispatch = data.get("dispatch", {}) or {}
        logging = data.get("logging", {}) or {}

        return cls(
            models=ModelsConfig.from_dict(data.get("models", {}) or {}),
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
                batch_streaming=dispatch.get("batch_streaming", False),
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
            disable_log_stats=data.get("disable_log_stats", False),
            max_cudagraph_capture_size=data.get("max_cudagraph_capture_size"),
            quantization=data.get("quantization"),
            speculative_model=data.get("speculative_model"),
            num_speculative_tokens=data.get("num_speculative_tokens"),
            dtype=data.get("dtype", "auto"),
            trust_remote_code=data.get("trust_remote_code", True),
            warmup=data.get("warmup", True),
            lora=cls._parse_lora_config(data.get("lora", {}) or {}),
        )

    @classmethod
    def _parse_lora_config(cls, data: dict[str, Any]) -> LoRAConfig:
        """Parse LoRA configuration section."""
        return LoRAConfig(
            enabled=data.get("enabled", False),
            max_loras=data.get("max_loras", 4),
            max_lora_rank=data.get("max_lora_rank", 64),
            base_path=data.get("base_path"),
        )

    @classmethod
    def _parse_ollama_config(cls, data: dict[str, Any]) -> OllamaConfig:
        """Parse Ollama engine configuration."""
        return OllamaConfig(
            model=data.get("model", ""),
            task=data.get("task", "generate"),
            host=data.get("host", "http://localhost:11434"),
            timeout=data.get("timeout", 300),
            models_path=data.get("models_path"),
            keep_alive=data.get("keep_alive", "5m"),
            num_ctx=data.get("num_ctx"),
            num_gpu=data.get("num_gpu"),
            warmup=data.get("warmup", True),
            auto_start=data.get("auto_start", True),
            binary_path=data.get("binary_path", "ollama"),
        )

    @classmethod
    def _parse_engines_config(cls, engines_data: dict[str, Any]) -> EnginesConfig:
        """Parse engines configuration section."""
        return EnginesConfig(
            native=cls._parse_native_config(engines_data.get("native", {}) or {}),
            vllm=cls._parse_vllm_config(engines_data.get("vllm", {}) or {}),
            ollama=cls._parse_ollama_config(engines_data.get("ollama", {}) or {}),
        )

    def apply_env_overrides(self) -> "InferenceConfig":
        """Apply environment variable overrides.

        Uses EnvConfigOverride strategy for env -> config mapping.
        """
        return apply_standard_overrides(self, cli_overrides=None)

    def apply_cli_overrides(
        self,
        host: str | None = None,
        port: int | None = None,
        handler: str | None = None,
        log_file: str | None = None,
        model_path: str | None = None,
        engine: str | None = None,
        overrides: dict[str, str] | None = None,
    ) -> "InferenceConfig":
        """Apply CLI argument overrides.

        Uses CliConfigOverride strategy. CLI takes precedence over env.

        Args:
            overrides: Generic key=value overrides using dotted paths,
                e.g. {"engines.vllm.gpu_memory_utilization": "0.05"}
        """
        cli = CliOverrides(
            host=host,
            port=port,
            handler=handler,
            log_file=log_file,
            model_path=model_path,
            engine=engine,
            generic=overrides,
        )
        return apply_standard_overrides(self, cli_overrides=cli)
