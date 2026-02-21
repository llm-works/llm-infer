"""Factory classes for creating engines and handlers.

Uses class-based registry pattern for clean extensibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

if TYPE_CHECKING:
    from .config import InferenceConfig
    from .handler import RequestHandler


# ---------------------------------------------------------------------------
# Engine Factories
# ---------------------------------------------------------------------------


class EngineFactory(ABC):
    """Base class for engine factories."""

    @abstractmethod
    def create(
        self, lg: Logger, config: InferenceConfig, on_progress: Any = None
    ) -> Any:
        """Create an engine instance."""
        ...

    @abstractmethod
    def warmup_enabled(self, config: InferenceConfig) -> bool:
        """Check if warmup is enabled for this engine type."""
        ...

    @abstractmethod
    def max_batch_size(self, config: InferenceConfig) -> int:
        """Get max batch size for handler configuration."""
        ...


class NativeEngineFactory(EngineFactory):
    """Factory for native inference engine."""

    def _import_native_deps(self) -> tuple[Any, Any, Any]:
        """Import native engine dependencies with helpful error message."""
        try:
            from ...engines.native import (
                EngineConfig,
                InferenceEngine,
                TransformerConfig,
            )

            return EngineConfig, InferenceEngine, TransformerConfig
        except ImportError as e:
            raise ImportError(
                "Native engine requested (backends.engine=native) but torch is not installed. "
                "Install with: pip install llm-infer[runtime]\n"
                "Or use Ollama engine: llm-infer serve --engine ollama"
            ) from e

    def _validate_model_path(self, config: InferenceConfig) -> None:
        """Validate model path is set."""
        if config.models.path is None:
            raise ValueError(
                "models.path is required for native engine "
                "(set via config, --model-path, or MODEL_PATH)"
            )

    def create(
        self, lg: Logger, config: InferenceConfig, on_progress: Any = None
    ) -> Any:
        engine_config_cls, engine_cls, transformer_config_cls = (
            self._import_native_deps()
        )
        self._validate_model_path(config)

        native_cfg = config.engines.native
        model_path = str(config.models.path)
        model_config = transformer_config_cls.from_hf_config(model_path)
        engine_config = engine_config_cls(
            model=model_config,
            model_path=model_path,
            num_blocks=native_cfg.num_blocks,
            block_size=native_cfg.block_size,
            max_batch_size=native_cfg.max_batch_size,
            attention_backend=native_cfg.attention_backend,
            linear_backend=config.backends.linear,
            torch_compile=native_cfg.torch_compile,
            warmup=native_cfg.warmup,
        )
        return engine_cls(lg, engine_config, on_progress=on_progress)

    def warmup_enabled(self, config: InferenceConfig) -> bool:
        return config.engines.native.warmup

    def max_batch_size(self, config: InferenceConfig) -> int:
        return config.engines.native.max_batch_size


class VLLMEngineFactory(EngineFactory):
    """Factory for vLLM-backed inference engine."""

    def _validate_model_path(self, config: InferenceConfig) -> None:
        """Validate model path is set."""
        if config.models.path is None:
            raise ValueError(
                "models.path is required for vLLM engine "
                "(set via config, --model-path, or MODEL_PATH)"
            )

    def create(
        self, lg: Logger, config: InferenceConfig, on_progress: Any = None
    ) -> Any:
        try:
            from ...engines.vllm import VLLMEngine
        except ImportError as e:
            raise ImportError(
                "vLLM engine requested (backends.engine=vllm) but vLLM is not installed. "
                "Install with: pip install vllm\n"
                "Or use Ollama engine: llm-infer serve --engine ollama"
            ) from e

        self._validate_model_path(config)
        vllm_cfg = replace(config.engines.vllm, model_path=str(config.models.path))
        return VLLMEngine(lg, vllm_cfg)

    def warmup_enabled(self, config: InferenceConfig) -> bool:
        return config.engines.vllm.warmup

    def max_batch_size(self, config: InferenceConfig) -> int:
        # vLLM handles batching internally
        return 1


class OllamaEngineFactory(EngineFactory):
    """Factory for Ollama-backed inference engine."""

    def _get_ollama_model_name(self, lg: Logger, config: InferenceConfig) -> str:
        """Get Ollama model name from model config.

        Resolution: models.yaml 'ollama' field > engines.ollama.model > model name as-is.
        """
        from pathlib import Path

        model_name = Path(config.models.path).name if config.models.path else None

        # Look up model config to get Ollama-specific name
        if model_name and (ollama_name := config.models.get(model_name).ollama):
            lg.debug(
                "resolved ollama model",
                extra={"model": model_name, "ollama": ollama_name},
            )
            return ollama_name

        # Fall back to engines.ollama.model if set
        if config.engines.ollama.model:
            return config.engines.ollama.model

        # Fall back to model name as-is
        if model_name:
            lg.warning(
                "no ollama field in model config, using as-is",
                extra={"model": model_name},
            )
            return model_name

        raise ValueError(
            "Model name required for Ollama. Set --model with 'ollama' field in models.yaml."
        )

    def create(
        self, lg: Logger, config: InferenceConfig, on_progress: Any = None
    ) -> Any:
        try:
            from ...engines.ollama import OllamaEngine
        except ImportError as e:
            raise ImportError(
                "Ollama engine requested (backends.engine=ollama) but httpx is not installed. "
                "Install with: pip install httpx\n"
                "Also ensure Ollama is installed: https://ollama.ai"
            ) from e

        ollama_model = self._get_ollama_model_name(lg, config)
        ollama_cfg = replace(config.engines.ollama, model=ollama_model)
        return OllamaEngine(lg, ollama_cfg)

    def warmup_enabled(self, config: InferenceConfig) -> bool:
        return config.engines.ollama.warmup

    def max_batch_size(self, config: InferenceConfig) -> int:
        # Ollama handles batching internally
        return 1


class VLLMServerEngineFactory(EngineFactory):
    """Factory for vLLM server-backed inference engine.

    Connects to a `vllm serve` process via OpenAI-compatible HTTP API.
    Like OllamaEngineFactory, resolves model path and creates the engine.
    """

    def _validate_model_path(self, config: InferenceConfig) -> None:
        """Validate model path is set."""
        if config.models.path is None:
            raise ValueError(
                "models.path is required for vllm-server engine "
                "(set via config, --model-path, or MODEL_PATH)"
            )

    def create(
        self, lg: Logger, config: InferenceConfig, on_progress: Any = None
    ) -> Any:
        try:
            from ...engines.vllm_server import VLLMServerEngine
        except ImportError as e:
            raise ImportError(
                "vLLM server engine requested (backends.engine=vllm-server) "
                "but httpx is not installed. "
                "Install with: pip install httpx\n"
                "Also ensure vLLM is installed: pip install vllm"
            ) from e

        self._validate_model_path(config)
        vllm_server_cfg = replace(
            config.engines.vllm_server, model_path=str(config.models.path)
        )
        return VLLMServerEngine(lg, vllm_server_cfg)

    def warmup_enabled(self, config: InferenceConfig) -> bool:
        return config.engines.vllm_server.warmup

    def max_batch_size(self, config: InferenceConfig) -> int:
        # Server handles batching internally
        return 1


# Engine factory registry
ENGINE_FACTORIES: dict[str, EngineFactory] = {
    "native": NativeEngineFactory(),
    "vllm": VLLMEngineFactory(),
    "vllm-server": VLLMServerEngineFactory(),
    "ollama": OllamaEngineFactory(),
}


def get_engine_factory(engine_type: str) -> EngineFactory:
    """Get engine factory by type."""
    if engine_type not in ENGINE_FACTORIES:
        available = ", ".join(ENGINE_FACTORIES.keys())
        raise ValueError(f"Unknown engine: {engine_type}. Available: {available}")
    return ENGINE_FACTORIES[engine_type]


# ---------------------------------------------------------------------------
# Handler Factories
# ---------------------------------------------------------------------------


class HandlerFactory(ABC):
    """Base class for handler factories."""

    @abstractmethod
    def create(
        self, lg: Logger, engine: Any, config: InferenceConfig
    ) -> RequestHandler:
        """Create a handler instance."""
        ...


class SequentialHandlerFactory(HandlerFactory):
    """Factory for sequential (one-at-a-time) handler."""

    def create(
        self, lg: Logger, engine: Any, config: InferenceConfig
    ) -> RequestHandler:
        from .handlers import SequentialHandler

        return SequentialHandler(engine)


class BoundedHandlerFactory(HandlerFactory):
    """Factory for bounded queue handler with batching."""

    def create(
        self, lg: Logger, engine: Any, config: InferenceConfig
    ) -> RequestHandler:
        from .handlers import BoundedQueueHandler

        engine_factory = get_engine_factory(config.backends.engine)
        return BoundedQueueHandler(
            engine,
            max_pending=config.dispatch.max_pending,
            max_batch_size=engine_factory.max_batch_size(config),
            batch_streaming=config.dispatch.batch_streaming,
        )


# Handler factory registry
HANDLER_FACTORIES: dict[str, HandlerFactory] = {
    "sequential": SequentialHandlerFactory(),
    "bounded": BoundedHandlerFactory(),
}


def get_handler_factory(handler_type: str) -> HandlerFactory:
    """Get handler factory by type."""
    if handler_type not in HANDLER_FACTORIES:
        available = ", ".join(HANDLER_FACTORIES.keys())
        raise ValueError(f"Unknown handler: {handler_type}. Available: {available}")
    return HANDLER_FACTORIES[handler_type]
