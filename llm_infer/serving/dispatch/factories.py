"""Factory classes for creating engines and handlers.

Uses class-based registry pattern for clean extensibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import InferenceConfig
    from .handler import RequestHandler


# ---------------------------------------------------------------------------
# Engine Factories
# ---------------------------------------------------------------------------


class EngineFactory(ABC):
    """Base class for engine factories."""

    @abstractmethod
    def create(self, lg: Any, config: InferenceConfig, on_progress: Any = None) -> Any:
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

    def create(self, lg: Any, config: InferenceConfig, on_progress: Any = None) -> Any:
        from ...pipelines import EngineConfig, InferenceEngine, ModelConfig

        if config.models.path is None:
            raise ValueError(
                "models.path is required for native engine "
                "(set via config, --model-path, or MODEL_PATH)"
            )
        native_cfg = config.engines.native
        model_path = str(config.models.path)
        model_config = ModelConfig.from_hf_config(model_path)
        engine_config = EngineConfig(
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

        return InferenceEngine(lg, engine_config, on_progress=on_progress)

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

    def create(self, lg: Any, config: InferenceConfig, on_progress: Any = None) -> Any:
        try:
            from ...pipelines.engines.vllm_engine import VLLMEngine
        except ImportError as e:
            raise ImportError(
                "vLLM engine requested (backends.engine=vllm) but vLLM is not installed. "
                "Install with: pip install vllm\nOr use native engine: backends.engine=native"
            ) from e

        self._validate_model_path(config)
        vllm_cfg = replace(config.engines.vllm, model_path=str(config.models.path))
        return VLLMEngine(lg, vllm_cfg)

    def warmup_enabled(self, config: InferenceConfig) -> bool:
        return config.engines.vllm.warmup

    def max_batch_size(self, config: InferenceConfig) -> int:
        # vLLM handles batching internally
        return 1


# Engine factory registry
ENGINE_FACTORIES: dict[str, EngineFactory] = {
    "native": NativeEngineFactory(),
    "vllm": VLLMEngineFactory(),
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
    def create(self, lg: Any, engine: Any, config: InferenceConfig) -> RequestHandler:
        """Create a handler instance."""
        ...


class SequentialHandlerFactory(HandlerFactory):
    """Factory for sequential (one-at-a-time) handler."""

    def create(self, lg: Any, engine: Any, config: InferenceConfig) -> RequestHandler:
        from .handlers import SequentialHandler

        return SequentialHandler(engine)


class BoundedHandlerFactory(HandlerFactory):
    """Factory for bounded queue handler with batching."""

    def create(self, lg: Any, engine: Any, config: InferenceConfig) -> RequestHandler:
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
