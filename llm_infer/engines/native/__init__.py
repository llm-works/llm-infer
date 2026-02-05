"""Native inference engine.

This module provides the custom native inference engine implementation,
used for learning/reference. It includes all the building blocks:
- Transformer model with paged attention
- KV cache management
- Tokenization
- Sampling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

if TYPE_CHECKING:
    from .engine import InferenceEngine

from .config import EngineConfig
from .engine import InferenceEngine, StreamingResult
from .generation import run_decode, run_prefill
from .model import (
    ModelArchitecture,
    TransformerConfig,
    TransformerModel,
    get_architecture,
)
from .scheduler import Request, RequestState, Scheduler

__all__ = [
    # Factory
    "create_native_engine",
    # Engine
    "EngineConfig",
    "InferenceEngine",
    "StreamingResult",
    # Generation
    "run_decode",
    "run_prefill",
    # Model
    "ModelArchitecture",
    "TransformerConfig",
    "TransformerModel",
    "get_architecture",
    # Scheduler
    "Request",
    "RequestState",
    "Scheduler",
]


def _parse_dtype(dtype_str: str) -> Any:
    """Parse dtype string to torch dtype."""
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_str, torch.float16)


def _build_engine_config(config: dict[str, Any]) -> tuple[EngineConfig, str]:
    """Build EngineConfig from config dict. Returns (engine_config, model_path)."""
    model_path = config.get("model", {}).get("path", "")
    backends = config.get("backends", {})
    engine = config.get("engine", {})
    model_config = TransformerConfig.from_hf_config(model_path)

    engine_cfg = EngineConfig(
        model=model_config,
        model_path=model_path,
        num_blocks=engine.get("num_blocks", 1024),
        block_size=engine.get("block_size", 16),
        max_batch_size=engine.get("max_batch_size", 1),
        device=engine.get("device", "cuda"),
        dtype=_parse_dtype(engine.get("dtype", "float16")),
        attention_backend=engine.get("attention_backend", "auto"),
        linear_backend=backends.get("linear", "pytorch"),
        torch_compile=engine.get("torch_compile", False),
        warmup=engine.get("warmup", False),
    )
    return engine_cfg, model_path


def _require_runtime_deps() -> None:
    """Check that local inference dependencies are installed."""
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    try:
        import safetensors  # noqa: F401
    except ImportError:
        missing.append("safetensors")

    if missing:
        raise ImportError(
            f"Local inference requires: {', '.join(missing)}. "
            "Install with: pip install llm-infer[runtime]"
        )


def create_native_engine(lg: Logger, config: dict[str, Any]) -> InferenceEngine:
    """Create native inference engine from config dictionary."""
    _require_runtime_deps()
    from ...serving.dispatch.main import ProgressTracker

    engine_cfg, _ = _build_engine_config(config)
    on_progress = ProgressTracker(lg)
    return InferenceEngine(lg, engine_cfg, on_progress=on_progress)
