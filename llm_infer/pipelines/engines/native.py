"""Native engine factory wrapper.

This module provides a factory function for creating the native inference engine,
maintaining compatibility with the engines module interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..engine import InferenceEngine


def _parse_dtype(dtype_str: str) -> Any:
    """Parse dtype string to torch dtype."""
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_str, torch.float16)


def _build_engine_config(config: dict[str, Any]) -> tuple[Any, Any]:
    """Build EngineConfig from config dict. Returns (engine_config, model_path)."""
    from ..config import EngineConfig
    from ..model import ModelConfig

    model_path = config.get("model", {}).get("path", "")
    backends = config.get("backends", {})
    engine = config.get("engine", {})
    model_config = ModelConfig.from_hf_config(model_path)

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


def create_native_engine(config: dict[str, Any], lg: Any = None) -> InferenceEngine:
    """Create native inference engine from config dictionary."""
    from ..engine import InferenceEngine

    engine_cfg, _ = _build_engine_config(config)

    on_progress = None
    if lg:
        from ...serving.dispatch.main import ProgressTracker

        on_progress = ProgressTracker(lg)

    return InferenceEngine(lg, engine_cfg, on_progress=on_progress)
