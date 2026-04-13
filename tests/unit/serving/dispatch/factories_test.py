"""Unit tests for serving/dispatch/factories.py.

Focuses on:
- _validate_model_path branches for each engine factory
- warmup_enabled / max_batch_size pass-throughs
- Registry: get_engine_factory / get_handler_factory
- Ollama model name resolution
- Adapter scanning for vllm-server / peft factories
- ConcurrentHttp _get_max_concurrent dispatch

Heavy create() paths that import torch/vllm/peft are exercised via
monkeypatched module-level imports rather than mocked instances.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.models.config import ModelConfig
from llm_infer.serving.dispatch.config import InferenceConfig
from llm_infer.serving.dispatch.factories import (
    BoundedHandlerFactory,
    ConcurrentHttpHandlerFactory,
    NativeEngineFactory,
    OllamaEngineFactory,
    PEFTEngineFactory,
    SequentialHandlerFactory,
    VLLMEngineFactory,
    VLLMServerEngineFactory,
    get_engine_factory,
    get_handler_factory,
)
from llm_infer.serving.dispatch.handlers.bounded import BoundedQueueHandler
from llm_infer.serving.dispatch.handlers.concurrent_http import ConcurrentHttpHandler
from llm_infer.serving.dispatch.handlers.sequential import SequentialHandler

pytestmark = pytest.mark.unit


@pytest.fixture
def lg() -> Logger:
    return MagicMock(spec=Logger)


def _config(
    *,
    model_path: Path | None = None,
    ollama_models: dict[str, str] | None = None,
) -> InferenceConfig:
    """Build a real InferenceConfig with sensible test defaults.

    Args:
        model_path: Optional models.path. The basename is used by Ollama
            resolution to look up per-model overrides.
        ollama_models: Map of model basename → ollama name (e.g.,
            {"qwen-7b": "qwen2.5:7b"}). Populates models.models so
            cfg.models.get(name).ollama returns the mapped value.
    """
    cfg = InferenceConfig()
    if model_path is not None:
        cfg.models.path = model_path
    if ollama_models:
        for name, ollama_name in ollama_models.items():
            cfg.models.models[name] = ModelConfig(name=name, ollama=ollama_name)
    return cfg


# ---------------------------------------------------------------------------
# _validate_model_path
# ---------------------------------------------------------------------------


class TestValidateModelPath:
    def test_native_missing(self) -> None:
        f = NativeEngineFactory()
        with pytest.raises(ValueError, match="models.path is required"):
            f._validate_model_path(_config())

    def test_vllm_missing(self) -> None:
        f = VLLMEngineFactory()
        with pytest.raises(ValueError, match="models.path is required"):
            f._validate_model_path(_config())

    def test_vllm_server_missing(self) -> None:
        f = VLLMServerEngineFactory()
        with pytest.raises(ValueError, match="models.path is required"):
            f._validate_model_path(_config())

    def test_peft_missing(self) -> None:
        f = PEFTEngineFactory()
        with pytest.raises(ValueError, match="models.path is required"):
            f._validate_model_path(_config())

    def test_validate_passes_with_path(self) -> None:
        f = NativeEngineFactory()
        cfg = _config(model_path=Path("/x"))
        f._validate_model_path(cfg)  # No exception


# ---------------------------------------------------------------------------
# warmup_enabled / max_batch_size
# ---------------------------------------------------------------------------


class TestEngineFactoryProperties:
    def test_native_warmup_enabled(self) -> None:
        f = NativeEngineFactory()
        cfg = _config()
        cfg.engines.native.warmup = True
        assert f.warmup_enabled(cfg) is True
        cfg.engines.native.warmup = False
        assert f.warmup_enabled(cfg) is False

    def test_native_max_batch_size(self) -> None:
        f = NativeEngineFactory()
        cfg = _config()
        cfg.engines.native.max_batch_size = 8
        assert f.max_batch_size(cfg) == 8

    def test_vllm_warmup_enabled(self) -> None:
        cfg = _config()
        cfg.engines.vllm.warmup = True
        assert VLLMEngineFactory().warmup_enabled(cfg) is True

    def test_vllm_max_batch_size_is_one(self) -> None:
        assert VLLMEngineFactory().max_batch_size(_config()) == 1

    def test_ollama_warmup_enabled(self) -> None:
        cfg = _config()
        cfg.engines.ollama.warmup = False
        assert OllamaEngineFactory().warmup_enabled(cfg) is False

    def test_ollama_max_batch_size_is_one(self) -> None:
        assert OllamaEngineFactory().max_batch_size(_config()) == 1

    def test_vllm_server_warmup_enabled(self) -> None:
        cfg = _config()
        cfg.engines.vllm_server.warmup = True
        assert VLLMServerEngineFactory().warmup_enabled(cfg) is True

    def test_vllm_server_max_batch_size_is_one(self) -> None:
        assert VLLMServerEngineFactory().max_batch_size(_config()) == 1

    def test_peft_warmup_enabled(self) -> None:
        cfg = _config()
        cfg.engines.peft.warmup = True
        assert PEFTEngineFactory().warmup_enabled(cfg) is True

    def test_peft_max_batch_size_is_one(self) -> None:
        assert PEFTEngineFactory().max_batch_size(_config()) == 1


# ---------------------------------------------------------------------------
# get_engine_factory / get_handler_factory registry
# ---------------------------------------------------------------------------


class TestEngineRegistry:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("native", NativeEngineFactory),
            ("vllm", VLLMEngineFactory),
            ("vllm-server", VLLMServerEngineFactory),
            ("ollama", OllamaEngineFactory),
            ("peft", PEFTEngineFactory),
        ],
    )
    def test_known_engines(self, name: str, cls: type) -> None:
        assert isinstance(get_engine_factory(name), cls)

    def test_unknown_engine_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine_factory("unknown")


class TestHandlerRegistry:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("sequential", SequentialHandlerFactory),
            ("bounded", BoundedHandlerFactory),
            ("concurrent_http", ConcurrentHttpHandlerFactory),
        ],
    )
    def test_known_handlers(self, name: str, cls: type) -> None:
        assert isinstance(get_handler_factory(name), cls)

    def test_unknown_handler_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown handler"):
            get_handler_factory("unknown")


# ---------------------------------------------------------------------------
# OllamaEngineFactory._get_ollama_model_name
# ---------------------------------------------------------------------------


class TestOllamaModelNameResolution:
    def test_model_yaml_field_takes_priority(self, lg: Logger) -> None:
        cfg = _config(
            model_path=Path("/models/qwen-7b"),
            ollama_models={"qwen-7b": "qwen2.5:7b"},
        )
        f = OllamaEngineFactory()
        assert f._get_ollama_model_name(lg, cfg) == "qwen2.5:7b"

    def test_falls_back_to_engine_config_model(self, lg: Logger) -> None:
        cfg = _config(model_path=Path("/models/qwen-7b"))
        cfg.engines.ollama.model = "explicit-model"
        f = OllamaEngineFactory()
        assert f._get_ollama_model_name(lg, cfg) == "explicit-model"

    def test_falls_back_to_path_basename(self, lg: Logger) -> None:
        cfg = _config(model_path=Path("/models/llama3"))
        cfg.engines.ollama.model = ""
        f = OllamaEngineFactory()
        assert f._get_ollama_model_name(lg, cfg) == "llama3"
        # Warning was emitted
        assert lg.warning.called  # type: ignore[attr-defined]

    def test_no_model_raises(self, lg: Logger) -> None:
        cfg = _config()  # path=None, model=""
        f = OllamaEngineFactory()
        with pytest.raises(ValueError, match="Model name required"):
            f._get_ollama_model_name(lg, cfg)


# ---------------------------------------------------------------------------
# VLLMServerEngineFactory._scan_adapters
# ---------------------------------------------------------------------------


class TestVLLMServerScanAdapters:
    def test_lora_disabled_returns_empty(self, lg: Logger) -> None:
        f = VLLMServerEngineFactory()
        cfg = _config()
        cfg.engines.vllm_server.lora.enabled = False
        assert f._scan_adapters(lg, cfg) == []

    def test_no_base_path_returns_empty(self, lg: Logger) -> None:
        f = VLLMServerEngineFactory()
        cfg = _config()
        cfg.engines.vllm_server.lora.enabled = True
        cfg.engines.vllm_server.lora.base_path = None
        assert f._scan_adapters(lg, cfg) == []

    def test_scans_when_enabled(self, lg: Logger, tmp_path: Path) -> None:
        f = VLLMServerEngineFactory()
        cfg = _config(model_path=Path("/models/qwen"))
        cfg.engines.vllm_server.lora.enabled = True
        cfg.engines.vllm_server.lora.base_path = str(tmp_path)
        # Empty dir -> empty list (but scan still ran)
        result = f._scan_adapters(lg, cfg)
        assert result == []


# ---------------------------------------------------------------------------
# PEFTEngineFactory._scan_adapters
# ---------------------------------------------------------------------------


class TestPEFTScanAdapters:
    def test_no_base_path_returns_empty(self, lg: Logger) -> None:
        f = PEFTEngineFactory()
        cfg = _config()
        cfg.engines.peft.adapter_base_path = None
        assert f._scan_adapters(lg, cfg) == []

    def test_scans_when_path_set(self, lg: Logger, tmp_path: Path) -> None:
        f = PEFTEngineFactory()
        cfg = _config(model_path=Path("/models/qwen"))
        cfg.engines.peft.adapter_base_path = str(tmp_path)
        result = f._scan_adapters(lg, cfg)
        assert result == []


# ---------------------------------------------------------------------------
# ConcurrentHttpHandlerFactory._get_max_concurrent
# ---------------------------------------------------------------------------


class TestConcurrentHttpHandlerFactory:
    def test_vllm_server(self) -> None:
        f = ConcurrentHttpHandlerFactory()
        cfg = _config()
        cfg.backends.engine = "vllm-server"
        cfg.engines.vllm_server.max_concurrent = 8
        assert f._get_max_concurrent(cfg) == 8

    def test_ollama(self) -> None:
        f = ConcurrentHttpHandlerFactory()
        cfg = _config()
        cfg.backends.engine = "ollama"
        cfg.engines.ollama.max_concurrent = 4
        assert f._get_max_concurrent(cfg) == 4

    def test_unsupported_engine_raises(self) -> None:
        f = ConcurrentHttpHandlerFactory()
        cfg = _config()
        cfg.backends.engine = "native"
        with pytest.raises(ValueError, match="only supports vllm-server/ollama"):
            f._get_max_concurrent(cfg)


# ---------------------------------------------------------------------------
# Handler create() - sequential / bounded / concurrent_http
# ---------------------------------------------------------------------------


class TestHandlerCreate:
    def test_sequential_handler_create(self, lg: Logger) -> None:
        f = SequentialHandlerFactory()
        engine = MagicMock()
        cfg = _config()
        handler = f.create(lg, engine, cfg)
        assert isinstance(handler, SequentialHandler)

    def test_bounded_handler_create(self, lg: Logger) -> None:
        f = BoundedHandlerFactory()
        engine = MagicMock()
        cfg = _config()
        cfg.backends.engine = "ollama"  # bounded uses engine factory.max_batch_size
        cfg.dispatch.max_pending = 5
        handler = f.create(lg, engine, cfg)
        assert isinstance(handler, BoundedQueueHandler)

    def test_concurrent_http_handler_create(self, lg: Logger) -> None:
        f = ConcurrentHttpHandlerFactory()
        engine = MagicMock()
        cfg = _config()
        cfg.backends.engine = "ollama"
        cfg.engines.ollama.max_concurrent = 2
        handler = f.create(lg, engine, cfg)
        assert isinstance(handler, ConcurrentHttpHandler)


# ---------------------------------------------------------------------------
# Engine create() - mocked import paths
# ---------------------------------------------------------------------------


class TestNativeEngineCreate:
    def test_import_error_raises_with_helpful_message(
        self, lg: Logger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = NativeEngineFactory()
        # Force ImportError on the lazy import; monkeypatch.delitem auto-restores
        # so a test failure can't pollute sys.modules for downstream tests.
        import sys

        monkeypatch.delitem(sys.modules, "llm_infer.engines.native", raising=False)
        monkeypatch.setattr(
            f,
            "_import_native_deps",
            MagicMock(side_effect=ImportError("torch missing")),
        )
        cfg = _config(model_path=Path("/x"))
        with pytest.raises(ImportError):
            f.create(lg, cfg)


class TestVLLMEngineCreate:
    def test_import_error_raises(
        self, lg: Logger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = VLLMEngineFactory()
        # Inject a fake module that raises on import
        import sys

        # Replace 'llm_infer.engines.vllm' with one that fails to define VLLMEngine
        original = sys.modules.get("llm_infer.engines.vllm")
        sys.modules["llm_infer.engines.vllm"] = SimpleNamespace()  # type: ignore[assignment]
        try:
            cfg = _config(model_path=Path("/x"))
            with pytest.raises(ImportError):
                f.create(lg, cfg)
        finally:
            if original is not None:
                sys.modules["llm_infer.engines.vllm"] = original
            else:
                del sys.modules["llm_infer.engines.vllm"]
