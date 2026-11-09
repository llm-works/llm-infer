# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for module imports and exports."""


class TestNativeEngineImports:
    """Test native engine module imports."""

    def test_import_native_engine(self) -> None:
        """Test native engine package imports correctly."""
        from llm_infer.engines import native

        assert hasattr(native, "InferenceEngine")
        assert hasattr(native, "EngineConfig")
        assert hasattr(native, "TransformerConfig")
        assert hasattr(native, "Scheduler")

    def test_import_sampler(self) -> None:
        """Test sampler module imports."""
        from llm_infer.engines.native.sampler import sample

        assert callable(sample)

    def test_import_guards(self) -> None:
        """Test guards module imports."""
        from llm_infer.engines.native.guards import GuardResult, RepetitionGuard

        assert GuardResult is not None
        assert RepetitionGuard is not None

    def test_import_kv_cache(self) -> None:
        """Test kv_cache module imports."""
        from llm_infer.engines.native.kv_cache import BlockPool, SequenceKVCache

        assert BlockPool is not None
        assert SequenceKVCache is not None

    def test_import_tokenizer(self) -> None:
        """Test tokenizer module imports."""
        from llm_infer.engines.native.tokenizer import (
            HuggingFaceTokenizer,
            TokenizerConfig,
        )

        assert HuggingFaceTokenizer is not None
        assert TokenizerConfig is not None


class TestBackendsImports:
    """Test backends module imports."""

    def test_import_linear_backends(self) -> None:
        """Test linear backends imports."""
        from llm_infer.engines.native.backends.linear import BackendRegistry

        assert BackendRegistry is not None

    def test_import_formats(self) -> None:
        """Test formats module imports."""
        from llm_infer.engines.native.backends.linear.formats import (
            AWQWeights,
            FP8Weights,
            QuantFormat,
        )

        assert AWQWeights is not None
        assert FP8Weights is not None
        assert QuantFormat is not None

    def test_import_pytorch_backends(self) -> None:
        """Test PyTorch backends import."""
        from llm_infer.engines.native.backends.linear.kernels import (
            PyTorchAWQBackend,
            PyTorchFP8Backend,
        )

        assert PyTorchAWQBackend is not None
        assert PyTorchFP8Backend is not None


class TestNativeEngineModulesImports:
    """Test native engine submodule imports."""

    def test_import_model_config(self) -> None:
        """Test model config imports."""
        from llm_infer.engines.native.model import TransformerConfig

        assert TransformerConfig is not None

    def test_import_scheduler(self) -> None:
        """Test scheduler imports."""
        from llm_infer.engines.native.scheduler import Request, RequestState, Scheduler

        assert Request is not None
        assert RequestState is not None
        assert Scheduler is not None

    def test_import_engine_config(self) -> None:
        """Test engine config imports."""
        from llm_infer.engines.native.config import EngineConfig

        assert EngineConfig is not None


class TestAttentionImports:
    """Test attention module imports."""

    def test_import_attention(self) -> None:
        """Test attention module imports."""
        from llm_infer.engines.native.attention import (
            FLASHINFER_AVAILABLE,
            NaiveAttentionBackend,
            apply_rope,
            get_attention_backend,
            precompute_rope_freqs,
            rotate_half,
        )

        assert callable(get_attention_backend)
        assert callable(apply_rope)
        assert callable(precompute_rope_freqs)
        assert callable(rotate_half)
        assert isinstance(FLASHINFER_AVAILABLE, bool)
        assert NaiveAttentionBackend is not None


class TestEnginesImports:
    """Test engines module imports."""

    def test_import_engines(self) -> None:
        """Test engines package imports."""
        from llm_infer import engines

        assert hasattr(engines, "create_engine")
        assert hasattr(engines, "InferenceEngineProtocol")

    def test_import_protocol(self) -> None:
        """Test protocol imports."""
        from llm_infer.engines.protocol import (
            InferenceEngineProtocol,
            StreamingResultProtocol,
        )

        assert InferenceEngineProtocol is not None
        assert StreamingResultProtocol is not None

    def test_import_vllm_engine(self) -> None:
        """Test vLLM engine imports."""
        from llm_infer.engines.vllm import VLLMEngine

        assert VLLMEngine is not None

    def test_import_ollama_engine(self) -> None:
        """Test Ollama engine imports."""
        from llm_infer.engines.ollama import OllamaEngine

        assert OllamaEngine is not None


class TestPublicAPIImports:
    """Test public API exports from llm_infer.api."""

    def test_all_exports_importable(self) -> None:
        """Test all __all__ exports are importable and correct types."""
        from dataclasses import is_dataclass
        from enum import EnumMeta
        from typing import Protocol

        from pydantic import BaseModel

        import llm_infer.api as api

        # Client types that are not Pydantic models
        client_types = {"Backend", "ChatResponse", "Factory", "LLMClient"}

        for name in api.__all__:
            obj = getattr(api, name)
            assert isinstance(obj, type | EnumMeta), f"{name} is not a class/enum"
            if isinstance(obj, type) and not isinstance(obj, EnumMeta):
                if name in client_types:
                    # Client types: Protocol, dataclass, or regular class
                    is_valid = (
                        is_dataclass(obj)
                        or issubclass(obj, Protocol)
                        or hasattr(obj, "__mro__")  # Any class
                    )
                    assert is_valid, f"{name} is not a valid client type"
                else:
                    assert issubclass(obj, BaseModel), f"{name} is not a Pydantic model"
