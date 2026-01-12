"""Unit tests for module imports and exports."""

import pytest

pytestmark = pytest.mark.unit


class TestPrimitivesImports:
    """Test primitives module imports."""

    def test_import_primitives(self) -> None:
        """Test primitives package imports correctly."""
        from llm_infer import primitives

        assert hasattr(primitives, "sampler")
        assert hasattr(primitives, "guards")
        assert hasattr(primitives, "kv_cache")

    def test_import_sampler(self) -> None:
        """Test sampler module imports."""
        from llm_infer.primitives.sampler import sample

        assert callable(sample)

    def test_import_guards(self) -> None:
        """Test guards module imports."""
        from llm_infer.primitives.guards import GuardResult, RepetitionGuard

        assert GuardResult is not None
        assert RepetitionGuard is not None

    def test_import_kv_cache(self) -> None:
        """Test kv_cache module imports."""
        from llm_infer.primitives.kv_cache import BlockPool, SequenceKVCache

        assert BlockPool is not None
        assert SequenceKVCache is not None

    def test_import_tokenizer(self) -> None:
        """Test tokenizer module imports."""
        from llm_infer.primitives.tokenizer import HuggingFaceTokenizer, TokenizerConfig

        assert HuggingFaceTokenizer is not None
        assert TokenizerConfig is not None


class TestBackendsImports:
    """Test backends module imports."""

    def test_import_linear_backends(self) -> None:
        """Test linear backends imports."""
        from llm_infer.backends.linear import get_backend, get_linear_backend

        assert callable(get_backend)
        assert callable(get_linear_backend)

    def test_import_formats(self) -> None:
        """Test formats module imports."""
        from llm_infer.backends.linear.formats import (
            AWQWeights,
            FP8Weights,
            QuantFormat,
        )

        assert AWQWeights is not None
        assert FP8Weights is not None
        assert QuantFormat is not None

    def test_import_pytorch_backends(self) -> None:
        """Test PyTorch backends import."""
        from llm_infer.backends.linear.kernels import (
            PyTorchAWQBackend,
            PyTorchFP8Backend,
        )

        assert PyTorchAWQBackend is not None
        assert PyTorchFP8Backend is not None


class TestPipelinesImports:
    """Test pipelines module imports."""

    def test_import_pipelines(self) -> None:
        """Test pipelines package imports."""
        from llm_infer import pipelines

        assert hasattr(pipelines, "model")

    def test_import_model_config(self) -> None:
        """Test model config imports."""
        from llm_infer.pipelines.model import ModelConfig

        assert ModelConfig is not None

    def test_import_scheduler(self) -> None:
        """Test scheduler imports."""
        from llm_infer.pipelines.scheduler import Request, RequestState, Scheduler

        assert Request is not None
        assert RequestState is not None
        assert Scheduler is not None

    def test_import_engine_config(self) -> None:
        """Test engine config imports."""
        from llm_infer.pipelines.config import EngineConfig

        assert EngineConfig is not None


class TestAttentionImports:
    """Test attention module imports."""

    def test_import_attention(self) -> None:
        """Test attention module imports."""
        from llm_infer.primitives.attention import (
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


class TestCircularImportRegression:
    """Regression tests for circular import issues.

    These tests verify that import paths work correctly without triggering
    circular import errors. Each test documents the import chain that was
    previously broken.
    """

    def test_client_api_no_circular_import(self) -> None:
        """Test client and api modules don't have circular imports.

        Previously broken chain:
            llm_infer.client → llm_infer.api → llm_infer.client (circular)

        Fixed by moving schemas to llm_infer.schemas.openai (leaf module).
        """
        # These imports must work in any order
        from llm_infer.api import ChatCompletionRequest
        from llm_infer.api import OpenAIClient as APIClient
        from llm_infer.client import ChatClient, ChatResponse, OpenAIClient

        assert OpenAIClient is APIClient  # Same class re-exported
        assert ChatClient is not None
        assert ChatResponse is not None
        assert ChatCompletionRequest is not None

    def test_schemas_importable_standalone(self) -> None:
        """Test schemas can be imported without pulling in client or serving."""
        from llm_infer.schemas.openai import (
            ChatCompletionRequest,
            ChatMessage,
            FinishReason,
            Role,
        )

        assert Role.USER.value == "user"
        assert FinishReason.STOP.value == "stop"
        assert ChatMessage is not None
        assert ChatCompletionRequest is not None

    def test_serving_api_openai_streaming_no_circular_import(self) -> None:
        """Test serving.api.openai.streaming imports without circular import.

        Previously broken chain:
            serving.api.openai.streaming
              → serving.api.__init__ (eager import of .routes)
                → .routes → ..dispatch.types
                  → ..dispatch.__init__ (eager import of .main)
                    → .main → ..api.routes (circular!)

        Fixed by lazy-loading run_server in dispatch/__init__.py.
        """
        from llm_infer.serving.api.openai.streaming import (
            create_chat_chunk,
            format_sse_done,
            format_sse_event,
            stream_chat_completion_sync,
        )

        assert callable(stream_chat_completion_sync)
        assert callable(create_chat_chunk)
        assert callable(format_sse_event)
        assert callable(format_sse_done)

    def test_serving_api_openai_router_importable(self) -> None:
        """Test OpenAI router can be imported for downstream use."""
        from llm_infer.serving.api.openai import create_openai_router

        assert callable(create_openai_router)

    def test_dispatch_run_server_lazy_import(self) -> None:
        """Test run_server is accessible via lazy import."""
        from llm_infer.serving.dispatch import run_server

        assert callable(run_server)

    def test_full_import_chain_client_to_streaming(self) -> None:
        """Test importing both client and streaming utilities together.

        This is the real-world use case: a proxy that needs both the client
        to call upstream and the streaming utilities to format responses.
        """
        from llm_infer.client import OpenAIClient
        from llm_infer.serving.api.openai.streaming import stream_chat_completion_sync

        assert OpenAIClient is not None
        assert callable(stream_chat_completion_sync)


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
        client_types = {"ChatClient", "ChatResponse", "OpenAIClient"}

        for name in api.__all__:
            obj = getattr(api, name)
            assert isinstance(obj, (type, EnumMeta)), f"{name} is not a class/enum"
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
