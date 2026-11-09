# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Import chains checked in a fresh interpreter, one subprocess per test."""


def _run_import_test(code: str) -> None:
    """Run import test code in a fresh subprocess to avoid module caching."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"Import test failed:\n{result.stderr or result.stdout}")


class TestCircularImportRegression:
    """Regression tests for circular import issues.

    These tests run imports in fresh subprocesses to avoid module caching
    that could mask circular import problems. Each test documents the import
    chain that was previously broken.
    """

    def test_client_api_no_circular_import(self) -> None:
        """Test client and api modules don't have circular imports.

        Previously broken chain:
            llm_infer.client → llm_infer.api → llm_infer.client (circular)

        Fixed by moving schemas to llm_infer.schemas.openai (leaf module).
        """
        _run_import_test("""
from llm_infer.client import Backend, ChatResponse, LLMClient
from llm_infer.api import ChatCompletionRequest, LLMClient as APIClient

assert LLMClient is APIClient  # Same class re-exported
assert Backend is not None
assert ChatResponse is not None
assert ChatCompletionRequest is not None
""")

    def test_schemas_importable_standalone(self) -> None:
        """Test schemas can be imported without pulling in client or serving."""
        _run_import_test("""
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
""")

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
        _run_import_test("""
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
""")

    def test_serving_api_openai_router_importable(self) -> None:
        """Test OpenAI router can be imported for downstream use."""
        _run_import_test("""
from llm_infer.serving.api.openai import create_openai_router

assert callable(create_openai_router)
""")

    def test_dispatch_run_server_lazy_import(self) -> None:
        """Test run_server is lazily imported (not loaded until accessed)."""
        _run_import_test("""
import sys
import llm_infer.serving.dispatch as dispatch

# Before accessing run_server, main module should NOT be loaded
assert "llm_infer.serving.dispatch.main" not in sys.modules

# Access run_server (triggers lazy import)
rs = dispatch.run_server
assert callable(rs)

# Now main module should be loaded
assert "llm_infer.serving.dispatch.main" in sys.modules
""")

    def test_full_import_chain_client_to_streaming(self) -> None:
        """Test importing both client and streaming utilities together.

        This is the real-world use case: a proxy that needs both the client
        to call upstream and the streaming utilities to format responses.
        """
        _run_import_test("""
from llm_infer.client import LLMClient
from llm_infer.serving.api.openai.streaming import stream_chat_completion_sync

assert LLMClient is not None
assert callable(stream_chat_completion_sync)
""")
