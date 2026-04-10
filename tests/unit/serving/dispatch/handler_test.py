"""Unit tests for serving/dispatch/handler.py base RequestHandler.

Uses SequentialHandler as the concrete subclass for testing the base class.
Existing handler_adapter_test.py covers _resolve_effective_adapter; this file
covers everything else: validation, params building, processing, streaming.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.serving.dispatch.handler import (
    AdapterError,
    _stable_adapter_int_id,
)
from llm_infer.serving.dispatch.handlers import SequentialHandler
from llm_infer.serving.dispatch.types import (
    RequestStatus,
    StreamChunk,
)

from ._helpers import ResponseQueueFake
from ._helpers import make_request as _request

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _handler(
    *, response_q: ResponseQueueFake | None = None, lg: Logger | None = None
) -> SequentialHandler:
    engine = MagicMock()
    engine.count_tokens.return_value = 5
    h = SequentialHandler(engine)
    if response_q is not None:
        h.set_response_queue(response_q)  # type: ignore[arg-type]
    if lg is not None:
        h.set_logger(lg)
    return h


# ---------------------------------------------------------------------------
# _stable_adapter_int_id
# ---------------------------------------------------------------------------


def test_stable_adapter_int_id_deterministic() -> None:
    a = _stable_adapter_int_id("my-adapter")
    b = _stable_adapter_int_id("my-adapter")
    assert a == b
    assert 0 <= a < 2**31


def test_stable_adapter_int_id_distinct() -> None:
    assert _stable_adapter_int_id("a") != _stable_adapter_int_id("b")


# ---------------------------------------------------------------------------
# Setters
# ---------------------------------------------------------------------------


class TestSetters:
    def test_set_logger(self) -> None:
        h = _handler()
        lg = MagicMock(spec=Logger)
        h.set_logger(lg)
        assert h._lg is lg

    def test_set_lora_base_path(self, tmp_path: Path) -> None:
        h = _handler()
        h.set_lora_base_path(str(tmp_path))
        assert h._lora_base_path == tmp_path

    def test_set_lora_base_path_none(self) -> None:
        h = _handler()
        h.set_lora_base_path(None)
        assert h._lora_base_path is None

    def test_set_response_queue(self) -> None:
        h = _handler()
        q = ResponseQueueFake()
        h.set_response_queue(q)  # type: ignore[arg-type]
        assert h._response_q is q

    def test_set_adapter_manager(self) -> None:
        h = _handler()
        mgr = MagicMock()
        h.set_adapter_manager(mgr)
        assert h.get_adapter_manager() is mgr

    def test_set_adapter_manager_none(self) -> None:
        h = _handler()
        h.set_adapter_manager(None)
        assert h.get_adapter_manager() is None


# ---------------------------------------------------------------------------
# _create_context
# ---------------------------------------------------------------------------


class TestCreateContext:
    def test_no_logger_returns_none(self) -> None:
        h = _handler()
        ctx = h._create_context(_request())
        assert ctx is None

    def test_with_logger_returns_context(self) -> None:
        h = _handler(lg=MagicMock(spec=Logger))
        ctx = h._create_context(_request("r1"))
        assert ctx is not None
        assert ctx.id == "r1"


# ---------------------------------------------------------------------------
# _validate_adapter_path
# ---------------------------------------------------------------------------


class TestValidateAdapterPath:
    def test_no_lora_base_path_raises(self) -> None:
        h = _handler()
        with pytest.raises(AdapterError, match="LoRA not configured"):
            h._validate_adapter_path("my-adapter")

    def test_invalid_key_raises(self, tmp_path: Path) -> None:
        h = _handler(lg=MagicMock(spec=Logger))
        h.set_lora_base_path(str(tmp_path))
        with pytest.raises(AdapterError, match="invalid adapter"):
            h._validate_adapter_path("../escape")

    def test_invalid_key_logs_warning_when_lg_set(self, tmp_path: Path) -> None:
        lg = MagicMock(spec=Logger)
        h = _handler(lg=lg)
        h.set_lora_base_path(str(tmp_path))
        with pytest.raises(AdapterError):
            h._validate_adapter_path("../escape")
        lg.warning.assert_called()  # type: ignore[attr-defined]

    def test_valid_key_returns_path(self, tmp_path: Path) -> None:
        h = _handler()
        h.set_lora_base_path(str(tmp_path))
        result = h._validate_adapter_path("my-adapter")
        assert result.name == "my-adapter"


# ---------------------------------------------------------------------------
# _check_adapter_enabled / _check_adapter_config_file
# ---------------------------------------------------------------------------


class TestCheckAdapterEnabled:
    def test_with_manager_resolved(self, tmp_path: Path) -> None:
        h = _handler()
        mgr = MagicMock()
        mgr.resolve.return_value = MagicMock()
        h.set_adapter_manager(mgr)
        # Should not raise
        h._check_adapter_enabled("ad", tmp_path / "ad")

    def test_with_manager_not_resolved_raises(self, tmp_path: Path) -> None:
        h = _handler()
        mgr = MagicMock()
        mgr.resolve.return_value = None
        h.set_adapter_manager(mgr)
        with pytest.raises(AdapterError, match="not enabled"):
            h._check_adapter_enabled("ad", tmp_path / "ad")

    def test_fallback_no_config_raises(self, tmp_path: Path) -> None:
        h = _handler()
        d = tmp_path / "ad"
        d.mkdir()
        with pytest.raises(AdapterError, match="no config.yaml"):
            h._check_adapter_enabled("ad", d)

    def test_fallback_malformed_yaml_raises(self, tmp_path: Path) -> None:
        h = _handler()
        d = tmp_path / "ad"
        d.mkdir()
        (d / "config.yaml").write_text("not: valid: yaml: [")
        with pytest.raises(AdapterError, match="failed to read"):
            h._check_adapter_enabled("ad", d)

    def test_fallback_non_mapping_raises(self, tmp_path: Path) -> None:
        h = _handler()
        d = tmp_path / "ad"
        d.mkdir()
        (d / "config.yaml").write_text("- a\n- b\n")
        with pytest.raises(AdapterError, match="must be a mapping"):
            h._check_adapter_enabled("ad", d)

    def test_fallback_disabled_raises(self, tmp_path: Path) -> None:
        h = _handler()
        d = tmp_path / "ad"
        d.mkdir()
        (d / "config.yaml").write_text("enabled: false\n")
        with pytest.raises(AdapterError, match="disabled"):
            h._check_adapter_enabled("ad", d)

    def test_fallback_enabled_passes(self, tmp_path: Path) -> None:
        h = _handler()
        d = tmp_path / "ad"
        d.mkdir()
        (d / "config.yaml").write_text("enabled: true\n")
        h._check_adapter_enabled("ad", d)  # No raise

    def test_fallback_default_enabled(self, tmp_path: Path) -> None:
        """If config has no `enabled` field, defaults to enabled."""
        h = _handler()
        d = tmp_path / "ad"
        d.mkdir()
        (d / "config.yaml").write_text("description: test\n")
        h._check_adapter_enabled("ad", d)  # No raise


# ---------------------------------------------------------------------------
# _import_lora_request_class
# ---------------------------------------------------------------------------


class TestImportLoraRequestClass:
    def test_import_failure_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        h = _handler(lg=MagicMock(spec=Logger))
        # Force ImportError on the lazy import inside the method
        import sys

        original = sys.modules.get("vllm.lora.request")
        sys.modules["vllm.lora.request"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(AdapterError, match="vLLM LoRA module not available"):
                h._import_lora_request_class("ad")
        finally:
            if original is not None:
                sys.modules["vllm.lora.request"] = original
            else:
                del sys.modules["vllm.lora.request"]


# ---------------------------------------------------------------------------
# _log_and_track_adapter
# ---------------------------------------------------------------------------


class TestLogAndTrackAdapter:
    def test_first_load_emits_info(self, tmp_path: Path) -> None:
        lg = MagicMock(spec=Logger)
        h = _handler(lg=lg)
        h._log_and_track_adapter("ad", tmp_path / "ad")
        lg.info.assert_called_once()  # type: ignore[attr-defined]
        # Marked as loaded
        assert "ad" in h._loaded_adapters  # type: ignore[operator]

    def test_subsequent_load_emits_debug(self, tmp_path: Path) -> None:
        lg = MagicMock(spec=Logger)
        h = _handler(lg=lg)
        h._log_and_track_adapter("ad", tmp_path / "ad")
        h._log_and_track_adapter("ad", tmp_path / "ad")
        # First call: info; subsequent: debug
        assert lg.info.call_count == 1  # type: ignore[attr-defined]
        assert lg.debug.call_count >= 1  # type: ignore[attr-defined]

    def test_with_manager_includes_metadata(self, tmp_path: Path) -> None:
        lg = MagicMock(spec=Logger)
        h = _handler(lg=lg)
        mgr = MagicMock()
        loaded = MagicMock()
        loaded.key = "ad-abc123def456"
        loaded.name = "ad"
        loaded.md5 = "abc123def456"
        loaded.mtime = "2026-01-01"
        loaded.peft_type = "LORA"
        mgr.resolve.return_value = loaded
        h.set_adapter_manager(mgr)
        h._log_and_track_adapter("ad", tmp_path / "ad")
        info_call = lg.info.call_args  # type: ignore[attr-defined]
        extra = info_call.kwargs["extra"]
        assert extra["key"] == "ad-abc123def456"
        assert extra["adapter_name"] == "ad"
        assert extra["peft_type"] == "LORA"

    def test_with_manager_unresolved_uses_adapter_field(self, tmp_path: Path) -> None:
        lg = MagicMock(spec=Logger)
        h = _handler(lg=lg)
        mgr = MagicMock()
        mgr.resolve.return_value = None
        h.set_adapter_manager(mgr)
        h._log_and_track_adapter("ad", tmp_path / "ad")
        extra = lg.info.call_args.kwargs["extra"]  # type: ignore[attr-defined]
        assert extra["adapter"] == "ad"


# ---------------------------------------------------------------------------
# _resolve_lora_request
# ---------------------------------------------------------------------------


class TestResolveLoraRequest:
    def test_none_returns_none_none(self) -> None:
        h = _handler()
        assert h._resolve_lora_request(None) == (None, None)

    def test_empty_string_raises(self) -> None:
        h = _handler()
        with pytest.raises(AdapterError, match="cannot be empty"):
            h._resolve_lora_request("")

    def test_adapter_not_found_returns_fallback(self, tmp_path: Path) -> None:
        h = _handler(lg=MagicMock(spec=Logger))
        h.set_lora_base_path(str(tmp_path))
        # tmp_path/ad doesn't exist
        result = h._resolve_lora_request("ad")
        assert result == (None, "ad")

    def test_adapter_resolves_path_string_when_vllm_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When vLLM LoRA not available, returns the path string for PEFT engine."""
        h = _handler()
        d = tmp_path / "ad"
        d.mkdir()
        (d / "config.yaml").write_text("enabled: true\n")
        h.set_lora_base_path(str(tmp_path))

        # Force ImportError on vLLM LoRA import
        import sys

        original = sys.modules.get("vllm.lora.request")
        sys.modules["vllm.lora.request"] = None  # type: ignore[assignment]
        try:
            result, fallback = h._resolve_lora_request("ad")
            assert isinstance(result, str)
            assert "ad" in result
            assert fallback is None
        finally:
            if original is not None:
                sys.modules["vllm.lora.request"] = original
            else:
                del sys.modules["vllm.lora.request"]


# ---------------------------------------------------------------------------
# _add_request_extensions
# ---------------------------------------------------------------------------


class TestAddRequestExtensions:
    def test_no_extensions(self) -> None:
        h = _handler()
        params: dict[str, Any] = {}
        h._add_request_extensions(params, _request())
        assert params == {}

    def test_tools(self) -> None:
        h = _handler()
        params: dict[str, Any] = {}
        h._add_request_extensions(params, _request(tools=[{"name": "x"}]))
        assert params["tools"] == [{"name": "x"}]

    def test_tool_choice(self) -> None:
        h = _handler()
        params: dict[str, Any] = {}
        h._add_request_extensions(params, _request(tool_choice="auto"))
        assert params["tool_choice"] == "auto"

    def test_response_format(self) -> None:
        h = _handler()
        params: dict[str, Any] = {}
        h._add_request_extensions(params, _request(response_format={"type": "json"}))
        assert params["response_format"] == {"type": "json"}

    def test_chat_template_kwargs(self) -> None:
        h = _handler()
        params: dict[str, Any] = {}
        h._add_request_extensions(params, _request(chat_template_kwargs={"a": 1}))
        assert params["chat_template_kwargs"] == {"a": 1}


# ---------------------------------------------------------------------------
# _build_engine_params
# ---------------------------------------------------------------------------


class TestBuildEngineParams:
    def test_basic_no_adapter(self) -> None:
        h = _handler()
        params, fallback = h._build_engine_params(_request("r1"))
        assert params["prompt"] == "hello"
        assert "lora_request" not in params
        assert fallback is None

    def test_with_fallback_adapter(self, tmp_path: Path) -> None:
        h = _handler(lg=MagicMock(spec=Logger))
        h.set_lora_base_path(str(tmp_path))
        params, fallback = h._build_engine_params(_request("r1", adapter="missing"))
        assert "lora_request" not in params
        assert fallback == "missing"


# ---------------------------------------------------------------------------
# _parse_generate_result
# ---------------------------------------------------------------------------


class TestParseGenerateResult:
    def test_string_result(self) -> None:
        h = _handler()
        text, tools, usage, adapter = h._parse_generate_result("hello")
        assert text == "hello"
        assert tools is None
        assert usage is None
        assert adapter is None

    def test_dict_result_full(self) -> None:
        h = _handler()
        result = {
            "content": "world",
            "tool_calls": [{"name": "f"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            "adapter": {"requested": "x"},
        }
        text, tools, usage, adapter = h._parse_generate_result(result)
        assert text == "world"
        assert tools == [{"name": "f"}]
        assert usage["prompt_tokens"] == 5
        assert adapter == {"requested": "x"}

    def test_dict_result_partial(self) -> None:
        h = _handler()
        text, tools, usage, adapter = h._parse_generate_result({"content": "x"})
        assert text == "x"
        assert tools is None


# ---------------------------------------------------------------------------
# _build_success_response
# ---------------------------------------------------------------------------


class TestBuildSuccessResponse:
    def test_with_usage(self) -> None:
        h = _handler()
        resp = h._build_success_response(
            _request("r1"),
            "result",
            None,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert resp.id == "r1"
        assert resp.status == RequestStatus.COMPLETED
        assert resp.result == "result"
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 5

    def test_without_usage_falls_back_to_count_tokens(self) -> None:
        h = _handler()
        h.engine.count_tokens.side_effect = [7, 3]  # type: ignore[attr-defined]
        resp = h._build_success_response(_request("r1"), "result", None)
        assert resp.prompt_tokens == 7
        assert resp.completion_tokens == 3

    def test_with_tool_calls(self) -> None:
        h = _handler()
        resp = h._build_success_response(
            _request("r1"),
            "",
            [{"name": "f"}],
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )
        assert resp.tool_calls == [{"name": "f"}]


# ---------------------------------------------------------------------------
# _process_blocking_request
# ---------------------------------------------------------------------------


class TestProcessBlockingRequest:
    def test_success_string_result(self) -> None:
        h = _handler()
        h.engine.generate.return_value = "result"  # type: ignore[attr-defined]
        resp = h._process_blocking_request(_request("r1"))
        assert resp.status == RequestStatus.COMPLETED
        assert resp.result == "result"

    def test_success_dict_result(self) -> None:
        h = _handler()
        h.engine.generate.return_value = {  # type: ignore[attr-defined]
            "content": "world",
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }
        resp = h._process_blocking_request(_request("r1"))
        assert resp.result == "world"
        assert resp.prompt_tokens == 2

    def test_adapter_error(self) -> None:
        h = _handler(lg=MagicMock(spec=Logger))
        h.engine.generate.side_effect = AdapterError("bad adapter")  # type: ignore[attr-defined]
        resp = h._process_blocking_request(_request("r1"))
        assert resp.status == RequestStatus.FAILED
        assert "bad adapter" in resp.error

    def test_generic_exception(self) -> None:
        h = _handler()
        h.engine.generate.side_effect = RuntimeError("boom")  # type: ignore[attr-defined]
        resp = h._process_blocking_request(_request("r1"))
        assert resp.status == RequestStatus.FAILED
        assert "boom" in resp.error

    def test_with_pre_request_fallback(self, tmp_path: Path) -> None:
        h = _handler(lg=MagicMock(spec=Logger))
        h.set_lora_base_path(str(tmp_path))
        h.engine.generate.return_value = "result"  # type: ignore[attr-defined]
        resp = h._process_blocking_request(_request("r1", adapter="missing"))
        assert resp.adapter is not None
        assert resp.adapter.fallback is True
        assert resp.adapter.requested == "missing"


# ---------------------------------------------------------------------------
# _build_adapter_info_from_result and _build_adapter_info
# ---------------------------------------------------------------------------


class TestBuildAdapterInfoFromResult:
    def test_engine_returned_dict(self) -> None:
        h = _handler()
        info = h._build_adapter_info_from_result(
            {
                "requested": "a",
                "actual": "b",
                "fallback": True,
                "mtime": "t",
                "md5": "m",
            },
            None,
        )
        assert info is not None
        assert info.requested == "a"
        assert info.actual == "b"
        assert info.fallback is True

    def test_pre_request_fallback(self) -> None:
        h = _handler()
        info = h._build_adapter_info_from_result(None, "missing-adapter")
        assert info is not None
        assert info.fallback is True
        assert info.requested == "missing-adapter"

    def test_no_adapter(self) -> None:
        h = _handler()
        assert h._build_adapter_info_from_result(None, None) is None


class TestBuildAdapterInfo:
    def test_no_stream_no_fallback(self) -> None:
        h = _handler()
        assert h._build_adapter_info(None, None) is None

    def test_stream_with_adapter_info(self) -> None:
        h = _handler()
        stream = MagicMock()
        stream.adapter_info = {"requested": "a", "fallback": False}
        info = h._build_adapter_info(None, stream)
        assert info is not None
        assert info.requested == "a"

    def test_stream_legacy_mismatch(self) -> None:
        h = _handler()
        stream = MagicMock(spec=["adapter_mismatch", "adapter_requested"])
        stream.adapter_info = None
        stream.adapter_mismatch = True
        stream.adapter_requested = "x"
        info = h._build_adapter_info(None, stream)
        assert info is not None
        assert info.requested == "x"
        assert info.fallback is True

    def test_fallback_only(self) -> None:
        h = _handler()
        info = h._build_adapter_info("fallback-key", None)
        assert info is not None
        assert info.fallback is True
        assert info.requested == "fallback-key"


# ---------------------------------------------------------------------------
# _process_streaming_request via _process_request dispatch
# ---------------------------------------------------------------------------


class TestProcessStreamingRequest:
    def _make_stream(self) -> MagicMock:
        stream = MagicMock()
        stream.__iter__ = lambda self: iter(["tok1", "tok2"])
        stream.finish_reason = "stop"
        stream.prompt_tokens = 5
        stream.completion_tokens = 2
        stream.tool_calls = None
        stream.adapter_info = None
        stream.adapter_mismatch = False
        return stream

    def test_streaming_success(self) -> None:
        q = ResponseQueueFake()
        h = _handler(response_q=q)
        h.engine.generate_stream_sync.return_value = self._make_stream()  # type: ignore[attr-defined]
        resp = h._process_streaming_request(_request("r1", stream=True))
        assert resp.status == RequestStatus.COMPLETED
        assert resp.prompt_tokens == 5
        # Token chunks + final chunk on the queue
        assert any(
            isinstance(item, StreamChunk) and not item.is_final for item in q.items
        )
        assert any(isinstance(item, StreamChunk) and item.is_final for item in q.items)

    def test_streaming_adapter_error(self) -> None:
        q = ResponseQueueFake()
        h = _handler(response_q=q, lg=MagicMock(spec=Logger))
        h.engine.generate_stream_sync.side_effect = AdapterError("bad")  # type: ignore[attr-defined]
        resp = h._process_streaming_request(_request("r1", stream=True))
        assert resp.status == RequestStatus.FAILED
        # Error chunk emitted
        assert any(
            isinstance(item, StreamChunk) and item.finish_reason == "error"
            for item in q.items
        )

    def test_streaming_generic_exception(self) -> None:
        q = ResponseQueueFake()
        h = _handler(response_q=q)
        h.engine.generate_stream_sync.side_effect = RuntimeError("boom")  # type: ignore[attr-defined]
        resp = h._process_streaming_request(_request("r1", stream=True))
        assert resp.status == RequestStatus.FAILED
        assert "boom" in resp.error


# ---------------------------------------------------------------------------
# _process_request dispatch
# ---------------------------------------------------------------------------


class TestProcessRequestDispatch:
    def test_blocking_no_response_q(self) -> None:
        h = _handler()
        h.engine.generate.return_value = "result"  # type: ignore[attr-defined]
        # Even if stream=True, no response_q -> blocking
        resp = h._process_request(_request("r1", stream=True))
        assert resp.status == RequestStatus.COMPLETED

    def test_streaming_with_response_q(self) -> None:
        q = ResponseQueueFake()
        h = _handler(response_q=q)
        stream = MagicMock()
        stream.__iter__ = lambda self: iter([])
        stream.finish_reason = "stop"
        stream.prompt_tokens = 0
        stream.completion_tokens = 0
        stream.tool_calls = None
        stream.adapter_info = None
        stream.adapter_mismatch = False
        h.engine.generate_stream_sync.return_value = stream  # type: ignore[attr-defined]
        resp = h._process_request(_request("r1", stream=True))
        assert resp.status == RequestStatus.COMPLETED


# ---------------------------------------------------------------------------
# sequence_stats default
# ---------------------------------------------------------------------------


def test_sequence_stats_default() -> None:
    """Base RequestHandler.sequence_stats() returns zeros — exercised via a
    handler that doesn't override it.  SequentialHandler doesn't override.
    """
    h = _handler()
    # SequentialHandler's parent default
    stats = h.sequence_stats()
    assert "active" in stats
    assert "total_tokens" in stats
