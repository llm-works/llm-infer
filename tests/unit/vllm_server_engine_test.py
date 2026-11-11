# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for vLLM server inference engine."""

import itertools
from typing import Any
from unittest.mock import MagicMock

import pytest


class TestVLLMServerEngineVerifyAdapters:
    """Test VLLMServerEngine._verify_adapters method."""

    def _create_engine_with_mocks(
        self,
        adapter_paths: dict[str, str],
        loaded_models: list[str],
    ) -> MagicMock:
        """Create a mock engine with pre-configured adapter_paths and vLLM response."""
        from llm_infer.engines.vllm_server import VLLMServerEngine

        # Create instance without calling __init__ (we'll set up state manually)
        engine = object.__new__(VLLMServerEngine)

        # Set up required attributes
        engine._lg = MagicMock()
        engine._adapter_paths = dict(adapter_paths)
        engine._adapter_metadata = {k: {} for k in adapter_paths}

        # Mock httpx client to return loaded_models
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": model} for model in loaded_models]
        }
        engine._client = MagicMock()
        engine._client.get.return_value = mock_response

        return engine

    def test_no_adapters_is_noop(self) -> None:
        """Test verification with no adapters does nothing."""
        engine = self._create_engine_with_mocks(
            adapter_paths={},
            loaded_models=["base-model"],
        )

        engine._verify_adapters()

        # Should not call /v1/models when no adapters
        engine._client.get.assert_not_called()
        engine._lg.warning.assert_not_called()

    def test_all_adapters_loaded_no_changes(self) -> None:
        """Test all adapters loaded by vLLM, no removal."""
        engine = self._create_engine_with_mocks(
            adapter_paths={
                "adapter-a": "/path/to/adapter-a",
                "adapter-b": "/path/to/adapter-b",
            },
            loaded_models=["base-model", "adapter-a", "adapter-b"],
        )

        engine._verify_adapters()

        # All adapters present, no warnings
        engine._client.get.assert_called_once_with("/v1/models")
        engine._lg.warning.assert_not_called()
        assert engine._adapter_paths == {
            "adapter-a": "/path/to/adapter-a",
            "adapter-b": "/path/to/adapter-b",
        }

    def test_missing_adapter_removed(self) -> None:
        """Test adapter not loaded by vLLM is removed from available list."""
        engine = self._create_engine_with_mocks(
            adapter_paths={
                "adapter-a": "/path/to/adapter-a",
                "adapter-b": "/path/to/adapter-b",
            },
            loaded_models=["base-model", "adapter-a"],  # adapter-b missing
        )

        engine._verify_adapters()

        # adapter-b should be removed
        assert engine._adapter_paths == {"adapter-a": "/path/to/adapter-a"}

        # Should log warnings
        assert engine._lg.warning.call_count == 2

        # First warning: specific adapter not loaded
        first_call = engine._lg.warning.call_args_list[0]
        assert "adapter not loaded by vLLM" in first_call[0][0]
        assert first_call[1]["extra"]["key"] == "adapter-b"

        # Second warning: verification complete summary
        second_call = engine._lg.warning.call_args_list[1]
        assert "verification complete" in second_call[0][0]
        assert second_call[1]["extra"]["removed"] == ["adapter-b"]
        assert second_call[1]["extra"]["available"] == ["adapter-a"]

    def test_all_adapters_missing(self) -> None:
        """Test all adapters missing from vLLM are removed."""
        engine = self._create_engine_with_mocks(
            adapter_paths={
                "adapter-a": "/path/to/adapter-a",
                "adapter-b": "/path/to/adapter-b",
            },
            loaded_models=["base-model"],  # No adapters loaded
        )

        engine._verify_adapters()

        # All adapters should be removed
        assert engine._adapter_paths == {}

        # Should log warning for each adapter + summary
        assert engine._lg.warning.call_count == 3

    def test_partial_adapters_missing(self) -> None:
        """Test some adapters loaded, others removed."""
        engine = self._create_engine_with_mocks(
            adapter_paths={
                "adapter-a": "/path/to/adapter-a",
                "adapter-b": "/path/to/adapter-b",
                "adapter-c": "/path/to/adapter-c",
            },
            loaded_models=["base-model", "adapter-b"],  # Only adapter-b loaded
        )

        engine._verify_adapters()

        # Only adapter-b should remain
        assert engine._adapter_paths == {"adapter-b": "/path/to/adapter-b"}

        # Should log warning for adapter-a and adapter-c + summary
        assert engine._lg.warning.call_count == 3


class TestVLLMServerEngineAdapterResponseVerification:
    """Test VLLMServerEngine adapter verification at request time."""

    def test_verify_adapter_response_no_adapter(self) -> None:
        """Test verification returns None when no adapter requested."""
        from llm_infer.engines.vllm_server import VLLMServerEngine

        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()

        result = engine._verify_adapter_response("base-model", None)

        assert result is None
        engine._lg.warning.assert_not_called()

    def test_verify_adapter_response_match(self) -> None:
        """Test verification returns success info when adapter matches."""
        from llm_infer.engines.vllm_server import VLLMServerEngine

        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()
        engine._adapter_metadata = {
            "my-adapter": {"mtime": "2026-01-01", "md5": "abc123"}
        }

        lora_request = MagicMock()
        lora_request.lora_name = "my-adapter"

        result = engine._verify_adapter_response("my-adapter", lora_request)

        assert result is not None
        assert result["requested"] == "my-adapter"
        assert result["actual"] == "my-adapter"
        assert result["fallback"] is False
        assert result["mtime"] == "2026-01-01"
        assert result["md5"] == "abc123"
        engine._lg.warning.assert_not_called()

    def test_verify_adapter_response_mismatch(self) -> None:
        """Test verification detects mismatch when vLLM used different model."""
        from llm_infer.engines.vllm_server import VLLMServerEngine

        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()

        lora_request = MagicMock()
        lora_request.lora_name = "my-adapter"

        result = engine._verify_adapter_response("base-model", lora_request)

        assert result is not None
        assert result["requested"] == "my-adapter"
        assert result["actual"] == "base-model"
        assert result["fallback"] is True
        engine._lg.warning.assert_called_once()
        call_extra = engine._lg.warning.call_args[1]["extra"]
        assert call_extra["requested"] == "my-adapter"
        assert call_extra["actual"] == "base-model"

    def test_parse_completion_response_with_mismatch(self) -> None:
        """Test response parsing includes adapter mismatch info."""
        from llm_infer.engines.vllm_server import VLLMServerEngine

        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()
        engine._max_model_len = None

        lora_request = MagicMock()
        lora_request.lora_name = "my-adapter"

        data = {
            "model": "base-model",  # Different from requested
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        result = engine._parse_completion_response(data, lora_request, 100)

        assert isinstance(result, dict)
        assert result["content"] == "Hello"
        assert result["adapter"]["fallback"] is True
        assert result["adapter"]["requested"] == "my-adapter"
        assert result["adapter"]["actual"] == "base-model"

    def test_parse_completion_response_success(self) -> None:
        """Test response parsing includes adapter success info with metadata."""
        from llm_infer.engines.vllm_server import VLLMServerEngine

        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()
        engine._max_model_len = None
        engine._adapter_metadata = {
            "my-adapter": {"mtime": "2026-01-01T00:00:00Z", "md5": "abc123def456"}
        }

        lora_request = MagicMock()
        lora_request.lora_name = "my-adapter"

        data = {
            "model": "my-adapter",  # Matches requested
            "choices": [{"message": {"content": "Hello"}}],
        }

        result = engine._parse_completion_response(data, lora_request, 100)

        # Returns dict with adapter info on success
        assert isinstance(result, dict)
        assert result["content"] == "Hello"
        assert result["adapter"]["fallback"] is False
        assert result["adapter"]["requested"] == "my-adapter"
        assert result["adapter"]["actual"] == "my-adapter"
        assert result["adapter"]["mtime"] == "2026-01-01T00:00:00Z"
        assert result["adapter"]["md5"] == "abc123def456"


class TestHitTokenLimit:
    """vLLM reports tool_calls for truncated output, so counts decide length."""

    def test_cap_reached(self) -> None:
        from llm_infer.engines.vllm_server import _hit_token_limit

        assert _hit_token_limit(10, 100, 100, None)

    def test_context_window_reached_without_cap(self) -> None:
        from llm_infer.engines.vllm_server import _hit_token_limit

        assert _hit_token_limit(3000, 1096, None, 4096)

    def test_under_both_limits(self) -> None:
        from llm_infer.engines.vllm_server import _hit_token_limit

        assert not _hit_token_limit(10, 50, 100, 4096)

    def test_no_limits_known(self) -> None:
        from llm_infer.engines.vllm_server import _hit_token_limit

        assert not _hit_token_limit(10, 50_000, None, None)


def _engine_for_payload() -> object:
    from llm_infer.engines.vllm_server import VLLMServerEngine

    engine = object.__new__(VLLMServerEngine)
    engine._lg = MagicMock()
    engine._model_name = "m"
    engine._config = MagicMock(chat_template_kwargs=None)
    engine._max_model_len = 4096
    engine._adapter_metadata = {}
    return engine


class TestVLLMServerMaxTokens:
    def test_payload_omits_max_tokens_when_uncapped(self) -> None:
        engine = _engine_for_payload()
        payload = engine._build_payload(None, "hi", None, 1.0, 1.0, None, False)
        assert "max_tokens" not in payload

    def test_payload_keeps_explicit_max_tokens(self) -> None:
        engine = _engine_for_payload()
        payload = engine._build_payload(None, "hi", 50, 1.0, 1.0, None, False)
        assert payload["max_tokens"] == 50

    def test_truncated_tool_call_reports_length(self) -> None:
        """vLLM says tool_calls; the context window says the output was cut off."""
        engine = _engine_for_payload()
        data = {
            "model": "m",
            "choices": [
                {
                    "message": {"content": "", "tool_calls": [{"id": "t"}]},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 4000, "completion_tokens": 96},
        }
        result = engine._parse_completion_response(data, None, None)
        assert result["finish_reason"] == "length"


def _sent_payload(engine: Any, **kwargs: Any) -> dict[str, Any]:
    """Run generate() against a stubbed server and return the payload it sent."""
    engine._post_chat_completions = MagicMock(
        return_value={"model": "m", "choices": [{"message": {"content": "ok"}}]}
    )
    engine.generate("hi", **kwargs)
    payload: dict[str, Any] = engine._post_chat_completions.call_args.args[0]
    return payload


def _engine_answering(status: int, body: dict[str, Any]) -> Any:
    """Engine whose HTTP client gets `status` / `body` from vLLM (in-memory transport)."""
    import httpx

    engine = _engine_for_payload()
    transport = httpx.MockTransport(lambda _request: httpx.Response(status, json=body))
    engine._client = httpx.Client(transport=transport, base_url="http://vllm")
    return engine


class TestVLLMServerErrorStatus:
    """vLLM's HTTP status reaches the caller instead of becoming a plain RuntimeError."""

    def test_client_error_keeps_status_and_message(self) -> None:
        from llm_infer.engines.protocol import EngineRequestError

        message = "max_tokens=40000 cannot be greater than max_model_len=32768."
        engine = _engine_answering(
            400, {"object": "error", "message": message, "code": 400}
        )
        with pytest.raises(EngineRequestError) as exc_info:
            engine._post_chat_completions({}, None)

        assert exc_info.value.status_code == 400
        assert exc_info.value.message == message

    def test_openai_error_format_message(self) -> None:
        """vLLM's OpenAI-style body nests the message under "error"."""
        from llm_infer.engines.protocol import EngineRequestError

        message = "min_tokens must be less than or equal to max_tokens=16, got 64."
        body = {"error": {"message": message, "type": "BadRequestError", "code": 400}}
        engine = _engine_answering(400, body)
        with pytest.raises(EngineRequestError) as exc_info:
            engine._post_chat_completions({}, None)

        assert exc_info.value.message == message

    def test_server_error_keeps_status(self) -> None:
        from llm_infer.engines.protocol import EngineRequestError

        engine = _engine_answering(500, {"object": "error", "message": "boom"})
        with pytest.raises(EngineRequestError) as exc_info:
            engine._post_chat_completions({}, None)

        assert exc_info.value.status_code == 500


class TestVLLMServerPassthrough:
    def test_unset_sampling_not_sent(self) -> None:
        """Left out, vLLM applies the model's generation_config."""
        payload = _sent_payload(_engine_for_payload())
        for key in ("temperature", "top_p", "top_k"):
            assert key not in payload

    def test_set_sampling_sent(self) -> None:
        payload = _sent_payload(
            _engine_for_payload(), temperature=0.3, top_p=0.9, top_k=20
        )
        assert (payload["temperature"], payload["top_p"], payload["top_k"]) == (
            0.3,
            0.9,
            20,
        )

    def test_extra_body_forwarded(self) -> None:
        extra = {"min_p": 0.05, "reasoning_effort": "low", "thinking_token_budget": 9}
        payload = _sent_payload(_engine_for_payload(), extra_body=extra)
        assert extra.items() <= payload.items()

    def test_extra_body_does_not_replace_fields_llm_infer_sets(self) -> None:
        payload = _sent_payload(
            _engine_for_payload(),
            max_tokens=10,
            extra_body={"max_tokens": 99, "model": "other", "stream": True},
        )
        assert (payload["max_tokens"], payload["model"], payload["stream"]) == (
            10,
            "m",
            False,
        )


class TestVLLMServerReasoningField:
    """vLLM returns reasoning as `reasoning`; older releases as `reasoning_content`."""

    @pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
    def test_completion_reasoning_kept(self, field: str) -> None:
        engine = _engine_for_payload()
        data = {
            "model": "m",
            "choices": [{"message": {"content": "answer", field: "why"}}],
        }
        result = engine._parse_completion_response(data, None, None)
        content = result["content"] if isinstance(result, dict) else result
        assert content == "<think>why</think>answer"

    @pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
    def test_stream_reasoning_kept(self, field: str) -> None:
        from llm_infer.engines.vllm_server import VLLMServerStreamingIterator

        it = VLLMServerStreamingIterator(
            MagicMock(), MagicMock(), "/v1/chat/completions", {}, None
        )
        text = it._bridge_reasoning_content({field: "why"})
        text += it._bridge_reasoning_content({"content": "answer"})
        assert text == "<think>why</think>answer"


class TestVLLMServerStreamingFinishReason:
    def _iterator(self, max_tokens: int | None, max_model_len: int | None) -> object:
        from llm_infer.engines.vllm_server import VLLMServerStreamingIterator

        payload = {} if max_tokens is None else {"max_tokens": max_tokens}
        return VLLMServerStreamingIterator(
            MagicMock(), MagicMock(), "/v1/chat/completions", payload, max_model_len
        )

    def _tool_call_finish(self) -> dict:
        tc = {"index": 0, "id": "t", "function": {"name": "f", "arguments": '{"a'}}
        return {
            "choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "tool_calls"}]
        }

    def test_tool_call_at_cap_reports_length(self) -> None:
        it = self._iterator(max_tokens=100, max_model_len=None)
        it._process_choice_delta(self._tool_call_finish())
        it._process_choice_delta(
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 100}}
        )
        assert it.finish_reason == "length"
        assert it.tool_calls is not None

    def test_tool_call_at_context_window_reports_length(self) -> None:
        it = self._iterator(max_tokens=None, max_model_len=4096)
        it._process_choice_delta(self._tool_call_finish())
        it._process_choice_delta(
            {"choices": [], "usage": {"prompt_tokens": 4000, "completion_tokens": 96}}
        )
        assert it.finish_reason == "length"

    def test_complete_tool_call_reports_tool_calls(self) -> None:
        it = self._iterator(max_tokens=None, max_model_len=4096)
        it._process_choice_delta(self._tool_call_finish())
        it._process_choice_delta(
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
        )
        assert it.finish_reason == "tool_calls"


_EXTRA_ARGS = (
    "--speculative-config",
    '{"method": "mtp", "num_speculative_tokens": 3}',
    "--max-num-batched-tokens",
    "8192",
)


class TestBuildServeCommand:
    """_build_serve_command() must only emit flags `vllm serve` accepts."""

    def _full_command(self) -> list[str]:
        """Serve command with every optional flag enabled."""
        from llm_infer.engines.vllm_server import VLLMServerEngine
        from llm_infer.serving.dispatch.config import (
            LoRAConfig,
            VLLMServerConfig,
            VLLMServerLoggingConfig,
        )

        # Skip __init__: it connects to (or starts) the server
        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()
        engine._config = VLLMServerConfig(
            model_path="/models/test",
            max_model_len=4096,
            quantization="awq",
            enforce_eager=True,
            reasoning_parser="qwen3",
            chat_template_kwargs={"enable_thinking": False},
            lora=LoRAConfig(enabled=True),
            logging=VLLMServerLoggingConfig(mode="capture"),
            extra_args=list(_EXTRA_ARGS),
        )
        engine._model_name = "test"
        engine._adapter_paths = {"a": "/adapters/a", "b": "/adapters/b"}
        return engine._build_serve_command()

    def test_installed_vllm_parses_command(self) -> None:
        """The installed vLLM's own `serve` parser accepts the command."""
        # Skip only without vllm; a moved module must fail, not skip
        pytest.importorskip("vllm", reason="vllm not installed")
        from vllm.entrypoints.cli.serve import ServeSubcommand
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser(prog="vllm")
        ServeSubcommand().subparser_init(parser.add_subparsers())

        # argparse exits with code 2 on an unrecognized flag or invalid choice
        parser.parse_args(self._full_command()[1:])

    def test_optional_flags_emitted_when_set(self) -> None:
        """Optional flags reach the command, so the parse check above covers them."""
        cmd = self._full_command()

        for flag in (
            "--max-model-len",
            "--quantization",
            "--enforce-eager",
            "--reasoning-parser",
            "--default-chat-template-kwargs",
            "--enable-lora",
            "--lora-modules",
            "--logging-config.formatter",
        ):
            assert flag in cmd

    def test_extra_args_appended_last(self) -> None:
        """Extra args come after every flag llm-infer sets, so they win."""
        assert self._full_command()[-len(_EXTRA_ARGS) :] == list(_EXTRA_ARGS)


class TestCheckedExtraArgs:
    """_checked_extra_args() validates user-supplied `vllm serve` arguments."""

    @pytest.mark.parametrize(
        "arg",
        [
            "--port",
            "--port=9000",
            "--served-model-name",
            "--served_model_name=x",
            "--host",
            "--host=0.0.0.0",
            "--uds",
            "--uds=/tmp/vllm.sock",
            "--api-key",
            "--api-key=secret",
        ],
    )
    def test_reserved_flags_rejected(self, arg: str) -> None:
        from llm_infer.engines.vllm_server import _checked_extra_args

        with pytest.raises(ValueError, match="llm-infer manages"):
            _checked_extra_args([arg, "9000"])

    def test_string_rejected(self) -> None:
        from llm_infer.engines.vllm_server import _checked_extra_args

        with pytest.raises(ValueError, match="must be a list"):
            _checked_extra_args("--enforce-eager")  # type: ignore[arg-type]

    def test_values_stringified(self) -> None:
        """YAML turns numbers into ints; argv needs strings."""
        from llm_infer.engines.vllm_server import _checked_extra_args

        assert _checked_extra_args(["--max-num-batched-tokens", 8192]) == [
            "--max-num-batched-tokens",
            "8192",
        ]

    def test_port_lookalike_allowed(self) -> None:
        """Only exact reserved names are rejected, not flags sharing a prefix."""
        from llm_infer.engines.vllm_server import _checked_extra_args

        args = ["--port-range", "1"]
        assert _checked_extra_args(args) == args

    def test_none_returns_empty_list(self) -> None:
        """YAML `extra_args: null` or `extra_args:` (no value) yields None."""
        from llm_infer.engines.vllm_server import _checked_extra_args

        assert _checked_extra_args(None) == []  # type: ignore[arg-type]


def _start_server(
    monkeypatch: pytest.MonkeyPatch,
    mode: str = "stderr",
    config_file: str | None = None,
) -> tuple[Any, MagicMock, MagicMock]:
    """Run _start_server() with Popen and the output reader mocked.

    Returns the engine, the Popen mock and the VLLMOutputReader mock.
    """
    from llm_infer.engines import vllm_server
    from llm_infer.serving.dispatch.config import (
        VLLMServerConfig,
        VLLMServerLoggingConfig,
    )

    popen = MagicMock()
    reader = MagicMock()
    monkeypatch.setattr(vllm_server.subprocess, "Popen", popen)
    monkeypatch.setattr(vllm_server, "VLLMOutputReader", reader)
    engine = object.__new__(vllm_server.VLLMServerEngine)
    engine._lg = MagicMock()
    engine._config = VLLMServerConfig(
        model_path="/models/test",
        logging=VLLMServerLoggingConfig(mode=mode, config_file=config_file),
    )
    engine._model_name = "test"
    engine._adapter_paths = {}
    engine._output = None
    engine._wait_for_server = MagicMock()
    if mode != "capture":
        # A real Popen leaves stdout None unless it is piped
        popen.return_value.stdout = None
    engine._start_server()
    return engine, popen, reader


def _serve_env(
    monkeypatch: pytest.MonkeyPatch,
    mode: str = "stderr",
    config_file: str | None = None,
) -> dict[str, str]:
    """The environment _start_server() gives `vllm serve`."""
    _, popen, _ = _start_server(monkeypatch, mode, config_file)
    return dict(popen.call_args.kwargs["env"])


class TestVLLMServerLoggingEnv:
    """vLLM's logging follows engines.vllm_server.logging, not serve's own setup."""

    def test_serve_process_setting_does_not_reach_vllm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """serve sets VLLM_CONFIGURE_LOGGING=0 for itself; `vllm serve` still logs."""
        monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")
        env = _serve_env(monkeypatch)

        assert env["VLLM_CONFIGURE_LOGGING"] == "1"
        assert "VLLM_LOGGING_CONFIG_PATH" not in env

    def test_drop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _serve_env(monkeypatch, "drop")["VLLM_CONFIGURE_LOGGING"] == "0"

    def test_config_file_reaches_vllm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = _serve_env(monkeypatch, "vllm", "/etc/vllm-logging.json")

        assert env["VLLM_CONFIGURE_LOGGING"] == "1"
        assert env["VLLM_LOGGING_CONFIG_PATH"] == "/etc/vllm-logging.json"

    def test_inherited_config_path_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The config decides vLLM's logging, not serve's environment."""
        monkeypatch.setenv("VLLM_LOGGING_CONFIG_PATH", "/elsewhere.json")

        assert "VLLM_LOGGING_CONFIG_PATH" not in _serve_env(monkeypatch)


class TestVLLMServerOutputModes:
    """Where the `vllm serve` process's output goes, per logging mode."""

    @pytest.mark.parametrize(
        ("mode", "stream"),
        [("stderr", "stderr"), ("stdout", "stdout"), ("drop", "stderr")],
    )
    def test_passthrough(
        self, monkeypatch: pytest.MonkeyPatch, mode: str, stream: str
    ) -> None:
        """The process writes to serve's own stream; nothing reads it."""
        import sys

        _, popen, reader = _start_server(monkeypatch, mode)

        kwargs = popen.call_args.kwargs
        assert kwargs["stdout"] is getattr(sys, stream)
        assert kwargs["stderr"] is getattr(sys, stream)
        assert "--logging-config.formatter" not in popen.call_args.args[0]
        reader.assert_not_called()

    def test_vllm_mode_passes_other_output_to_stderr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        _, popen, reader = _start_server(monkeypatch, "vllm", "/etc/vllm-logging.json")

        assert popen.call_args.kwargs["stdout"] is sys.stderr
        reader.assert_not_called()

    def test_capture_pipes_json_records_to_reader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both streams into one pipe, vLLM formats JSON, a reader is started."""
        import subprocess

        engine, popen, reader = _start_server(monkeypatch, "capture")

        kwargs = popen.call_args.kwargs
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.STDOUT
        cmd = popen.call_args.args[0]
        flag = cmd.index("--logging-config.formatter")
        assert cmd[flag + 1] == "json"
        reader.assert_called_once_with(engine._lg, popen.return_value.stdout)
        reader.return_value.start.assert_called_once()

    def test_override_checked_at_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """-o sets the field after the load-time check; the start checks again."""
        from llm_infer.engines import vllm_server
        from llm_infer.serving.dispatch.config import VLLMServerConfig

        popen = MagicMock()
        monkeypatch.setattr(vllm_server.subprocess, "Popen", popen)
        engine = object.__new__(vllm_server.VLLMServerEngine)
        engine._lg = MagicMock()
        engine._config = VLLMServerConfig(model_path="/models/test")
        engine._config.logging.mode = "vllm"  # as -o would, without config_file

        with pytest.raises(ValueError, match="mode vllm requires config_file"):
            engine._start_server()
        popen.assert_not_called()


class TestVLLMServerLoggingConfig:
    def test_parsed_from_section(self) -> None:
        from llm_infer.serving.dispatch.config import VLLMServerConfig

        config = VLLMServerConfig.from_dict(
            {"logging": {"mode": "vllm", "config_file": "/etc/vllm-logging.json"}}
        )

        assert config.logging.mode == "vllm"
        assert config.logging.config_file == "/etc/vllm-logging.json"

    def test_defaults_without_section(self) -> None:
        """Without config, output passes through to stderr: safe for production."""
        from llm_infer.serving.dispatch.config import VLLMServerConfig

        config = VLLMServerConfig.from_dict({})

        assert config.logging.mode == "stderr"
        assert config.logging.config_file is None

    def test_unknown_mode_rejected(self) -> None:
        from llm_infer.serving.dispatch.config import VLLMServerLoggingConfig

        with pytest.raises(ValueError, match="mode must be one of"):
            VLLMServerLoggingConfig(mode="file")

    def test_vllm_mode_requires_config_file(self) -> None:
        from llm_infer.serving.dispatch.config import VLLMServerLoggingConfig

        with pytest.raises(ValueError, match="mode vllm requires config_file"):
            VLLMServerLoggingConfig(mode="vllm")

    def test_config_file_requires_vllm_mode(self) -> None:
        """A file another mode would ignore is an error, not a silent no-op."""
        from llm_infer.serving.dispatch.config import VLLMServerLoggingConfig

        with pytest.raises(ValueError, match="config_file requires mode vllm"):
            VLLMServerLoggingConfig(mode="capture", config_file="/x.json")


class TestVLLMOutputParseLine:
    """parse_line(): one line of captured `vllm serve` output to a log call."""

    def test_vllm_record_keeps_level_and_fields(self) -> None:
        from llm_infer.engines.vllm_server_output import parse_line

        line = (
            '{"asctime": "2026-11-10 22:00:49,701", "levelname": "ERROR", '
            '"name": "vllm.engine.core", "processName": "MainProcess", '
            '"process": 42, "message": "boom", "exc_info": "Traceback ...", '
            '"vllm_process_name": "EngineCore"}'
        )

        assert parse_line(line) == (
            "error",
            "boom",
            {
                "logger": "vllm.engine.core",
                "vllm_process": "EngineCore",
                "pid": 42,
                "traceback": "Traceback ...",
            },
        )

    def test_extra_avoids_log_record_attributes(self) -> None:
        """Logger rejects extra keys that are LogRecord attributes (ReservedKeyError)."""
        import logging

        from llm_infer.engines.vllm_server_output import parse_line

        reserved = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
        line = (
            '{"asctime": "t", "levelname": "INFO", "name": "vllm.x", '
            '"processName": "MainProcess", "process": 42, "message": "m", '
            '"exc_info": "tb", "vllm_process_name": "EngineCore"}'
        )

        assert not set(parse_line(line)[2]) & reserved

    @pytest.mark.parametrize(
        ("levelname", "method"),
        [
            ("DEBUG", "debug"),
            ("INFO", "info"),
            ("WARNING", "warning"),
            ("CRITICAL", "critical"),
            ("NOTICE", "debug"),
        ],
    )
    def test_level_translated(self, levelname: str, method: str) -> None:
        from llm_infer.engines.vllm_server_output import parse_line

        line = f'{{"levelname": "{levelname}", "message": "m"}}'

        assert parse_line(line)[0] == method

    @pytest.mark.parametrize(
        "line",
        [
            # Another library's output (torch's C++ logging)
            "W1110 22:00:47.317000 2328594 torch/_opaque_base.py:6] deprecated",
            # A record cut by an interleaved write
            '{"levelname": "INFO", "message": "par',
            # JSON, but not a log record
            "[1, 2]",
            '{"levelname": "INFO"}',
        ],
    )
    def test_other_output_logged_raw_at_debug(self, line: str) -> None:
        from llm_infer.engines.vllm_server_output import parse_line

        assert parse_line(line) == ("debug", line, {})


class TestVLLMServerRelease:
    """release() / reacquire(): stop and start the backend while serve runs."""

    def _engine(self, owns: bool) -> Any:
        from llm_infer.engines.vllm_server import VLLMServerEngine

        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()
        engine._owns_process = owns
        engine._process = None
        return engine

    def test_external_backend_is_not_managed(self) -> None:
        from llm_infer.engines.protocol import BackendNotManagedError

        engine = self._engine(owns=False)

        assert engine.manages_backend is False
        with pytest.raises(BackendNotManagedError):
            engine.release()
        with pytest.raises(BackendNotManagedError):
            engine.reacquire()

    def test_release_stops_server(self) -> None:
        engine = self._engine(owns=True)
        engine._stop_server = MagicMock()

        engine.release()

        engine._stop_server.assert_called_once()

    def test_reacquire_starts_and_verifies(self) -> None:
        engine = self._engine(owns=True)
        for name in ("_start_server", "_verify_connection", "_verify_adapters"):
            setattr(engine, name, MagicMock())

        engine.reacquire()

        engine._start_server.assert_called_once()
        engine._verify_connection.assert_called_once()

    def test_failed_reacquire_stops_what_it_started(self) -> None:
        engine = self._engine(owns=True)
        engine._start_server = MagicMock(side_effect=RuntimeError("exited"))
        engine._stop_server = MagicMock()

        with pytest.raises(RuntimeError):
            engine.reacquire()

        engine._stop_server.assert_called_once()

    def test_reacquire_while_running_is_a_noop(self) -> None:
        engine = self._engine(owns=True)
        engine._process = MagicMock()
        engine._start_server = MagicMock()

        engine.reacquire()

        engine._start_server.assert_not_called()


class TestVLLMServerSetAdapters:
    """set_adapters(): the adapters the next server start registers."""

    def _engine(self) -> Any:
        from llm_infer.engines.vllm_server import VLLMServerEngine
        from llm_infer.serving.dispatch.config import LoRAConfig, VLLMServerConfig

        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()
        engine._config = VLLMServerConfig(
            model_path="/models/test", lora=LoRAConfig(enabled=True)
        )
        engine._model_name = "test"
        engine._process = None
        engine._adapter_paths = {}
        engine._adapter_metadata = {}
        return engine

    @staticmethod
    def _adapter(key: str) -> Any:
        from pathlib import Path

        from llm_infer.serving.adapters import LoadedAdapter

        return LoadedAdapter(key=key, name=key, path=Path(f"/adapters/{key}"))

    def _lora_modules(self, engine: Any) -> list[str]:
        """The adapter specs after --lora-modules, up to the next flag."""
        cmd = engine._build_serve_command()
        specs = cmd[cmd.index("--lora-modules") + 1 :]
        return list(itertools.takewhile(lambda arg: not arg.startswith("--"), specs))

    def test_replaces_adapters_for_next_start(self) -> None:
        engine = self._engine()
        engine.set_adapters([self._adapter("old"), self._adapter("kept")])

        engine.set_adapters([self._adapter("kept"), self._adapter("new")])

        assert self._lora_modules(engine) == [
            "kept=/adapters/kept",
            "new=/adapters/new",
        ]
        assert set(engine._adapter_metadata) == {"kept", "new"}

    def test_empty_list_registers_no_adapters(self) -> None:
        engine = self._engine()
        engine.set_adapters([self._adapter("old")])

        engine.set_adapters([])

        assert "--lora-modules" not in engine._build_serve_command()

    def test_refused_while_server_runs(self) -> None:
        engine = self._engine()
        engine.set_adapters([self._adapter("old")])
        engine._process = MagicMock()

        with pytest.raises(RuntimeError):
            engine.set_adapters([self._adapter("new")])

        assert engine._adapter_paths == {"old": "/adapters/old"}
