# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for vLLM server inference engine."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


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
        from llm_infer.serving.dispatch.config import LoRAConfig, VLLMServerConfig

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
        ):
            assert flag in cmd

    def test_extra_args_appended_last(self) -> None:
        """Extra args come after every flag llm-infer sets, so they win."""
        assert self._full_command()[-len(_EXTRA_ARGS) :] == list(_EXTRA_ARGS)


class TestCheckedExtraArgs:
    """_checked_extra_args() validates user-supplied `vllm serve` arguments."""

    @pytest.mark.parametrize(
        "arg",
        ["--port", "--port=9000", "--served-model-name", "--served_model_name=x"],
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
