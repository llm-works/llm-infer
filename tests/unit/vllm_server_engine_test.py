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
        assert first_call[1]["extra"]["adapter_id"] == "adapter-b"

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
        """Test verification returns no mismatch when no adapter requested."""
        from llm_infer.engines.vllm_server import VLLMServerEngine

        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()

        mismatch, requested = engine._verify_adapter_response("base-model", None)

        assert mismatch is False
        assert requested is None
        engine._lg.warning.assert_not_called()

    def test_verify_adapter_response_match(self) -> None:
        """Test verification returns no mismatch when adapter matches."""
        from llm_infer.engines.vllm_server import VLLMServerEngine

        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()
        engine._adapter_metadata = {
            "my-adapter": {"mtime": "2026-01-01", "md5": "abc123"}
        }

        # Mock lora_request with lora_name attribute
        lora_request = MagicMock()
        lora_request.lora_name = "my-adapter"

        mismatch, requested = engine._verify_adapter_response(
            "my-adapter", lora_request
        )

        assert mismatch is False
        assert requested is None
        engine._lg.warning.assert_not_called()

    def test_verify_adapter_response_mismatch(self) -> None:
        """Test verification detects mismatch when vLLM used different model."""
        from llm_infer.engines.vllm_server import VLLMServerEngine

        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()

        lora_request = MagicMock()
        lora_request.lora_name = "my-adapter"

        mismatch, requested = engine._verify_adapter_response(
            "base-model", lora_request
        )

        assert mismatch is True
        assert requested == "my-adapter"
        engine._lg.warning.assert_called_once()
        call_extra = engine._lg.warning.call_args[1]["extra"]
        assert call_extra["requested"] == "my-adapter"
        assert call_extra["actual"] == "base-model"

    def test_parse_completion_response_with_mismatch(self) -> None:
        """Test response parsing includes adapter mismatch info."""
        from llm_infer.engines.vllm_server import VLLMServerEngine

        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()

        lora_request = MagicMock()
        lora_request.lora_name = "my-adapter"

        data = {
            "model": "base-model",  # Different from requested
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        result = engine._parse_completion_response(data, lora_request)

        assert isinstance(result, dict)
        assert result["content"] == "Hello"
        assert result["adapter_mismatch"] is True
        assert result["adapter_requested"] == "my-adapter"

    def test_parse_completion_response_no_mismatch(self) -> None:
        """Test response parsing has no mismatch when adapter matches."""
        from llm_infer.engines.vllm_server import VLLMServerEngine

        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()
        engine._adapter_metadata = {
            "my-adapter": {"mtime": "2026-01-01", "md5": "abc123"}
        }

        lora_request = MagicMock()
        lora_request.lora_name = "my-adapter"

        data = {
            "model": "my-adapter",  # Matches requested
            "choices": [{"message": {"content": "Hello"}}],
        }

        result = engine._parse_completion_response(data, lora_request)

        # Returns string when no extra info needed
        assert result == "Hello"
