"""Unit tests for Ollama inference engine."""

from unittest.mock import MagicMock

import pytest

from llm_infer.engines.ollama import OllamaStreamingIterator
from llm_infer.serving.dispatch.config import OllamaConfig

pytestmark = pytest.mark.unit


# -----------------------------------------------------------------------------
# OllamaConfig tests
# -----------------------------------------------------------------------------


class TestOllamaConfigFromDict:
    """Test OllamaConfig.from_dict factory."""

    def test_default_values(self) -> None:
        """Test config uses defaults for empty dict."""
        config = OllamaConfig.from_dict({})
        assert config.model == ""
        assert config.task == "generate"
        assert config.host == "http://localhost:11434"
        assert config.timeout == 300
        assert config.models_path is None
        assert config.keep_alive == "5m"
        assert config.num_ctx is None
        assert config.num_gpu is None
        assert config.warmup is True
        assert config.auto_start is True
        assert config.binary_path == "ollama"

    def test_custom_values(self) -> None:
        """Test config parses custom values."""
        config = OllamaConfig.from_dict(
            {
                "model": "llama3.2:3b",
                "task": "embed",
                "host": "http://remote:11434",
                "timeout": 600,
                "models_path": "/custom/models",
                "keep_alive": "10m",
                "num_ctx": 4096,
                "num_gpu": 1,
                "warmup": False,
                "auto_start": False,
                "binary_path": "/usr/local/bin/ollama",
            }
        )
        assert config.model == "llama3.2:3b"
        assert config.task == "embed"
        assert config.host == "http://remote:11434"
        assert config.timeout == 600
        assert config.models_path == "/custom/models"
        assert config.keep_alive == "10m"
        assert config.num_ctx == 4096
        assert config.num_gpu == 1
        assert config.warmup is False
        assert config.auto_start is False
        assert config.binary_path == "/usr/local/bin/ollama"

    def test_model_parameter_override(self) -> None:
        """Test model parameter overrides dict value."""
        config = OllamaConfig.from_dict(
            {"model": "from-dict"},
            model="from-param",
        )
        assert config.model == "from-param"

    def test_model_parameter_fallback(self) -> None:
        """Test model falls back to dict when param empty."""
        config = OllamaConfig.from_dict(
            {"model": "from-dict"},
            model="",
        )
        assert config.model == "from-dict"


# -----------------------------------------------------------------------------
# OllamaStreamingIterator tests
# -----------------------------------------------------------------------------


class TestOllamaStreamingIteratorProcessLine:
    """Test OllamaStreamingIterator._process_stream_line method."""

    def _create_iterator(self) -> OllamaStreamingIterator:
        """Create iterator with mocked dependencies."""
        mock_lg = MagicMock()
        mock_client = MagicMock()
        return OllamaStreamingIterator(
            lg=mock_lg,
            client=mock_client,
            url="/api/generate",
            payload={"model": "test"},
        )

    def test_empty_line_returns_none(self) -> None:
        """Test empty lines are skipped."""
        iterator = self._create_iterator()
        assert iterator._process_stream_line("") is None
        assert iterator._process_stream_line("   ") is None
        assert iterator._process_stream_line("\n") is None

    def test_generate_format_extracts_response(self) -> None:
        """Test parsing /api/generate response format."""
        iterator = self._create_iterator()
        result = iterator._process_stream_line('{"response": "Hello", "done": false}')
        assert result == "Hello"

    def test_chat_format_extracts_content(self) -> None:
        """Test parsing /api/chat response format."""
        iterator = self._create_iterator()
        result = iterator._process_stream_line(
            '{"message": {"content": "World"}, "done": false}'
        )
        assert result == "World"

    def test_empty_response_returns_none(self) -> None:
        """Test empty response text returns None."""
        iterator = self._create_iterator()
        result = iterator._process_stream_line('{"response": "", "done": false}')
        assert result is None

    def test_done_true_sets_finished(self) -> None:
        """Test done=true sets finish state and raises StopIteration."""
        iterator = self._create_iterator()
        with pytest.raises(StopIteration):
            iterator._process_stream_line(
                '{"done": true, "prompt_eval_count": 10, "eval_count": 20}'
            )
        assert iterator._finished is True
        assert iterator.prompt_tokens == 10
        assert iterator.completion_tokens == 20
        assert iterator.finish_reason == "stop"

    def test_done_with_length_reason(self) -> None:
        """Test done_reason=length sets finish_reason correctly."""
        iterator = self._create_iterator()
        with pytest.raises(StopIteration):
            iterator._process_stream_line(
                '{"done": true, "done_reason": "length", "eval_count": 100}'
            )
        assert iterator.finish_reason == "length"

    def test_done_with_final_content_returns_it(self) -> None:
        """Test done=true with final content returns that content."""
        iterator = self._create_iterator()
        # When done=true has final response text, it should return it before stopping
        result = iterator._process_stream_line(
            '{"response": "final", "done": true, "eval_count": 1}'
        )
        assert result == "final"
        assert iterator._finished is True

    def test_malformed_json_raises(self) -> None:
        """Test malformed JSON raises JSONDecodeError."""
        import json

        iterator = self._create_iterator()
        with pytest.raises(json.JSONDecodeError):
            iterator._process_stream_line("{malformed json")


class TestOllamaStreamingIteratorNext:
    """Test OllamaStreamingIterator.__next__ method."""

    def test_json_decode_error_cleanup_and_reraise(self) -> None:
        """Test JSONDecodeError triggers cleanup and raises RuntimeError."""
        mock_lg = MagicMock()
        mock_client = MagicMock()

        iterator = OllamaStreamingIterator(
            lg=mock_lg,
            client=mock_client,
            url="/api/generate",
            payload={"model": "test"},
        )

        # Mock the stream to return malformed JSON
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter(["{malformed json"])

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_response)
        mock_stream_ctx.__exit__ = MagicMock(return_value=None)
        mock_client.stream.return_value = mock_stream_ctx

        with pytest.raises(RuntimeError, match="Ollama returned malformed JSON"):
            next(iterator)

        # Verify cleanup was called
        assert iterator._finished is True
        mock_lg.warning.assert_called_once()
        # Verify warning includes context
        call_kwargs = mock_lg.warning.call_args
        assert "malformed JSON" in call_kwargs[0][0]

    def test_iterates_through_chunks(self) -> None:
        """Test iterator yields chunks from stream."""
        mock_lg = MagicMock()
        mock_client = MagicMock()

        iterator = OllamaStreamingIterator(
            lg=mock_lg,
            client=mock_client,
            url="/api/generate",
            payload={"model": "test"},
        )

        # Mock stream with multiple chunks
        lines = [
            '{"response": "Hello", "done": false}',
            '{"response": " ", "done": false}',
            '{"response": "World", "done": false}',
            '{"done": true, "eval_count": 3}',
        ]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter(lines)

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_response)
        mock_stream_ctx.__exit__ = MagicMock(return_value=None)
        mock_client.stream.return_value = mock_stream_ctx

        chunks = list(iterator)
        assert chunks == ["Hello", " ", "World"]
        assert iterator.completion_tokens == 3

    def test_skips_empty_chunks(self) -> None:
        """Test iterator skips empty response chunks."""
        mock_lg = MagicMock()
        mock_client = MagicMock()

        iterator = OllamaStreamingIterator(
            lg=mock_lg,
            client=mock_client,
            url="/api/generate",
            payload={"model": "test"},
        )

        lines = [
            '{"response": "", "done": false}',  # Empty, should skip
            '{"response": "content", "done": false}',
            '{"done": true}',
        ]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter(lines)

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_response)
        mock_stream_ctx.__exit__ = MagicMock(return_value=None)
        mock_client.stream.return_value = mock_stream_ctx

        chunks = list(iterator)
        assert chunks == ["content"]


class TestOllamaStreamingIteratorContextManager:
    """Test OllamaStreamingIterator context manager protocol."""

    def test_context_manager_cleanup(self) -> None:
        """Test context manager calls cleanup on exit."""
        mock_lg = MagicMock()
        mock_client = MagicMock()

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=MagicMock())
        mock_stream_ctx.__exit__ = MagicMock(return_value=None)
        mock_client.stream.return_value = mock_stream_ctx

        iterator = OllamaStreamingIterator(
            lg=mock_lg,
            client=mock_client,
            url="/api/generate",
            payload={"model": "test"},
        )

        # Start the stream
        iterator._start_stream()

        # Use as context manager
        with iterator:
            pass  # Exit immediately

        # Verify cleanup was called
        mock_stream_ctx.__exit__.assert_called_once()


# -----------------------------------------------------------------------------
# OllamaEngineFactory tests
# -----------------------------------------------------------------------------


class TestOllamaEngineFactoryGetModelName:
    """Test OllamaEngineFactory._get_ollama_model_name method."""

    def _create_mock_config(
        self,
        model_path: str | None = None,
        model_ollama_field: str | None = None,
        engines_ollama_model: str = "",
    ) -> MagicMock:
        """Create mock InferenceConfig."""
        config = MagicMock()
        config.models.path = model_path

        # Mock models.get() to return a ModelConfig with ollama field
        mock_model_config = MagicMock()
        mock_model_config.ollama = model_ollama_field
        config.models.get.return_value = mock_model_config

        config.engines.ollama.model = engines_ollama_model
        return config

    def test_resolves_from_models_yaml_ollama_field(self) -> None:
        """Test resolution from models.yaml ollama field."""
        from llm_infer.serving.dispatch.factories import OllamaEngineFactory

        factory = OllamaEngineFactory()
        mock_lg = MagicMock()
        config = self._create_mock_config(
            model_path="/models/qwen2.5-7b",
            model_ollama_field="qwen2.5:7b",
        )

        result = factory._get_ollama_model_name(mock_lg, config)

        assert result == "qwen2.5:7b"
        config.models.get.assert_called_with("qwen2.5-7b")

    def test_falls_back_to_engines_ollama_model(self) -> None:
        """Test fallback to engines.ollama.model when no ollama field."""
        from llm_infer.serving.dispatch.factories import OllamaEngineFactory

        factory = OllamaEngineFactory()
        mock_lg = MagicMock()
        config = self._create_mock_config(
            model_path="/models/some-model",
            model_ollama_field=None,  # No ollama field in models.yaml
            engines_ollama_model="fallback:latest",
        )

        result = factory._get_ollama_model_name(mock_lg, config)

        assert result == "fallback:latest"

    def test_falls_back_to_model_name_as_is(self) -> None:
        """Test fallback to model name as-is when no mappings."""
        from llm_infer.serving.dispatch.factories import OllamaEngineFactory

        factory = OllamaEngineFactory()
        mock_lg = MagicMock()
        config = self._create_mock_config(
            model_path="/models/llama3.2",
            model_ollama_field=None,
            engines_ollama_model="",
        )

        result = factory._get_ollama_model_name(mock_lg, config)

        assert result == "llama3.2"
        # Should log warning about using model name as-is
        mock_lg.warning.assert_called_once()

    def test_raises_when_no_model(self) -> None:
        """Test raises ValueError when no model specified."""
        from llm_infer.serving.dispatch.factories import OllamaEngineFactory

        factory = OllamaEngineFactory()
        mock_lg = MagicMock()
        config = self._create_mock_config(
            model_path=None,
            model_ollama_field=None,
            engines_ollama_model="",
        )

        with pytest.raises(ValueError, match="Model name required"):
            factory._get_ollama_model_name(mock_lg, config)

    def test_falls_back_when_model_not_in_models_yaml(self) -> None:
        """Test fallback when model isn't defined in models.yaml."""
        from llm_infer.serving.dispatch.factories import OllamaEngineFactory

        factory = OllamaEngineFactory()
        mock_lg = MagicMock()

        # Model not in models.yaml - models.get() returns config with ollama=None
        config = MagicMock()
        config.models.path = "/models/unknown-model"
        config.models.get.return_value = MagicMock(ollama=None)
        config.engines.ollama.model = "fallback:latest"

        result = factory._get_ollama_model_name(mock_lg, config)

        assert result == "fallback:latest"
        config.models.get.assert_called_with("unknown-model")


class TestOllamaEngineFactoryWarmup:
    """Test OllamaEngineFactory.warmup_enabled method."""

    def test_warmup_enabled_true(self) -> None:
        """Test warmup_enabled returns config value."""
        from llm_infer.serving.dispatch.factories import OllamaEngineFactory

        factory = OllamaEngineFactory()
        config = MagicMock()
        config.engines.ollama.warmup = True

        assert factory.warmup_enabled(config) is True

    def test_warmup_enabled_false(self) -> None:
        """Test warmup_enabled returns config value."""
        from llm_infer.serving.dispatch.factories import OllamaEngineFactory

        factory = OllamaEngineFactory()
        config = MagicMock()
        config.engines.ollama.warmup = False

        assert factory.warmup_enabled(config) is False


class TestOllamaEngineFactoryMaxBatchSize:
    """Test OllamaEngineFactory.max_batch_size method."""

    def test_returns_one(self) -> None:
        """Test max_batch_size returns 1 (Ollama handles batching internally)."""
        from llm_infer.serving.dispatch.factories import OllamaEngineFactory

        factory = OllamaEngineFactory()
        config = MagicMock()

        assert factory.max_batch_size(config) == 1
