"""Unit tests for ServeTool model override functionality."""

from collections.abc import Generator
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from llm_infer.cli.tools.serve import ServeTool
from llm_infer.models.config import ModelConfig, ModelsConfig
from llm_infer.serving.dispatch.config import VLLMConfig

pytestmark = pytest.mark.unit


@dataclass
class MockEnginesConfig:
    """Mock engines config for testing."""

    vllm: VLLMConfig = field(default_factory=VLLMConfig)


@dataclass
class MockInferenceConfig:
    """Mock inference config for testing."""

    models: ModelsConfig = field(default_factory=ModelsConfig)
    engines: MockEnginesConfig = field(default_factory=MockEnginesConfig)


class TestApplyModelOverrides:
    """Tests for ServeTool._apply_model_overrides."""

    @pytest.fixture
    def serve_tool(self) -> Generator[tuple[ServeTool, MagicMock], None, None]:
        """Create ServeTool instance with mocked logger."""
        with patch.object(ServeTool, "lg", new_callable=lambda: MagicMock()) as mock_lg:
            tool = ServeTool(parent=None)
            yield tool, mock_lg

    @pytest.fixture
    def config_with_model(self) -> MockInferenceConfig:
        """Create config with a model that has vllm overrides."""
        model_cfg = ModelConfig(
            name="test-model",
            task="embed",
            max_model_len=512,
            vllm={"enable_prefix_caching": False, "gpu_memory_utilization": 0.5},
        )
        # Mark max_model_len as explicitly set
        model_cfg._max_model_len_set = True

        models = ModelsConfig(models={"test-model": model_cfg})
        return MockInferenceConfig(models=models)

    def test_applies_task_override(
        self, serve_tool: tuple[ServeTool, MagicMock]
    ) -> None:
        """Test that task is overridden from model config."""
        tool, mock_lg = serve_tool
        model_cfg = ModelConfig(name="embed-model", task="embed")
        models = ModelsConfig(models={"embed-model": model_cfg})
        config = MockInferenceConfig(models=models)

        tool._apply_model_overrides(config, "embed-model")

        assert config.engines.vllm.task == "embed"
        mock_lg.debug.assert_called()

    def test_applies_max_model_len_override(
        self, serve_tool: tuple[ServeTool, MagicMock]
    ) -> None:
        """Test that max_model_len is overridden when explicitly set."""
        tool, mock_lg = serve_tool
        model_cfg = ModelConfig(name="test-model", max_model_len=1024)
        model_cfg._max_model_len_set = True
        models = ModelsConfig(models={"test-model": model_cfg})
        config = MockInferenceConfig(models=models)

        tool._apply_model_overrides(config, "test-model")

        assert config.engines.vllm.max_model_len == 1024

    def test_applies_vllm_overrides(
        self,
        serve_tool: tuple[ServeTool, MagicMock],
        config_with_model: MockInferenceConfig,
    ) -> None:
        """Test that vllm dict overrides are applied."""
        tool, mock_lg = serve_tool
        tool._apply_model_overrides(config_with_model, "test-model")

        assert config_with_model.engines.vllm.enable_prefix_caching is False
        assert config_with_model.engines.vllm.gpu_memory_utilization == 0.5

    def test_warns_on_unknown_vllm_key(
        self, serve_tool: tuple[ServeTool, MagicMock]
    ) -> None:
        """Test that unknown vllm keys trigger a warning."""
        tool, mock_lg = serve_tool
        model_cfg = ModelConfig(
            name="test-model",
            vllm={"nonexistent_key": "value"},
        )
        models = ModelsConfig(models={"test-model": model_cfg})
        config = MockInferenceConfig(models=models)

        tool._apply_model_overrides(config, "test-model")

        mock_lg.warning.assert_called_once()
        call_args = mock_lg.warning.call_args
        assert "unknown vllm config key" in call_args[0][0]
        assert call_args[1]["extra"]["key"] == "nonexistent_key"

    def test_skips_missing_model(self, serve_tool: tuple[ServeTool, MagicMock]) -> None:
        """Test that missing model config is handled gracefully."""
        tool, mock_lg = serve_tool
        config = MockInferenceConfig()

        # Should not raise
        tool._apply_model_overrides(config, "nonexistent-model")

        mock_lg.debug.assert_not_called()
        mock_lg.warning.assert_not_called()

    def test_empty_vllm_overrides(
        self, serve_tool: tuple[ServeTool, MagicMock]
    ) -> None:
        """Test that empty vllm dict doesn't cause issues."""
        tool, mock_lg = serve_tool
        model_cfg = ModelConfig(name="test-model", vllm={})
        models = ModelsConfig(models={"test-model": model_cfg})
        config = MockInferenceConfig(models=models)

        # Should not raise
        tool._apply_model_overrides(config, "test-model")

    def test_all_valid_vllm_keys(self, serve_tool: tuple[ServeTool, MagicMock]) -> None:
        """Test that all VLLMConfig fields can be overridden."""
        tool, mock_lg = serve_tool
        valid_overrides = {
            "task": "embed",
            "gpu_memory_utilization": 0.8,
            "max_num_seqs": 128,
            "enable_prefix_caching": False,
            "enforce_eager": True,
            "tensor_parallel_size": 2,
        }
        model_cfg = ModelConfig(name="test-model", vllm=valid_overrides)
        models = ModelsConfig(models={"test-model": model_cfg})
        config = MockInferenceConfig(models=models)

        tool._apply_model_overrides(config, "test-model")

        assert config.engines.vllm.task == "embed"
        assert config.engines.vllm.gpu_memory_utilization == 0.8
        assert config.engines.vllm.max_num_seqs == 128
        assert config.engines.vllm.enable_prefix_caching is False
        assert config.engines.vllm.enforce_eager is True
        assert config.engines.vllm.tensor_parallel_size == 2
        mock_lg.warning.assert_not_called()


class TestModelConfigVllmField:
    """Tests for ModelConfig vllm field parsing."""

    def test_vllm_defaults_to_empty_dict(self) -> None:
        """Test that vllm field defaults to empty dict."""
        config = ModelConfig(name="test")
        assert config.vllm == {}

    def test_from_dict_parses_vllm(self) -> None:
        """Test that from_dict correctly parses vllm section."""
        data = {
            "task": "embed",
            "vllm": {
                "enable_prefix_caching": False,
                "gpu_memory_utilization": 0.7,
            },
        }
        config = ModelConfig.from_dict("test-model", data)

        assert config.vllm == {
            "enable_prefix_caching": False,
            "gpu_memory_utilization": 0.7,
        }

    def test_from_dict_handles_none_vllm(self) -> None:
        """Test that from_dict handles None vllm gracefully."""
        data = {"task": "generate", "vllm": None}
        config = ModelConfig.from_dict("test-model", data)
        assert config.vllm == {}

    def test_from_dict_handles_missing_vllm(self) -> None:
        """Test that from_dict handles missing vllm key."""
        data = {"task": "generate"}
        config = ModelConfig.from_dict("test-model", data)
        assert config.vllm == {}
