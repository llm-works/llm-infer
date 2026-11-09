# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Adapter base model compatibility, read from adapter_config.json on disk."""

from pathlib import Path
from unittest.mock import MagicMock

from llm_infer.serving.adapters import AdapterManager


class TestBaseModelCompatibility:
    """Test adapter base model compatibility checking."""

    def test_compatible_base_model(self, tmp_path: Path) -> None:
        """Adapter with matching base model is loaded."""
        adapter_dir = tmp_path / "adapters" / "test-adapter"
        adapter_dir.mkdir(parents=True)

        # Create adapter_config.json with matching base model
        adapter_config = {"base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct"}
        (adapter_dir / "adapter_config.json").write_text(
            __import__("json").dumps(adapter_config)
        )

        lg = MagicMock()
        model_path = tmp_path / "models" / "Qwen2.5-3B-Instruct"
        manager = AdapterManager(lg, tmp_path / "adapters", base_model_path=model_path)

        # Check compatibility directly - returns (compatible, peft_type)
        compatible, peft_type = manager._check_base_model_compatibility(
            adapter_dir, "test-adapter"
        )
        assert compatible is True
        assert peft_type is None  # No peft_type in config

    def test_incompatible_base_model(self, tmp_path: Path) -> None:
        """Adapter with different base model is skipped."""
        adapter_dir = tmp_path / "adapters" / "test-adapter"
        adapter_dir.mkdir(parents=True)

        # Create adapter_config.json with different base model
        adapter_config = {"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}
        (adapter_dir / "adapter_config.json").write_text(
            __import__("json").dumps(adapter_config)
        )

        lg = MagicMock()
        model_path = tmp_path / "models" / "Qwen2.5-3B-Instruct"
        manager = AdapterManager(lg, tmp_path / "adapters", base_model_path=model_path)

        compatible, peft_type = manager._check_base_model_compatibility(
            adapter_dir, "test-adapter"
        )
        assert compatible is False

    def test_case_insensitive_comparison(self, tmp_path: Path) -> None:
        """Base model comparison is case-insensitive."""
        adapter_dir = tmp_path / "adapters" / "test-adapter"
        adapter_dir.mkdir(parents=True)

        # HuggingFace path with different case
        adapter_config = {"base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct"}
        (adapter_dir / "adapter_config.json").write_text(
            __import__("json").dumps(adapter_config)
        )

        lg = MagicMock()
        # Local directory with lowercase
        model_path = tmp_path / "models" / "qwen2.5-3b-instruct"
        manager = AdapterManager(lg, tmp_path / "adapters", base_model_path=model_path)

        compatible, peft_type = manager._check_base_model_compatibility(
            adapter_dir, "test-adapter"
        )
        assert compatible is True

    def test_missing_adapter_config_allows(self, tmp_path: Path) -> None:
        """Adapter without adapter_config.json is allowed."""
        adapter_dir = tmp_path / "adapters" / "test-adapter"
        adapter_dir.mkdir(parents=True)
        # No adapter_config.json created

        lg = MagicMock()
        model_path = tmp_path / "models" / "Qwen2.5-3B-Instruct"
        manager = AdapterManager(lg, tmp_path / "adapters", base_model_path=model_path)

        compatible, peft_type = manager._check_base_model_compatibility(
            adapter_dir, "test-adapter"
        )
        assert compatible is True

    def test_missing_base_model_path_in_manager_allows(self, tmp_path: Path) -> None:
        """Manager without base_model_path allows all adapters."""
        adapter_dir = tmp_path / "adapters" / "test-adapter"
        adapter_dir.mkdir(parents=True)

        adapter_config = {"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}
        (adapter_dir / "adapter_config.json").write_text(
            __import__("json").dumps(adapter_config)
        )

        lg = MagicMock()
        # No base_model_path provided
        manager = AdapterManager(lg, tmp_path / "adapters", base_model_path=None)

        compatible, peft_type = manager._check_base_model_compatibility(
            adapter_dir, "test-adapter"
        )
        assert compatible is True

    def test_missing_base_model_field_in_config_allows(self, tmp_path: Path) -> None:
        """Adapter config without base_model_name_or_path is allowed."""
        adapter_dir = tmp_path / "adapters" / "test-adapter"
        adapter_dir.mkdir(parents=True)

        # Config without base_model_name_or_path
        adapter_config = {"r": 16, "lora_alpha": 32}
        (adapter_dir / "adapter_config.json").write_text(
            __import__("json").dumps(adapter_config)
        )

        lg = MagicMock()
        model_path = tmp_path / "models" / "Qwen2.5-3B-Instruct"
        manager = AdapterManager(lg, tmp_path / "adapters", base_model_path=model_path)

        compatible, peft_type = manager._check_base_model_compatibility(
            adapter_dir, "test-adapter"
        )
        assert compatible is True

    def test_peft_type_extracted(self, tmp_path: Path) -> None:
        """Test peft_type is extracted from adapter_config.json."""
        adapter_dir = tmp_path / "adapters" / "test-adapter"
        adapter_dir.mkdir(parents=True)

        adapter_config = {
            "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
            "peft_type": "PROMPT_TUNING",
        }
        (adapter_dir / "adapter_config.json").write_text(
            __import__("json").dumps(adapter_config)
        )

        lg = MagicMock()
        model_path = tmp_path / "models" / "Qwen2.5-3B-Instruct"
        manager = AdapterManager(lg, tmp_path / "adapters", base_model_path=model_path)

        compatible, peft_type = manager._check_base_model_compatibility(
            adapter_dir, "test-adapter"
        )
        assert compatible is True
        assert peft_type == "PROMPT_TUNING"
