"""Unit tests for AdapterManager scan/load and validate_adapter_key.

These tests focus on coverage gaps in the existing tests/unit/adapters_test.py:
- validate_adapter_key path traversal checks
- AdapterManager.scan() filesystem interaction (real tmp_path dirs)
- _read_config malformed YAML / non-dict handling
- _load_adapter PEFT-only fallback path
- _log_changes added/removed diff
- peft_type filter
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.serving.adapters import AdapterManager, validate_adapter_key

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# validate_adapter_key
# ---------------------------------------------------------------------------


class TestValidateAdapterKey:
    def test_empty_key_rejected(self, tmp_path: Path) -> None:
        assert validate_adapter_key("", tmp_path) is None

    def test_dot_rejected(self, tmp_path: Path) -> None:
        assert validate_adapter_key(".", tmp_path) is None

    def test_forward_slash_rejected(self, tmp_path: Path) -> None:
        assert validate_adapter_key("foo/bar", tmp_path) is None

    def test_backslash_rejected(self, tmp_path: Path) -> None:
        assert validate_adapter_key("foo\\bar", tmp_path) is None

    def test_parent_traversal_rejected(self, tmp_path: Path) -> None:
        assert validate_adapter_key("..", tmp_path) is None
        assert validate_adapter_key("foo..bar", tmp_path) is None
        assert validate_adapter_key("../escape", tmp_path) is None

    def test_valid_key_returns_resolved_path(self, tmp_path: Path) -> None:
        (tmp_path / "my-adapter").mkdir()
        result = validate_adapter_key("my-adapter", tmp_path)
        assert result is not None
        assert result.name == "my-adapter"

    def test_nonexistent_valid_key_still_returns_path(self, tmp_path: Path) -> None:
        # Validation does not require existence
        result = validate_adapter_key("not-deployed", tmp_path)
        assert result is not None


# ---------------------------------------------------------------------------
# Helpers to build adapter directories
# ---------------------------------------------------------------------------


def _make_adapter_dir(
    parent: Path,
    name: str,
    *,
    config_yaml: str | None = None,
    adapter_config_json: dict | None = None,
    weights_file: str | bytes = "fake-weights",
) -> Path:
    """Create an adapter directory with optional config files."""
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    if config_yaml is not None:
        (d / "config.yaml").write_text(config_yaml)
    if adapter_config_json is not None:
        (d / "adapter_config.json").write_text(json.dumps(adapter_config_json))
    if isinstance(weights_file, bytes):
        (d / "adapter_model.safetensors").write_bytes(weights_file)
    else:
        (d / "adapter_model.safetensors").write_text(weights_file)
    return d


@pytest.fixture
def lg() -> Logger:
    return MagicMock(spec=Logger)


# ---------------------------------------------------------------------------
# AdapterManager.scan() — _is_scannable branches
# ---------------------------------------------------------------------------


class TestScanScannability:
    def test_no_base_path(self, lg: Logger) -> None:
        mgr = AdapterManager(lg, base_path=None)
        assert mgr.scan() == 0

    def test_base_path_does_not_exist(self, lg: Logger, tmp_path: Path) -> None:
        mgr = AdapterManager(lg, base_path=str(tmp_path / "missing"))
        assert mgr.scan() == 0

    def test_base_path_is_a_file(self, lg: Logger, tmp_path: Path) -> None:
        f = tmp_path / "not-a-dir"
        f.write_text("hi")
        mgr = AdapterManager(lg, base_path=str(f))
        assert mgr.scan() == 0
        # Warning logged
        assert lg.warning.called  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# AdapterManager.scan() — full scan with real filesystem
# ---------------------------------------------------------------------------


class TestScanWithFilesystem:
    def test_scan_loads_adapter_with_config_yaml(
        self, lg: Logger, tmp_path: Path
    ) -> None:
        _make_adapter_dir(
            tmp_path,
            "ad-aaaaaaaaaaaa",
            config_yaml="enabled: true\ndescription: my desc\n",
            adapter_config_json={"peft_type": "LORA"},
        )
        mgr = AdapterManager(lg, base_path=str(tmp_path))
        assert mgr.scan() == 1
        loaded = mgr.list()
        assert len(loaded) == 1
        assert loaded[0].description == "my desc"
        assert loaded[0].enabled is True

    def test_scan_loads_peft_only_adapter_without_config_yaml(
        self, lg: Logger, tmp_path: Path
    ) -> None:
        """Adapter with only adapter_config.json should be accepted."""
        _make_adapter_dir(
            tmp_path,
            "ad-bbbbbbbbbbbb",
            adapter_config_json={"peft_type": "LORA"},
        )
        mgr = AdapterManager(lg, base_path=str(tmp_path))
        assert mgr.scan() == 1
        loaded = mgr.list()[0]
        assert loaded.enabled is True
        assert loaded.description is None

    def test_scan_skips_directory_with_neither_config(
        self, lg: Logger, tmp_path: Path
    ) -> None:
        d = tmp_path / "no-configs"
        d.mkdir()
        (d / "weights.bin").write_text("x")
        mgr = AdapterManager(lg, base_path=str(tmp_path))
        assert mgr.scan() == 0

    def test_scan_skips_disabled_adapter(self, lg: Logger, tmp_path: Path) -> None:
        _make_adapter_dir(
            tmp_path,
            "ad-cccccccccccc",
            config_yaml="enabled: false\n",
            adapter_config_json={"peft_type": "LORA"},
        )
        mgr = AdapterManager(lg, base_path=str(tmp_path))
        assert mgr.scan() == 0

    def test_scan_skips_files_only_directories(
        self, lg: Logger, tmp_path: Path
    ) -> None:
        # File at top level should be skipped
        (tmp_path / "stray-file.txt").write_text("x")
        _make_adapter_dir(
            tmp_path,
            "ad-dddddddddddd",
            adapter_config_json={"peft_type": "LORA"},
        )
        mgr = AdapterManager(lg, base_path=str(tmp_path))
        assert mgr.scan() == 1

    def test_scan_with_malformed_config_yaml_skips(
        self, lg: Logger, tmp_path: Path
    ) -> None:
        _make_adapter_dir(
            tmp_path,
            "ad-eeeeeeeeeeee",
            config_yaml="not: valid: yaml: [",  # malformed
            adapter_config_json={"peft_type": "LORA"},
        )
        mgr = AdapterManager(lg, base_path=str(tmp_path))
        assert mgr.scan() == 0
        # Warning was logged
        assert lg.warning.called  # type: ignore[attr-defined]

    def test_scan_with_non_mapping_config_yaml_skips(
        self, lg: Logger, tmp_path: Path
    ) -> None:
        _make_adapter_dir(
            tmp_path,
            "ad-ffffffffffff",
            config_yaml="- just\n- a\n- list\n",
            adapter_config_json={"peft_type": "LORA"},
        )
        mgr = AdapterManager(lg, base_path=str(tmp_path))
        assert mgr.scan() == 0


# ---------------------------------------------------------------------------
# _read_adapter_config_json error path
# ---------------------------------------------------------------------------


class TestReadAdapterConfigJson:
    def test_malformed_json_logs_warning(self, lg: Logger, tmp_path: Path) -> None:
        d = tmp_path / "ad-1"
        d.mkdir()
        (d / "adapter_config.json").write_text("{not valid json")
        mgr = AdapterManager(lg, base_path=str(tmp_path))
        result = mgr._read_adapter_config_json(d, "ad-1")
        assert result == {"base_model_name_or_path": None, "peft_type": None}
        assert lg.warning.called  # type: ignore[attr-defined]

    def test_missing_json_returns_default(self, lg: Logger, tmp_path: Path) -> None:
        d = tmp_path / "ad-2"
        d.mkdir()
        mgr = AdapterManager(lg, base_path=str(tmp_path))
        result = mgr._read_adapter_config_json(d, "ad-2")
        assert result == {"base_model_name_or_path": None, "peft_type": None}


# ---------------------------------------------------------------------------
# peft_type filter
# ---------------------------------------------------------------------------


class TestPeftTypeFilter:
    def test_lora_filter_keeps_lora(self, lg: Logger, tmp_path: Path) -> None:
        _make_adapter_dir(
            tmp_path,
            "lora-aaaaaaaaaaaa",
            adapter_config_json={"peft_type": "LORA"},
        )
        mgr = AdapterManager(
            lg,
            base_path=str(tmp_path),
            peft_type_filter=AdapterManager.LORA_TYPES,
        )
        assert mgr.scan() == 1

    def test_lora_filter_rejects_prompt_tuning(
        self, lg: Logger, tmp_path: Path
    ) -> None:
        _make_adapter_dir(
            tmp_path,
            "pt-aaaaaaaaaaaa",
            adapter_config_json={"peft_type": "PROMPT_TUNING"},
        )
        mgr = AdapterManager(
            lg,
            base_path=str(tmp_path),
            peft_type_filter=AdapterManager.LORA_TYPES,
        )
        assert mgr.scan() == 0


# ---------------------------------------------------------------------------
# _log_changes diff behavior
# ---------------------------------------------------------------------------


class TestLogChanges:
    def test_first_scan_logs_all_as_loaded(self, lg: Logger, tmp_path: Path) -> None:
        _make_adapter_dir(
            tmp_path,
            "ad-aaaaaaaaaaaa",
            adapter_config_json={"peft_type": "LORA"},
        )
        mgr = AdapterManager(lg, base_path=str(tmp_path))
        mgr.scan()
        # info called with "adapter loaded" once
        info_msgs = [
            c.args[0]
            for c in lg.info.call_args_list
            if c.args  # type: ignore[attr-defined]
        ]
        assert info_msgs.count("adapter loaded") == 1
        assert info_msgs.count("adapter unloaded") == 0

    def test_second_scan_with_no_changes_logs_nothing(
        self, lg: Logger, tmp_path: Path
    ) -> None:
        _make_adapter_dir(
            tmp_path,
            "ad-aaaaaaaaaaaa",
            adapter_config_json={"peft_type": "LORA"},
        )
        mgr = AdapterManager(lg, base_path=str(tmp_path))
        mgr.scan()
        lg.info.reset_mock()  # type: ignore[attr-defined]
        mgr.scan()
        info_msgs = [
            c.args[0]
            for c in lg.info.call_args_list
            if c.args  # type: ignore[attr-defined]
        ]
        assert info_msgs == []

    def test_removed_adapter_logged_as_unloaded(
        self, lg: Logger, tmp_path: Path
    ) -> None:
        d = _make_adapter_dir(
            tmp_path,
            "ad-aaaaaaaaaaaa",
            adapter_config_json={"peft_type": "LORA"},
        )
        mgr = AdapterManager(lg, base_path=str(tmp_path))
        mgr.scan()
        lg.info.reset_mock()  # type: ignore[attr-defined]

        # Remove the adapter directory
        (d / "adapter_config.json").unlink()
        (d / "adapter_model.safetensors").unlink()
        d.rmdir()

        mgr.scan()
        info_msgs = [
            c.args[0]
            for c in lg.info.call_args_list
            if c.args  # type: ignore[attr-defined]
        ]
        assert "adapter unloaded" in info_msgs
        assert "adapter loaded" not in info_msgs

    def test_added_adapter_logged_as_loaded(self, lg: Logger, tmp_path: Path) -> None:
        _make_adapter_dir(
            tmp_path,
            "ad-aaaaaaaaaaaa",
            adapter_config_json={"peft_type": "LORA"},
        )
        mgr = AdapterManager(lg, base_path=str(tmp_path))
        mgr.scan()
        lg.info.reset_mock()  # type: ignore[attr-defined]

        # Add a second adapter
        _make_adapter_dir(
            tmp_path,
            "ad-bbbbbbbbbbbb",
            adapter_config_json={"peft_type": "LORA"},
        )
        mgr.scan()
        loaded_keys = [
            c.kwargs["extra"]["key"]
            for c in lg.info.call_args_list  # type: ignore[attr-defined]
            if c.args and c.args[0] == "adapter loaded"
        ]
        assert "ad-bbbbbbbbbbbb" in loaded_keys
        assert "ad-aaaaaaaaaaaa" not in loaded_keys
