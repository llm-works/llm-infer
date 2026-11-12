# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""RequestHandler's adapter checks against adapter directories on disk.

Uses SequentialHandler as the concrete subclass for testing the base class.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.serving.dispatch.handler import AdapterError
from llm_infer.serving.dispatch.handlers import SequentialHandler


def _handler(*, lg: Logger | None = None) -> SequentialHandler:
    engine = MagicMock()
    engine.count_tokens.return_value = 5
    h = SequentialHandler(engine)
    if lg is not None:
        h.set_logger(lg)
    return h


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
