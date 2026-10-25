# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for BootSequence.warmup wiring."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_infer.serving.dispatch import main
from llm_infer.serving.dispatch.config import InferenceConfig
from llm_infer.serving.dispatch.main import BootSequence

pytestmark = pytest.mark.unit


def _boot(engine: str) -> BootSequence:
    config = InferenceConfig()
    config.backends.engine = engine
    config.engines.vllm_server.warmup = True
    config.engines.vllm_server.max_concurrent = 8
    config.engines.vllm_server.max_num_seqs = 4
    boot = BootSequence(config, MagicMock())
    boot._engine = MagicMock()
    return boot


def test_warmup_passes_factory_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    base = MagicMock(return_value=["baseline"])
    adapters = MagicMock()
    monkeypatch.setattr(main, "warmup_base_model", base)
    monkeypatch.setattr(main, "warmup_adapters", adapters)
    boot = _boot("vllm-server")

    boot.warmup()

    base.assert_called_once_with(boot._lg, boot._engine, 4)
    adapters.assert_called_once_with(
        boot._lg, boot._engine, boot._adapter_manager, ["baseline"], 4
    )


def test_warmup_disabled_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    base = MagicMock()
    monkeypatch.setattr(main, "warmup_base_model", base)
    boot = _boot("vllm-server")
    boot._config.engines.vllm_server.warmup = False

    boot.warmup()

    base.assert_not_called()
