"""Unit tests for logging_setup.py."""

from __future__ import annotations

import logging
import os

import pytest

from llm_infer.logging_setup import configure_third_party_logging

pytestmark = pytest.mark.unit


def test_default_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear env to ensure setdefault paths execute
    for k in (
        "TORCH_LOGS",
        "TORCH_COMPILE_DEBUG",
        "TRANSFORMERS_VERBOSITY",
        "TOKENIZERS_PARALLELISM",
        "VLLM_CONFIGURE_LOGGING",
    ):
        monkeypatch.delenv(k, raising=False)

    configure_third_party_logging()

    assert os.environ["TORCH_LOGS"] == "-all"
    assert os.environ["TORCH_COMPILE_DEBUG"] == "0"
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert os.environ["VLLM_CONFIGURE_LOGGING"] == "0"


def test_torch_warning_level_sets_python_loggers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCH_LOGS", raising=False)
    configure_third_party_logging(torch_level="warning")
    assert logging.getLogger("torch._inductor").level == logging.WARNING


def test_torch_debug_does_not_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCH_LOGS", raising=False)
    configure_third_party_logging(torch_level="debug")
    assert logging.getLogger("torch._inductor").level == logging.DEBUG


def test_torch_error_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORCH_LOGS", raising=False)
    configure_third_party_logging(torch_level="error")
    assert logging.getLogger("torch._inductor").level == logging.ERROR


def test_transformers_warning_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)
    configure_third_party_logging(transformers_level="warning")
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "warning"


def test_unknown_torch_level_falls_back_to_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCH_LOGS", raising=False)
    configure_third_party_logging(torch_level="unknown-level")
    assert logging.getLogger("torch._inductor").level == logging.WARNING
