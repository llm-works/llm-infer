# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for max_tokens=None resolution in the native and PEFT engines.

None means "generate up to the remaining context window"; engines that need a
concrete budget derive it from the model's context length.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm_infer.engines.native.engine import InferenceEngine
from llm_infer.engines.peft import PEFTEngine, PEFTStreamingIterator

pytestmark = pytest.mark.unit


def _native_engine(context_len: int) -> InferenceEngine:
    engine = object.__new__(InferenceEngine)
    engine.config = SimpleNamespace(model=SimpleNamespace(max_seq_len=context_len))
    return engine


class TestNativeResolveMaxTokens:
    def test_explicit_cap_passes_through(self) -> None:
        assert _native_engine(4096).resolve_max_tokens(50, 100) == 50

    def test_none_uses_remaining_context(self) -> None:
        assert _native_engine(4096).resolve_max_tokens(None, 1000) == 3096

    def test_prompt_filling_context_raises(self) -> None:
        with pytest.raises(ValueError, match="fills the context window"):
            _native_engine(4096).resolve_max_tokens(None, 4096)


class TestPEFTResolveMaxNewTokens:
    def _model(self, **config: int) -> SimpleNamespace:
        return SimpleNamespace(config=SimpleNamespace(**config))

    def test_explicit_cap_passes_through(self) -> None:
        engine = object.__new__(PEFTEngine)
        model = self._model(max_position_embeddings=4096)
        assert engine._resolve_max_new_tokens(model, 50, 100) == 50

    def test_none_uses_remaining_context(self) -> None:
        engine = object.__new__(PEFTEngine)
        model = self._model(max_position_embeddings=4096)
        assert engine._resolve_max_new_tokens(model, None, 1000) == 3096

    def test_none_without_context_length_raises(self) -> None:
        engine = object.__new__(PEFTEngine)
        with pytest.raises(ValueError, match="max_position_embeddings"):
            engine._resolve_max_new_tokens(self._model(), None, 1000)


class TestPEFTStreamingFinishReason:
    def _drain(self, max_new_tokens: int, generated: int) -> PEFTStreamingIterator:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(generated))
        it = PEFTStreamingIterator(iter(["a", "b"]), 3, max_new_tokens, tokenizer)
        list(it)
        return it

    def test_budget_used_reports_length(self) -> None:
        assert self._drain(max_new_tokens=2, generated=2).finish_reason == "length"

    def test_under_budget_reports_stop(self) -> None:
        assert self._drain(max_new_tokens=10, generated=2).finish_reason == "stop"
