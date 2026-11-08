# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for the native engine's HuggingFace tokenizer wrapper."""

from __future__ import annotations

from typing import Any

import pytest

from llm_infer.engines.native.tokenizer import huggingface
from llm_infer.engines.native.tokenizer.huggingface import HuggingFaceTokenizer

TOKEN_IDS = [151644, 872, 198, 6023]


class _Transformers5Tokenizer:
    """Stands in for AutoTokenizer: apply_chat_template as transformers 5 has it.

    With tokenize=True it returns a mapping (input_ids, attention_mask) unless
    return_dict=False asks for the token ids alone.
    """

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        assert kwargs.get("tokenize") is True
        if kwargs.get("return_dict", True):
            return {"input_ids": TOKEN_IDS, "attention_mask": [1] * len(TOKEN_IDS)}
        return list(TOKEN_IDS)


@pytest.fixture
def tokenizer(monkeypatch: pytest.MonkeyPatch) -> HuggingFaceTokenizer:
    monkeypatch.setattr(
        huggingface.AutoTokenizer,
        "from_pretrained",
        lambda _path: _Transformers5Tokenizer(),
    )
    return HuggingFaceTokenizer("unused")


class TestEncodeChat:
    def test_returns_token_ids(self, tokenizer: HuggingFaceTokenizer) -> None:
        assert tokenizer.encode_chat("hi") == TOKEN_IDS

    def test_message_list(self, tokenizer: HuggingFaceTokenizer) -> None:
        messages = [{"role": "user", "content": "hi"}]
        assert tokenizer.encode_chat(messages) == TOKEN_IDS
