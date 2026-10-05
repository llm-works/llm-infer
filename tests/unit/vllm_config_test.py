# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for VLLMConfig kwargs against the installed vLLM."""

import dataclasses

import pytest

from llm_infer.serving.dispatch.config import LoRAConfig, VLLMConfig

pytestmark = pytest.mark.unit


def _full_config() -> VLLMConfig:
    """Config with every optional field set, so every kwarg is emitted."""
    return VLLMConfig(
        model_path="/models/test",
        max_model_len=4096,
        max_num_batched_tokens=8192,
        max_cudagraph_capture_size=16,
        quantization="awq",
        spec_model="/models/draft",
        spec_tokens=4,
        lora=LoRAConfig(enabled=True),
    )


class TestToLlmKwargs:
    """to_llm_kwargs() must only emit keys vLLM's EngineArgs accepts."""

    def test_keys_are_engine_args_fields(self) -> None:
        """Every emitted key is a field of the installed vLLM's EngineArgs."""
        arg_utils = pytest.importorskip("vllm.engine.arg_utils")
        accepted = {f.name for f in dataclasses.fields(arg_utils.EngineArgs)}

        unknown = set(_full_config().to_llm_kwargs()) - accepted

        assert not unknown, f"EngineArgs rejects: {sorted(unknown)}"

    def test_optional_keys_emitted_when_set(self) -> None:
        """Optional fields reach the kwargs, so the field check above covers them."""
        kwargs = _full_config().to_llm_kwargs()

        for key in ("spec_model", "spec_tokens", "enable_lora", "quantization"):
            assert key in kwargs
