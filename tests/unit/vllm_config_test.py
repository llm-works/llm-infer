# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for VLLMConfig parsing and kwargs against the installed vLLM."""

import dataclasses

import pytest

from llm_infer.serving.dispatch.config import InferenceConfig, LoRAConfig, VLLMConfig

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
        # Skip only without vllm; a moved module must fail, not skip
        pytest.importorskip("vllm", reason="vllm not installed")
        from vllm.engine.arg_utils import EngineArgs

        accepted = {f.name for f in dataclasses.fields(EngineArgs)}

        unknown = set(_full_config().to_llm_kwargs()) - accepted

        assert not unknown, f"EngineArgs rejects: {sorted(unknown)}"

    def test_optional_keys_emitted_when_set(self) -> None:
        """Optional fields reach the kwargs, so the field check above covers them."""
        kwargs = _full_config().to_llm_kwargs()

        for key in ("spec_model", "spec_tokens", "enable_lora", "quantization"):
            assert key in kwargs

    def test_full_config_sets_every_optional_field(self) -> None:
        """A new None-default field must be added to _full_config() to be checked."""
        # gpu_memory_gb is converted to gpu_memory_utilization, never emitted itself
        config = _full_config()
        unset = [
            f.name
            for f in dataclasses.fields(VLLMConfig)
            if f.default is None
            and f.name != "gpu_memory_gb"
            and getattr(config, f.name) is None
        ]

        assert not unset, f"_full_config() leaves unset: {unset}"


class TestRemovedKeys:
    """Keys vLLM 0.30 removed or renamed are rejected, not silently dropped."""

    @pytest.mark.parametrize(
        ("key", "hint"),
        [
            ("swap_space", "removed"),
            ("speculative_model", "spec_model"),
            ("num_speculative_tokens", "spec_tokens"),
        ],
    )
    def test_from_dict_rejects(self, key: str, hint: str) -> None:
        with pytest.raises(ValueError, match=f"{key}.*{hint}"):
            VLLMConfig.from_dict({key: 1})

    def test_inference_config_rejects(self) -> None:
        with pytest.raises(ValueError, match="speculative_model.*spec_model"):
            InferenceConfig.from_dict(
                {"engines": {"vllm": {"speculative_model": "/m"}}}
            )

    def test_new_keys_accepted(self) -> None:
        config = VLLMConfig.from_dict({"spec_model": "/m", "spec_tokens": 4})

        assert (config.spec_model, config.spec_tokens) == ("/m", 4)
