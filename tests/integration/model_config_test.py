# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""TransformerConfig loaded from a config.json on disk."""

import json
import tempfile
from pathlib import Path

import pytest

from llm_infer.engines.native.model.config import TransformerConfig


class TestTransformerConfigFromHfConfig:
    """Test TransformerConfig.from_hf_config factory."""

    def test_load_from_config_json(self) -> None:
        """Test loading config from HF config.json."""
        hf_config = {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(hf_config, f)

            config = TransformerConfig.from_hf_config(tmpdir)

            assert config.num_layers == 32
            assert config.num_heads == 32
            assert config.num_kv_heads == 8
            assert config.hidden_size == 4096
            assert config.vocab_size == 128256
            assert config.rope_theta == 500000.0

    def test_missing_config_raises(self) -> None:
        """Test that missing config.json raises ValueError."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(ValueError, match="No config.json found"),
        ):
            TransformerConfig.from_hf_config(tmpdir)

    def test_head_dim_computed(self) -> None:
        """Test head_dim is computed from hidden_size when not provided."""
        hf_config = {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 32000,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(hf_config, f)

            config = TransformerConfig.from_hf_config(tmpdir)
            assert config.head_dim == 4096 // 32  # 128
