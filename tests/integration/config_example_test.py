# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""The documented configuration example and the packaged etc/ use only keys
llm-infer reads.

Config parsing ignores keys it doesn't know, so a removed or misspelled key
loads without error and does nothing. These tests load each file through
appinfra (resolving !include and !path), parse it with the real parsing
code, and fail on every key that code never read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from appinfra.config import Config

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "docs" / "examples" / "llm-infer.yaml"
PACKAGED = REPO / "llm_infer" / "etc" / "llm-infer.yaml"

# Read by other code than InferenceConfig.from_dict, or checked on its own
_OTHER_READERS = (
    "logging",  # appinfra's logging section
    "admin",  # AdminConfig (pydantic, extra="forbid"): unknown keys fail parsing
    "third_party_logging",  # cli/tools/serve.py, _configure_logging
    "compat",  # cli/tools/compat.py
    "models.templates",  # holds YAML anchors only; never read as config
)


class _Tracked(dict):  # type: ignore[type-arg]
    """A config section that records the keys the parsing code reads."""

    def __init__(self, data: dict[str, Any], path: str, read: set[str]) -> None:
        super().__init__(
            {key: _track(value, _join(path, key), read) for key, value in data.items()}
        )
        self._path = path
        self._read = read

    def _mark(self, key: str) -> None:
        self._read.add(_join(self._path, key))

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        self._mark(key)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            self._mark(key)
        return super().get(key, default)

    def items(self) -> Any:
        # Iterating name -> entry maps (the models catalog) reads every name
        for key in self:
            self._mark(key)
        return super().items()


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _track(value: Any, path: str, read: set[str]) -> Any:
    if isinstance(value, dict):
        return _Tracked(value, path, read)
    return value


def _key_paths(data: dict[str, Any], path: str = "") -> list[str]:
    paths = []
    for key, value in data.items():
        key_path = _join(path, key)
        paths.append(key_path)
        if isinstance(value, dict):
            paths.extend(_key_paths(value, key_path))
    return paths


def _model_vllm_keys(raw: dict[str, Any]) -> set[str]:
    """Per-model vllm keys serve applies to the vllm or vllm_server engine.

    serve copies them onto either engine config that has the attribute
    (cli/tools/serve.py, _apply_vllm_overrides); they aren't read at parse time.
    """
    from dataclasses import fields

    from llm_infer.serving.dispatch.config import VLLMConfig, VLLMServerConfig

    known = {f.name for f in fields(VLLMConfig)} | {
        f.name for f in fields(VLLMServerConfig)
    }
    models = raw.get("models") or {}
    entries = {
        f"models.models.{name}": entry or {}
        for name, entry in (models.get("models") or {}).items()
    }
    entries["models.defaults"] = models.get("defaults") or {}
    return {
        f"{path}.vllm.{key}"
        for path, entry in entries.items()
        for key in (entry.get("vllm") or {})
        if key in known
    }


def _unread_keys(path: Path) -> list[str]:
    """Keys in the config at path that llm-infer never reads."""
    from llm_infer.serving.dispatch.config import InferenceConfig

    raw = Config.from_path(path, enable_env_overrides=False).to_dict()
    read: set[str] = set()
    InferenceConfig.from_dict(_Tracked(raw, "", read))
    read |= _model_vllm_keys(raw)

    def owned_elsewhere(key: str) -> bool:
        return any(key == p or key.startswith(f"{p}.") for p in _OTHER_READERS)

    unread = [k for k in _key_paths(raw) if k not in read and not owned_elsewhere(k)]
    # Report a section once, not each key below it
    return [k for k in unread if not any(k.startswith(f"{u}.") for u in unread)]


@pytest.mark.parametrize("path", [EXAMPLE, PACKAGED], ids=["example", "packaged"])
def test_every_key_is_read(path: Path) -> None:
    assert _unread_keys(path) == []


def test_example_values_reach_the_config() -> None:
    """What the docs say the example sets is what serve gets."""
    from llm_infer.serving.dispatch.config import InferenceConfig

    raw = Config.from_path(EXAMPLE, enable_env_overrides=False).to_dict()
    config = InferenceConfig.from_dict(raw)

    assert config.models.default == "qwen2.5-7b-instruct"
    assert config.models.locations == [Path("/srv/models")]
    qwen = config.models.get("qwen2.5-7b-instruct")
    assert qwen.think.enable_suffix == " /think"  # from the template anchor
    assert qwen.vllm == {"gpu_memory_gb": 20.0, "enable_prefix_caching": True}
    assert config.models.get("bge-small-en-v1.5").task == "embed"
    assert config.backends.engine == "vllm-server"
    assert config.engines.vllm_server.logging.mode == "stderr"


def test_default_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """INFRA_MODELS_DEFAULT replaces models.default, as the docs say."""
    from llm_infer.serving.dispatch.config import InferenceConfig

    monkeypatch.setenv("INFRA_MODELS_DEFAULT", "bge-small-en-v1.5")
    raw = Config.from_path(EXAMPLE).to_dict()

    assert InferenceConfig.from_dict(raw).models.default == "bge-small-en-v1.5"


def test_removed_and_misspelled_keys_are_reported(tmp_path: Path) -> None:
    """The check catches what parsing ignores: a removed section, a typo."""
    config = tmp_path / "llm-infer.yaml"
    config.write_text(
        "models:\n"
        "  default: qwen\n"
        "  selection:\n"
        "    generate:\n"
        "      default: qwen\n"
        "  models:\n"
        "    qwen:\n"
        "      max_model_len: 4096\n"
        "      vllm:\n"
        "        enforce_eager: true\n"
        "        enforce_eagre: true\n"
        "dispatch:\n"
        "  max_pendng: 10\n"
    )

    assert _unread_keys(config) == [
        "models.selection",
        "models.models.qwen.vllm.enforce_eagre",
        "dispatch.max_pendng",
    ]
