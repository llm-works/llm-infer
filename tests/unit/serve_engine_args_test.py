# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for passing `llm-infer serve ... -- <args>` to `vllm serve`."""

import argparse
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_infer.cli.tools.serve import ServeTool
from llm_infer.serving.dispatch.config import VLLMServerConfig

pytestmark = pytest.mark.unit


def _parse(argv: list[str]) -> argparse.Namespace:
    """Parse `llm-infer <argv>` with the parser shape appinfra builds for tools."""
    root = argparse.ArgumentParser(prog="llm-infer")
    serve = root.add_subparsers(dest="tool").add_parser("serve")
    ServeTool(parent=None).add_args(serve)
    return root.parse_args(argv)


class TestEngineArgsParsing:
    """Everything after `--` lands in engine_args, unparsed."""

    def test_args_after_double_dash(self) -> None:
        args = _parse(
            ["serve", "--engine", "vllm-server", "--", "--speculative-config", "{}"]
        )
        assert args.engine == "vllm-server"
        assert args.engine_args == ["--speculative-config", "{}"]

    def test_own_flags_after_double_dash_are_passed_through(self) -> None:
        args = _parse(["serve", "--", "--engine", "vllm", "--port", "9000"])
        assert args.engine is None
        assert args.port is None
        assert args.engine_args == ["--engine", "vllm", "--port", "9000"]

    def test_no_double_dash(self) -> None:
        assert _parse(["serve", "--engine", "vllm-server"]).engine_args == []


class TestApplyEngineArgs:
    """ServeTool._apply_engine_args() routes engine_args to vllm-server only."""

    @pytest.fixture
    def tool(self) -> Generator[tuple[ServeTool, MagicMock], None, None]:
        with patch.object(ServeTool, "lg", new_callable=lambda: MagicMock()) as lg:
            yield ServeTool(parent=None), lg

    def _config(self, engine: str) -> SimpleNamespace:
        return SimpleNamespace(
            backends=SimpleNamespace(engine=engine),
            engines=SimpleNamespace(
                vllm_server=VLLMServerConfig(extra_args=["--from-yaml"])
            ),
        )

    def test_appended_after_configured_extra_args(
        self, tool: tuple[ServeTool, MagicMock]
    ) -> None:
        serve, _ = tool
        serve._parsed_args = argparse.Namespace(engine_args=["--from-cli"])
        config = self._config("vllm-server")

        assert serve._apply_engine_args(config) is True
        assert config.engines.vllm_server.extra_args == ["--from-yaml", "--from-cli"]

    @pytest.mark.parametrize("engine", ["vllm", "ollama", "native", "peft"])
    def test_other_engine_rejected(
        self, tool: tuple[ServeTool, MagicMock], engine: str
    ) -> None:
        serve, lg = tool
        serve._parsed_args = argparse.Namespace(engine_args=["--from-cli"])
        config = self._config(engine)

        assert serve._apply_engine_args(config) is False
        assert config.engines.vllm_server.extra_args == ["--from-yaml"]
        lg.error.assert_called_once()

    def test_no_engine_args_is_noop(self, tool: tuple[ServeTool, MagicMock]) -> None:
        serve, lg = tool
        serve._parsed_args = argparse.Namespace(engine_args=[])
        config = self._config("ollama")

        assert serve._apply_engine_args(config) is True
        lg.error.assert_not_called()
