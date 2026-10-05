# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for the CLI entry point."""

import os

import pytest

from llm_infer.cli import cli

pytestmark = pytest.mark.unit

_PYTHON = "/opt/envs/ml/bin/python"
_BIN = "/opt/envs/ml/bin"


class TestEnsureInterpreterBinOnPath:
    """_ensure_interpreter_bin_on_path() puts the interpreter's bin dir on PATH."""

    def test_prepends_missing_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli.sys, "executable", _PYTHON)
        monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))

        cli._ensure_interpreter_bin_on_path()

        assert os.environ["PATH"] == os.pathsep.join([_BIN, "/usr/bin", "/bin"])

    def test_keeps_path_when_dir_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli.sys, "executable", _PYTHON)
        path = os.pathsep.join(["/usr/bin", _BIN])
        monkeypatch.setenv("PATH", path)

        cli._ensure_interpreter_bin_on_path()

        assert os.environ["PATH"] == path

    def test_sets_path_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli.sys, "executable", _PYTHON)
        monkeypatch.delenv("PATH", raising=False)

        cli._ensure_interpreter_bin_on_path()

        assert os.environ["PATH"] == _BIN

    def test_noop_without_executable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli.sys, "executable", "")
        monkeypatch.setenv("PATH", "/usr/bin")

        cli._ensure_interpreter_bin_on_path()

        assert os.environ["PATH"] == "/usr/bin"
