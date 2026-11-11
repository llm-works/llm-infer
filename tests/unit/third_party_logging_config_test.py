# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Third-party logging levels come from the third_party_logging section."""

import argparse
from unittest.mock import patch

from llm_infer.cli.tools.serve import ServeTool
from llm_infer.serving.dispatch.config import InferenceConfig


def test_levels_read_from_their_section() -> None:
    config = InferenceConfig.from_dict(
        {"third_party_logging": {"torch": "debug", "transformers": "info"}}
    )

    assert config.third_party_logging.torch == "debug"
    assert config.third_party_logging.transformers == "info"


def test_appinfra_logging_section_not_read() -> None:
    """`logging` is appinfra's section; levels there don't apply."""
    config = InferenceConfig.from_dict({"logging": {"torch": "debug"}})

    assert config.third_party_logging.torch == "warning"


def test_serve_applies_cli_overrides_to_levels() -> None:
    """`-o third_party_logging.<lib>=<level>` reaches the logging setup."""
    tool = ServeTool(parent=None)
    tool._parsed_args = argparse.Namespace(
        overrides=["third_party_logging.torch=debug", "api.port=9000"]
    )
    raw_config = {"third_party_logging": {"torch": "info", "transformers": "error"}}

    with patch("llm_infer.logging_setup.configure_third_party_logging") as configure:
        tool._configure_logging(raw_config)

    configure.assert_called_once_with(torch_level="debug", transformers_level="error")
