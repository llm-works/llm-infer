# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Third-party logging levels come from the third_party_logging section."""

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
