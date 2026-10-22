# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Shared fixtures for the unit tests."""

import pytest
from appinfra.log import Logger, create_lg


@pytest.fixture
def lg() -> Logger:
    """A real appinfra logger; appinfra.testing destroys it after the test."""
    return create_lg("test", "debug")
