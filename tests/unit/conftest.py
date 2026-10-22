# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Shared fixtures for the unit tests."""

from collections.abc import Iterator

import pytest
from appinfra.log import Logger, LoggerFactory, create_lg


@pytest.fixture
def lg() -> Iterator[Logger]:
    """A real appinfra logger, destroyed after the test.

    appinfra refuses to create a logger name this process already holds, so the
    next test in the same worker can only create "test" once this one is gone.
    """
    yield create_lg("test", "debug")
    LoggerFactory.destroy("test")
