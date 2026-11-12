# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""DaemonThreadExecutor arguments; tests with running workers are integration."""

from __future__ import annotations

import pytest

from llm_infer.serving.dispatch.handlers.daemon_executor import DaemonThreadExecutor


def test_rejects_non_positive_max_workers() -> None:
    with pytest.raises(ValueError, match="max_workers must be positive"):
        DaemonThreadExecutor(0, "t")
