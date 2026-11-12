# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""ConcurrentHttpHandler arguments; tests with running workers are integration."""

from unittest.mock import MagicMock

import pytest

from llm_infer.serving.dispatch.handlers import ConcurrentHttpHandler


class TestConcurrentHttpHandlerValidation:
    """Tests for __init__ validation."""

    def test_rejects_zero_max_pending(self) -> None:
        """__init__ rejects max_pending <= 0."""
        with pytest.raises(ValueError, match="max_pending must be positive"):
            ConcurrentHttpHandler(MagicMock(), max_pending=0)

    def test_rejects_negative_max_pending(self) -> None:
        """__init__ rejects negative max_pending."""
        with pytest.raises(ValueError, match="max_pending must be positive"):
            ConcurrentHttpHandler(MagicMock(), max_pending=-1)

    def test_rejects_zero_max_concurrent(self) -> None:
        """__init__ rejects max_concurrent <= 0."""
        with pytest.raises(ValueError, match="max_concurrent must be positive"):
            ConcurrentHttpHandler(MagicMock(), max_concurrent=0)

    def test_rejects_negative_max_concurrent(self) -> None:
        """__init__ rejects negative max_concurrent."""
        with pytest.raises(ValueError, match="max_concurrent must be positive"):
            ConcurrentHttpHandler(MagicMock(), max_concurrent=-1)
