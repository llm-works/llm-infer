# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Stream resolvers for response processing.

Resolvers process stream events into output.
"""

from .base import BaseResolver
from .terminal import TerminalResolver

__all__ = [
    "BaseResolver",
    "TerminalResolver",
]
