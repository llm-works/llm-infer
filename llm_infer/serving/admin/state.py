# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Backend state shared between the main process and the HTTP subprocess."""

from __future__ import annotations

import multiprocessing as mp
from enum import IntEnum
from typing import Any


class BackendState(IntEnum):
    """Whether the inference backend can take requests."""

    INITIALIZING = 0  # Engine still loading at startup
    SERVING = 1
    RELEASING = 2  # Admin release in progress: stopping the backend
    RELEASED = 3  # Backend stopped, GPU memory freed; inference answers 503
    LOADING = 4  # Admin reacquire in progress: starting the backend

    @property
    def label(self) -> str:
        return self.name.lower()


class SharedBackendState:
    """BackendState in shared memory.

    Created in the main process before the HTTP subprocess starts, and handed to
    it as a constructor argument, so both see the same value. The main process
    writes it; the subprocess only reads it.
    """

    def __init__(self) -> None:
        self._value: Any = mp.Value("i", BackendState.INITIALIZING)

    def get(self) -> BackendState:
        return BackendState(self._value.value)

    def set(self, state: BackendState) -> None:
        self._value.value = state
