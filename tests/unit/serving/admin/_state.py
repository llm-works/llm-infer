# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""In-process stand-in for SharedBackendState."""

from llm_infer.serving.admin.state import BackendState


class LocalBackendState:
    """SharedBackendState's get/set without shared memory.

    SharedBackendState keeps the value in a multiprocessing.Value, which creates
    a file under /dev/shm. The controller and the gate middleware only call
    get() and set(), so their unit tests hand them this instead.
    """

    def __init__(self) -> None:
        self._value = BackendState.INITIALIZING

    def get(self) -> BackendState:
        return self._value

    def set(self, state: BackendState) -> None:
        self._value = state
