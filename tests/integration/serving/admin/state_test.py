# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""SharedBackendState in shared memory."""

from __future__ import annotations

from llm_infer.serving.admin.state import BackendState, SharedBackendState


def test_shared_state_starts_initializing() -> None:
    assert SharedBackendState().get() == BackendState.INITIALIZING
