# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Real ``mp.Process(forkserver)`` spawn covering the Py3.14 pickle boundary.

Py3.14 switched the POSIX default start method from ``fork`` to
``forkserver``, which requires everything the parent hands to the FastAPI
child to pickle cleanly. dispatch/main._build_routes wraps route/router
factories in ``appinfra.subprocess.Lazy`` so the child imports the
factory by qualname and constructs the handler locally — nested closures
never cross the boundary.

This test reproduces that boundary end-to-end: it spins a real
forkserver-backed subprocess, hands it the same Lazy-wrapped bundle
_build_routes emits, and asserts the child resolves every factory.
Guards against #151.

Marked ``integration`` because it exercises real OS process spawn and
cross-process IPC, not because it needs external services.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest
from appinfra.subprocess import Lazy

from llm_infer.serving.api.openai.router import OpenAIRouterConfig

pytestmark = pytest.mark.integration


def _resolve_all(payload: list[Lazy]) -> None:
    """Child target: resolve every Lazy so factories run in the subprocess."""
    for item in payload:
        item.resolve()


def test_lazy_bundle_survives_forkserver_spawn() -> None:
    ctx = mp.get_context("forkserver")
    ready = ctx.Value("b", False)  # mp.Value must come from the same context
    payload = [
        Lazy("llm_infer.serving.api.routes:create_health_handler", ready),
        Lazy("llm_infer.serving.api.routes:create_routes", "some-model"),
        Lazy(
            "llm_infer.serving.api.openai.router:create_openai_router",
            OpenAIRouterConfig(model_name="some-model"),
        ),
        Lazy("llm_infer.serving.api.adapters:create_adapter_router"),
    ]
    proc = ctx.Process(target=_resolve_all, args=(payload,))
    proc.start()
    proc.join(timeout=30)
    assert proc.exitcode == 0, f"forkserver child exited {proc.exitcode}"
