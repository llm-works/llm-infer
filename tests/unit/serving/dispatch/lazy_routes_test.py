# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Regression tests for the Py3.14 forkserver pickle boundary.

dispatch/main._build_routes wraps route/router factories in
appinfra.subprocess.Lazy so the child imports the factory by qualname and
constructs it locally — nested closures inside a factory never cross the
pickle boundary. These tests fail fast if a qualname string in
_build_routes / _add_lora_routes goes stale (typo, factory rename,
module move).
"""

from __future__ import annotations

import multiprocessing as mp

import pytest
from appinfra.subprocess import Lazy
from fastapi import APIRouter

from llm_infer.serving.api.openai.router import OpenAIRouterConfig

pytestmark = pytest.mark.unit


def test_health_factory_qualname_resolves() -> None:
    ready = mp.Value("b", False)
    handler = Lazy(
        "llm_infer.serving.api.routes:create_health_handler", ready
    ).resolve()
    assert callable(handler)


def test_routes_factory_qualname_resolves() -> None:
    router = Lazy("llm_infer.serving.api.routes:create_routes", "some-model").resolve()
    assert isinstance(router, APIRouter)


def test_openai_router_factory_qualname_resolves() -> None:
    router = Lazy(
        "llm_infer.serving.api.openai.router:create_openai_router",
        OpenAIRouterConfig(model_name="some-model"),
    ).resolve()
    assert isinstance(router, APIRouter)


def test_adapter_router_factory_qualname_resolves() -> None:
    router = Lazy("llm_infer.serving.api.adapters:create_adapter_router").resolve()
    assert isinstance(router, APIRouter)
