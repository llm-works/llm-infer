# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Health handlers of serving/api/routes.py; the HTTP endpoints are tested in
integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_infer.serving.api.routes import create_health_handler, health_handler


@pytest.mark.asyncio
async def test_legacy_health_handler() -> None:
    resp = await health_handler()
    assert resp.status == "ok"


@pytest.mark.asyncio
async def test_create_health_handler_no_ready_flag() -> None:
    h = create_health_handler(None)
    resp = await h()
    assert resp.status == "ok"


@pytest.mark.asyncio
async def test_create_health_handler_initializing() -> None:
    flag = MagicMock()
    flag.value = False
    h = create_health_handler(flag)
    resp = await h()
    assert resp.status == "initializing"


@pytest.mark.asyncio
async def test_create_health_handler_ready() -> None:
    flag = MagicMock()
    flag.value = True
    h = create_health_handler(flag)
    resp = await h()
    assert resp.status == "ok"
