"""Tests for NativeVertexBackend — Vertex REST substrate for cachedContents.

Verifies that the backend wires yaml (retry / rate_limit / auth / timeout)
into the wire path via the client substrate: RetryHelper retries transient
statuses, RateLimiter is acquired before every attempt (including retries),
non-transient statuses propagate as BackendRequestError, and cache/generate
URLs are built from ``project`` + ``region`` as documented for Vertex REST.

httpx is exercised with ``httpx.MockTransport`` through the constructor's
optional ``transport=`` seam — no network. ``from_yaml`` is exercised with
an ``api_key`` auth block so no SA credential file / google-auth import is
touched by the test suite.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from llm_infer.client.backends.context import BackendContext, RetryConfig
from llm_infer.client.backends.vertex_native import (
    NativeVertexBackend,
    NativeVertexFactory,
)
from llm_infer.client.errors import BackendRequestError


class _StubAuth:
    """Minimal AuthProvider — records how many times it's called so we can
    assert the backend re-fetches headers on each retry attempt."""

    def __init__(self) -> None:
        self.calls = 0

    def headers(self) -> dict[str, str]:
        self.calls += 1
        return {"Authorization": "Bearer fake-token"}

    async def headers_async(self) -> dict[str, str]:
        return self.headers()


class _Recorder:
    """Sequential response scripter for MockTransport. Each request pops the
    next scripted response; unexpected extra requests raise."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError(
                f"unexpected extra request: {request.method} {request.url}"
            )
        return self._responses.pop(0)


def _make_backend(
    responses: list[httpx.Response],
    *,
    retry: RetryConfig | None = None,
    rate_limiter: Any = None,
    service_tier: str | None = None,
    lg: Any = None,
) -> tuple[NativeVertexBackend, _Recorder, _StubAuth]:
    """Build a backend wired to a scripted MockTransport."""
    recorder = _Recorder(responses)
    transport = httpx.MockTransport(recorder)
    auth = _StubAuth()
    ctx = BackendContext(rate_limiter=rate_limiter, retry=retry, request_timeout=5.0)
    backend = NativeVertexBackend(
        lg or Mock(),
        ctx,
        auth,
        project="p",
        region="us-central1",
        service_tier=service_tier,
        transport=transport,
    )
    return backend, recorder, auth


def _resp(status: int, body: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(status, json=body or {})


class TestCacheCreate:
    @pytest.mark.asyncio
    async def test_returns_cache_name_from_response(self) -> None:
        backend, recorder, auth = _make_backend(
            [
                _resp(
                    200,
                    {
                        "name": "projects/p/locations/us-central1/cachedContents/abc",
                        "usageMetadata": {"totalTokenCount": 2500},
                    },
                )
            ]
        )
        name, usage = await backend.cache_create(
            model="gemini-2.5-flash-lite",
            system="you are careful",
            user_text="corpus",
            ttl_seconds=120,
        )
        assert name == "projects/p/locations/us-central1/cachedContents/abc"
        assert usage == {"totalTokenCount": 2500}
        assert auth.calls == 1
        (req,) = recorder.requests
        assert req.method == "POST"
        assert str(req.url) == (
            "https://us-central1-aiplatform.googleapis.com/v1"
            "/projects/p/locations/us-central1/cachedContents"
        )
        assert req.headers["authorization"] == "Bearer fake-token"

    @pytest.mark.asyncio
    async def test_raises_when_response_has_no_name(self) -> None:
        backend, _, _ = _make_backend([_resp(200, {})])
        with pytest.raises(BackendRequestError):
            await backend.cache_create(
                model="gemini-2.5-flash-lite",
                system="s",
                user_text="u",
                ttl_seconds=60,
            )


class TestGenerateContent:
    @pytest.mark.asyncio
    async def test_body_carries_cached_content_and_generation_config(self) -> None:
        backend, recorder, _ = _make_backend([_resp(200, {"candidates": []})])
        payload = await backend.generate_content(
            model="gemini-2.5-flash-lite",
            cache_name="projects/p/locations/us-central1/cachedContents/abc",
            user_text="extract claims",
            generation_config={"temperature": 0.7, "maxOutputTokens": 16384},
        )
        assert payload == {"candidates": []}
        (req,) = recorder.requests
        assert str(req.url) == (
            "https://us-central1-aiplatform.googleapis.com/v1"
            "/projects/p/locations/us-central1"
            "/publishers/google/models/gemini-2.5-flash-lite:generateContent"
        )
        body = json.loads(req.content)
        assert (
            body["cachedContent"]
            == "projects/p/locations/us-central1/cachedContents/abc"
        )
        assert body["generationConfig"] == {
            "temperature": 0.7,
            "maxOutputTokens": 16384,
        }
        assert body["contents"] == [
            {"role": "user", "parts": [{"text": "extract claims"}]}
        ]


class TestCacheDelete:
    @pytest.mark.asyncio
    async def test_delete_swallows_error(self) -> None:
        backend, recorder, _ = _make_backend([_resp(404, {"error": "gone"})])
        await backend.cache_delete(
            "projects/p/locations/us-central1/cachedContents/missing"
        )
        assert len(recorder.requests) == 1
        assert recorder.requests[0].method == "DELETE"


class TestPersistentClient:
    """One ``httpx.AsyncClient`` is created at ``__init__`` and reused for
    all POST/DELETE calls — no per-request TCP+TLS handshake."""

    @pytest.mark.asyncio
    async def test_reuses_single_client_across_calls(self) -> None:
        backend, recorder, _ = _make_backend(
            [
                _resp(
                    200, {"name": "projects/p/locations/us-central1/cachedContents/abc"}
                ),
                _resp(200, {"candidates": []}),
                _resp(204),
            ]
        )
        client_ref = backend._client
        assert isinstance(client_ref, httpx.AsyncClient)
        await backend.cache_create(model="m", system="s", user_text="u", ttl_seconds=60)
        await backend.generate_content(
            model="m", cache_name="c", user_text="u", generation_config={}
        )
        await backend.cache_delete(
            "projects/p/locations/us-central1/cachedContents/abc"
        )
        assert backend._client is client_ref
        assert len(recorder.requests) == 3


class TestRetryLogCarriesModel:
    """RetryHelper duck-types ``request.model`` / ``request.id`` for its log
    line. NativeVertexBackend threads ``model`` into ``_RetryCtx`` so retries
    surface which model they're calling."""

    @pytest.mark.asyncio
    async def test_retry_warning_extra_carries_model(self) -> None:
        lg = Mock()
        retry = RetryConfig(base=0.001, factor=1.0, max_delay=0.01, timeout=5.0)
        backend, _, _ = _make_backend(
            [_resp(500, {"error": "boom"}), _resp(200, {"candidates": []})],
            retry=retry,
            lg=lg,
        )
        await backend.generate_content(
            model="gemini-2.5-flash-lite",
            cache_name="c",
            user_text="u",
            generation_config={},
        )
        warning_calls = [
            c
            for c in lg.warning.call_args_list
            if c.args and c.args[0] == "transient error, retrying"
        ]
        assert warning_calls, "expected a retry warning"
        extras = [c.kwargs.get("extra", {}) for c in warning_calls]
        assert any(e.get("model") == "gemini-2.5-flash-lite" for e in extras)
        assert any(
            isinstance(e.get("req"), str) and e["req"].startswith("native-")
            for e in extras
        )


class TestRetryOnTransient:
    @pytest.mark.asyncio
    async def test_429_then_success(self) -> None:
        retry = RetryConfig(base=0.001, factor=1.0, max_delay=0.01, timeout=5.0)
        backend, recorder, auth = _make_backend(
            [
                _resp(429, {"error": "rate limited"}),
                _resp(
                    200, {"name": "projects/p/locations/us-central1/cachedContents/abc"}
                ),
            ],
            retry=retry,
        )
        name, _ = await backend.cache_create(
            model="gemini-2.5-flash-lite",
            system="s",
            user_text="u",
            ttl_seconds=60,
        )
        assert name.endswith("/abc")
        # Auth is re-acquired on each attempt so retries never carry a
        # stale bearer token past its refresh skew.
        assert auth.calls == 2
        assert len(recorder.requests) == 2

    @pytest.mark.asyncio
    async def test_500_then_success(self) -> None:
        retry = RetryConfig(base=0.001, factor=1.0, max_delay=0.01, timeout=5.0)
        backend, recorder, _ = _make_backend(
            [
                _resp(500, {"error": "boom"}),
                _resp(200, {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}),
            ],
            retry=retry,
        )
        payload = await backend.generate_content(
            model="m",
            cache_name="c",
            user_text="u",
            generation_config={},
        )
        assert payload["candidates"]
        assert len(recorder.requests) == 2

    @pytest.mark.asyncio
    async def test_400_does_not_retry(self) -> None:
        retry = RetryConfig(base=0.001, factor=1.0, max_delay=0.01, timeout=5.0)
        backend, recorder, _ = _make_backend(
            [_resp(400, {"error": "bad request"})], retry=retry
        )
        with pytest.raises(BackendRequestError) as ei:
            await backend.cache_create(
                model="m", system="s", user_text="u", ttl_seconds=60
            )
        assert ei.value.status_code == 400
        assert len(recorder.requests) == 1


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_slot_acquired_before_each_request_including_retry(self) -> None:
        rl = Mock()
        rl.next = Mock(return_value=0.0)
        retry = RetryConfig(base=0.001, factor=1.0, max_delay=0.01, timeout=5.0)
        backend, recorder, _ = _make_backend(
            [
                _resp(429, {"error": "rl"}),
                _resp(
                    200, {"name": "projects/p/locations/us-central1/cachedContents/x"}
                ),
            ],
            retry=retry,
            rate_limiter=rl,
        )
        await backend.cache_create(model="m", system="s", user_text="u", ttl_seconds=60)
        # RateLimiter.next() is called before each HTTP attempt — one per try.
        assert rl.next.call_count == 2
        assert len(recorder.requests) == 2


class TestServiceTier:
    """Vertex Priority is a request-header signal (``X-Vertex-AI-LLM-Shared-
    Request-Type``) — the native backend must send it per attempt when
    ``service_tier: priority`` is configured, and must warn when the
    response's ``x-gemini-service-tier`` shows a silent downgrade."""

    @pytest.mark.asyncio
    async def test_priority_header_sent_on_every_request(self) -> None:
        backend, recorder, _ = _make_backend(
            [
                _resp(
                    200, {"name": "projects/p/locations/us-central1/cachedContents/x"}
                ),
            ],
            service_tier="priority",
        )
        await backend.cache_create(model="m", system="s", user_text="u", ttl_seconds=60)
        (req,) = recorder.requests
        assert req.headers.get("x-vertex-ai-llm-shared-request-type") == "priority"

    @pytest.mark.asyncio
    async def test_no_priority_header_when_service_tier_none(self) -> None:
        backend, recorder, _ = _make_backend(
            [_resp(200, {"name": "projects/p/locations/us-central1/cachedContents/x"})]
        )
        await backend.cache_create(model="m", system="s", user_text="u", ttl_seconds=60)
        (req,) = recorder.requests
        assert req.headers.get("x-vertex-ai-llm-shared-request-type") is None

    @pytest.mark.asyncio
    async def test_priority_header_on_delete(self) -> None:
        backend, recorder, _ = _make_backend([_resp(200, {})], service_tier="priority")
        await backend.cache_delete("projects/p/locations/us-central1/cachedContents/x")
        (req,) = recorder.requests
        assert req.headers.get("x-vertex-ai-llm-shared-request-type") == "priority"

    @pytest.mark.asyncio
    async def test_downgrade_logs_warning(self) -> None:
        lg = Mock()
        response = httpx.Response(
            200,
            json={"name": "projects/p/locations/us-central1/cachedContents/x"},
            headers={"x-gemini-service-tier": "standard"},
        )
        backend, _, _ = _make_backend([response], service_tier="priority", lg=lg)
        await backend.cache_create(model="m", system="s", user_text="u", ttl_seconds=60)
        # Exactly one downgrade warning fired with the served tier in extra.
        warnings = [c for c in lg.warning.call_args_list if "downgraded" in c.args[0]]
        assert len(warnings) == 1
        assert warnings[0].kwargs["extra"]["tier_served"] == "standard"

    @pytest.mark.asyncio
    async def test_no_downgrade_warning_when_served_matches(self) -> None:
        lg = Mock()
        response = httpx.Response(
            200,
            json={"name": "projects/p/locations/us-central1/cachedContents/x"},
            headers={"x-gemini-service-tier": "priority"},
        )
        backend, _, _ = _make_backend([response], service_tier="priority", lg=lg)
        await backend.cache_create(model="m", system="s", user_text="u", ttl_seconds=60)
        warnings = [c for c in lg.warning.call_args_list if "downgraded" in c.args[0]]
        assert warnings == []

    @pytest.mark.asyncio
    async def test_no_downgrade_warning_when_header_absent(self) -> None:
        # Vertex OpenAI-compat doesn't emit the served-tier header for
        # gemini-2.5-flash; native REST may share that gap. Absence must
        # NOT be treated as a downgrade.
        lg = Mock()
        backend, _, _ = _make_backend(
            [_resp(200, {"name": "projects/p/locations/us-central1/cachedContents/x"})],
            service_tier="priority",
            lg=lg,
        )
        await backend.cache_create(model="m", system="s", user_text="u", ttl_seconds=60)
        warnings = [c for c in lg.warning.call_args_list if "downgraded" in c.args[0]]
        assert warnings == []

    def test_invalid_service_tier_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid service_tier"):
            NativeVertexBackend(
                Mock(),
                BackendContext(request_timeout=5.0),
                _StubAuth(),
                project="p",
                region="us-central1",
                service_tier="gold",
            )


class TestNativeVertexFactory:
    """``NativeVertexFactory.create`` is a config parser — it wires the yaml
    backend block into BackendContext + AuthProvider. Exercised with an
    ``api_key`` auth block so google-auth / SA files are not needed.
    """

    def test_builds_context_and_auth_from_backend_block(self) -> None:
        cfg = {
            "type": "openai",
            "base_url": "https://us-central1-aiplatform.googleapis.com/v1beta/...",
            "auth": {"mode": "api_key", "api_key": "test-key-not-a-secret"},
            "rate_limit": {"per_minute": 120},
            "retry": {"timeout": 60, "base": 2.0, "factor": 2.0, "max_delay": 10.0},
            "timeout": 45.0,
            "service_tier": "priority",
        }
        backend = NativeVertexFactory(Mock()).create(
            cfg, project="p", region="us-central1"
        )
        assert backend._ctx.request_timeout == 45.0
        assert backend._ctx.retry is not None
        assert backend._ctx.retry.base == 2.0
        assert backend._ctx.retry.max_delay == 10.0
        assert backend._ctx.rate_limiter is not None
        assert backend._ctx.rate_limiter.per_minute == 120
        assert backend._service_tier == "priority"

    def test_raises_when_auth_missing(self) -> None:
        with pytest.raises(ValueError, match="auth"):
            NativeVertexFactory(Mock()).create(
                {"type": "openai", "base_url": "https://foo"},
                project="p",
                region="us-central1",
            )
