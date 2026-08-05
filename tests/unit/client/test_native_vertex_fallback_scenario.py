"""Fallback scenarios composing ``NativeVertexBackend`` primitives with a
plain ``ChatClient`` fallback — the pattern llm-xray uses.

These tests are NOT unit tests of a single class. They validate that the
primitives shipped in PR 1 (``cache_create`` / ``generate_content`` /
``cache_delete``) plus the existing ``RetryHelper`` layering are enough to
build the "native primary → chat fallback" pattern in caller code, with
no llm-infer helper. See arc-backend-expansion-2026-08 decision #6:
the drain/fallback helper originally scoped as PR 3 was dropped because
this pattern is ~10 lines at the call site.

Two granularities are exercised, both viable on the same primitives:

* Whole-drain fallback (llm-xray's current shape): any slice failure
  discards partial native results and re-runs the whole batch on the
  fallback. Preserves batch-atomic billing.
* Per-slice salvage: a failing slice falls back individually; other
  slices keep their native results. Salvages cache benefit at the cost
  of mixed-surface billing complexity.

Retry-then-fallback layering is verified: transient errors inside the
native call retry per ``RetryConfig`` first; only after ``RetryHelper``
exhausts does the ``BackendError`` propagate as the fallback signal.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from appinfra.dot_dict import DotDict  # noqa: F401  (kept for parity with other tests)

from llm_infer.client.backends.context import BackendContext, RetryConfig
from llm_infer.client.backends.vertex_native import NativeVertexBackend
from llm_infer.client.errors import BackendError, BackendRequestError
from llm_infer.client.types import ChatResponse

pytestmark = pytest.mark.unit


class _StubAuth:
    """Stub AuthProvider — every call returns a constant bearer header."""

    def headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer fake-token"}

    async def headers_async(self) -> dict[str, str]:
        return self.headers()


class _Recorder:
    """Sequential response scripter. Pops the next response per request;
    excess requests raise so scenarios that over-invoke fail loudly."""

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


class _AlwaysFails:
    """Persistent-500 scripter — used to drive RetryHelper to exhaustion
    without accounting for exact attempt counts (which depend on the
    ``timeout``/``base`` interplay in ``RetryConfig``)."""

    def __init__(self, status: int = 500) -> None:
        self._status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self._status, json={"error": "always"})


def _resp(status: int, body: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(status, json=body or {})


def _make_native(
    responses: list[httpx.Response],
    *,
    retry: RetryConfig | None = None,
    lg: Any = None,
) -> tuple[NativeVertexBackend, _Recorder]:
    """Build a NativeVertexBackend wired to a scripted MockTransport."""
    recorder = _Recorder(responses)
    return _wrap_native(recorder, retry=retry, lg=lg), recorder


def _make_native_always_fails(
    *,
    retry: RetryConfig | None = None,
    lg: Any = None,
) -> tuple[NativeVertexBackend, _AlwaysFails]:
    """Backend backed by a persistent-500 transport — used when the test
    wants RetryHelper to exhaust without asserting the exact retry count."""
    handler = _AlwaysFails()
    return _wrap_native(handler, retry=retry, lg=lg), handler


def _wrap_native(
    handler: Any, *, retry: RetryConfig | None, lg: Any
) -> NativeVertexBackend:
    ctx = BackendContext(retry=retry, request_timeout=5.0)
    return NativeVertexBackend(
        lg or Mock(),
        ctx,
        _StubAuth(),
        project="p",
        region="us-central1",
        transport=httpx.MockTransport(handler),
    )


def _cache_ok(
    name: str = "projects/p/locations/us-central1/cachedContents/abc",
) -> httpx.Response:
    return _resp(200, {"name": name, "usageMetadata": {"totalTokenCount": 2500}})


def _generate_ok(text: str) -> httpx.Response:
    return _resp(
        200,
        {"candidates": [{"content": {"parts": [{"text": text}]}}]},
    )


def _delete_ok() -> httpx.Response:
    return _resp(204)


def _extract_text(payload: dict[str, Any]) -> str:
    """Pull the assistant text out of a generateContent response."""
    return payload["candidates"][0]["content"]["parts"][0]["text"]


def _mock_fallback_chat(replies: list[str]) -> Mock:
    """A ChatClient stand-in whose ``chat_async`` returns scripted content
    per call. Signature-permissive — accepts anything, so scenarios don't
    have to mock every ChatClient kwarg."""
    fallback = Mock()
    scripted = list(replies)
    call_log: list[dict[str, Any]] = []

    async def _chat_async(*args: Any, **kwargs: Any) -> ChatResponse:
        call_log.append(kwargs)
        if not scripted:
            raise AssertionError("fallback chat_async called more times than scripted")
        return ChatResponse(
            content=scripted.pop(0),
            model=kwargs.get("model", "fallback-model"),
            provider="fallback",
        )

    fallback.chat_async = AsyncMock(side_effect=_chat_async)
    fallback.call_log = call_log
    return fallback


# =========================================================================
# Scenario 1: happy path — native serves all slices, fallback never invoked
# =========================================================================


class TestNativeHappyPath:
    """When every native call succeeds, the fallback is never touched."""

    @pytest.mark.asyncio
    async def test_all_slices_served_by_native(self) -> None:
        slices = ["claims", "entities", "questions"]
        native, recorder = _make_native(
            [
                _cache_ok(),
                _generate_ok("claims-result"),
                _generate_ok("entities-result"),
                _generate_ok("questions-result"),
                _delete_ok(),
            ]
        )
        fallback = _mock_fallback_chat([])

        results = await _run_whole_drain_with_fallback(
            native, fallback, model="gemini-2.5-flash-lite", slices=slices
        )

        assert [_extract_text(r) for r in results] == [
            "claims-result",
            "entities-result",
            "questions-result",
        ]
        fallback.chat_async.assert_not_called()
        # cache_create + 3× generate_content + cache_delete = 5 HTTP calls
        assert len(recorder.requests) == 5
        assert recorder.requests[0].url.path.endswith("/cachedContents")
        assert recorder.requests[-1].method == "DELETE"


# =========================================================================
# Scenario 2: cache_create fails post-retry → whole batch on fallback
# =========================================================================


class TestCacheCreateFailure:
    """cache_create fails → ``BackendError`` propagates; the caller catches
    and runs all slices on the fallback. Two variants: no-retry
    (single-shot fail) and retry-exhausted (persistent failure)."""

    @pytest.mark.asyncio
    async def test_no_retry_500_triggers_whole_drain_fallback(self) -> None:
        # retry=None → first 500 raises immediately, no attempt count games.
        slices = ["claims", "entities", "questions"]
        native, recorder = _make_native([_resp(500, {"error": "boom"})])
        fallback = _mock_fallback_chat(["fb-claims", "fb-entities", "fb-questions"])

        results = await _run_whole_drain_with_fallback(
            native, fallback, model="gemini-2.5-flash-lite", slices=slices
        )

        assert [r.content for r in results] == [
            "fb-claims",
            "fb-entities",
            "fb-questions",
        ]
        assert fallback.chat_async.await_count == 3
        user_texts = [call["messages"][0]["content"] for call in fallback.call_log]
        assert user_texts == slices

    @pytest.mark.asyncio
    async def test_retry_exhausted_500_triggers_whole_drain_fallback(self) -> None:
        # Persistent 500 with a short retry budget — RetryHelper burns through
        # its window and re-raises BackendRequestError, which the caller's
        # try/except turns into fallback dispatch.
        retry = RetryConfig(base=0.001, factor=1.0, max_delay=0.01, timeout=0.05)
        slices = ["claims", "entities", "questions"]
        native, handler = _make_native_always_fails(retry=retry)
        fallback = _mock_fallback_chat(["fb-claims", "fb-entities", "fb-questions"])

        results = await _run_whole_drain_with_fallback(
            native, fallback, model="gemini-2.5-flash-lite", slices=slices
        )

        assert [r.content for r in results] == [
            "fb-claims",
            "fb-entities",
            "fb-questions",
        ]
        assert fallback.chat_async.await_count == 3
        # RetryHelper made > 1 attempt before giving up.
        assert len(handler.requests) > 1


# =========================================================================
# Scenario 3: mid-drain failure — whole-drain fallback shape (llm-xray)
# =========================================================================


class TestMidDrainWholeFallback:
    """Slice 2/3 fails; caller discards native results and re-runs the whole
    batch on the fallback. Matches llm-xray's atomic-billing contract:
    partial native results are never surfaced."""

    @pytest.mark.asyncio
    async def test_second_slice_failure_reruns_all_on_fallback(self) -> None:
        slices = ["claims", "entities", "questions"]
        # RetryConfig=None → generate_content raises on first 500 (no retry
        # budget), which lets us script exactly one 500 for the failing slice
        # without accounting for retries.
        native, recorder = _make_native(
            [
                _cache_ok(),
                _generate_ok("claims-result"),  # slice 1 succeeds on native
                _resp(500, {"error": "boom"}),  # slice 2 fails
                _generate_ok("questions-result"),  # scripted but MUST NOT be called
                _delete_ok(),
            ]
        )
        # gather() cancels the third generate_content task as soon as the
        # second raises, so the scripted third response goes unused.
        fallback = _mock_fallback_chat(["fb-claims", "fb-entities", "fb-questions"])

        results = await _run_whole_drain_with_fallback(
            native, fallback, model="gemini-2.5-flash-lite", slices=slices
        )

        assert [r.content for r in results] == [
            "fb-claims",
            "fb-entities",
            "fb-questions",
        ]
        # All 3 slices went to fallback, including slice 1 whose native
        # result was discarded per the whole-drain contract.
        assert fallback.chat_async.await_count == 3
        # cache_delete was still invoked in finally after the drain raised.
        delete_calls = [r for r in recorder.requests if r.method == "DELETE"]
        assert len(delete_calls) == 1


# =========================================================================
# Scenario 4: mid-drain failure — per-slice salvage shape
# =========================================================================


class TestMidDrainPerSliceSalvage:
    """Slice 2/3 fails; slices 1 and 3 keep their native results, only
    slice 2 falls back. Salvages cache benefit at the caller's discretion —
    the primitives don't decide this, the caller does."""

    @pytest.mark.asyncio
    async def test_only_failing_slice_falls_back(self) -> None:
        slices = ["claims", "entities", "questions"]
        native, recorder = _make_native(
            [
                _cache_ok(),
                _generate_ok("claims-result"),  # slice 1: native
                _resp(500, {"error": "boom"}),  # slice 2: native fails → fallback
                _generate_ok("questions-result"),  # slice 3: native
                _delete_ok(),
            ]
        )
        fallback = _mock_fallback_chat(["fb-entities"])

        results = await _run_per_slice_salvage(
            native, fallback, model="gemini-2.5-flash-lite", slices=slices
        )

        contents = [
            _extract_text(r) if isinstance(r, dict) else r.content for r in results
        ]
        assert contents == ["claims-result", "fb-entities", "questions-result"]
        # Only slice 2 was sent to fallback.
        assert fallback.chat_async.await_count == 1
        assert fallback.call_log[0]["messages"][0]["content"] == "entities"
        # cache_delete still runs.
        assert any(r.method == "DELETE" for r in recorder.requests)


# =========================================================================
# Scenario 5: retry-then-fallback — native retries transient errors first
# =========================================================================


class TestRetryThenFallback:
    """Transient errors inside cache_create retry per ``RetryConfig`` first.
    Only when ``RetryHelper`` exhausts does the exception propagate as the
    fallback signal — same layering as ``RetryChatClient`` on the chat
    surface. The fallback path only sees BackendError post-exhaustion."""

    @pytest.mark.asyncio
    async def test_transient_retries_before_fallback_engages(self) -> None:
        retry = RetryConfig(base=0.001, factor=1.0, max_delay=0.01, timeout=5.0)
        # cache_create fails 2× (transient), then succeeds. Then the 3 slices
        # all succeed. No fallback engagement — retry absorbed the failures.
        native, recorder = _make_native(
            [
                _resp(500, {"error": "temp"}),  # cache_create retry 1
                _resp(503, {"error": "temp"}),  # cache_create retry 2
                _cache_ok(),  # cache_create success
                _generate_ok("claims-result"),
                _generate_ok("entities-result"),
                _generate_ok("questions-result"),
                _delete_ok(),
            ],
            retry=retry,
        )
        fallback = _mock_fallback_chat([])

        results = await _run_whole_drain_with_fallback(
            native,
            fallback,
            model="gemini-2.5-flash-lite",
            slices=["claims", "entities", "questions"],
        )

        # Native served everything after retry absorbed the two 500/503s.
        assert [_extract_text(r) for r in results] == [
            "claims-result",
            "entities-result",
            "questions-result",
        ]
        fallback.chat_async.assert_not_called()
        # 3 cache_create attempts (2 fail + 1 success) + 3 generate + 1 delete
        assert len(recorder.requests) == 7

    @pytest.mark.asyncio
    async def test_persistent_transient_exhausts_retry_then_falls_back(self) -> None:
        retry = RetryConfig(base=0.001, factor=1.0, max_delay=0.01, timeout=0.05)
        native, handler = _make_native_always_fails(retry=retry)
        fallback = _mock_fallback_chat(["fb-a", "fb-b"])

        # cache_create's retry budget exhausts and raises BackendRequestError.
        with pytest.raises(BackendRequestError):
            await _cache_create_or_raise(native, model="gemini-2.5-flash-lite")

        # RetryHelper attempted more than once before giving up.
        assert len(handler.requests) > 1

        # Caller catches, runs slices on fallback — this half of the flow is
        # what the try/except boundary above provides.
        results = [
            await fallback.chat_async(messages=[{"role": "user", "content": s}])
            for s in ("a", "b")
        ]
        assert [r.content for r in results] == ["fb-a", "fb-b"]


# =========================================================================
# Caller patterns under test
#
# These are inline reference implementations — the point of these tests
# is that they are ~15 lines each and live in the caller, not in llm-infer.
# =========================================================================


async def _run_whole_drain_with_fallback(
    native: NativeVertexBackend,
    fallback: Any,
    *,
    model: str,
    slices: list[str],
    ttl_seconds: int = 60,
    system: str = "you are careful",
    prefix: str = "corpus",
) -> list[Any]:
    """Whole-drain fallback: any native failure discards partial native
    results and re-runs every slice on the fallback. Matches llm-xray's
    atomic-billing contract."""
    try:
        cache_name, _usage = await native.cache_create(
            model=model, system=system, user_text=prefix, ttl_seconds=ttl_seconds
        )
        try:
            tasks = [
                native.generate_content(
                    model=model,
                    cache_name=cache_name,
                    user_text=slice_text,
                    generation_config={},
                )
                for slice_text in slices
            ]
            return list(await asyncio.gather(*tasks))
        finally:
            await native.cache_delete(cache_name)
    except BackendError:
        return [
            await fallback.chat_async(
                messages=[{"role": "user", "content": slice_text}],
                system=system,
                model=model,
            )
            for slice_text in slices
        ]


async def _run_per_slice_salvage(
    native: NativeVertexBackend,
    fallback: Any,
    *,
    model: str,
    slices: list[str],
    ttl_seconds: int = 60,
    system: str = "you are careful",
    prefix: str = "corpus",
) -> list[Any]:
    """Per-slice salvage: a failing slice falls back individually; other
    slices keep their native results. Different granularity than
    whole-drain — same primitives."""
    try:
        cache_name, _ = await native.cache_create(
            model=model, system=system, user_text=prefix, ttl_seconds=ttl_seconds
        )
    except BackendError:
        return [
            await fallback.chat_async(
                messages=[{"role": "user", "content": s}], system=system, model=model
            )
            for s in slices
        ]

    try:
        results: list[Any] = []
        for slice_text in slices:
            try:
                r = await native.generate_content(
                    model=model,
                    cache_name=cache_name,
                    user_text=slice_text,
                    generation_config={},
                )
                results.append(r)
            except BackendError:
                fb = await fallback.chat_async(
                    messages=[{"role": "user", "content": slice_text}],
                    system=system,
                    model=model,
                )
                results.append(fb)
        return results
    finally:
        await native.cache_delete(cache_name)


async def _cache_create_or_raise(
    native: NativeVertexBackend, *, model: str
) -> tuple[str, dict[str, Any]]:
    """Isolate the cache_create call so retry-then-fallback tests can
    assert the exception surface without running the drain."""
    return await native.cache_create(
        model=model, system="s", user_text="u", ttl_seconds=60
    )


# _ensure request bodies serialize (touch json to silence unused import warnings)
_ = json
