# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Vertex REST backend for ``cachedContents`` + ``generateContent``.

:class:`~.providers.gemini.GeminiBackend` routes through Vertex's
OpenAI-compat surface, which silently ignores the ``cachedContent`` request
field. To use explicit context caching we have to speak Vertex's native
REST directly. This backend owns that wire path while reusing the rest of
the client substrate: rate limiter, retry helper, auth provider, request
timeout — all driven off the same yaml backend block the OpenAI-compat
Gemini backend consumes.

Public API is shaped for the cached-extraction path — three async methods
for the lifecycle we care about:

- :meth:`cache_create`: allocate a ``cachedContents`` handle for a shared
  (system, prefix) so slice calls read cached input at ~10% of standard
  rate.
- :meth:`generate_content`: fire one ``generateContent`` against a cache
  handle with a per-slice user turn and generation config.
- :meth:`cache_delete`: best-effort cleanup; TTL expires the cache
  otherwise.

Errors are translated to :class:`~..errors.BackendRequestError` /
:class:`~..errors.BackendUnavailableError` so
:meth:`~..retry.RetryHelper.call_async` classifies 429 / 5xx / timeouts as
transient and backs off per the yaml ``retry: {timeout, base, factor,
max_delay}`` block. Non-transient failures propagate; callers translate to
their own error type.

Sibling to :class:`~.base.Backend` without inheriting from it: the
chat-shaped abstract methods don't fit Vertex's cache lifecycle (allocate
→ reference N times → delete), and the caller owns that lifecycle
explicitly.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import httpx
from appinfra.dot_dict import DotDict
from appinfra.log import Logger

from ..errors import BackendRequestError, BackendUnavailableError
from ..retry import RetryHelper
from ..types import ChatRequest, LLMCallbacks
from .auth import AuthProvider, auth_from_config
from .context import BackendContext, context_from_config
from .vertex_common import (
    SERVED_TIER_HEADER,
    VERTEX_PRIORITY_HEADER,
    validate_service_tier,
)

_PROVIDER = "vertex_native"


def _make_retry_request(model: str | None) -> ChatRequest:
    """Build a ChatRequest carrying only the log metadata the retry helper
    surfaces (``.id`` / ``.model``). ``messages=[]`` marks the absence of a
    chat body — this backend's calls are cache/generate REST, not chat. The
    ``native-`` id prefix distinguishes these entries in retry logs."""
    return ChatRequest(messages=[], model=model, id=f"native-{uuid.uuid4().hex[:12]}")


class NativeVertexBackend:
    """Vertex native REST client for cachedContents + generateContent.

    Owns retry/backoff, rate-limit, auth, and request timeout via the
    shared client substrate. Public methods raise
    :class:`~..errors.BackendRequestError` /
    :class:`~..errors.BackendUnavailableError` on non-2xx responses /
    network failures; the retry helper transparently retries transient
    ones (429, 5xx, unavailability) per ``ctx.retry``.
    """

    def __init__(
        self,
        lg: Logger,
        ctx: BackendContext,
        auth: AuthProvider,
        project: str,
        region: str,
        service_tier: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        callbacks: LLMCallbacks | None = None,
    ) -> None:
        if not project or not region:
            raise ValueError(
                f"NativeVertexBackend: project and region are required "
                f"(got project={project!r}, region={region!r})"
            )
        self._lg = lg
        self._ctx = ctx
        self._auth = auth
        self._project = project
        self._region = region
        self._service_tier = validate_service_tier(service_tier)
        # Tests inject ``httpx.MockTransport`` via ``transport=`` to intercept
        # without hitting the network. Persistent client amortizes TCP+TLS.
        self._client = httpx.AsyncClient(
            timeout=ctx.request_timeout, transport=transport
        )
        self._retry = RetryHelper(lg, ctx, provider=_PROVIDER)
        # Callbacks (on_retry / on_error) plumb through to the retry helper.
        self._callbacks = callbacks

    async def aclose(self) -> None:
        """Close the underlying httpx client. Optional — httpx's finalizer
        cleans up sockets on drop; call this to release connections early
        (tests, backend swap)."""
        await self._client.aclose()

    async def cache_create(
        self,
        model: str,
        system: str,
        user_text: str,
        ttl_seconds: int,
    ) -> tuple[str, dict[str, Any]]:
        """Allocate a ``cachedContents`` handle. Returns
        ``(resource_name, usageMetadata)`` — usageMetadata carries
        ``totalTokenCount`` (all tokens written to cache, billed at input
        rate on Vertex). Empty dict when the field is absent.

        Note: this operation retries on transient errors (429/5xx/timeout).
        If the server succeeds but the client times out before receiving the
        response, a retry will create a duplicate cache. The TTL ensures
        orphaned caches expire; callers needing stricter idempotency should
        add a ``displayName`` and reconcile before retry.
        """
        body = {
            "model": self._model_ref(model),
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "ttl": f"{ttl_seconds}s",
        }
        payload = await self._post_json(self._cache_create_url(), body, model=model)
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise BackendRequestError(
                f"cache create returned no name: {json.dumps(payload)[:400]}"
            )
        usage = payload.get("usageMetadata")
        if not isinstance(usage, dict):
            usage = {}
        self._lg.debug(
            "vertex native cache created",
            extra={
                "cache": name,
                "ttl_s": ttl_seconds,
                "cache_tokens": usage.get("totalTokenCount"),
            },
        )
        return name, usage

    async def generate_content(
        self,
        model: str,
        cache_name: str,
        user_text: str,
        generation_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Fire one ``generateContent`` against a cache handle. Returns the
        parsed JSON response body."""
        body = {
            "cachedContent": cache_name,
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": generation_config,
        }
        return await self._post_json(self._generate_url(model), body, model=model)

    async def cache_delete(self, cache_name: str) -> None:
        """Best-effort delete. Failures are logged; TTL expires the cache."""
        try:
            await self._delete(self._cache_delete_url(cache_name))
        except (BackendRequestError, BackendUnavailableError) as e:
            self._lg.warning(
                "vertex native cache delete failed",
                extra={"cache": cache_name, "exception": e},
            )

    # ---- URL helpers -------------------------------------------------------

    def _host(self) -> str:
        # Global endpoint has no region prefix. Regional endpoints prepend
        # `{region}-`. cachedContents observed working on both — global is
        # the only path where newer flash-lite generations are reachable
        # before regional rollout completes.
        if self._region == "global":
            return "aiplatform.googleapis.com"
        return f"{self._region}-aiplatform.googleapis.com"

    def _cache_create_url(self) -> str:
        return (
            f"https://{self._host()}/v1"
            f"/projects/{self._project}/locations/{self._region}/cachedContents"
        )

    def _cache_delete_url(self, cache_name: str) -> str:
        return f"https://{self._host()}/v1/{cache_name}"

    def _generate_url(self, model: str) -> str:
        return (
            f"https://{self._host()}/v1"
            f"/projects/{self._project}/locations/{self._region}"
            f"/publishers/google/models/{model}:generateContent"
        )

    def _model_ref(self, model: str) -> str:
        return f"projects/{self._project}/locations/{self._region}/publishers/google/models/{model}"

    # ---- HTTP core ---------------------------------------------------------

    async def _post_json(
        self, url: str, body: dict[str, Any], *, model: str | None = None
    ) -> dict[str, Any]:
        """POST retried through ``RetryHelper.call_async``. Rate limit and
        auth are re-acquired on every attempt so a stale token or a
        cache-hit rate slot doesn't survive a retry."""

        async def _once() -> dict[str, Any]:
            await self._acquire_slot()
            headers = await self._build_headers()
            try:
                resp = await self._client.post(url, json=body, headers=headers)
            except httpx.TimeoutException as e:
                raise BackendUnavailableError(f"POST {url} timed out: {e}") from e
            except httpx.HTTPError as e:
                raise BackendUnavailableError(f"POST {url} network error: {e}") from e
            payload = _decode_json(url, resp, method="POST")
            self._detect_downgrade(url, resp)
            return payload

        return await self._retry.call_async(
            _once, request=_make_retry_request(model), callbacks=self._callbacks
        )

    async def _delete(self, url: str, *, model: str | None = None) -> None:
        """DELETE retried through ``RetryHelper.call_async``."""

        async def _once() -> None:
            await self._acquire_slot()
            headers = await self._build_headers()
            try:
                resp = await self._client.delete(url, headers=headers)
            except httpx.TimeoutException as e:
                raise BackendUnavailableError(f"DELETE {url} timed out: {e}") from e
            except httpx.HTTPError as e:
                raise BackendUnavailableError(f"DELETE {url} network error: {e}") from e
            if not (200 <= resp.status_code < 300):
                raise BackendRequestError(
                    f"DELETE {url} -> {resp.status_code}: {resp.text[:400]}",
                    status_code=resp.status_code,
                )

        await self._retry.call_async(
            _once, request=_make_retry_request(model), callbacks=self._callbacks
        )

    async def _acquire_slot(self) -> None:
        """Block on the rate limiter without pinning the event loop. Skips
        cleanly when no limiter is configured."""
        rl = self._ctx.rate_limiter
        if rl is None:
            return
        await asyncio.to_thread(rl.next)

    async def _build_headers(self) -> dict[str, str]:
        """Auth headers plus the Vertex Priority tier header when configured.
        Called per-attempt so retries carry a fresh token; the tier header is
        constant but rides along for uniformity."""
        headers = await self._auth.headers_async()
        if self._service_tier == "priority":
            headers[VERTEX_PRIORITY_HEADER] = "priority"
        return headers

    def _detect_downgrade(self, url: str, resp: httpx.Response) -> None:
        """Vertex Priority can silently fall back to Standard under capacity
        pressure; the served tier is reported in ``x-gemini-service-tier``.
        Log WARN so downgrades surface without a callback wired in — mirrors
        the OpenAI-compat GeminiBackend."""
        if self._service_tier != "priority":
            return
        served = resp.headers.get(SERVED_TIER_HEADER)
        if served is None or served == "priority":
            return
        self._lg.warning(
            "vertex priority downgraded",
            extra={
                "url": url,
                "tier_requested": "priority",
                "tier_served": served,
            },
        )


class NativeVertexFactory:
    """Builds :class:`NativeVertexBackend` from a yaml backend block.

    Separate from :class:`~.factory.BackendFactory` because
    :class:`NativeVertexBackend` is a sibling to :class:`~.base.Backend`
    (cache lifecycle rather than chat) and needs ``project`` / ``region``
    as explicit kwargs — the chat surface has no equivalent.

    The block shape matches what ``BackendFactory`` consumes for the
    OpenAI-compat Vertex path (``auth`` / ``rate_limit`` / ``retry`` /
    ``timeout`` / ``service_tier``), so a single yaml block can serve
    both the chat path and the native-cache path without duplication.
    """

    def __init__(self, lg: Logger) -> None:
        self._lg = lg

    def create(
        self,
        config: DotDict,
        *,
        project: str,
        region: str,
        callbacks: LLMCallbacks | None = None,
    ) -> NativeVertexBackend:
        """Build a backend from the yaml block. Raises ``ValueError`` when
        no ``auth`` block is present (native Vertex needs SA credentials)."""
        ctx = self._create_context(config)
        auth = auth_from_config(self._lg, config.get("auth"))
        if auth is None:
            raise ValueError(
                "NativeVertexBackend requires an auth block; got none in backend yaml"
            )
        return NativeVertexBackend(
            self._lg,
            ctx,
            auth,
            project=project,
            region=region,
            service_tier=config.get("service_tier"),
            callbacks=callbacks,
        )

    def _create_context(self, config: DotDict) -> BackendContext:
        return context_from_config(self._lg, config)


def _decode_json(
    url: str, resp: httpx.Response, method: str = "POST"
) -> dict[str, Any]:
    if 200 <= resp.status_code < 300:
        try:
            payload = resp.json()
        except ValueError as e:
            raise BackendRequestError(
                f"{method} {url}: invalid JSON body: {resp.text[:400]}",
                status_code=resp.status_code,
            ) from e
        if not isinstance(payload, dict):
            raise BackendRequestError(
                f"{method} {url}: JSON body is not an object: {type(payload).__name__}",
                status_code=resp.status_code,
            )
        return payload
    raise BackendRequestError(
        f"{method} {url} -> {resp.status_code}: {resp.text[:400]}",
        status_code=resp.status_code,
    )
