"""Unit tests for FallbackClient and fallback helpers."""

from unittest.mock import MagicMock

import pytest

from llm_infer.client.backends import RetryConfig
from llm_infer.client.errors import (
    BackendRequestError,
    BackendTimeoutError,
    ConfigError,
    FallbackAmbiguityError,
)
from llm_infer.client.fallback import FallbackClient
from llm_infer.client.fallback_helper import detect_cycles, parse_fallback_key
from llm_infer.client.router import ResolvedTarget
from llm_infer.client.types import ChatResponse

pytestmark = pytest.mark.unit


class TestDetectCycles:
    """Tests for detect_cycles helper."""

    def test_no_cycles_returns_empty(self) -> None:
        """Config without cycles returns empty set."""
        fallbacks = {
            "gpt-4o": "claude-sonnet",
            "claude-sonnet": "gemini-pro",
        }
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert cycles == set()
        lg.warning.assert_not_called()

    def test_simple_cycle_detected(self) -> None:
        """Simple A->B->A cycle is detected."""
        fallbacks = {
            "a": "b",
            "b": "a",
        }
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert cycles == {"a", "b"}
        lg.warning.assert_called_once()
        call_args = lg.warning.call_args
        assert "cycle" in call_args[1]["extra"]

    def test_longer_cycle_detected(self) -> None:
        """Longer A->B->C->A cycle is detected."""
        fallbacks = {
            "a": "b",
            "b": "c",
            "c": "a",
        }
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert cycles == {"a", "b", "c"}
        lg.warning.assert_called_once()

    def test_self_loop_detected(self) -> None:
        """Self-loop A->A is detected."""
        fallbacks = {"a": "a"}
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert "a" in cycles
        lg.warning.assert_called_once()


class TestFallbackClientImport:
    """Test FallbackClient can be imported."""

    def test_import_from_client_package(self) -> None:
        """FallbackClient is exported from client package."""
        from llm_infer.client import FallbackClient

        assert FallbackClient is not None

    def test_import_directly(self) -> None:
        """FallbackClient can be imported directly."""
        from llm_infer.client.fallback import FallbackClient

        assert FallbackClient is not None


class TestFallbackClientLogging:
    """Tests for FallbackClient logging behavior."""

    @pytest.fixture
    def mock_router(self) -> MagicMock:
        """Create a mock router with resolve and get_client methods."""
        router = MagicMock()

        # resolve() returns ResolvedTarget with model and backend
        def mock_resolve(model: str | None = None, backend: str | None = None):
            return ResolvedTarget(
                model=model or "default-model", backend="test-backend"
            )

        router.resolve = mock_resolve
        return router

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        """Create a mock logger."""
        return MagicMock()

    def test_logs_warning_on_fallback(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Should log warning with details when falling back to another model."""
        # Setup: first model fails with 500, second succeeds
        mock_client = MagicMock()
        call_count = 0

        def mock_chat(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendRequestError("Internal Server Error", status_code=500)
            return ChatResponse(
                content="success", model="claude-sonnet", provider="anthropic"
            )

        mock_client._chat = mock_chat
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        # Act
        response = client.chat([{"role": "user", "content": "hello"}], model="gpt-4o")

        # Assert - response succeeded via fallback
        assert response.content == "success"

        # Assert - warning logged with correct fields
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert call_args[0][0] == "model request failed, trying fallback"

        extra = call_args[1]["extra"]
        assert extra["failed_model"] == "gpt-4o"
        assert extra["fallback_model"] == "claude-sonnet"
        assert extra["error_type"] == "BackendRequestError"
        assert extra["status_code"] == 500
        assert "Internal Server Error" in extra["error"]
        assert extra["attempt"] == 1

    def test_logs_multiple_fallbacks_in_chain(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Should log each fallback attempt in a chain."""
        # Setup: first two models fail, third succeeds
        mock_client = MagicMock()
        call_count = 0

        def mock_chat(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendRequestError("Service Unavailable", status_code=503)
            if call_count == 2:
                raise BackendTimeoutError("Request timed out")
            return ChatResponse(
                content="finally worked", model="gemini-pro", provider="google"
            )

        mock_client._chat = mock_chat
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {
            "gpt-4o": "claude-sonnet",
            "claude-sonnet": "gemini-pro",
        }
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        # Act
        response = client.chat([{"role": "user", "content": "hello"}], model="gpt-4o")

        # Assert - succeeded on third model
        assert response.content == "finally worked"

        # Assert - two warning logs (one per fallback)
        assert mock_logger.warning.call_count == 2

        # First fallback: gpt-4o -> claude-sonnet
        first_call = mock_logger.warning.call_args_list[0]
        assert first_call[1]["extra"]["failed_model"] == "gpt-4o"
        assert first_call[1]["extra"]["fallback_model"] == "claude-sonnet"
        assert first_call[1]["extra"]["status_code"] == 503
        assert first_call[1]["extra"]["attempt"] == 1

        # Second fallback: claude-sonnet -> gemini-pro
        second_call = mock_logger.warning.call_args_list[1]
        assert second_call[1]["extra"]["failed_model"] == "claude-sonnet"
        assert second_call[1]["extra"]["fallback_model"] == "gemini-pro"
        assert second_call[1]["extra"]["error_type"] == "BackendTimeoutError"
        assert second_call[1]["extra"]["status_code"] is None  # Timeout has no status
        assert second_call[1]["extra"]["attempt"] == 2

    def test_logs_error_when_all_models_fail(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Should log error when entire fallback chain is exhausted."""
        # Setup: all models fail
        mock_client = MagicMock()
        mock_client._chat = MagicMock(
            side_effect=BackendRequestError("Server Error", status_code=500)
        )
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        # Act & Assert - should raise after exhausting chain
        with pytest.raises(BackendRequestError):
            client.chat([{"role": "user", "content": "hello"}], model="gpt-4o")

        # Assert - warning for fallback attempt + error for chain exhaustion
        mock_logger.warning.assert_called_once()
        mock_logger.error.assert_called_once()

        error_call = mock_logger.error.call_args
        assert error_call[0][0] == "all fallback models failed"
        assert error_call[1]["extra"]["original_model"] == "gpt-4o"
        assert "Server Error" in error_call[1]["extra"]["final_error"]

    def test_no_logging_when_first_model_succeeds(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Should not log anything when first model succeeds."""
        mock_client = MagicMock()
        mock_client._chat = MagicMock(
            return_value=ChatResponse(
                content="success", model="gpt-4o", provider="openai"
            )
        )
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        # Act
        response = client.chat([{"role": "user", "content": "hello"}], model="gpt-4o")

        # Assert
        assert response.content == "success"
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_fallback_on_rate_limit(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """429 triggers fallback: by this layer the inner retry is exhausted."""
        mock_client = MagicMock()
        call_count = 0

        def mock_chat(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendRequestError("Rate limited", status_code=429)
            return ChatResponse(
                content="success", model="claude-sonnet", provider="anthropic"
            )

        mock_client._chat = mock_chat
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        # Act
        response = client.chat([{"role": "user", "content": "hello"}], model="gpt-4o")

        # Assert - served by the fallback model
        assert response.content == "success"
        assert call_count == 2

        # Assert - fallback logged with the 429
        mock_logger.warning.assert_called_once()
        extra = mock_logger.warning.call_args[1]["extra"]
        assert extra["failed_model"] == "gpt-4o"
        assert extra["fallback_model"] == "claude-sonnet"
        assert extra["status_code"] == 429
        mock_logger.error.assert_not_called()

    def test_cyclic_fallback_retries_until_success(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Cyclic fallback retries round-robin until one model succeeds."""
        mock_client = MagicMock()
        call_count = 0

        def mock_chat(request):
            nonlocal call_count
            call_count += 1
            if call_count < 4:  # Fail first 3 attempts (a, b, a)
                raise BackendRequestError("Service Unavailable", status_code=503)
            return ChatResponse(content="success", model="b", provider="test")

        mock_client._chat = mock_chat
        mock_router.get_client = MagicMock(return_value=mock_client)

        # Cyclic fallback: a -> b -> a (round-robin)
        fallbacks = {"a": "b", "b": "a"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        response = client.chat([{"role": "user", "content": "hello"}], model="a")

        assert response.content == "success"
        assert call_count == 4  # a fails, b fails, a fails, b succeeds


class TestFallbackOnRateLimitAllPaths:
    """429 → fallback across stream/async paths (sync chat covered above)."""

    @pytest.fixture
    def mock_router(self) -> MagicMock:
        """Create a mock router with resolve and get_client methods."""
        router = MagicMock()

        def mock_resolve(model: str | None = None, backend: str | None = None):
            return ResolvedTarget(
                model=model or "default-model", backend="test-backend"
            )

        router.resolve = mock_resolve
        return router

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        """Create a mock logger."""
        return MagicMock()

    def test_stream_fallback_on_rate_limit(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Pre-token 429 on a sync stream is served by the fallback model."""
        mock_client = MagicMock()
        call_count = 0

        def mock_chat_stream(request, holder):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendRequestError("Rate limited", status_code=429)
            return iter(["hello", " world"])

        mock_client._chat_stream = mock_chat_stream
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        stream = client.chat_stream(
            [{"role": "user", "content": "hello"}], model="gpt-4o"
        )
        tokens = list(stream)

        assert tokens == ["hello", " world"]
        assert call_count == 2
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args[1]["extra"]["status_code"] == 429

    @pytest.mark.asyncio
    async def test_async_fallback_on_rate_limit(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """429 on async chat is served by the fallback model."""
        mock_client = MagicMock()
        call_count = 0

        async def mock_chat_async(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendRequestError("Rate limited", status_code=429)
            return ChatResponse(
                content="success", model="claude-sonnet", provider="anthropic"
            )

        mock_client._chat_async = mock_chat_async
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        response = await client.chat_async(
            [{"role": "user", "content": "hello"}], model="gpt-4o"
        )

        assert response.content == "success"
        assert call_count == 2
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args[1]["extra"]["status_code"] == 429

    @pytest.mark.asyncio
    async def test_async_stream_fallback_on_rate_limit(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Pre-token 429 on an async stream is served by the fallback model."""
        mock_client = MagicMock()
        call_count = 0

        def mock_chat_stream_async(request, holder):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendRequestError("Rate limited", status_code=429)

            async def gen():
                yield "hello"
                yield " world"

            return gen()

        mock_client._chat_stream_async = mock_chat_stream_async
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        stream = client.chat_stream_async(
            [{"role": "user", "content": "hello"}], model="gpt-4o"
        )
        tokens = [token async for token in stream]

        assert tokens == ["hello", " world"]
        assert call_count == 2
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args[1]["extra"]["status_code"] == 429

    def test_mid_stream_rate_limit_propagates(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """429 after the first token propagates: partial output can't be replayed."""

        def mock_chat_stream(request, holder):
            yield "partial"
            raise BackendRequestError("Rate limited", status_code=429)

        mock_client = MagicMock()
        mock_client._chat_stream = mock_chat_stream
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        stream = client.chat_stream(
            [{"role": "user", "content": "hello"}], model="gpt-4o"
        )
        tokens = []
        with pytest.raises(BackendRequestError) as exc_info:
            for token in stream:
                tokens.append(token)

        assert tokens == ["partial"]
        assert exc_info.value.status_code == 429
        mock_logger.warning.assert_not_called()


class TestNoRetryWarning:
    """Construction-time warning for backends without retry config."""

    def _router(self, retry: RetryConfig | None) -> MagicMock:
        """Mock router exposing a real clients mapping with the given retry."""
        router = MagicMock()
        client = MagicMock()
        client.backend.ctx.retry = retry
        router.clients = {"primary": client}
        return router

    def test_warns_when_backend_has_no_retry(self) -> None:
        """retry: None means fallback engages on the first transient error."""
        lg = MagicMock()
        FallbackClient(lg, self._router(None), {"a": "b"})

        lg.warning.assert_called_once()
        msg = lg.warning.call_args[0][0]
        assert "no retry config" in msg
        assert lg.warning.call_args[1]["extra"]["backend"] == "primary"

    def test_no_warning_when_retry_configured(self) -> None:
        """Backends with a retry budget construct silently."""
        lg = MagicMock()
        FallbackClient(lg, self._router(RetryConfig()), {"a": "b"})

        lg.warning.assert_not_called()


class TestParseFallbackKey:
    """Tests for the parse_fallback_key helper."""

    def test_bare_key(self) -> None:
        assert parse_fallback_key("gpt-4o") == ("gpt-4o", None)

    def test_qualified_key(self) -> None:
        assert parse_fallback_key("gpt-4o@openai") == ("gpt-4o", "openai")

    def test_model_name_with_slashes_stays_intact(self) -> None:
        """OpenRouter-style ``provider/model`` MUST NOT split on ``/``."""
        assert parse_fallback_key("openai/gpt-4o@openrouter") == (
            "openai/gpt-4o",
            "openrouter",
        )

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValueError, match="both model and backend"):
            parse_fallback_key("@backend")

    def test_empty_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="both model and backend"):
            parse_fallback_key("model@")


class TestDetectCyclesWithQualifiedKeys:
    """detect_cycles treats bare and qualified keys as distinct strings."""

    def test_bare_and_qualified_are_distinct(self) -> None:
        """``M`` and ``M@b`` are separate nodes; no spurious cycle."""
        fallbacks = {"a": "b", "a@primary": "c"}
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert cycles == set()
        lg.warning.assert_not_called()

    def test_cycle_across_qualified_keys(self) -> None:
        """A cycle through qualified nodes is still detected."""
        fallbacks = {"a@primary": "b@backup", "b@backup": "a@primary"}
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert cycles == {"a@primary", "b@backup"}
        lg.warning.assert_called_once()


class TestAmbiguityCheck:
    """Eager @ ambiguity validation at FallbackClient construction."""

    def _router(
        self,
        catalogs: dict[str, list[str]],
    ) -> MagicMock:
        """Router mock exposing a real clients Mapping and a discovery stub."""
        router = MagicMock()

        # Real dict so isinstance(clients, Mapping) is True and iteration works.
        clients: dict[str, MagicMock] = {}
        for name in catalogs:
            client = MagicMock()
            client.backend.ctx.retry = RetryConfig()
            clients[name] = client
        router.clients = clients

        discovery = MagicMock()
        discovery.get_models_for_backend = lambda name: list(catalogs.get(name, []))
        router.discovery = discovery
        return router

    def test_bare_model_in_one_backend_ok(self) -> None:
        """Unambiguous bare model: construction succeeds."""
        router = self._router({"openai": ["gpt-4o"], "anthropic": ["claude-sonnet"]})
        FallbackClient(MagicMock(), router, {"gpt-4o": "claude-sonnet"})

    def test_bare_model_in_two_backends_raises(self) -> None:
        """Two backends serve the same bare model → FallbackAmbiguityError."""
        router = self._router(
            {
                "openai_a": ["gpt-4o"],
                "openai_b": ["gpt-4o"],
                "anthropic": ["claude-sonnet"],
            }
        )
        with pytest.raises(FallbackAmbiguityError) as exc:
            FallbackClient(MagicMock(), router, {"gpt-4o": "claude-sonnet"})

        assert exc.value.model == "gpt-4o"
        assert exc.value.backends == ["openai_a", "openai_b"]
        # Error message names the qualified options
        msg = str(exc.value)
        assert "gpt-4o@openai_a" in msg
        assert "gpt-4o@openai_b" in msg

    def test_qualified_ref_bypasses_ambiguity(self) -> None:
        """A qualified key/value pins the backend, so ambiguity doesn't apply."""
        router = self._router({"a": ["gpt-4o"], "b": ["gpt-4o"]})
        FallbackClient(MagicMock(), router, {"gpt-4o@a": "gpt-4o@b"})

    def test_ambiguity_from_value_reference(self) -> None:
        """A bare model appearing only as a value also triggers the check."""
        router = self._router(
            {"one": ["only-here"], "two": ["shared"], "three": ["shared"]}
        )
        with pytest.raises(FallbackAmbiguityError) as exc:
            FallbackClient(MagicMock(), router, {"only-here": "shared"})
        assert exc.value.model == "shared"

    def test_unknown_backend_in_qualified_raises(self) -> None:
        """``model@bogus`` where bogus is not a configured backend → ConfigError."""
        router = self._router({"openai": ["gpt-4o"]})
        with pytest.raises(ConfigError, match="unknown"):
            FallbackClient(MagicMock(), router, {"gpt-4o": "claude@bogus_backend"})

    def test_malformed_ref_raises_at_construction(self) -> None:
        """An ``@`` with an empty side surfaces immediately from parsing."""
        router = self._router({"openai": ["gpt-4o"]})
        with pytest.raises(ValueError, match="both model and backend"):
            FallbackClient(MagicMock(), router, {"gpt-4o": "@nowhere"})

    def test_probe_failure_treated_as_empty_catalog(self) -> None:
        """A backend that fails to probe is silently treated as empty (no false-positive)."""
        router = MagicMock()
        client = MagicMock()
        client.backend.ctx.retry = RetryConfig()
        router.clients = {"flaky": client, "ok": client}

        def probe(name: str) -> list[str]:
            if name == "flaky":
                raise RuntimeError("network down")
            return ["gpt-4o"]

        router.discovery = MagicMock()
        router.discovery.get_models_for_backend = probe

        # No ambiguity (only "ok" reports gpt-4o); construction succeeds.
        FallbackClient(MagicMock(), router, {"gpt-4o": "gpt-4o@ok"})


class TestQualifiedRouting:
    """The ``@backend`` suffix overrides model→backend routing at call time."""

    def _router_with_backends(
        self, catalogs: dict[str, list[str]]
    ) -> tuple[MagicMock, dict[str, MagicMock]]:
        """Router that records which backend each call was routed to."""
        clients: dict[str, MagicMock] = {}
        for name in catalogs:
            c = MagicMock()
            c.backend.ctx.retry = RetryConfig()
            clients[name] = c
        router = MagicMock()
        router.clients = clients

        discovery = MagicMock()
        discovery.get_models_for_backend = lambda n: list(catalogs.get(n, []))
        router.discovery = discovery

        # Model→backend from catalog (first-wins), for bare resolution.
        model_to_backend: dict[str, str] = {}
        for backend, models in catalogs.items():
            for m in models:
                model_to_backend.setdefault(m, backend)

        def resolve(
            model: str | None = None, backend: str | None = None
        ) -> ResolvedTarget:
            if backend is not None:
                return ResolvedTarget(model=model, backend=backend)
            resolved_backend = model_to_backend.get(model or "", "") or next(
                iter(catalogs)
            )
            return ResolvedTarget(model=model, backend=resolved_backend)

        router.resolve = resolve
        router.get_client = lambda backend=None, model=None: clients[backend]
        return router, clients

    def test_qualified_value_routes_to_named_backend(self) -> None:
        """Fallback value ``model@backend`` calls that backend, not the model's default."""
        router, clients = self._router_with_backends(
            {"anthropic": ["claude"], "backup": ["claude"]}
        )
        # First call (primary) fails; fallback pins to "backup".
        call_log: list[str] = []

        def make_chat(backend_name: str):
            def _chat(req):
                call_log.append(backend_name)
                if backend_name == "anthropic":
                    raise BackendRequestError("boom", status_code=500)
                return ChatResponse(content="ok", model="claude", provider="anthropic")

            return _chat

        clients["anthropic"]._chat = make_chat("anthropic")
        clients["backup"]._chat = make_chat("backup")

        # Bare "claude" would be ambiguous, so pin both entries.
        client = FallbackClient(
            MagicMock(),
            router,
            {"claude@anthropic": "claude@backup"},
        )
        resp = client.chat(
            [{"role": "user", "content": "hi"}], model="claude@anthropic"
        )
        assert resp.content == "ok"
        assert call_log == ["anthropic", "backup"]

    def test_qualified_lookup_wins_over_bare(self) -> None:
        """When both ``model`` and ``model@backend`` keys exist, qualified wins."""
        router, clients = self._router_with_backends(
            {"primary": ["gpt-4o"], "cheap": ["claude"], "premium": ["claude-opus"]}
        )
        # Primary path: gpt-4o (on 'primary') fails, then goes to claude-opus@premium.
        # A stray bare 'gpt-4o' -> 'cheap-fallback' entry should NOT be used
        # because 'gpt-4o@primary' is more specific.
        call_log: list[str] = []

        def _chat_primary(req):
            call_log.append("primary")
            raise BackendRequestError("boom", status_code=500)

        def _chat_premium(req):
            call_log.append("premium")
            return ChatResponse(content="ok", model="claude-opus", provider="anthropic")

        def _chat_cheap(req):
            call_log.append("cheap")
            return ChatResponse(
                content="wrong path", model="claude", provider="anthropic"
            )

        clients["primary"]._chat = _chat_primary
        clients["premium"]._chat = _chat_premium
        clients["cheap"]._chat = _chat_cheap

        client = FallbackClient(
            MagicMock(),
            router,
            {
                "gpt-4o": "claude",  # bare: NOT taken because qualified match exists
                "gpt-4o@primary": "claude-opus@premium",  # qualified: taken
            },
        )
        resp = client.chat([{"role": "user", "content": "hi"}], model="gpt-4o")
        assert resp.content == "ok"
        # 'cheap' backend must NEVER be called — qualified match preempted bare.
        assert "cheap" not in call_log
        assert call_log == ["primary", "premium"]

    def test_chain_semantics_with_at_syntax(self) -> None:
        """Multi-hop chain where each hop is @-qualified."""
        router, clients = self._router_with_backends(
            {"a": ["m1"], "b": ["m2"], "c": ["m3"]}
        )
        call_log: list[str] = []

        def _make(name: str, succeed: bool):
            def _chat(req):
                call_log.append(name)
                if not succeed:
                    raise BackendRequestError("boom", status_code=503)
                return ChatResponse(content=name, model=req.model, provider="test")

            return _chat

        clients["a"]._chat = _make("a", False)
        clients["b"]._chat = _make("b", False)
        clients["c"]._chat = _make("c", True)

        client = FallbackClient(
            MagicMock(),
            router,
            {"m1@a": "m2@b", "m2@b": "m3@c"},
        )
        resp = client.chat([{"role": "user", "content": "hi"}], model="m1@a")
        assert resp.content == "c"
        assert call_log == ["a", "b", "c"]
