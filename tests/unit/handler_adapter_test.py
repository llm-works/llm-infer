"""Unit tests for handler adapter resolution logic."""

from unittest.mock import MagicMock

import pytest

from llm_infer.serving.dispatch.handlers import SequentialHandler

pytestmark = pytest.mark.unit


class TestResolveEffectiveAdapter:
    """Test RequestHandler._resolve_effective_adapter method."""

    def _create_handler_with_mock_manager(
        self, available_adapters: set[str]
    ) -> SequentialHandler:
        """Create a mock handler with adapter manager.

        Uses SequentialHandler (concrete) with mocked engine.
        Mocks resolve() to return LoadedAdapter-like objects for available adapters.
        """
        mock_engine = MagicMock()
        handler = SequentialHandler(mock_engine)
        handler._adapter_manager = MagicMock()

        def mock_resolve(key: str) -> MagicMock | None:
            if key in available_adapters:
                adapter = MagicMock()
                adapter.key = key  # Return the same key (no version resolution in test)
                return adapter
            return None

        handler._adapter_manager.resolve = mock_resolve
        return handler

    def _create_handler_without_manager(self) -> SequentialHandler:
        """Create handler without adapter manager."""
        mock_engine = MagicMock()
        handler = SequentialHandler(mock_engine)
        handler._adapter_manager = None
        return handler

    def _create_request(
        self, model: str | None = None, adapter: str | None = None
    ) -> MagicMock:
        """Create a mock request with model and adapter fields."""
        request = MagicMock()
        request.model = model
        request.adapter = adapter
        return request

    def test_explicit_adapter_takes_priority(self) -> None:
        """Explicit adapter field should be used even if model is also set."""
        handler = self._create_handler_with_mock_manager({"adapter-a", "adapter-b"})
        request = self._create_request(model="adapter-a", adapter="adapter-b")

        result = handler._resolve_effective_adapter(request)

        assert result == "adapter-b"

    def test_model_field_fallback_to_adapter(self) -> None:
        """Model field should be used as adapter if it matches a known adapter."""
        handler = self._create_handler_with_mock_manager({"my-adapter"})
        request = self._create_request(model="my-adapter", adapter=None)

        result = handler._resolve_effective_adapter(request)

        assert result == "my-adapter"

    def test_model_field_not_adapter(self) -> None:
        """Model field should not be used if it doesn't match a known adapter."""
        handler = self._create_handler_with_mock_manager({"other-adapter"})
        request = self._create_request(model="unknown-model", adapter=None)

        result = handler._resolve_effective_adapter(request)

        assert result is None

    def test_reserved_auto_not_adapter(self) -> None:
        """Reserved 'auto' should not be looked up as adapter."""
        handler = self._create_handler_with_mock_manager({"auto"})  # Even if exists
        request = self._create_request(model="auto", adapter=None)

        result = handler._resolve_effective_adapter(request)

        assert result is None

    def test_reserved_default_not_adapter(self) -> None:
        """Reserved 'default' should not be looked up as adapter."""
        handler = self._create_handler_with_mock_manager({"default"})  # Even if exists
        request = self._create_request(model="default", adapter=None)

        result = handler._resolve_effective_adapter(request)

        assert result is None

    def test_no_model_no_adapter(self) -> None:
        """No model or adapter specified should return None."""
        handler = self._create_handler_with_mock_manager({"some-adapter"})
        request = self._create_request(model=None, adapter=None)

        result = handler._resolve_effective_adapter(request)

        assert result is None

    def test_no_adapter_manager(self) -> None:
        """When adapter manager is not set, model should not resolve to adapter."""
        handler = self._create_handler_without_manager()
        request = self._create_request(model="my-adapter", adapter=None)

        result = handler._resolve_effective_adapter(request)

        assert result is None

    def test_explicit_adapter_without_manager(self) -> None:
        """Explicit adapter field works even without adapter manager."""
        handler = self._create_handler_without_manager()
        request = self._create_request(model=None, adapter="my-adapter")

        result = handler._resolve_effective_adapter(request)

        assert result == "my-adapter"
