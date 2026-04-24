"""Model and backend resolution for LLMRouter.

Handles model-to-backend mapping, lazy discovery, and "auto"/"default"
reserved model name resolution.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from appinfra.log import Logger

from .errors import BackendUnavailableError

if TYPE_CHECKING:
    from .client import LLMClient
    from .discovery import ModelDiscovery


class ModelResolver:
    """Resolves models to backends and handles reserved model names.

    Supports:
    - Static model-to-backend mapping
    - Lazy discovery via ModelDiscovery
    - Reserved names: "auto" (probe backend), "default" (use configured default)
    """

    def __init__(
        self,
        lg: Logger,
        model_to_backend: dict[str, str],
        default: str,
        discovery: ModelDiscovery | None = None,
    ) -> None:
        """Initialize resolver.

        Args:
            lg: Logger instance.
            model_to_backend: Mutable mapping from model ID to backend name.
            default: Default backend name.
            discovery: Optional ModelDiscovery for lazy discovery.
        """
        self._lg = lg
        self._model_to_backend = model_to_backend
        self._default = default
        self._discovery = discovery

    @property
    def models(self) -> dict[str, str]:
        """Current model-to-backend mapping."""
        return self._model_to_backend

    def resolve_backend(self, model: str) -> str:
        """Resolve backend for a model, using lazy discovery if needed.

        Args:
            model: Model ID to look up.

        Returns:
            Backend name (falls back to default if not found).
        """
        if model in self._model_to_backend:
            return self._model_to_backend[model]

        if self._discovery is not None:
            found = self._discovery.get_backend_for_model(model)
            if found is not None:
                self._model_to_backend = self._discovery.models
                return found

        return self._default

    def resolve_model(
        self, client: LLMClient, model: str | None, *, retry: bool = True
    ) -> str | None:
        """Resolve model name, handling reserved names.

        Args:
            client: Target client to resolve model for.
            model: Model name (may be reserved like "auto" or "default").
            retry: If True and client has backoff, retry on backend failure.

        Returns:
            Resolved model name, or None if no model configured.
        """
        if model is None:
            return client.default_model

        if model == "default":
            default = client.default_model
            if default == "auto" or default is None:
                return self._resolve_auto_model(client, retry=retry)
            return default

        if model == "auto":
            return self._resolve_auto_model(client, retry=retry)

        return model

    def _list_models_with_retry(
        self, client: LLMClient, *, retry: bool = True
    ) -> list[str]:
        """List models from backend, retrying if backoff is configured."""
        backoff = client.backoff
        timeout = client.timeout
        start_time = time.time()

        while True:
            try:
                models = client.backend.list_models()
                if backoff is not None:
                    backoff.reset()
                return models
            except BackendUnavailableError as e:
                if backoff is None or not retry:
                    raise

                elapsed = time.time() - start_time
                if timeout > 0 and elapsed >= timeout:
                    self._lg.error(
                        "model discovery timed out",
                        extra={"error": str(e), "elapsed": elapsed},
                    )
                    raise

                delay = backoff.next_delay()
                self._lg.warning(
                    "backend unavailable for model discovery, retrying",
                    extra={"error": str(e), "delay": delay, "elapsed": elapsed},
                )
                time.sleep(delay)

    def _resolve_auto_model(
        self, client: LLMClient, *, retry: bool = True
    ) -> str | None:
        """Resolve "auto" to an actual model by probing the backend.

        Resolution order:
            1. If only one model available, use it
            2. If backend has configured default_model (non-auto), use it
            3. Use first model from list_models()
        """
        try:
            models = self._list_models_with_retry(client, retry=retry)
        except BackendUnavailableError as e:
            self._lg.warning(
                "failed to discover models for auto resolution",
                extra={"error": str(e)},
            )
            return client.default_model if client.default_model != "auto" else None
        except Exception as e:
            self._lg.warning(
                "failed to discover models for auto resolution",
                extra={"exception": e},
            )
            return client.default_model if client.default_model != "auto" else None

        if not models:
            return None

        if len(models) == 1:
            return models[0]

        default = client.default_model
        if default and default != "auto" and default in models:
            return default

        return models[0]
