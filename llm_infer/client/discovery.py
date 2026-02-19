"""Lazy model discovery for multi-backend routing.

ModelDiscovery handles probing backends for available models on-demand,
avoiding startup errors when backends are configured but not running.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from .exceptions import ModelConflictError

if TYPE_CHECKING:
    from .client import LLMClient


class ModelDiscovery:
    """Lazy model discovery from backends.

    Builds a model-to-backend routing table by:
    1. Using config-specified `models` lists immediately (no probing)
    2. Probing backends lazily when a model is requested that isn't in the table
       (only if lazy_probe=True)

    This avoids startup errors when backends are configured but not running.
    Only backends that are actually used get probed.

    Example:
        discovery = ModelDiscovery(lg, clients, configs)

        # Get backend for a model (probes lazily if needed)
        backend = discovery.get_backend_for_model("gpt-4")

        # Explicitly probe a backend
        discovery.discover_backend("openai")

        # Check current routing table
        print(discovery.models)
    """

    def __init__(
        self,
        lg: Logger,
        clients: dict[str, LLMClient],
        configs: dict[str, dict[str, Any]],
        lazy_probe: bool = True,
    ) -> None:
        """Initialize model discovery.

        Args:
            lg: Logger instance.
            clients: Backend name to LLMClient mapping.
            configs: Backend name to config mapping.
            lazy_probe: If True, probe backends lazily when models are requested.
                If False, only use config-specified models (no backend probing).
        """
        self._lg = lg
        self._clients = clients
        self._configs = configs
        self._lazy_probe = lazy_probe
        self._model_to_backend: dict[str, str] = {}
        self._discovered_backends: set[str] = set()

        # Pre-populate from config (no probing)
        self._load_from_config()

    def _load_from_config(self) -> None:
        """Load model mappings from config-specified models lists.

        Raises:
            ModelConflictError: If the same model appears in multiple backend configs.
        """
        for name, config in self._configs.items():
            config_models: list[str] = config.get("models", [])
            for model in config_models:
                if model in self._model_to_backend:
                    existing = self._model_to_backend[model]
                    raise ModelConflictError(model, existing, name)
                self._model_to_backend[model] = name

    @property
    def models(self) -> dict[str, str]:
        """Current model-to-backend mapping (read-only copy)."""
        return dict(self._model_to_backend)

    @property
    def discovered_backends(self) -> set[str]:
        """Set of backends that have been probed (read-only copy)."""
        return set(self._discovered_backends)

    def get_backend_for_model(self, model: str) -> str | None:
        """Get backend for a model, discovering lazily if needed.

        Args:
            model: Model ID to look up.

        Returns:
            Backend name if found, None otherwise.
        """
        # Check existing routing table
        if model in self._model_to_backend:
            return self._model_to_backend[model]

        # Try lazy discovery of unprobed backends (if enabled)
        if self._lazy_probe:
            return self._discover_model(model)

        return None

    def _discover_model(self, model: str) -> str | None:
        """Probe unprobed backends to find a model.

        Args:
            model: Model ID to search for.

        Returns:
            Backend name if found, None otherwise.
        """
        for name in self._clients:
            if name in self._discovered_backends:
                continue

            self.discover_backend(name)

            if model in self._model_to_backend:
                return self._model_to_backend[model]

        return None

    def discover_backend(self, name: str) -> set[str]:
        """Probe a specific backend for its models.

        Args:
            name: Backend name to probe.

        Returns:
            Set of discovered model IDs.

        Raises:
            ValueError: If backend name is not known.
        """
        if name not in self._clients:
            raise ValueError(f"Unknown backend: {name}")

        if name in self._discovered_backends:
            return {m for m, b in self._model_to_backend.items() if b == name}

        self._discovered_backends.add(name)
        models = self._probe_backend(name)
        self._add_models_to_routing(name, models)
        return models

    def _probe_backend(self, name: str) -> set[str]:
        """Query backend for available models."""
        client = self._clients[name]
        try:
            models = set(client.backend.list_models())
            self._lg.debug(
                f"discovered models from backend '{name}'",
                extra={"models": sorted(models), "count": len(models)},
            )
            return models
        except Exception as e:
            self._lg.warning(
                f"failed to discover models from backend '{name}'",
                extra={"exception": e},
            )
            return set()

    def _add_models_to_routing(self, name: str, models: set[str]) -> None:
        """Add discovered models to routing table, skipping conflicts."""
        for model in models:
            if model in self._model_to_backend:
                existing = self._model_to_backend[model]
                if existing != name:
                    self._lg.debug(
                        f"model '{model}' already routed to '{existing}', "
                        f"ignoring discovery from '{name}'"
                    )
                continue
            self._model_to_backend[model] = name

    def discover_all(self) -> dict[str, str]:
        """Probe all backends for models.

        Returns:
            Complete model-to-backend mapping after discovery.
        """
        for name in self._clients:
            self.discover_backend(name)
        return self.models
