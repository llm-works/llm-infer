"""Model discovery and resolution for multi-backend routing.

ModelDiscovery handles:
1. Model→backend routing (which backend serves which model)
2. Model name resolution ("auto"/"default" → actual model)
3. Lazy probing of backends for available models
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from appinfra.dot_dict import DotDict
from appinfra.log import Logger

from .errors import ModelConflictError

if TYPE_CHECKING:
    from .backends import Backend

# Reserved model names that require resolution
# - "auto": Probe backend via list_models(), pick first available
# - "default": Use configured default_model for this backend
RESERVED_MODEL_NAMES = frozenset({"auto", "default"})


class ModelDiscovery:
    """Model discovery and resolution for multi-backend routing.

    Handles:
    - Model→backend routing table (from config or lazy discovery)
    - "auto"/"default" model name resolution
    - Lazy probing via get_models_for_backend()

    Example:
        discovery = ModelDiscovery(lg, backends, configs)

        # Routing: which backend serves "gpt-4"?
        backend = discovery.get_backend_for_model("gpt-4")

        # Resolution: what model to use for "auto" on backend "local"?
        model = discovery.resolve_model("local", "auto", default_model=None)
    """

    def __init__(
        self,
        lg: Logger,
        backends: dict[str, Backend],
        configs: dict[str, DotDict],
    ) -> None:
        """Initialize model discovery.

        Args:
            lg: Logger instance.
            backends: Backend name to Backend instance mapping.
            configs: Backend name to config (DotDict) mapping.
        """
        self._lg = lg
        self._backends = backends
        self._configs = configs
        self._model_to_backend: dict[str, str] = {}
        self._backend_models: dict[str, list[str]] = {}
        self._discovered_backends: set[str] = set()
        self._auto_cache: dict[str, str | None] = {}

        # Pre-populate from config (no probing)
        self._load_from_config()

    def _load_from_config(self) -> None:
        """Load model mappings from config-specified models lists.

        Raises:
            ModelConflictError: If the same model appears in multiple backend configs.
        """
        for name, config in self._configs.items():
            config_models: list[str] = config.get("models", [])
            if config_models:
                self._backend_models[name] = list(config_models)
                self._discovered_backends.add(name)
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

    # =========================================================================
    # Routing: model → backend
    # =========================================================================

    def get_backend_for_model(self, model: str) -> str | None:
        """Get backend for a model from the routing table.

        Does NOT probe backends. Returns None for unknown models, letting
        the router fall back to the default backend.

        Args:
            model: Model ID to look up.

        Returns:
            Backend name if found in routing table, None otherwise.
        """
        return self._model_to_backend.get(model)

    # =========================================================================
    # Resolution: "auto"/"default" → actual model
    # =========================================================================

    def resolve_model(
        self,
        backend_name: str,
        model: str | None,
        default_model: str | None = None,
    ) -> str | None:
        """Resolve model name to actual model for a backend.

        Handles reserved names:
        - None or "default": Use default_model config
        - "auto": Probe backend, pick first available model

        Args:
            backend_name: Backend to resolve for.
            model: Model name from request (may be None or reserved).
            default_model: Configured default model for this backend.

        Returns:
            Resolved model name, or None if no model available.
        """
        # None or "default" → use configured default
        if model is None or model == "default":
            model = default_model

        # "auto" → probe and pick first
        if model == "auto":
            return self._resolve_auto(backend_name)

        return model

    def _resolve_auto(self, backend_name: str) -> str | None:
        """Resolve 'auto' by probing backend for available models.

        Caches the result to avoid repeated probing.

        Args:
            backend_name: Backend to probe.

        Returns:
            First available model, or None if no models found.
        """
        if backend_name in self._auto_cache:
            return self._auto_cache[backend_name]

        models = self.get_models_for_backend(backend_name)
        result = models[0] if models else None
        self._auto_cache[backend_name] = result

        if result:
            self._lg.debug(
                f"resolved 'auto' to '{result}' for backend '{backend_name}'",
                extra={"available_models": models},
            )
        else:
            self._lg.warning(
                f"no models found for backend '{backend_name}', 'auto' resolution failed"
            )

        return result

    # =========================================================================
    # Probing: lazy model discovery
    # =========================================================================

    def get_models_for_backend(self, name: str) -> list[str]:
        """Get models for a backend, probing lazily if needed.

        Args:
            name: Backend name.

        Returns:
            List of model IDs available on this backend.

        Raises:
            ValueError: If backend name is not known.
        """
        if name not in self._backends:
            raise ValueError(f"Unknown backend: {name}")

        if name not in self._discovered_backends:
            self._discover_backend(name)

        return list(self._backend_models.get(name, []))

    def _discover_backend(self, name: str) -> None:
        """Probe a backend for its models."""
        self._discovered_backends.add(name)
        backend = self._backends[name]
        try:
            models = backend.list_models()
            self._backend_models[name] = models
            self._lg.debug(
                f"discovered models from backend '{name}'",
                extra={"models": models, "count": len(models)},
            )
            self._add_models_to_routing(name, models)
        except Exception as e:
            self._lg.warning(
                f"failed to discover models from backend '{name}'",
                extra={"exception": e},
            )
            self._backend_models[name] = []

    def _add_models_to_routing(self, name: str, models: list[str]) -> None:
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
        for name in self._backends:
            if name not in self._discovered_backends:
                self._discover_backend(name)
        return self.models

    def clear_auto_cache(self, backend_name: str | None = None) -> None:
        """Clear cached auto resolution.

        Args:
            backend_name: Specific backend to clear, or None for all.
        """
        if backend_name is not None:
            self._auto_cache.pop(backend_name, None)
        else:
            self._auto_cache.clear()
