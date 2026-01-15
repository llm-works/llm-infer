"""Configuration override strategies.

Implements Strategy pattern for applying configuration overrides from
different sources (environment variables, CLI arguments, etc.) with
explicit precedence ordering.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import InferenceConfig


class ConfigOverride(ABC):
    """Abstract base for configuration override strategies."""

    @abstractmethod
    def apply(self, config: InferenceConfig) -> None:
        """Apply overrides to the configuration in-place."""
        pass


class EnvConfigOverride(ConfigOverride):
    """Apply overrides from environment variables.

    Environment variable mappings:
    - NUM_BLOCKS -> engines.native.num_blocks
    - BLOCK_SIZE -> engines.native.block_size
    - MAX_BATCH_SIZE -> engines.native.max_batch_size
    - MAX_PENDING -> dispatch.max_pending
    - HANDLER -> dispatch.handler
    - HOST -> api.host
    - PORT -> api.port
    """

    # Mapping: (env_var, config_path, type_converter)
    MAPPINGS: list[tuple[str, str, type]] = [
        ("NUM_BLOCKS", "engines.native.num_blocks", int),
        ("BLOCK_SIZE", "engines.native.block_size", int),
        ("MAX_BATCH_SIZE", "engines.native.max_batch_size", int),
        ("MAX_PENDING", "dispatch.max_pending", int),
        ("HANDLER", "dispatch.handler", str),
        ("HOST", "api.host", str),
        ("PORT", "api.port", int),
    ]

    def apply(self, config: InferenceConfig) -> None:
        """Apply environment variable overrides."""
        for env_var, config_path, type_conv in self.MAPPINGS:
            if env_val := os.environ.get(env_var):
                try:
                    converted = type_conv(env_val)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid value for {env_var}={env_val!r} "
                        f"(expected {type_conv.__name__} for {config_path}): {e}"
                    ) from e
                self._set_nested(config, config_path, converted)

    def _set_nested(self, config: InferenceConfig, path: str, value: Any) -> None:
        """Set a nested attribute using dot notation."""
        parts = path.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


@dataclass
class CliOverrides:
    """Container for CLI override values."""

    host: str | None = None
    port: int | None = None
    handler: str | None = None
    log_file: str | None = None
    model_path: str | None = None
    engine: str | None = None
    generic: dict[str, str] | None = None  # key=value overrides via -o flag


class CliConfigOverride(ConfigOverride):
    """Apply overrides from CLI arguments.

    CLI arguments take precedence over environment variables.
    """

    def __init__(self, overrides: CliOverrides) -> None:
        self.overrides = overrides

    def apply(self, config: InferenceConfig) -> None:
        """Apply CLI argument overrides."""
        if self.overrides.host is not None:
            config.api.host = self.overrides.host
        if self.overrides.port is not None:
            config.api.port = self.overrides.port
        if self.overrides.handler is not None:
            config.dispatch.handler = self.overrides.handler
        if self.overrides.log_file is not None:
            config.api.log_file = self.overrides.log_file
        if self.overrides.model_path is not None:
            config.models.path = Path(self.overrides.model_path)
        if self.overrides.engine is not None:
            config.backends.engine = self.overrides.engine
        # Apply generic key=value overrides last (highest priority)
        if self.overrides.generic:
            self._apply_generic(config, self.overrides.generic)

    def _apply_generic(
        self, config: InferenceConfig, overrides: dict[str, str]
    ) -> None:
        """Apply generic dotted-path overrides with type inference."""
        for key, value in overrides.items():
            converted = self._convert_value(value)
            self._set_nested(config, key, converted)

    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Handle null/none
        if value.lower() in ("null", "none"):
            return None
        # Handle booleans
        if value.lower() in ("true", "yes"):
            return True
        if value.lower() in ("false", "no"):
            return False
        # Handle integers
        try:
            return int(value)
        except ValueError:
            pass
        # Handle floats
        try:
            return float(value)
        except ValueError:
            pass
        # Default to string
        return value

    def _set_nested(self, config: InferenceConfig, path: str, value: Any) -> None:
        """Set a nested attribute using dot notation."""
        parts = path.split(".")
        obj: Any = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


class ConfigOverrideChain:
    """Chain of config overrides applied in order.

    Default precedence (applied in order, later overrides earlier):
    1. Environment variables
    2. CLI arguments
    """

    def __init__(self, overrides: list[ConfigOverride] | None = None) -> None:
        self._overrides = overrides or []

    def add(self, override: ConfigOverride) -> ConfigOverrideChain:
        """Add an override to the chain."""
        self._overrides.append(override)
        return self

    def apply_all(self, config: InferenceConfig) -> InferenceConfig:
        """Apply all overrides in order."""
        for override in self._overrides:
            override.apply(config)
        return config


def apply_standard_overrides(
    config: InferenceConfig,
    cli_overrides: CliOverrides | None = None,
) -> InferenceConfig:
    """Apply standard override chain: env -> CLI.

    Args:
        config: The base configuration to modify.
        cli_overrides: Optional CLI argument overrides.

    Returns:
        The modified configuration.
    """
    chain = ConfigOverrideChain().add(EnvConfigOverride())
    if cli_overrides:
        chain.add(CliConfigOverride(cli_overrides))
    return chain.apply_all(config)
