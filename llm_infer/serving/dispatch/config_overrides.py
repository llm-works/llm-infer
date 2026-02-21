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
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from .config import InferenceConfig


def parse_override_args(overrides: list[str] | None) -> dict[str, str] | None:
    """Parse KEY=VALUE override arguments into a dict.

    Args:
        overrides: List of "KEY=VALUE" strings from CLI (e.g., -o flag).

    Returns:
        Dict mapping keys to values, or None if no overrides.

    Raises:
        ValueError: If any override is malformed (missing '=' or empty key).
    """
    if not overrides:
        return None
    result = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item!r} (expected KEY=VALUE)")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override format: {item!r} (key cannot be empty)")
        result[key] = value.strip()
    return result


def _get_nested_target(obj: Any, path: str) -> tuple[Any, str]:
    """Get the target object and final attribute name for a dotted path.

    Args:
        obj: Root object to traverse.
        path: Dot-separated path (e.g., "engines.vllm.gpu_memory_utilization").

    Returns:
        Tuple of (target_object, final_attr_name).

    Raises:
        ValueError: If any part of the path doesn't exist.
    """
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"Invalid config path: {path!r} (no attribute {part!r})")
        target = getattr(target, part)
    if not hasattr(target, parts[-1]):
        raise ValueError(f"Invalid config path: {path!r} (no attribute {parts[-1]!r})")
    return target, parts[-1]


def _get_field_type(obj: Any, attr: str) -> type | None:
    """Get the type annotation for a field on an object.

    Args:
        obj: Object to inspect.
        attr: Attribute name.

    Returns:
        The type annotation, or None if not found.
    """
    try:
        hints = get_type_hints(type(obj))
        return hints.get(attr)
    except (TypeError, NameError, AttributeError):
        # TypeError: object has no __annotations__
        # NameError: forward reference cannot be resolved
        # AttributeError: type object has no attribute
        return None


def _normalize_type(field_type: type | None) -> tuple[type | None, bool]:
    """Normalize a type annotation to its base type.

    Handles Optional[T] (T | None) by extracting T and noting nullability.
    Supports both modern syntax (int | None) and typing module (Optional[int]).

    Args:
        field_type: The type annotation to normalize.

    Returns:
        Tuple of (base_type, is_nullable).
    """
    if field_type is None:
        return None, True

    origin = get_origin(field_type)

    # Handle Union types: both modern (int | None) and typing.Union/Optional
    if origin is type(int | str) or origin is Union:
        args = get_args(field_type)
        non_none_args = [a for a in args if a is not type(None)]
        is_nullable = type(None) in args
        if len(non_none_args) == 1:
            return non_none_args[0], is_nullable
        # Multiple non-None types - can't determine single base type
        return None, is_nullable

    return field_type, False


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute using dot notation.

    Args:
        obj: Root object to traverse.
        path: Dot-separated path (e.g., "engines.vllm.gpu_memory_utilization").
        value: Value to set.

    Raises:
        ValueError: If any part of the path doesn't exist on the object.
    """
    target, attr = _get_nested_target(obj, path)
    setattr(target, attr, value)


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
                _set_nested_attr(config, config_path, converted)


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
        """Apply generic dotted-path overrides with type-aware conversion."""
        for key, value in overrides.items():
            # Get target and expected type
            target, attr = _get_nested_target(config, key)
            field_type = _get_field_type(target, attr)
            base_type, is_nullable = _normalize_type(field_type)

            # Convert with type awareness
            converted = self._convert_value_typed(value, base_type, is_nullable, key)
            setattr(target, attr, converted)

    def _convert_value_typed(
        self, value: str, expected_type: type | None, is_nullable: bool, path: str
    ) -> Any:
        """Convert string value to the expected type.

        Args:
            value: String value to convert.
            expected_type: Expected type (or None if unknown).
            is_nullable: Whether None is allowed.
            path: Config path (for error messages).

        Returns:
            Converted value.

        Raises:
            ValueError: If conversion fails or type doesn't match.
        """
        # Handle null/none
        if value.lower() in ("null", "none"):
            if not is_nullable and expected_type is not None:
                raise ValueError(
                    f"Invalid value for {path}: 'null' not allowed "
                    f"(field type is {expected_type.__name__}, not Optional)"
                )
            return None

        # If we know the expected type, convert directly to it
        if expected_type is not None:
            return self._convert_to_type(value, expected_type, path)

        # Fall back to inference if type is unknown
        return self._convert_value_inferred(value)

    def _convert_to_type(self, value: str, expected_type: type, path: str) -> Any:
        """Convert string value to a specific type.

        Args:
            value: String value to convert.
            expected_type: Target type.
            path: Config path (for error messages).

        Returns:
            Converted value.

        Raises:
            ValueError: If conversion fails.
        """
        try:
            if expected_type is bool:
                # Special handling for booleans (can't just call bool())
                if value.lower() in ("true", "yes", "1"):
                    return True
                if value.lower() in ("false", "no", "0"):
                    return False
                raise ValueError("expected boolean (true/false/yes/no)")

            if expected_type is int:
                # Handle scientific notation for ints
                f = float(value)
                if not f.is_integer():
                    raise ValueError(f"expected integer, got {value!r}")
                return int(f)

            if expected_type is float:
                return float(value)

            if expected_type is str:
                return value

            # For other types, try direct conversion
            return expected_type(value)

        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid value for {path}: {value!r} cannot be converted to "
                f"{expected_type.__name__} ({e})"
            ) from e

    def _convert_value_inferred(self, value: str) -> Any:
        """Convert string value using type inference (fallback).

        Used when the target field's type annotation is unknown.
        """
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
        # Handle floats (convert whole numbers like 1e3 to int)
        try:
            f = float(value)
            if f.is_integer():
                return int(f)
            return f
        except ValueError:
            pass
        # Default to string
        return value


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
