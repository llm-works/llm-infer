"""Unit tests for adapter management and versioned resolution."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_infer.serving.adapters import AdapterManager, LoadedAdapter, parse_adapter_key

pytestmark = pytest.mark.unit


class TestParseAdapterKey:
    """Test parse_adapter_key function."""

    def test_versioned_key(self) -> None:
        """Parse key with 12-char hex md5 suffix."""
        name, md5 = parse_adapter_key("jokester-p-sft-abc123def456")
        assert name == "jokester-p-sft"
        assert md5 == "abc123def456"

    def test_versioned_key_lowercase_hex(self) -> None:
        """MD5 suffix must be lowercase hex."""
        name, md5 = parse_adapter_key("adapter-0123456789ab")
        assert name == "adapter"
        assert md5 == "0123456789ab"

    def test_unversioned_key_simple(self) -> None:
        """Simple name without dashes."""
        name, md5 = parse_adapter_key("simple")
        assert name == "simple"
        assert md5 is None

    def test_unversioned_key_with_dashes(self) -> None:
        """Name with dashes but no md5 suffix."""
        name, md5 = parse_adapter_key("jokester-p-sft")
        assert name == "jokester-p-sft"
        assert md5 is None

    def test_suffix_too_short(self) -> None:
        """Suffix less than 12 chars is not a valid md5."""
        name, md5 = parse_adapter_key("adapter-abc123")
        assert name == "adapter-abc123"
        assert md5 is None

    def test_suffix_too_long(self) -> None:
        """Suffix more than 12 chars is not a valid md5."""
        name, md5 = parse_adapter_key("adapter-abc123def4567890")
        assert name == "adapter-abc123def4567890"
        assert md5 is None

    def test_suffix_uppercase_hex(self) -> None:
        """Uppercase hex is not matched (md5 should be lowercase)."""
        name, md5 = parse_adapter_key("adapter-ABC123DEF456")
        assert name == "adapter-ABC123DEF456"
        assert md5 is None

    def test_suffix_non_hex(self) -> None:
        """Non-hex characters in suffix position."""
        name, md5 = parse_adapter_key("adapter-ghijklmnopqr")
        assert name == "adapter-ghijklmnopqr"
        assert md5 is None


class TestLoadedAdapter:
    """Test LoadedAdapter dataclass."""

    def test_create_with_all_fields(self) -> None:
        """Create adapter with all fields."""
        adapter = LoadedAdapter(
            key="jokester-p-sft-abc123def456",
            name="jokester-p-sft",
            path=Path("/adapters/jokester-p-sft-abc123def456"),
            md5="abc123def456",
            mtime="2025-01-15T10:30:00+00:00",
            enabled=True,
            description="Test adapter",
        )
        assert adapter.key == "jokester-p-sft-abc123def456"
        assert adapter.name == "jokester-p-sft"
        assert adapter.md5 == "abc123def456"

    def test_name_equals_key_for_unversioned(self) -> None:
        """For unversioned adapters, name equals key."""
        adapter = LoadedAdapter(
            key="simple-adapter",
            name="simple-adapter",
            path=Path("/adapters/simple-adapter"),
        )
        assert adapter.name == adapter.key


class TestAdapterManagerVersions:
    """Test AdapterManager with versioned adapters."""

    def _create_manager(self, adapters: list[LoadedAdapter]) -> AdapterManager:
        """Create manager with pre-populated adapters for testing."""
        lg = MagicMock()
        manager = AdapterManager(lg, base_path=None)
        for adapter in adapters:
            manager._adapters[adapter.key] = adapter
        manager._rebuild_versions_index()
        return manager

    def test_resolve_exact_match(self) -> None:
        """Exact key match returns that adapter."""
        adapter = LoadedAdapter(
            key="jokester-p-sft-abc123def456",
            name="jokester-p-sft",
            path=Path("/adapters/jokester-p-sft-abc123def456"),
            mtime="2025-01-15T10:30:00+00:00",
        )
        manager = self._create_manager([adapter])

        result = manager.resolve("jokester-p-sft-abc123def456")

        assert result is adapter

    def test_resolve_by_name_single_version(self) -> None:
        """Name lookup returns only version."""
        adapter = LoadedAdapter(
            key="jokester-p-sft-abc123def456",
            name="jokester-p-sft",
            path=Path("/adapters/jokester-p-sft-abc123def456"),
            mtime="2025-01-15T10:30:00+00:00",
        )
        manager = self._create_manager([adapter])

        result = manager.resolve("jokester-p-sft")

        assert result is adapter

    def test_resolve_by_name_multiple_versions_returns_latest(self) -> None:
        """Name lookup with multiple versions returns highest mtime."""
        older = LoadedAdapter(
            key="jokester-p-sft-111111111111",
            name="jokester-p-sft",
            path=Path("/adapters/jokester-p-sft-111111111111"),
            mtime="2025-01-10T10:00:00+00:00",
        )
        newer = LoadedAdapter(
            key="jokester-p-sft-222222222222",
            name="jokester-p-sft",
            path=Path("/adapters/jokester-p-sft-222222222222"),
            mtime="2025-01-15T10:00:00+00:00",
        )
        manager = self._create_manager([older, newer])

        result = manager.resolve("jokester-p-sft")

        assert result is newer

    def test_resolve_not_found(self) -> None:
        """Unknown key returns None."""
        adapter = LoadedAdapter(
            key="existing-adapter",
            name="existing-adapter",
            path=Path("/adapters/existing-adapter"),
        )
        manager = self._create_manager([adapter])

        result = manager.resolve("unknown-adapter")

        assert result is None

    def test_is_available_full_key(self) -> None:
        """is_available works with full key."""
        adapter = LoadedAdapter(
            key="jokester-p-sft-abc123def456",
            name="jokester-p-sft",
            path=Path("/adapters/jokester-p-sft-abc123def456"),
        )
        manager = self._create_manager([adapter])

        assert manager.is_available("jokester-p-sft-abc123def456") is True
        assert manager.is_available("unknown") is False

    def test_is_available_by_name(self) -> None:
        """is_available works with name (base key)."""
        adapter = LoadedAdapter(
            key="jokester-p-sft-abc123def456",
            name="jokester-p-sft",
            path=Path("/adapters/jokester-p-sft-abc123def456"),
        )
        manager = self._create_manager([adapter])

        assert manager.is_available("jokester-p-sft") is True

    def test_resolve_path_by_name(self) -> None:
        """resolve_path works with name."""
        adapter = LoadedAdapter(
            key="jokester-p-sft-abc123def456",
            name="jokester-p-sft",
            path=Path("/adapters/jokester-p-sft-abc123def456"),
        )
        manager = self._create_manager([adapter])

        result = manager.resolve_path("jokester-p-sft")

        assert result == Path("/adapters/jokester-p-sft-abc123def456")

    def test_list_returns_all_versions(self) -> None:
        """list() returns all versions, not just latest."""
        v1 = LoadedAdapter(
            key="adapter-111111111111",
            name="adapter",
            path=Path("/adapters/adapter-111111111111"),
            mtime="2025-01-10T10:00:00+00:00",
        )
        v2 = LoadedAdapter(
            key="adapter-222222222222",
            name="adapter",
            path=Path("/adapters/adapter-222222222222"),
            mtime="2025-01-15T10:00:00+00:00",
        )
        manager = self._create_manager([v1, v2])

        result = manager.list()

        assert len(result) == 2
        assert v1 in result
        assert v2 in result

    def test_multiple_adapters_different_names(self) -> None:
        """Multiple adapters with different names resolve independently."""
        adapter_a = LoadedAdapter(
            key="adapter-a-111111111111",
            name="adapter-a",
            path=Path("/adapters/adapter-a-111111111111"),
        )
        adapter_b = LoadedAdapter(
            key="adapter-b-222222222222",
            name="adapter-b",
            path=Path("/adapters/adapter-b-222222222222"),
        )
        manager = self._create_manager([adapter_a, adapter_b])

        assert manager.resolve("adapter-a") is adapter_a
        assert manager.resolve("adapter-b") is adapter_b

    def test_unversioned_adapter_resolves_by_key(self) -> None:
        """Unversioned adapter (name == key) resolves correctly."""
        adapter = LoadedAdapter(
            key="simple-adapter",
            name="simple-adapter",
            path=Path("/adapters/simple-adapter"),
        )
        manager = self._create_manager([adapter])

        # Both should work since name == key
        assert manager.resolve("simple-adapter") is adapter

    def test_versions_index_sorted_by_mtime_desc(self) -> None:
        """Versions index is sorted by mtime descending (newest first)."""
        oldest = LoadedAdapter(
            key="adapter-aaaaaaaaaaaa",
            name="adapter",
            path=Path("/adapters/adapter-aaaaaaaaaaaa"),
            mtime="2025-01-01T10:00:00+00:00",
        )
        middle = LoadedAdapter(
            key="adapter-bbbbbbbbbbbb",
            name="adapter",
            path=Path("/adapters/adapter-bbbbbbbbbbbb"),
            mtime="2025-01-10T10:00:00+00:00",
        )
        newest = LoadedAdapter(
            key="adapter-cccccccccccc",
            name="adapter",
            path=Path("/adapters/adapter-cccccccccccc"),
            mtime="2025-01-20T10:00:00+00:00",
        )
        # Add in random order
        manager = self._create_manager([middle, newest, oldest])

        # First in versions list should be newest
        assert manager._versions["adapter"][0] == "adapter-cccccccccccc"
        assert manager._versions["adapter"][1] == "adapter-bbbbbbbbbbbb"
        assert manager._versions["adapter"][2] == "adapter-aaaaaaaaaaaa"

    def test_none_mtime_sorts_last(self) -> None:
        """Adapters with None mtime sort after those with mtime."""
        with_mtime = LoadedAdapter(
            key="adapter-111111111111",
            name="adapter",
            path=Path("/adapters/adapter-111111111111"),
            mtime="2025-01-10T10:00:00+00:00",
        )
        without_mtime = LoadedAdapter(
            key="adapter-222222222222",
            name="adapter",
            path=Path("/adapters/adapter-222222222222"),
            mtime=None,
        )
        manager = self._create_manager([without_mtime, with_mtime])

        # One with mtime should be first (newer)
        assert manager._versions["adapter"][0] == "adapter-111111111111"
        assert manager._versions["adapter"][1] == "adapter-222222222222"
