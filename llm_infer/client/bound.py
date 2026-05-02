"""BoundChatClient - ChatClient wrapper with pre-bound kwargs.

Example:
    router = Factory(lg).from_config(config)
    exploration = BoundChatClient(router, role="exploration")
    synthesis = BoundChatClient(router, role="synthesis")

    # Both use the same router, different roles
    exploration.chat(messages)  # role="exploration" merged
    synthesis.chat(messages)    # role="synthesis" merged
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Self

from .base import ChatClient
from .types import ChatResponse, ChatStream, ChatStreamSync


class _Unset(Enum):
    """Sentinel for detecting unset parameters in BoundChatClient."""

    UNSET = auto()


_UNSET = _Unset.UNSET


class BoundChatClient(ChatClient):
    """ChatClient wrapper that binds kwargs to every call.

    Creates a view of a ChatClient with pre-bound arguments that are merged
    into every chat call. Useful for binding routing parameters (role, backend)
    without modifying the underlying client.

    The bound client delegates resource management to the wrapped client.
    Closing the bound client closes the underlying client.
    """

    def __init__(self, client: ChatClient, **kwargs: Any) -> None:
        """Create a bound view of a ChatClient.

        Args:
            client: The ChatClient to wrap.
            **kwargs: Arguments to merge into every chat call.
        """
        self._client = client
        self._bound_kwargs = kwargs

    @property
    def client(self) -> ChatClient:
        """The underlying ChatClient."""
        return self._client

    @property
    def bound_kwargs(self) -> dict[str, Any]:
        """The bound kwargs (read-only copy)."""
        return dict(self._bound_kwargs)

    def with_chat_args(self, **kwargs: Any) -> BoundChatClient:
        """Create a new BoundChatClient with additional bound kwargs.

        Args:
            **kwargs: Additional arguments to merge.

        Returns:
            New BoundChatClient with merged kwargs.
        """
        merged = {**self._bound_kwargs, **kwargs}
        return BoundChatClient(self._client, **merged)

    def can_call(self) -> bool:
        """Check if a call is allowed (delegates to wrapped client)."""
        return self._client.can_call()

    def _merge_kwargs(
        self,
        messages: list[dict[str, Any]],
        model: str | None | _Unset,
        system: str | None | _Unset,
        temperature: float | _Unset,
        max_tokens: int | None | _Unset,
        tools: list[dict[str, Any]] | None | _Unset,
        tool_choice: str | dict[str, Any] | None | _Unset,
        think: bool | None | _Unset,
        adapter: str | None | _Unset,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Merge bound kwargs with call-time kwargs."""
        call_kwargs: dict[str, Any] = {**self._bound_kwargs, "messages": messages}
        for name, value in [
            ("model", model),
            ("system", system),
            ("temperature", temperature),
            ("max_tokens", max_tokens),
            ("tools", tools),
            ("tool_choice", tool_choice),
            ("think", think),
            ("adapter", adapter),
        ]:
            if value is not _UNSET:
                call_kwargs[name] = value
        call_kwargs.update(kwargs)
        return call_kwargs

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None | _Unset = _UNSET,
        system: str | None | _Unset = _UNSET,
        temperature: float | _Unset = _UNSET,
        max_tokens: int | None | _Unset = _UNSET,
        tools: list[dict[str, Any]] | None | _Unset = _UNSET,
        tool_choice: str | dict[str, Any] | None | _Unset = _UNSET,
        think: bool | None | _Unset = _UNSET,
        adapter: str | None | _Unset = _UNSET,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request with bound kwargs merged."""
        call_kwargs = self._merge_kwargs(
            messages,
            model,
            system,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            think,
            adapter,
            **kwargs,
        )
        return self._client.chat(**call_kwargs)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None | _Unset = _UNSET,
        system: str | None | _Unset = _UNSET,
        temperature: float | _Unset = _UNSET,
        max_tokens: int | None | _Unset = _UNSET,
        tools: list[dict[str, Any]] | None | _Unset = _UNSET,
        tool_choice: str | dict[str, Any] | None | _Unset = _UNSET,
        think: bool | None | _Unset = _UNSET,
        adapter: str | None | _Unset = _UNSET,
        **kwargs: Any,
    ) -> ChatStreamSync:
        """Stream chat completion tokens with bound kwargs merged."""
        call_kwargs = self._merge_kwargs(
            messages,
            model,
            system,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            think,
            adapter,
            **kwargs,
        )
        return self._client.chat_stream(**call_kwargs)

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None | _Unset = _UNSET,
        system: str | None | _Unset = _UNSET,
        temperature: float | _Unset = _UNSET,
        max_tokens: int | None | _Unset = _UNSET,
        tools: list[dict[str, Any]] | None | _Unset = _UNSET,
        tool_choice: str | dict[str, Any] | None | _Unset = _UNSET,
        think: bool | None | _Unset = _UNSET,
        adapter: str | None | _Unset = _UNSET,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request (async) with bound kwargs merged."""
        call_kwargs = self._merge_kwargs(
            messages,
            model,
            system,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            think,
            adapter,
            **kwargs,
        )
        return await self._client.chat_async(**call_kwargs)

    def chat_stream_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None | _Unset = _UNSET,
        system: str | None | _Unset = _UNSET,
        temperature: float | _Unset = _UNSET,
        max_tokens: int | None | _Unset = _UNSET,
        tools: list[dict[str, Any]] | None | _Unset = _UNSET,
        tool_choice: str | dict[str, Any] | None | _Unset = _UNSET,
        think: bool | None | _Unset = _UNSET,
        adapter: str | None | _Unset = _UNSET,
        **kwargs: Any,
    ) -> ChatStream:
        """Stream chat completion tokens (async) with bound kwargs merged."""
        call_kwargs = self._merge_kwargs(
            messages,
            model,
            system,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            think,
            adapter,
            **kwargs,
        )
        return self._client.chat_stream_async(**call_kwargs)

    def close(self) -> None:
        """Close the wrapped client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the wrapped client (async)."""
        await self._client.aclose()

    def __enter__(self) -> Self:
        """Enter sync context manager."""
        self._client.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit sync context manager."""
        self._client.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        await self._client.__aexit__(exc_type, exc_val, exc_tb)
