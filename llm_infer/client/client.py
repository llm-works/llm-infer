"""Unified multi-backend LLM client.

This module provides the main LLMClient facade that unifies access to
different LLM backends with a consistent API.

Example:
    # Quick start with OpenAI-compatible server
    with LLMClient.openai(base_url="http://localhost:8000/v1") as client:
        response = client.chat([{"role": "user", "content": "Hello"}])
        print(response)

    # Async streaming with Anthropic
    async with LLMClient.anthropic() as client:
        async for token in client.chat_stream_async(messages):
            print(token, end="")

    # From YAML configuration
    config = {
        "default": "local",
        "backends": {
            "local": {
                "type": "openai_compatible",
                "base_url": "http://localhost:8000/v1",
            },
            "anthropic": {
                "type": "anthropic",
                "model": "claude-sonnet-4-20250514",
            },
        },
    }
    client = LLMClient.from_config(config)
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Self

from llm_infer.client.backends import Backend, OpenAICompatibleBackend, create_backend
from llm_infer.client.types import ChatResponse


class LLMClient:
    """Unified multi-backend LLM client with sync/async support.

    This client provides a consistent interface across different LLM backends
    (OpenAI-compatible, Anthropic, etc.) with support for both synchronous
    and asynchronous operations.

    The client delegates all operations to an underlying Backend instance,
    which handles the actual API communication.

    Attributes:
        backend: The underlying backend instance.

    Resource Management:
        Use context managers for proper resource cleanup:

        # Sync
        with LLMClient.openai() as client:
            ...

        # Async
        async with LLMClient.anthropic() as client:
            ...
    """

    def __init__(
        self,
        backend: Backend,
        default_model: str | None = None,
    ) -> None:
        """Initialize the client with a backend.

        Args:
            backend: The backend to use for API calls.
            default_model: Default model to use if not specified per-request.
        """
        self._backend = backend
        self._default_model = default_model

    @property
    def backend(self) -> Backend:
        """The underlying backend instance."""
        return self._backend

    @property
    def last_response(self) -> ChatResponse | None:
        """Last response with usage stats.

        This is populated after both streaming and non-streaming requests,
        providing access to usage statistics and metadata.
        """
        return self._backend.last_response

    # =========================================================================
    # Factory methods
    # =========================================================================

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LLMClient:
        """Create from YAML/JSON config dict.

        Supports multi-backend configuration with a default backend.

        Config format:
            default: backend_name  # Which backend to use by default
            backends:
              backend_name:
                type: openai_compatible
                base_url: http://localhost:8000/v1
                model: qwen2.5-72b
              another_backend:
                type: anthropic
                model: claude-sonnet-4-20250514

        Args:
            config: Configuration dictionary.

        Returns:
            Configured LLMClient instance.

        Raises:
            ValueError: If configuration is invalid.
        """
        backends_config = config.get("backends", {})
        default_name = config.get("default")

        if not backends_config:
            # Single backend config (no "backends" wrapper)
            return cls.from_backend_config(config)

        if not default_name:
            # Use first backend as default
            default_name = next(iter(backends_config.keys()))

        if default_name not in backends_config:
            raise ValueError(f"Default backend '{default_name}' not found in backends")

        backend_config = backends_config[default_name]
        backend = create_backend(backend_config)

        return cls(backend=backend, default_model=backend_config.get("model"))

    @classmethod
    def from_backend_config(cls, config: dict[str, Any]) -> LLMClient:
        """Create from single backend config.

        Args:
            config: Backend configuration with 'type' key.

        Returns:
            Configured LLMClient instance.
        """
        backend = create_backend(config)
        return cls(backend=backend, default_model=config.get("model"))

    @classmethod
    def openai(
        cls,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> LLMClient:
        """Create client for OpenAI-compatible API.

        Args:
            base_url: API base URL.
            model: Default model name.
            api_key: Optional API key.
            timeout: Request timeout in seconds.

        Returns:
            LLMClient configured for OpenAI-compatible API.
        """
        backend = OpenAICompatibleBackend(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )
        return cls(backend=backend, default_model=model)

    @classmethod
    def anthropic(
        cls,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> LLMClient:
        """Create client for Anthropic Claude API.

        Requires: pip install llm-infer[anthropic]

        Args:
            model: Claude model name.
            api_key: Anthropic API key (uses env var if not provided).
            max_tokens: Default max tokens for responses.
            timeout: Request timeout in seconds.

        Returns:
            LLMClient configured for Anthropic API.

        Raises:
            ImportError: If anthropic package is not installed.
        """
        from llm_infer.client.backends.anthropic import AnthropicBackend

        backend = AnthropicBackend(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return cls(backend=backend, default_model=model)

    # =========================================================================
    # Sync API
    # =========================================================================

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request and return content (sync).

        This is the simple API that returns just the generated text.
        For full response with usage stats, use chat_full().

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Returns:
            Generated text content.
        """
        response = self.chat_full(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        return response.content

    def chat_full(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request and return full response (sync).

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, thinking, tool_calls, etc.
        """
        return self._backend.chat(
            messages=messages,
            model=model or self._default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream chat completion tokens (sync).

        Yields tokens as they arrive. After iteration, access last_response
        for usage statistics.

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Yields:
            String tokens as they arrive.
        """
        yield from self._backend.chat_stream(
            messages=messages,
            model=model or self._default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    # =========================================================================
    # Async API
    # =========================================================================

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request and return content (async).

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Returns:
            Generated text content.
        """
        response = await self.chat_full_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        return response.content

    async def chat_full_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request and return full response (async).

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, thinking, tool_calls, etc.
        """
        return await self._backend.chat_async(
            messages=messages,
            model=model or self._default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    async def chat_stream_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens (async).

        Yields tokens as they arrive. After iteration, access last_response
        for usage statistics.

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Yields:
            String tokens as they arrive.
        """
        async for token in self._backend.chat_stream_async(
            messages=messages,
            model=model or self._default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        ):
            yield token

    # =========================================================================
    # Resource management
    # =========================================================================

    def close(self) -> None:
        """Close sync resources."""
        self._backend.close()

    async def aclose(self) -> None:
        """Close all resources (sync and async)."""
        await self._backend.aclose()

    def __enter__(self) -> Self:
        """Enter sync context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit sync context manager."""
        self.close()

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        await self.aclose()
