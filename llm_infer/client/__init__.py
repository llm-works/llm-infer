"""Multi-backend LLM client with sync/async support.

This package provides a unified client for interacting with different LLM
backends (OpenAI-compatible, Anthropic) with support for both synchronous
and asynchronous operations.

Quick Start:
    from appinfra.log import Logger
    from llm_infer.client import Factory

    lg = Logger("my-app")
    factory = Factory(lg)

    # Single backend - returns LLMClient
    with factory.openai(base_url="http://localhost:8000/v1") as client:
        response = client.chat([{"role": "user", "content": "Hello"}])
        print(response)

    # Multi-backend config - returns LLMRouter
    config = {
        "default": "local",
        "backends": {
            "local": {"type": "openai_compatible", "base_url": "http://localhost:8000/v1"},
            "openai": {"type": "openai", "base_url": "https://api.openai.com/v1"},
        },
    }
    with factory.from_config(config) as router:
        response = router.chat(messages)                    # Uses default
        response = router.chat(messages, backend="openai")  # Routes to OpenAI

    # Async with Anthropic
    async with factory.anthropic() as client:
        async for token in client.chat_stream_async(messages):
            print(token, end="")

    # With llm-infer extensions
    with factory.openai() as client:
        response = client.chat_full(
            messages=[{"role": "user", "content": "Think about this"}],
            think=True,
            adapter_id="my-lora",
        )
        print(response.content)
        print(response.thinking)  # Separated thinking content

Classes:
    - LLMClient: Single-backend client
    - LLMRouter: Multi-backend router with backend selection
    - Factory: Creates clients and routers from config

Backends:
    - OpenAICompatibleBackend: Works with OpenAI, llm-infer, vLLM, Ollama
    - AnthropicBackend: Anthropic Claude API (requires: pip install llm-infer[anthropic])

llm-infer Extensions:
    - adapter_id: LoRA adapter selection for vLLM
    - think: Thinking mode with <think> block extraction
    - tools/tool_choice: Function calling support
"""

from llm_infer.client.backends import Backend, OpenAICompatibleBackend
from llm_infer.client.client import LLMClient
from llm_infer.client.exceptions import (
    BackendError,
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from llm_infer.client.factory import Factory
from llm_infer.client.router import LLMRouter
from llm_infer.client.types import ChatResponse

__all__ = [
    # Factory (primary entry point)
    "Factory",
    # Client facade
    "LLMClient",
    # Router (multi-backend)
    "LLMRouter",
    # Response types
    "ChatResponse",
    # Backend base class
    "Backend",
    "OpenAICompatibleBackend",
    # Exceptions
    "BackendError",
    "BackendRequestError",
    "BackendTimeoutError",
    "BackendUnavailableError",
]

# Optional SAIA integration (requires: pip install llm-infer[saia])
try:
    from llm_infer.client.saia import SAIAAdapter  # noqa: F401

    __all__.append("SAIAAdapter")
except ImportError:
    pass
