# Client Package

`llm_infer.client` is a Python client library for LLM inference with a unified interface across
backends. Built for autonomous agents and production use.

## Overview

**Core Features:**
- **Multiple backends** - OpenAI, Anthropic, and any OpenAI-compatible API
- **Sync, async, streaming** - All execution modes supported
- **Rate limiting** - Per-backend request throttling
- **Retry with backoff** - Configurable exponential backoff on failures
- **Model routing** - Route requests to backends by model name
- **Extensible** - Register custom backends via `Factory.register()`
- **Protocol extensions** - LoRA adapter selection, thinking mode

**Installation:**
- Client-only (lightweight): `pip install llm-infer`
- With Anthropic: `pip install llm-infer[anthropic]`

**Supported Backends:**
- **OpenAI-compatible**: OpenAI, llm-infer server, vLLM, Ollama
- **Anthropic**: Claude models

**Public API paths:**
```python
# Preferred: import from root package
from llm_infer import client
factory = client.Factory(lg)

# Or import specific classes directly
from llm_infer.client import Factory, LLMClient, ChatResponse, Backend

# Alternative path (deprecated, use llm_infer.client instead)
from llm_infer.api import Factory, LLMClient, ChatResponse, Backend
```

## Available Types

| Type | Kind | Description |
|------|------|-------------|
| `Factory` | Class | Factory for creating LLMClient instances |
| `LLMClient` | Class | Unified client facade |
| `Backend` | ABC | Abstract base class for backend implementations |
| `OpenAICompatibleBackend` | Class | Backend for OpenAI-compatible APIs |
| `ChatResponse` | Dataclass | Response with content, usage, thinking, tool_calls |
| `BackendError` | Exception | Base exception for all backend errors |
| `BackendUnavailableError` | Exception | Connection failed |
| `BackendTimeoutError` | Exception | Request timed out |
| `BackendRequestError` | Exception | HTTP error from backend |

## Quick Start

### Sync Usage (Recommended for Scripts)

```python
from appinfra.log import Logger
from llm_infer.client import Factory

lg = Logger("my-app")
factory = Factory(lg)

with factory.openai(base_url="http://localhost:8000/v1") as client:
    response = client.chat(
        messages=[{"role": "user", "content": "Hello!"}],
        system="You are a helpful assistant.",
        temperature=0.7,
    )
    print(response.content)
    print(f"Tokens used: {response.usage.total_tokens}")
```

### Async Usage

```python
from appinfra.log import Logger
from llm_infer.client import Factory

lg = Logger("my-app")
factory = Factory(lg)

async with factory.openai(base_url="http://localhost:8000/v1") as client:
    response = await client.chat_async(
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.content)
```

### Streaming

```python
from appinfra.log import Logger
from llm_infer.client import Factory

lg = Logger("my-app")
factory = Factory(lg)

# Sync streaming
with factory.openai() as client:
    for token in client.chat_stream(
        messages=[{"role": "user", "content": "Tell me a story"}],
        max_tokens=500,
    ):
        print(token, end="", flush=True)

    # Access usage stats after streaming completes (with null check)
    if client.last_response and client.last_response.usage:
        print(f"\nTokens: {client.last_response.usage.total_tokens}")

# Async streaming
async with factory.openai() as client:
    async for token in client.chat_stream_async(messages):
        print(token, end="", flush=True)
```

### Using Anthropic Backend

```python
from appinfra.log import Logger
from llm_infer.client import Factory

lg = Logger("my-app")
factory = Factory(lg)

# Requires: pip install llm-infer[anthropic]
async with factory.anthropic(model="claude-sonnet-4-20250514") as client:
    response = await client.chat_async(
        messages=[{"role": "user", "content": "Hello!"}],
        system="You are a helpful assistant.",
    )
    print(response.content)
```

## llm-infer Extensions

The client supports llm-infer specific extensions for enhanced functionality.

### LoRA Adapter Selection

```python
with factory.openai() as client:
    response = client.chat(
        messages=[{"role": "user", "content": "Translate to French: Hello"}],
        adapter="translation-lora",  # Select LoRA adapter
    )
    print(response.content)
```

### Thinking Mode

Enable thinking mode to get separated reasoning content:

```python
with factory.openai() as client:
    response = client.chat(
        messages=[{"role": "user", "content": "What is 15 * 23?"}],
        think=True,  # Enable thinking mode
    )
    print(f"Thinking: {response.thinking}")
    print(f"Answer: {response.content}")
```

### Tool Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }
]

with factory.openai() as client:
    response = client.chat(
        messages=[{"role": "user", "content": "What's the weather in NYC?"}],
        tools=tools,
        tool_choice="auto",
    )

    if response.has_tool_calls():
        for tool_call in response.tool_calls:
            print(f"Call: {tool_call.function.name}({tool_call.function.arguments})")
```

## API Reference

### Factory

Factory class for creating LLMClient instances. Requires a `Logger` instance.

```python
from appinfra.log import Logger
from llm_infer.client import Factory

lg = Logger("my-app")
factory = Factory(lg)

# OpenAI-compatible API
factory.openai(
    base_url: str = "http://localhost:8000/v1",
    model: str = "default",
    api_key: str | None = None,
    timeout: float = 120.0,
    rate_limit: dict = {"per_minute": 60},  # Rate limiting config
) -> LLMClient

# Anthropic Claude API
factory.anthropic(
    model: str = "claude-sonnet-4-20250514",
    api_key: str | None = None,
    max_tokens: int = 4096,
    timeout: float = 120.0,
    rate_limit: dict = {"per_minute": 60},  # Rate limiting config
) -> LLMClient

# From configuration dict
factory.from_config(config: dict) -> LLMClient
factory.from_backend_config(config: dict) -> LLMClient

# Register custom backend (class method)
Factory.register(name: str, backend_class: type[Backend]) -> None
```

### LLMClient

The client facade that delegates to backend implementations. Create instances using `Factory`.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `chat(messages, ...)` | `ChatResponse` | Sync chat completion |
| `chat_stream(messages, ...)` | `Iterator[str]` | Sync streaming |
| `chat_async(messages, ...)` | `ChatResponse` | Async chat completion |
| `chat_stream_async(messages, ...)` | `AsyncIterator[str]` | Async streaming |

#### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[dict]` | required | Chat messages with `role` and `content` |
| `model` | `str \| None` | `None` | Model override (uses client default if None) |
| `temperature` | `float` | `1.0` | Sampling temperature (0.0 to 2.0) |
| `max_tokens` | `int \| None` | `None` | Maximum tokens to generate |
| `system` | `str \| None` | `None` | System prompt |
| `adapter` | `str \| None` | `None` | LoRA adapter name (OpenAI-compatible only) |
| `think` | `bool \| None` | `None` | Enable thinking mode |
| `tools` | `list[dict] \| None` | `None` | Tool definitions |
| `tool_choice` | `str \| dict \| None` | `None` | Tool use control |

### ChatResponse

```python
@dataclass
class ChatResponse:
    content: str                              # Generated text
    usage: ChatCompletionUsage | None = None  # Token usage stats
    finish_reason: FinishReason | None = None # Why generation stopped
    model: str | None = None                  # Model that generated response

    # llm-infer extensions
    thinking: str | None = None               # Extracted <think> content
    tool_calls: list[ToolCall] | None = None  # Function calls

    def has_tool_calls(self) -> bool: ...     # Check if tool calls present
```

### Configuration Format

For `Factory.from_config()`:

```yaml
# Multi-backend configuration with rate limiting and retry
default: local

rate_limit:
  per_minute: 60  # Requests per minute per backend

retry:
  enabled: true
  timeout: 120    # Total retry timeout in seconds
  backoff:
    base: 1.0     # Initial delay
    max: 60.0     # Maximum delay

backends:
  local:
    type: openai_compatible
    base_url: http://localhost:8000/v1
    model: qwen2.5-72b
    timeout: 120.0
    # Per-backend overrides (optional)
    rate_limit:
      per_minute: 120
  anthropic:
    type: anthropic
    model: claude-sonnet-4-20250514
    max_tokens: 4096

# Single backend configuration
type: openai_compatible
base_url: http://localhost:8000/v1
model: default
```

## Usage Patterns

### Building a Proxy

```python
from uuid import uuid4
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from appinfra.log import Logger
from llm_infer.api import FinishReason
from llm_infer.client import Factory
from llm_infer.serving.api.openai.streaming import stream_chat_completion

app = FastAPI()
lg = Logger("proxy")
factory = Factory(lg)
client = factory.openai(base_url="http://backend:8000/v1")

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    messages = request["messages"]

    if request.get("stream"):
        return StreamingResponse(
            stream_chat_completion(
                request_id=str(uuid4()),
                model="proxied-model",
                token_iterator=client.chat_stream(messages),
                get_finish_reason=lambda: FinishReason.STOP,
            ),
            media_type="text/event-stream",
        )
    else:
        response = client.chat(messages)
        return {"choices": [{"message": {"content": response.content}}]}

@app.on_event("shutdown")
async def shutdown():
    await client.aclose()
```

### Mocking in Tests

Use the `Backend` ABC for type-safe mocking:

```python
from llm_infer.client import Backend, ChatResponse, LLMClient

class MockBackend(Backend):
    def __init__(self, responses: list[str]):
        self._responses = iter(responses)
        self._last_response: ChatResponse | None = None

    @property
    def last_response(self) -> ChatResponse | None:
        return self._last_response

    def chat(self, messages, **kwargs) -> ChatResponse:
        content = next(self._responses)
        self._last_response = ChatResponse(content=content)
        return self._last_response

    def chat_stream(self, messages, **kwargs):
        content = next(self._responses)
        for char in content:
            yield char
        self._last_response = ChatResponse(content=content)

    async def chat_async(self, messages, **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def chat_stream_async(self, messages, **kwargs):
        async def gen():
            for token in self.chat_stream(messages, **kwargs):
                yield token
        return gen()

    @classmethod
    def from_config(cls, lg: Logger, config: dict) -> "MockBackend":
        return cls(responses=[])

# Use in tests
backend = MockBackend(["Hello!", "Goodbye!"])
client = LLMClient(lg=lg, backend=backend)
assert client.chat([{"role": "user", "content": "Hi"}]).content == "Hello!"
```

### With External APIs

```python
from appinfra.log import Logger
from llm_infer.client import Factory

lg = Logger("my-app")
factory = Factory(lg)

# OpenAI
client = factory.openai(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4",
)

# Anthropic
client = factory.anthropic(
    api_key="sk-ant-...",
    model="claude-sonnet-4-20250514",
)
```

## Error Handling

The client provides a clean exception hierarchy:

```python
from appinfra.log import Logger
from llm_infer.client import (
    Factory,
    BackendError,
    BackendUnavailableError,
    BackendTimeoutError,
    BackendRequestError,
)

lg = Logger("my-app")
factory = Factory(lg)

with factory.openai() as client:
    try:
        response = client.chat(messages=[{"role": "user", "content": "Hello"}])
        print(response.content)
    except BackendUnavailableError:
        print("Server not reachable")
    except BackendTimeoutError:
        print("Request timed out")
    except BackendRequestError as e:
        print(f"HTTP error {e.status_code}: {e}")
    except BackendError as e:
        print(f"Backend error: {e}")
```

## Resource Management

Always use context managers to ensure proper cleanup:

```python
from appinfra.log import Logger
from llm_infer.client import Factory

lg = Logger("my-app")
factory = Factory(lg)

# Sync - closes sync HTTP client
with factory.openai() as client:
    response = client.chat(messages)
    print(response.content)

# Async - closes both sync and async HTTP clients
async with factory.openai() as client:
    response = await client.chat_async(messages)
    print(response.content)

# Manual cleanup if not using context managers
client = factory.openai()
try:
    response = client.chat(messages)
    print(response.content)
finally:
    client.close()  # or await client.aclose() for async
```

## Sequence Diagrams

### Streaming Flow

```text
Client                    Backend                   API
  │                         │                        │
  │─── chat_stream() ──────>│                        │
  │                         │─── POST /chat/completions ──>│
  │                         │                        │
  │                         │<─── SSE: role chunk ───│
  │                         │<─── SSE: token chunk ──│
  │<── yield token ─────────│<─── SSE: token chunk ──│
  │<── yield token ─────────│<─── SSE: finish chunk ─│
  │<── yield token ─────────│<─── SSE: [DONE] ───────│
  │                         │                        │
  │─── last_response ──────>│─── ChatResponse        │
```

### Non-Streaming Flow

```text
Client                    Backend                   API
  │                         │                        │
  │─── chat() ─────────────>│                        │
  │                         │─── POST /chat/completions ──>│
  │                         │                        │
  │                         │<─── JSON response ─────│
  │<── ChatResponse ────────│                        │
```
