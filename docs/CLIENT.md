# Multi-Backend LLM Client

This document describes the unified LLM client supporting multiple backends with sync/async
operations.

## Overview

The `llm_infer.client` module provides a multi-backend client for interacting with LLM APIs. It
supports both synchronous and asynchronous operations, SSE streaming, and llm-infer specific
extensions like LoRA adapters and thinking mode.

**Supported Backends:**
- **OpenAI-compatible**: OpenAI, llm-infer server, vLLM, Ollama
- **Anthropic**: Claude models (requires `pip install llm-infer[anthropic]`)

**Public API path:**
```python
from llm_infer.client import LLMClient, ChatResponse, Backend
# or
from llm_infer.api import LLMClient, ChatResponse, Backend
```

## Available Types

| Type | Kind | Description |
|------|------|-------------|
| `LLMClient` | Class | Unified client facade with factory methods |
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
from llm_infer.client import LLMClient

with LLMClient.openai(base_url="http://localhost:8000/v1") as client:
    # Simple: returns content string
    response = client.chat(
        messages=[{"role": "user", "content": "Hello!"}],
        system="You are a helpful assistant.",
        temperature=0.7,
    )
    print(response)

    # Full response with metadata
    response = client.chat_full(messages=[{"role": "user", "content": "Hello!"}])
    print(response.content)
    print(f"Tokens used: {response.usage.total_tokens}")
```

### Async Usage

```python
from llm_infer.client import LLMClient

async with LLMClient.openai(base_url="http://localhost:8000/v1") as client:
    response = await client.chat_async(
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response)
```

### Streaming

```python
# Sync streaming
with LLMClient.openai() as client:
    for token in client.chat_stream(
        messages=[{"role": "user", "content": "Tell me a story"}],
        max_tokens=500,
    ):
        print(token, end="", flush=True)

    # Access usage stats after streaming completes
    print(f"\nTokens: {client.last_response.usage.total_tokens}")

# Async streaming
async with LLMClient.openai() as client:
    async for token in client.chat_stream_async(messages):
        print(token, end="", flush=True)
```

### Using Anthropic Backend

```python
from llm_infer.client import LLMClient

# Requires: pip install llm-infer[anthropic]
async with LLMClient.anthropic(model="claude-sonnet-4-20250514") as client:
    response = await client.chat_async(
        messages=[{"role": "user", "content": "Hello!"}],
        system="You are a helpful assistant.",
    )
    print(response)
```

## llm-infer Extensions

The client supports llm-infer specific extensions for enhanced functionality.

### LoRA Adapter Selection

```python
with LLMClient.openai() as client:
    response = client.chat_full(
        messages=[{"role": "user", "content": "Translate to French: Hello"}],
        adapter_id="translation-lora",  # Select LoRA adapter
    )
```

### Thinking Mode

Enable thinking mode to get separated reasoning content:

```python
with LLMClient.openai() as client:
    response = client.chat_full(
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

with LLMClient.openai() as client:
    response = client.chat_full(
        messages=[{"role": "user", "content": "What's the weather in NYC?"}],
        tools=tools,
        tool_choice="auto",
    )

    if response.has_tool_calls():
        for tool_call in response.tool_calls:
            print(f"Call: {tool_call.function.name}({tool_call.function.arguments})")
```

## API Reference

### LLMClient

The main client facade that delegates to backend implementations.

#### Factory Methods

```python
# OpenAI-compatible API
LLMClient.openai(
    base_url: str = "http://localhost:8000/v1",
    model: str = "default",
    api_key: str | None = None,
    timeout: float = 120.0,
) -> LLMClient

# Anthropic Claude API
LLMClient.anthropic(
    model: str = "claude-sonnet-4-20250514",
    api_key: str | None = None,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> LLMClient

# From configuration dict
LLMClient.from_config(config: dict) -> LLMClient
LLMClient.from_backend_config(config: dict) -> LLMClient
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `chat(messages, ...)` | `str` | Sync chat, returns content |
| `chat_full(messages, ...)` | `ChatResponse` | Sync chat, returns full response |
| `chat_stream(messages, ...)` | `Iterator[str]` | Sync streaming |
| `chat_async(messages, ...)` | `str` | Async chat, returns content |
| `chat_full_async(messages, ...)` | `ChatResponse` | Async chat, returns full response |
| `chat_stream_async(messages, ...)` | `AsyncIterator[str]` | Async streaming |

#### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[dict]` | required | Chat messages with `role` and `content` |
| `model` | `str \| None` | `None` | Model override (uses client default if None) |
| `temperature` | `float` | `1.0` | Sampling temperature (0.0 to 2.0) |
| `max_tokens` | `int \| None` | `None` | Maximum tokens to generate |
| `system` | `str \| None` | `None` | System prompt |
| `adapter_id` | `str \| None` | `None` | LoRA adapter name (OpenAI-compatible only) |
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

For `LLMClient.from_config()`:

```yaml
# Multi-backend configuration
default: local  # Which backend to use
backends:
  local:
    type: openai_compatible
    base_url: http://localhost:8000/v1
    model: qwen2.5-72b
    timeout: 120.0
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

from llm_infer.api import FinishReason
from llm_infer.client import LLMClient
from llm_infer.serving.api.openai.streaming import stream_chat_completion

app = FastAPI()
client = LLMClient.openai(base_url="http://backend:8000/v1")

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
        response = client.chat_full(messages)
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
    def from_config(cls, config):
        return cls([])

# Use in tests
backend = MockBackend(["Hello!", "Goodbye!"])
client = LLMClient(backend=backend)
assert client.chat([{"role": "user", "content": "Hi"}]) == "Hello!"
```

### With External APIs

```python
# OpenAI
client = LLMClient.openai(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4",
)

# Anthropic
client = LLMClient.anthropic(
    api_key="sk-ant-...",
    model="claude-sonnet-4-20250514",
)
```

## Error Handling

The client provides a clean exception hierarchy:

```python
from llm_infer.client import (
    LLMClient,
    BackendError,
    BackendUnavailableError,
    BackendTimeoutError,
    BackendRequestError,
)

with LLMClient.openai() as client:
    try:
        response = client.chat(messages=[{"role": "user", "content": "Hello"}])
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
# Sync - closes sync HTTP client
with LLMClient.openai() as client:
    response = client.chat(messages)

# Async - closes both sync and async HTTP clients
async with LLMClient.openai() as client:
    response = await client.chat_async(messages)

# Manual cleanup if not using context managers
client = LLMClient.openai()
try:
    response = client.chat(messages)
finally:
    client.close()  # or await client.aclose() for async
```

## Sequence Diagrams

### Streaming Flow

```
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

```
Client                    Backend                   API
  │                         │                        │
  │─── chat_full() ────────>│                        │
  │                         │─── POST /chat/completions ──>│
  │                         │                        │
  │                         │<─── JSON response ─────│
  │<── ChatResponse ────────│                        │
```
