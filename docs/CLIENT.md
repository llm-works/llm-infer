# OpenAI-Compatible Client

This document describes the async client for consuming SSE streams from OpenAI-compatible APIs.

## Overview

The `llm_infer.client` module provides a reusable async client for calling OpenAI-compatible APIs,
with full support for SSE streaming. This enables downstream packages (proxies, frontends) to
consume
streaming responses without reimplementing SSE parsing.

**Public API path:**
```python
from llm_infer.client import OpenAIClient, ChatClient, ChatResponse
# or
from llm_infer.api import OpenAIClient, ChatClient, ChatResponse
```

## Available Types

| Type | Kind | Description |
|------|------|-------------|
| `OpenAIClient` | Class | Async client with `chat()` and `chat_stream()` methods |
| `ChatClient` | Protocol | Interface for type-safe mocking and dependency injection |
| `ChatResponse` | Dataclass | Response with `content`, `usage`, and `finish_reason` |

## Quick Start

### Non-Streaming Request

```python
from llm_infer.client import OpenAIClient

client = OpenAIClient(base_url="http://localhost:8000/v1")

response = await client.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    system="You are a helpful assistant.",
    temperature=0.7,
)

print(response.content)
print(f"Tokens used: {response.usage.total_tokens}")
```

### Streaming Request

```python
async for token in client.chat_stream(
    messages=[{"role": "user", "content": "Tell me a story"}],
    max_tokens=500,
):
    print(token, end="", flush=True)

# Access usage stats after streaming completes
print(f"\nTokens: {client.last_response.usage.total_tokens}")
```

## API Reference

### OpenAIClient

```python
class OpenAIClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None: ...

    async def chat(
        self,
        messages: list[ChatMessage] | list[dict],
        system: str | None = None,
        model: str = "default",
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ChatResponse: ...

    async def chat_stream(
        self,
        messages: list[ChatMessage] | list[dict],
        system: str | None = None,
        model: str = "default",
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]: ...

    @property
    def last_response(self) -> ChatResponse | None: ...
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | `http://localhost:8000/v1` | API base URL |
| `api_key` | `str \| None` | `None` | Bearer token for Authorization header |
| `timeout` | `float` | `120.0` | Request timeout in seconds |

**Method Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `list[dict]` | Chat messages with `role` and `content` |
| `system` | `str \| None` | System prompt (prepended to messages) |
| `model` | `str` | Model name for the request |
| `temperature` | `float` | Sampling temperature (0.0 to 2.0) |
| `max_tokens` | `int \| None` | Maximum tokens to generate |
| `**kwargs` | `Any` | Additional API parameters (top_p, etc.) |

### ChatResponse

```python
@dataclass
class ChatResponse:
    content: str
    usage: ChatCompletionUsage | None = None
    finish_reason: FinishReason | None = None
```

### ChatClient Protocol

The `ChatClient` protocol enables type-safe mocking and dependency injection:

```python
@runtime_checkable
class ChatClient(Protocol):
    @property
    def last_response(self) -> ChatResponse | None: ...

    async def chat(self, messages, **kwargs) -> ChatResponse: ...

    async def chat_stream(self, messages, **kwargs) -> AsyncIterator[str]: ...
```

## Usage Patterns

### Building a Proxy

Compose with the existing `stream_chat_completion()` helper to proxy streaming responses:

```python
from uuid import uuid4
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from llm_infer.api import FinishReason
from llm_infer.client import OpenAIClient
from llm_infer.serving.api.openai.streaming import stream_chat_completion

app = FastAPI()
client = OpenAIClient(base_url="http://backend:8000/v1")

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
        response = await client.chat(messages)
        return {"choices": [{"message": {"content": response.content}}]}
```

### Mocking in Tests

Use the `ChatClient` protocol for type-safe mocking:

```python
from collections.abc import AsyncIterator
from llm_infer.client import ChatClient, ChatResponse

class MockChatClient:
    def __init__(self, responses: list[str]):
        self._responses = iter(responses)
        self._last_response: ChatResponse | None = None

    @property
    def last_response(self) -> ChatResponse | None:
        return self._last_response

    async def chat(self, messages, **kwargs) -> ChatResponse:
        content = next(self._responses)
        self._last_response = ChatResponse(content=content)
        return self._last_response

    async def chat_stream(self, messages, **kwargs) -> AsyncIterator[str]:
        content = next(self._responses)
        for char in content:
            yield char
        self._last_response = ChatResponse(content=content)

# Type checks correctly
client: ChatClient = MockChatClient(["Hello!", "Goodbye!"])
```

### With Authentication

```python
client = OpenAIClient(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    timeout=60.0,
)

response = await client.chat(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4",
)
```

## Error Handling

The client raises standard `httpx` exceptions:

```python
import httpx
from llm_infer.client import OpenAIClient

client = OpenAIClient()

try:
    response = await client.chat(messages=[...])
except httpx.HTTPStatusError as e:
    print(f"HTTP error: {e.response.status_code}")
except httpx.RequestError as e:
    print(f"Request failed: {e}")
except ValueError as e:
    print(f"Invalid response: {e}")  # e.g., empty choices array
```

## Sequence Diagrams

### Streaming Flow

```
Client                    httpx                     API
  │                         │                        │
  │─── build payload ───────│                        │
  │─── POST /chat/completions ──────────────────────>│
  │                         │                        │
  │                         │<─── SSE: role chunk ───│
  │                         │<─── SSE: token chunk ──│
  │<── yield token ─────────│<─── SSE: token chunk ──│
  │<── yield token ─────────│<─── SSE: finish chunk ─│
  │<── yield token ─────────│<─── SSE: [DONE] ───────│
  │                         │                        │
  │─── finalize ChatResponse                         │
  │─── store as last_response                        │
```

### Non-Streaming Flow

```
Client                    httpx                     API
  │                         │                        │
  │─── build payload ───────│                        │
  │─── POST /chat/completions ──────────────────────>│
  │                         │                        │
  │                         │<─── JSON response ─────│
  │<── parse response ──────│                        │
  │─── create ChatResponse                           │
  │─── return                                        │
```
