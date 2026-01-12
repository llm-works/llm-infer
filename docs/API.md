# Public API Reference

This document describes the public API surface of `llm-infer` for downstream consumers.

## Design Principle

`llm-infer` exposes OpenAI-compatible Pydantic schemas through a clean, stable public API path.
This enables downstream projects (proxies, clients, tooling) to import schemas without depending on
internal module structure.

**Public API path:**
```python
from llm_infer.api import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
```

**Alternative (schemas only, no client):**
```python
from llm_infer.schemas.openai import ChatCompletionRequest, ChatMessage, Role
```

## Available Schemas

### Enums

| Schema | Description |
|--------|-------------|
| `Role` | Chat message roles: `SYSTEM`, `USER`, `ASSISTANT`, `TOOL` |
| `FinishReason` | Generation stop reasons: `STOP`, `LENGTH`, `CONTENT_FILTER` |

### Chat Messages

| Schema | Description |
|--------|-------------|
| `ChatMessage` | A single chat message with `role`, `content`, and optional `name` |

### Chat Completions (Non-Streaming)

| Schema | Description |
|--------|-------------|
| `ChatCompletionRequest` | Request body for `/v1/chat/completions` |
| `ChatCompletionResponse` | Non-streaming response |
| `ChatCompletionChoice` | A single completion choice |
| `ChatCompletionUsage` | Token usage statistics |

### Chat Completions (Streaming)

| Schema | Description |
|--------|-------------|
| `ChatCompletionChunk` | SSE streaming chunk |
| `ChatCompletionChunkChoice` | A single streaming chunk choice |
| `ChatCompletionChunkDelta` | Delta content in streaming response |

### Legacy Completions

| Schema | Description |
|--------|-------------|
| `CompletionRequest` | Request body for `/v1/completions` |
| `CompletionResponse` | Non-streaming response |
| `CompletionChoice` | A single completion choice |
| `CompletionChunk` | SSE streaming chunk |
| `CompletionChunkChoice` | Streaming chunk choice |

### Models Endpoint

| Schema | Description |
|--------|-------------|
| `ModelInfo` | Model information for `/v1/models` |
| `ModelList` | Response for `GET /v1/models` |

## Usage Examples

### Building a Proxy

```python
from llm_infer.api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    Role,
)

# Parse incoming request
request = ChatCompletionRequest.model_validate(incoming_json)

# Access message content
for msg in request.messages:
    if msg.role == Role.USER:
        print(f"User said: {msg.content}")

# Build response
response = ChatCompletionResponse(
    id="chatcmpl-123",
    created=1234567890,
    model="my-model",
    choices=[...],
    usage=ChatCompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    ),
)
```

### Streaming Response Handling

```python
from llm_infer.api import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    Role,
)

# Parse streaming chunk
chunk = ChatCompletionChunk.model_validate(sse_data)

for choice in chunk.choices:
    if choice.delta.content:
        print(choice.delta.content, end="", flush=True)
    if choice.finish_reason:
        print(f"\n[Finished: {choice.finish_reason}]")
```

### Request Validation

```python
from pydantic import ValidationError
from llm_infer.api import ChatCompletionRequest

try:
    request = ChatCompletionRequest.model_validate(user_input)
except ValidationError as e:
    # Handle validation errors
    print(f"Invalid request: {e}")
```

## Stability Guarantee

The `llm_infer.api` module is the public API surface. We aim to maintain backward compatibility
for all exports from this module. Internal module paths (`llm_infer.serving.*`) are implementation
details and may change without notice.
