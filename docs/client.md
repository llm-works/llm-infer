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
- **Google**: Gemini via AI Studio or Vertex AI OpenAI-compat endpoint
  (auto-selects `GeminiBackend` when `provider: google` or a
  `googleapis.com` `base_url` is detected)
- **Vertex AI native REST**: `cachedContents` + `generateContent` via
  `type: vertex_native` (sibling to `Backend`, built through
  `Factory.vertex_natives_from_config()`)

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
| `Factory` | Class | Factory for creating clients, routers, and embedding clients from config |
| `ChatClient` | ABC | Abstract base for all chat clients (`LLMClient`, `LLMRouter`, `FallbackClient`) |
| `LLMClient` | Class | Single-backend client with rate limiting, retry, and callbacks |
| `LLMRouter` | Class | Multi-backend router with strategy-based routing and model discovery |
| `FallbackClient` | Class | Wraps `LLMRouter` with cross-provider model fallback on transient errors |
| `EmbeddingClient` | Class | Embedding client with retry and callbacks |
| `Backend` | ABC | Abstract base class for chat backend implementations |
| `OpenAICompatibleBackend` | Class | Backend for OpenAI-compatible APIs |
| `Provider` | Enum | Detected provider (`openai`, `anthropic`, `google`, `xai`, `local`, `unknown`) |
| `LLMCallbacks` | Dataclass | Six lifecycle hooks: `on_request`, `on_response`, `on_retry`, `on_error`, `on_before_send`, `on_after_send` |
| `EmbeddingCallbacks` | Dataclass | Embedding equivalents of the retry-loop-level hooks |
| `ChatRequest` | Dataclass | Request carrier with 8-hex-char `id` for log correlation |
| `ChatResponse` | Dataclass | Response with content, usage, thinking, tool_calls, and a `request` backreference |
| `SendContext` / `SendResult` | Dataclass | HTTP-level context/result passed to `on_before_send`/`on_after_send` |
| `RoutingStrategy` | Protocol | Custom backend-selection logic; built-in `DefaultStrategy` is round-robin |
| `BackendError` | Exception | Base exception for all backend errors |
| `BackendUnavailableError` | Exception | Connection failed |
| `BackendTimeoutError` | Exception | Request timed out |
| `BackendRequestError` | Exception | HTTP error from backend |
| `ConfigError` | Exception | Base exception for configuration problems |
| `ModelConflictError` | Exception | Same model registered by two backends |

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
async with factory.openai(base_url="http://localhost:8000/v1") as client:
    messages = [{"role": "user", "content": "Tell me a story"}]
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
async with factory.anthropic(default_model="claude-sonnet-4-20250514") as client:
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

Factory class for creating clients, routers, and embedding clients from
config. Requires a `Logger` instance. Every constructor accepts an optional
`callbacks=` argument (see [Callbacks](#callbacks)).

```python
from appinfra.log import Logger
from appinfra.yaml import SecretStr
from llm_infer.client import Factory, LLMCallbacks, EmbeddingCallbacks

lg = Logger("my-app")
factory = Factory(lg)


# OpenAI-compatible API
def openai(
    base_url: str = "http://localhost:8000/v1",
    default_model: str | None = None,
    api_key: str | SecretStr | None = None,
    timeout: float = 120.0,
    rate_limit: dict[str, Any] | None = None,
    callbacks: LLMCallbacks | None = None,
) -> LLMClient: ...


# Anthropic Claude API
def anthropic(
    default_model: str = "claude-sonnet-4-20250514",
    api_key: str | SecretStr | None = None,
    max_tokens: int = 4096,
    timeout: float = 120.0,
    rate_limit: dict[str, Any] | None = None,
    callbacks: LLMCallbacks | None = None,
) -> LLMClient: ...


# OpenAI-compatible embeddings
def embeddings(
    base_url: str = "http://localhost:8001/v1",
    model: str = "default",
    api_key: str | SecretStr | None = None,
    timeout: float = 120.0,
    retry: RetryConfig | None = None,
    rate_limit: dict[str, Any] | None = None,
    dimensions: int | None = None,
    callbacks: EmbeddingCallbacks | None = None,
) -> EmbeddingClient: ...


# Google Generative AI / Vertex embeddings
def embeddings_google(
    api_key: str | SecretStr | None = None,
    model: str = "gemini-embedding-001",
    task_type: str = "RETRIEVAL_DOCUMENT",
    *,
    callbacks: EmbeddingCallbacks | None = None,
) -> EmbeddingClient: ...


# From configuration dict
def from_config(config: dict, callbacks: LLMCallbacks | None = None) -> LLMRouter: ...
def from_backend_config(
    config: dict, name: str, callbacks: LLMCallbacks | None = None
) -> LLMClient: ...
def embeddings_from_config(config: dict, name: str = "default") -> EmbeddingClient: ...


# Native Vertex REST (cachedContents + generateContent)
def vertex_natives_from_config(config: dict) -> dict[str, NativeVertexBackend]: ...
def vertex_native_from_backend_config(
    config: dict, name: str
) -> NativeVertexBackend: ...
```

`from_config` always returns an `LLMRouter`. A bare single-backend config
(no `backends:` block) is wrapped in a one-entry router so the calling code
gets the same surface either way. Non-chat types listed in
`NON_CHAT_BACKEND_TYPES` (currently `vertex_native`) are filtered out of
the router and must be built through their dedicated accessor.

### LLMClient

The client facade that delegates to backend implementations. Create instances using `Factory`.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `chat(messages, ...)` | `ChatResponse` | Sync chat completion |
| `chat_stream(messages, ...)` | `Iterator[str]` | Sync streaming |
| `chat_async(messages, ...)` | `ChatResponse` | Async chat completion |
| `chat_stream_async(messages, ...)` | `AsyncIterator[str]` | Async streaming |
| `with_callbacks(callbacks)` | `LLMClient` | Return a copy with callbacks configured |
| `last_response` (property) | `ChatResponse \| None` | Last response captured after a streaming call (falls back to `backend.last_response`) |
| `close()` / `aclose()` | — | Release sync / async HTTP resources |

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
    content: str  # Generated text
    usage: ChatCompletionUsage | None = None  # Token usage stats
    finish_reason: FinishReason | None = None  # Why generation stopped
    model: str | None = None  # Model that generated response
    request: ChatRequest | None = None  # Backreference to the originating request

    # llm-infer extensions
    thinking: str | None = None  # Extracted <think> content
    tool_calls: list[ToolCall] | None = None  # Function calls

    def has_tool_calls(self) -> bool: ...  # Check if tool calls present
```

### LLMRouter

Multi-backend router returned by `Factory.from_config()`. Single-backend
configs (no `backends:` block) are wrapped in a one-entry router so calling
code gets a consistent surface. Routes each request to a named backend via a
`RoutingStrategy` (default: round-robin over enabled backends that serve the
requested model). Model discovery is lazy — backends are probed on first use.

```python
router = Factory(lg).from_config(config)

# Use the default backend
response = router.chat(messages)

# Route to a specific backend
response = router.chat(messages, backend="anthropic")

# Route by model (model-to-backend resolution via ModelDiscovery)
response = router.chat(messages, model="claude-sonnet-4-20250514")

# Inspect routing without sending
target = router.resolve(model="gpt-4")  # -> ResolvedTarget(backend=..., model=...)

# Read-only views into current state
router.clients  # Mapping[str, LLMClient]
router.default  # default backend name
router.models  # Mapping[str, str] (model -> backend)
```

Custom routing strategies (weighted, latency-aware, etc.) implement the
`RoutingStrategy` protocol and are wired via `strategy: {factory: myapp.routing}`
in config. The built-in `DefaultStrategy` covers round-robin.

### FallbackClient

Wraps an `LLMRouter` and transparently retries on transient errors (5xx,
timeout, unavailable, 429) with an equivalent model, until the chain is
exhausted. For 429s the inner `RetryHelper` backs off against the same model
first; fallback engages once that budget is exhausted.

```python
from llm_infer.client import Factory, FallbackClient

router = Factory(lg).from_config(config)
fallbacks = {
    "gpt-4o": "claude-sonnet-4-20250514",
    "claude-sonnet-4-20250514": "gemini-2.0-pro",  # chains implicitly
}
client = FallbackClient(lg, router, fallbacks)
response = client.chat(messages, model="gpt-4o")
```

**Pinning a fallback to a backend** — keys and values accept `model@backend`
to pin a step to a specific backend. `@` is used (not `/`) to avoid clashing
with OpenRouter's `provider/model` names:

```python
fallbacks = {
    "gpt-4o@openai_primary": "gpt-4o@openai_backup",
    "gpt-4o@openai_backup": "claude-sonnet-4-20250514",
}
```

Bare refs are accepted without cross-backend probing: declared-config
collisions are already caught upstream by `ModelDiscovery` as
`ModelConflictError`, and a bare ref that no backend declares resolves at
request time via the router's default. Qualified `model@backend` refs are
validated at construction to name a configured backend; unknown backends
raise `ConfigError`. Cycles (`A → B → A`) are detected and retried
round-robin until one succeeds. A backend configured without `retry` falls
back on its first transient error (a warning is logged at construction).

### EmbeddingClient

Wraps an embedding backend and adds retry, callbacks, and client-level
defaults. Built via `Factory.embeddings()`, `Factory.embeddings_google()`,
or `Factory.embeddings_from_config()`.

```python
from llm_infer.client import EmbeddingClient
from llm_infer.client.backends import RetryConfig

client = Factory(lg).embeddings(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="text-embedding-3-small",
    dimensions=384,  # client-level default; per-call overrides win
    retry=RetryConfig(timeout=120.0),
)

with client:
    result = client.embed("Hello world")
    result = client.embed("Hello world", dimensions=1536)
    batch = client.embed_batch(["one", "two", "three"])

# Async
async with client:
    result = await client.embed_async("Hello world")
    batch = await client.embed_batch_async(["one", "two"])
```

Token-source semantics:

| Backend | `result.prompt_tokens`  | `count_tokens()`  |
|---------|-------------------------|-------------------|
| OpenAI  | populated from response | tiktoken (local)  |
| Google  | `None`                  | `countTokens` API |

### Callbacks

`LLMCallbacks` exposes six hooks split across two levels. Pass one via
`callbacks=` on any `Factory` constructor, or attach later with
`LLMClient.with_callbacks()`.

**Retry-loop level** (fires once per retry-loop entry, before backoff):

| Hook | Signature | Purpose |
|------|-----------|---------|
| `on_request` | `(request, retry)` | Fires before each attempt; `retry=0` for first |
| `on_response` | `(request, response)` | Fires after a successful response (after stream completes for streaming) |
| `on_retry` | `(request, exception, attempt, delay_seconds)` | Fires on a transient error that will be retried, before the backoff sleep |
| `on_error` | `(request, exception)` | Fires after a terminal failure |

**HTTP level** (fires at actual send, after any backoff delay):

| Hook | Signature | Purpose |
|------|-----------|---------|
| `on_before_send` | `(SendContext)` | Immediately before HTTP request |
| `on_after_send` | `(SendContext, SendResult)` | After HTTP response/error |

`SendContext` carries `attempt`, `retry_reason`, `delay_seconds`, `model`,
`backend`, and `req_id`. `SendResult` carries `status_code`, `error`, and
`elapsed_ms`.

```python
from llm_infer.client import Factory, LLMCallbacks

# lg, messages from Quick Start; metrics is your observability client


def log_attempt(req, retry):
    lg.info("chat.request", extra={"req_id": req.id, "retry": retry})


def record_http(ctx, result):
    metrics.histogram(
        "llm.http_ms",
        result.elapsed_ms,
        tags={"backend": ctx.backend, "status": result.status_code},
    )


callbacks = LLMCallbacks(on_request=log_attempt, on_after_send=record_http)
with Factory(lg).openai(callbacks=callbacks) as client:
    client.chat(messages)
```

`EmbeddingCallbacks` mirrors the retry-loop-level surface for
`EmbeddingClient`.

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
    # provider: <name>  # optional; overrides URL-based auto-detection.
                        # Valid: anthropic, openai, xai, google, local, unknown.
                        # Mismatch with URL-detected value logs a WARN.
    # Per-backend overrides (optional)
    rate_limit:
      per_minute: 120
  anthropic:
    type: anthropic
    model: claude-sonnet-4-20250514
    max_tokens: 4096
  vertex_direct:
    # Native Vertex REST (cachedContents + generateContent).
    # Built by Factory.vertex_natives_from_config(config), NOT from_config —
    # from_config filters non-chat types out of the router.
    type: vertex_native
    project: my-gcp-project
    region: us-central1
    auth:
      mode: gcp_sa
      credentials_path: /path/to/sa.json
    service_tier: priority   # optional
    timeout: 60

# Single backend configuration
type: openai_compatible
base_url: http://localhost:8000/v1
model: default
```

## Usage Patterns

### Building a Gateway

`llm_infer.client` is a library, not a standalone gateway process. When a
single OpenAI-compatible HTTP surface in front of an `LLMRouter` is needed
(so downstream clients can point at one URL), wrap the router in FastAPI.
Routing, model discovery, retry, fallback, and callbacks are already
covered by the library; the wrapper adds the HTTP surface, hot-reload
wiring, `/health`, and `/metrics`.

```python
from uuid import uuid4
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from appinfra.log import Logger
from llm_infer.api import FinishReason
from llm_infer.client import Factory, FallbackClient
from llm_infer.serving.api.openai.streaming import stream_chat_completion

app = FastAPI()
lg = Logger("gateway")
factory = Factory(lg)
router = factory.from_config(load_config())
client = FallbackClient(lg, router, fallbacks={"gpt-4o": "claude-sonnet-4-20250514"})


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    messages = request["messages"]
    model = request.get("model")

    if request.get("stream"):
        return StreamingResponse(
            stream_chat_completion(
                request_id=str(uuid4()),
                model=model or "default",
                token_iterator=client.chat_stream(messages, model=model),
                get_finish_reason=lambda: FinishReason.STOP,
            ),
            media_type="text/event-stream",
        )
    response = client.chat(messages, model=model)
    return {
        "id": f"chatcmpl-{uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.content},
                "finish_reason": response.finish_reason.value
                if response.finish_reason
                else "stop",
            }
        ],
        "usage": response.usage.model_dump() if response.usage else None,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": b}
            for m, b in router.models.items()
        ],
    }


@app.on_event("shutdown")
async def shutdown():
    await router.aclose()
```

**Runtime config reload.** The library does not ship a control plane; the
router is immutable after construction. To add/remove/enable backends
without restarting the process, subscribe to `appinfra`'s YAML watcher on
the config file, rebuild an `LLMRouter` (and any `FallbackClient` wrapper)
from the new config in the reload handler, and atomically swap the module
reference used by request handlers. Drain in-flight requests against the
previous router before calling `aclose()` on it.

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
