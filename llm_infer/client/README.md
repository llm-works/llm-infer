# llm_infer.client

Multi-backend LLM client with sync/async support, routing strategies, and model discovery.

## Core Types

### Chat

| Module | Class | Purpose |
|--------|-------|---------|
| `base.py` | `ChatClient` | Abstract base class for all chat clients |
| `client.py` | `LLMClient` | Single-backend client with rate limiting and backoff |
| `router.py` | `LLMRouter` | Multi-backend router with strategy-based routing |
| `factory.py` | `Factory` | Creates clients/routers from config |

### Embedding

| Module | Class | Purpose |
|--------|-------|---------|
| `embedding.py` | `EmbeddingClient` | Embedding client with retry support |
| `backends/embedding.py` | `Backend` | Abstract base class for embedding backends |
| `backends/embedding.py` | `EmbeddingResult` | Result with embedding vector, model, token count |

## Routing System

| Module | Purpose |
|--------|---------|
| `strategy.py` | `RoutingStrategy` protocol and core types (`RoutingContext`, `RoutingDecision`, `RoutingResult`) |
| `strategies.py` | Built-in strategies: `DefaultStrategy`, `FallbackStrategy` |
| `resolver.py` | `ModelResolver` — model-to-backend resolution, lazy discovery, "auto"/"default" handling |
| `router_helper.py` | Internal helpers for routing loop (setup, success/error handling) |

## Other Modules

| Module | Purpose |
|--------|---------|
| `types.py` | `ChatRequest`, `ChatResponse`, `AdapterInfo` |
| `errors.py` | `BackendError`, `BackendRequestError`, `BackendTimeoutError`, `BackendUnavailableError` |
| `discovery.py` | `ModelDiscovery` — lazy model probing across backends |
| `backends/` | Chat backends (OpenAI, Anthropic, Gemini) and embedding backends (OpenAI, Google) |
| `saia.py` | SAIA adapter integration (optional) |

## Configuration

### Multi-Backend Router

```yaml
default: primary                    # Required: default backend name

rate_limit:                         # Optional: applies to all backends
  per_minute: 60

retry:                              # Optional: exponential backoff on failure
  backoff:
    base: 1.0                       # Initial delay (seconds)
    max: 60                         # Max delay (seconds)
  timeout: 120                      # Request timeout (seconds, 0 = no timeout)

backends:
  primary:
    enabled: true                   # Optional, default true
    type: openai_compatible         # or: openai, anthropic
    base_url: http://localhost:8000/v1
    api_key: sk-...                 # Optional (or use env var via config loader)
    model: llama-3.1-8b             # Optional: default model
    models:                         # Optional: restrict to these models
      - llama-3.1-8b
      - llama-3.1-70b

  fallback:
    type: anthropic
    api_key: sk-ant-...
```

### Backend Types

| Type | Description |
|------|-------------|
| `openai_compatible` | Any OpenAI-compatible API (vLLM, Ollama, llm-infer, etc.) |
| `openai` | OpenAI API (defaults to `https://api.openai.com/v1`) |
| `anthropic` | Anthropic Claude API |

### Single Backend

For simple single-backend use, omit `backends`:

```yaml
type: openai_compatible
base_url: http://localhost:8000/v1
model: llama-3.1-8b
```

### Model Discovery

By default, backends are lazily probed for available models on first use. To skip
probing and use explicit models, specify a `models` list per backend.

Reserved model names:
- `auto` — probe backend and pick first available model
- `default` — use backend's configured `model` (falls back to auto)

## Usage

```python
from appinfra.log import Logger
from llm_infer.client import Factory

lg = Logger("my-app")
factory = Factory(lg)

# Single backend
with factory.openai(base_url="https://api.openai.com/v1") as client:
    response = client.chat([{"role": "user", "content": "Hello"}])

# Multi-backend from config
with factory.from_config(config) as router:
    response = router.chat(messages)
    response = router.chat(messages, backend="fallback")  # Explicit backend
    response = router.chat(messages, model="gpt-4")       # Route by model
```

### Async

```python
async with factory.from_config(config) as router:
    response = await router.chat_async(messages)
    async for token in router.chat_stream_async(messages):
        print(token, end="")
```

### With Strategy

Configure routing strategies in the config:

```yaml
# Built-in fallback strategy
strategy:
  type: fallback
  order: [primary, fallback]

# Custom strategy from external package (expects module.Factory class)
strategy:
  factory: myapp.routing
  priority_order: [fast, reliable]
```

```python
# Strategy is loaded automatically from config
router = factory.from_config(config)

# Fallback strategy will retry on transient errors (429, 5xx, timeout)
response = router.chat(messages)
```

## Embeddings

Generate vector embeddings with OpenAI or Google backends.

### Basic Usage

```python
from appinfra.log import Logger
from llm_infer.client.backends import embedding

lg = Logger("my-app")

# OpenAI embeddings
backend = embedding.OpenAIBackend(
    lg,
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="text-embedding-3-small",
)

with backend:
    result = backend.embed("Hello world")
    print(f"Dimensions: {len(result.embedding)}")  # 1536
    print(f"Tokens: {result.prompt_tokens}")       # from API response

    # Batch embedding
    results = backend.embed_batch(["Text one", "Text two", "Text three"])
```

### Google Embeddings

```python
from llm_infer.client.backends import embedding
from llm_infer.client.backends.providers.google import GoogleEmbeddingTaskType

backend = embedding.GoogleBackend(
    lg,
    api_key="AIza...",
    model="gemini-embedding-001",
    task_type=GoogleEmbeddingTaskType.RETRIEVAL_DOCUMENT,
)

with backend:
    result = backend.embed("Hello world")
    print(f"Dimensions: {len(result.embedding)}")  # 3072
    print(f"Tokens: {result.prompt_tokens}")       # None (use count_tokens)
```

### With Retry

Wrap a backend with `EmbeddingClient` for retry on transient errors:

```python
from llm_infer.client import EmbeddingClient
from llm_infer.client.backends import RetryConfig, embedding

backend = embedding.OpenAIBackend(...)
client = EmbeddingClient(lg, backend, retry=RetryConfig(timeout=120.0))

with client:
    result = client.embed("Hello world")  # Retries on 429, 5xx, timeout
```

### Token Counting

For cost tracking, use `count_tokens()`:

```python
# OpenAI: uses tiktoken locally (no API call)
tokens = backend.count_tokens("Hello world")

# Google: calls countTokens API
tokens = backend.count_tokens("Hello world")

# Convenience pattern for either backend
result = backend.embed(text)
tokens = result.prompt_tokens or backend.count_tokens(text)
```

| Backend | `prompt_tokens` in result | `count_tokens()` |
|---------|---------------------------|------------------|
| OpenAI  | ✓ (from API response)     | tiktoken (local) |
| Google  | None                      | countTokens API  |

### Async

```python
async with backend:
    result = await backend.embed_async("Hello world")
    results = await backend.embed_batch_async(["One", "Two"])
    tokens = await backend.count_tokens_async("Hello world")
```
