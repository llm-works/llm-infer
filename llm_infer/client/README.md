# llm_infer.client

Multi-backend LLM client with sync/async support, routing strategies, model
discovery, and structured lifecycle callbacks. Backends: OpenAI, Anthropic,
Google (AI Studio + Vertex OpenAI-compat), Vertex AI native REST, and any
OpenAI-compatible server (vLLM, Ollama, `llm-infer serve`).

## Core Types

### Chat

| Module | Class | Purpose |
|--------|-------|---------|
| `base.py` | `ChatClient` | Abstract base class for all chat clients |
| `client.py` | `LLMClient` | Single-backend client with rate limiting and backoff |
| `router.py` | `LLMRouter` | Multi-backend router with strategy-based routing |
| `fallback.py` | `FallbackClient` | Cross-provider model fallback on transient errors |
| `factory.py` | `Factory` | Creates clients/routers from config |

### Embedding

| Module | Class | Purpose |
|--------|-------|---------|
| `embedding.py` | `EmbeddingClient` | Embedding client with retry and callbacks |
| `backends/embedding.py` | `Backend` | Abstract base class for embedding backends |
| `backends/embedding.py` | `EmbeddingResult` | Result with embedding vector, model, token count |

## Routing System

| Module | Purpose |
|--------|---------|
| `strategy.py` | `RoutingStrategy` protocol and core types (`RoutingContext`, `RoutingDecision`, `RoutingResult`, `DecisionType`) |
| `strategies.py` | Built-in strategy: `DefaultStrategy` (round-robin over enabled backends) |
| `discovery.py` | `ModelDiscovery` — lazy model probing and model-to-backend resolution |
| `router_helper.py` | Internal helpers for routing loop (setup, success/error handling) |

## Other Modules

| Module | Purpose |
|--------|---------|
| `types.py` | `ChatRequest`, `ChatResponse`, `ChatStream`/`ChatStreamSync`, `AdapterInfo`, `LLMCallbacks`, `SendContext`, `SendResult`, `Provider` |
| `errors.py` | `BackendError`, `BackendRequestError`, `BackendTimeoutError`, `BackendUnavailableError`, `ConfigError`, `FallbackAmbiguityError`, `ModelConflictError` |
| `retry.py` | `RetryBase`, `RetryHelper` — exponential backoff for transient errors |
| `bound.py` | `BoundChatClient` — wraps a `ChatClient` to inject context (used by `with_callbacks()`) |
| `embedding.py` | `EmbeddingClient` (see [Embeddings](#embeddings)) |
| `backends/` | Chat backends (OpenAI, Anthropic, Gemini, native Vertex) and embedding backends (OpenAI, Google) |
| `backends/auth.py` | `AuthProvider` (bearer/GCP SA) — resolves the `auth:` block on any backend |
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
    api_key: sk-...                 # Optional; str or SecretStr
    # provider: openai              # Optional; overrides URL-based auto-detect.
                                    # Valid: openai, anthropic, xai, google, local, unknown.
                                    # Mismatch with URL-detected value logs a WARN.
    # auth:                         # Optional; alternative to api_key for
    #   mode: gcp_sa                # non-bearer schemes (e.g., GCP SA for Vertex).
    #   credentials_path: /path/to/sa.json
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
| `vertex_native` | Vertex AI native REST (`cachedContents` + `generateContent`). Sibling to `Backend` — built via `Factory.vertex_natives_from_config()`, not routed through `LLMRouter`. |

When `provider: google` (or auto-detected from a `googleapis.com` `base_url`),
the OpenAI-compatible path returns a specialized `GeminiBackend` that also
honors an optional `service_tier` field (e.g. `priority`).

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
    response = router.chat(messages, model="gpt-4")  # Route by model
```

### Async

```python
async with factory.from_config(config) as router:
    response = await router.chat_async(messages)
    async for token in router.chat_stream_async(messages):
        print(token, end="")
```

### With Strategy

`LLMRouter` resolves each request to a backend via a `RoutingStrategy`. The only
built-in strategy is `default` (round-robin over enabled backends matching the
requested model). Custom strategies live in external packages.

```yaml
# Built-in default strategy (round-robin) — also the implicit default
strategy:
  type: default

# Custom strategy from external package (module must export a Factory class)
strategy:
  factory: myapp.routing
  priority_order: [fast, reliable]
```

For cross-provider model fallback on transient errors (5xx, timeout,
unavailable), wrap the router with [`FallbackClient`](#cross-provider-fallback)
instead of writing a custom strategy.

### Cross-Provider Fallback

`FallbackClient` wraps an `LLMRouter` and retries with an equivalent model on
transient errors. Fallback pairs chain implicitly:

```python
from llm_infer.client import Factory, FallbackClient

router = Factory(lg).from_config(config)
fallbacks = {
    "gpt-4o": "claude-sonnet-4-20250514",
    "claude-sonnet-4-20250514": "gemini-2.0-pro",  # chains: gpt-4o -> claude -> gemini
}
client = FallbackClient(lg, router, fallbacks)

response = client.chat(messages, model="gpt-4o")
```

Rate-limit errors (429) are first retried with backoff against the same model
(per the backend's `retry` config); once that budget is exhausted, fallback
engages like any other transient error. A backend configured without `retry`
falls back on its first transient error — a warning is logged at construction.
Cycles (A->B->A) are detected and retried round-robin until one succeeds.

#### Pinning a fallback to a backend (`model@backend`)

Keys and values may take the form `model@backend` to pin a fallback step to a
specific backend. `@` is used rather than `/` to avoid clashing with
OpenRouter's `provider/model` names. When the same model is served by more
than one backend, `FallbackClient` raises `FallbackAmbiguityError` at
construction and requires the caller to disambiguate:

```python
# gpt-4o is served by two backends: qualify to pick one
fallbacks = {
    "gpt-4o@openai_primary": "gpt-4o@openai_backup",
    "gpt-4o@openai_backup": "claude-sonnet-4-20250514",
}
```

Lookup is qualified-first: at each step `FallbackClient` first tries
`f"{model}@{resolved_backend}"` before falling back to the bare key, so bare
and qualified entries mix cleanly within a single map. Ambiguity is checked
eagerly at `__init__` (backend catalogs are probed once and cached), which
surfaces misconfiguration at wire-up instead of at 3 AM.

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
    print(f"Tokens: {result.prompt_tokens}")  # from API response

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
    print(f"Tokens: {result.prompt_tokens}")  # None (use count_tokens)
```

### With Retry

Wrap a backend with `EmbeddingClient` for retry on transient errors:

```python
from llm_infer.client import EmbeddingClient
from llm_infer.client.backends import RetryConfig, embedding

backend = embedding.OpenAIBackend(...)
client = EmbeddingClient(
    lg,
    backend,
    retry=RetryConfig(timeout=120.0),
    model="text-embedding-3-small",  # client-level default
    dimensions=384,  # client-level default (per-call overrides win)
)

with client:
    result = client.embed("Hello world")  # uses defaults
    result = client.embed("Hello world", dimensions=1536)  # per-call override
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

## Observability

Each `ChatRequest` carries an 8-hex-char `id` for log correlation. `LLMClient`
logs request/response (model, backend, timing, tokens) and `RetryHelper` logs
retry attempts at DEBUG level — set your logger accordingly to trace a request
through retries and fallbacks:

```python
import logging

logging.getLogger("my-app").setLevel(logging.DEBUG)
```

For structured observability hooks (cost tracking, tracing, latency
histograms), use `LLMClient.with_callbacks()` or pass `callbacks=` to any
`Factory` method. `LLMCallbacks` exposes six hooks split across two levels:

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
| `on_before_send` | `(SendContext)` | Immediately before HTTP request; includes `attempt`, `retry_reason`, `delay_seconds` |
| `on_after_send` | `(SendContext, SendResult)` | After HTTP response/error; includes `status_code`, `error`, `elapsed_ms` |

```python
from llm_infer.client import Factory, LLMCallbacks


def record_latency(ctx, result):
    metrics.histogram(
        "llm.http_ms",
        result.elapsed_ms,
        tags={"backend": ctx.backend, "attempt": ctx.attempt},
    )


callbacks = LLMCallbacks(on_after_send=record_latency)
with Factory(lg).openai(callbacks=callbacks) as client:
    client.chat(messages)
```

`EmbeddingClient` mirrors this surface via `EmbeddingCallbacks` and its own
`with_callbacks()`.

### Configuration Hot-Reload

llm-infer does not ship a control plane. When the backend catalog needs to
change without restarting the process, wire `appinfra`'s YAML watcher to a
handler that rebuilds the router from the new config and atomically swaps it
into your request path; discard the previous `LLMRouter` after in-flight
requests drain.
