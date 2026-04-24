# llm_infer.client

Multi-backend LLM client with sync/async support, routing strategies, and model discovery.

## Core Types

| Module | Class | Purpose |
|--------|-------|---------|
| `base.py` | `ChatClient` | Abstract base class for all chat clients |
| `client.py` | `LLMClient` | Single-backend client with rate limiting and backoff |
| `router.py` | `LLMRouter` | Multi-backend router with strategy-based routing |
| `factory.py` | `Factory` | Creates clients/routers from config |

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
| `backends/` | Backend implementations (OpenAI-compatible, Anthropic) |
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

# Custom strategy from external package
strategy:
  factory: mypackage.billing:BudgetStrategyFactory
  budget_limit: 100
```

```python
# Strategy is loaded automatically from config
router = factory.from_config(config)

# Fallback strategy will retry on transient errors (429, 5xx, timeout)
response = router.chat(messages)
```
