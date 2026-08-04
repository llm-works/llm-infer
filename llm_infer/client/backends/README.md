# Backends

Backend implementations for LLM chat and embedding APIs.

## Structure

```
backends/
├── base.py          # Abstract base class for chat backends
├── embedding.py     # Abstract base class for embedding backends
├── context.py       # BackendContext, RetryConfig
├── factory.py       # BackendFactory (creates backends from config)
├── mixins.py        # AsyncRequestTrackingMixin
├── provider.py      # Provider enum, ProviderDetector
├── vertex_native.py # NativeVertexBackend (sibling — cache lifecycle, not chat())
└── providers/       # Concrete provider implementations
```

## Chat Backends

All chat backends inherit from `Backend` (base.py) and implement:
- `chat()` / `chat_async()` - non-streaming
- `chat_stream()` / `chat_stream_async()` - streaming

```python
from llm_infer.client.backends import OpenAICompatibleBackend, BackendContext

backend = OpenAICompatibleBackend(
    lg=logger,
    name="openai",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    default_model="gpt-4o",
)

response = backend.chat(ChatRequest(messages=[...]))
```

## Embedding Backends

Embedding backends inherit from `embedding.Backend` and implement:
- `embed()` / `embed_async()` - single text
- `embed_batch()` / `embed_batch_async()` - multiple texts
- `count_tokens()` / `count_tokens_async()` - token counting

```python
from llm_infer.client.backends import embedding

backend = embedding.OpenAIBackend(
    lg=logger,
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="text-embedding-3-small",
)

result = backend.embed("Hello world")
# result.embedding: list[float]
# result.prompt_tokens: int | None
```

## Token Counting

Embedding backends provide `count_tokens()` for cost tracking:

| Backend | `prompt_tokens` in result | `count_tokens()` |
|---------|---------------------------|------------------|
| OpenAI  | ✓ (from API)              | tiktoken (local) |
| Google  | None                      | countTokens API  |

```python
result = backend.embed(text)
tokens = result.prompt_tokens or backend.count_tokens(text)
```

## Available Backends

**Chat:**
- `OpenAICompatibleBackend` - OpenAI, vLLM, any /v1/chat/completions
- `AnthropicBackend` - Native Claude API
- `GeminiBackend` - Google Gemini (OpenAI-compatible wrapper)

**Embedding:**
- `embedding.OpenAIBackend` - OpenAI /v1/embeddings
- `embedding.GoogleBackend` - Google Generative AI embedContent

**Native (non-chat lifecycle):**
- `NativeVertexBackend` - Vertex REST for `cachedContents` + `generateContent`

See `providers/README.md` for provider-specific details.

## NativeVertexBackend

Sibling to `Backend` — does NOT inherit from it and is not routable through
`BackendFactory`. The Vertex OpenAI-compat surface silently ignores the
`cachedContent` request field, so explicit context caching requires speaking
Vertex's native REST directly. Chat-shaped abstract methods don't fit a cache
lifecycle (allocate → reference N times → delete), and the caller owns that
lifecycle explicitly.

```python
from llm_infer.client.backends import NativeVertexFactory

backend = NativeVertexFactory(lg).create(
    config,  # DotDict — same yaml block shape as the Vertex OpenAI-compat backend
    project="my-gcp-project",
    region="us-central1",
)

name, usage = await backend.cache_create(
    model="gemini-2.5-flash-lite",
    system="you are careful",
    user_text="<large shared prefix>",
    ttl_seconds=600,
)
try:
    for slice_text in slices:
        payload = await backend.generate_content(
            model="gemini-2.5-flash-lite",
            cache_name=name,
            user_text=slice_text,
            generation_config={"temperature": 0.2, "maxOutputTokens": 8192},
        )
        ...
finally:
    await backend.cache_delete(name)
```

Reuses the shared substrate: `RetryHelper` (429/5xx/timeouts), `RateLimiter`,
`AuthProvider`, and `BackendContext.request_timeout`. Errors surface as
`BackendRequestError` / `BackendUnavailableError`. `project` and `region` are
explicit kwargs so a single Vertex backend yaml block can be reused across
the chat path (`GeminiBackend`) and the native-cache path without duplication.
Configure `service_tier: priority` in yaml to send the Vertex Priority header;
observed downgrades log at WARN.
