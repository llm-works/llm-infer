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
- `OpenAICompatibleBackend` - OpenAI, Azure, vLLM, any /v1/chat/completions
- `AnthropicBackend` - Native Claude API
- `GeminiBackend` - Google Gemini (OpenAI-compatible wrapper)

**Embedding:**
- `embedding.OpenAIBackend` - OpenAI /v1/embeddings
- `embedding.GoogleBackend` - Google Generative AI embedContent

See `providers/README.md` for provider-specific details.
