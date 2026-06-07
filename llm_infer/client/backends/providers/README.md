# Provider Backends

Concrete backend implementations for specific LLM providers.

## Chat Backends

### OpenAICompatibleBackend

Works with OpenAI, Azure OpenAI, vLLM, and any server implementing `/v1/chat/completions`.

```python
from llm_infer.client.backends import OpenAICompatibleBackend

backend = OpenAICompatibleBackend(
    lg=logger,
    name="openai",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    default_model="gpt-4o",
)
```

Features:
- Streaming and non-streaming
- Tool/function calling
- Thinking mode (for o1/o3 models)
- Structured output (response_format)

### AnthropicBackend

Native Claude API backend using the official Anthropic SDK.

```python
from llm_infer.client.backends import AnthropicBackend

backend = AnthropicBackend(
    lg=logger,
    name="claude",
    api_key="sk-ant-...",
    default_model="claude-sonnet-4-20250514",
)
```

Requires: `pip install llm-infer[anthropic]`

Key differences from OpenAI:
- System messages passed separately (not in messages array)
- Thinking uses `extended_thinking` with `budget_tokens`
- Native Anthropic tool format

### GeminiBackend

Google Gemini via OpenAI-compatible API with normalization.

```python
from llm_infer.client.backends import GeminiBackend

backend = GeminiBackend(
    lg=logger,
    name="gemini",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    api_key="AIza...",
    default_model="gemini-2.5-flash",
)
```

Normalizes Gemini 2.5 behavior:
- Thinking disabled by default (other providers require opt-in)
- `think=True` enables thinking via `reasoning_effort`

## Embedding Backends

### OpenAIEmbeddingBackend

OpenAI-compatible `/v1/embeddings` API.

```python
from llm_infer.client.backends.providers.openai import OpenAIEmbeddingBackend

backend = OpenAIEmbeddingBackend(
    lg=logger,
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="text-embedding-3-small",
)

result = backend.embed("Hello world")
# result.embedding: 1536 dimensions
# result.prompt_tokens: from API response
```

Token counting uses tiktoken (local, no API call).
Requires: `pip install llm-infer[embedding]`

### GoogleEmbeddingBackend

Google Generative AI embedContent API.

```python
from llm_infer.client.backends.providers.google import (
    GoogleEmbeddingBackend,
    GoogleEmbeddingTaskType,
)

backend = GoogleEmbeddingBackend(
    lg=logger,
    api_key="AIza...",
    model="gemini-embedding-001",
    task_type=GoogleEmbeddingTaskType.RETRIEVAL_DOCUMENT,
)

result = backend.embed("Hello world")
# result.embedding: 3072 dimensions
# result.prompt_tokens: None (use count_tokens())
```

Task types optimize embeddings for specific use cases:
- `RETRIEVAL_DOCUMENT` - for documents in a retrieval corpus
- `RETRIEVAL_QUERY` - for search queries
- `SEMANTIC_SIMILARITY` - for comparing text similarity
- `CLASSIFICATION` - for text classification
- `CLUSTERING` - for clustering similar texts

Token counting requires an API call to `countTokens`.

## Adding New Providers

1. Inherit from `Backend` (chat) or `embedding.Backend` (embeddings)
2. Implement all abstract methods (sync and async)
3. Translate errors to `BackendError` hierarchy
4. Add to `providers/__init__.py` exports
