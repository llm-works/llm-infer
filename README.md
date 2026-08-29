# llm-infer

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Coverage](https://img.shields.io/badge/coverage-53%25-yellow.svg)
[![Typed](https://img.shields.io/badge/typed-PEP%20561-brightgreen.svg)](https://peps.python.org/pep-0561/)
[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-brightgreen)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/llm-works/llm-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/llm-works/llm-infer/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llm-infer.svg)](https://pypi.org/project/llm-infer/)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

Unified CLI and client library for local LLM inference. Wraps Ollama, vLLM, and a native engine
behind a single interface.

**Components:**

- **CLI & Server** - Single command to serve models via Ollama, vLLM, or native torch engine
- **Client Package** - Standard interface to multiple LLM backends (OpenAI, Anthropic, local servers)
- **Native Engine** - Custom torch implementation for learning and experimentation

## Quick Start

Serve a model locally on Ollama (the simplest path — CPU or GPU, no local
model weights to manage):

```bash
# 1. Install the Ollama binary (once per machine)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the model
ollama pull qwen2.5:0.5b

# 3. Install llm-infer and serve — llm-infer manages the ollama daemon
pip install 'llm-infer[runtime]'
llm-infer serve --model qwen2.5:0.5b

# 4. Query from another terminal
llm-infer query "What is the capital of France?"
```

See [docs/usage.md](docs/usage.md) for the per-engine walkthroughs (Ollama,
vLLM, native, PEFT) and [llm_infer/etc/README.md](llm_infer/etc/README.md)
for the bundled configuration and customization patterns.

## Client Package

`llm_infer.client` is a Python client library for LLM inference with a unified interface across
backends. Built for autonomous agents and production use. Combined with `llm-infer serve`, the
same package covers both self-hosted inference (vLLM, Ollama, native) and multi-provider routing:
serve one or more local models behind an OpenAI-compatible endpoint, then route across those
plus cloud providers from a single client.

- **Multiple backends** - OpenAI, Anthropic, Google Gemini, Vertex AI (OpenAI-compat and native
  REST), and any OpenAI-compatible API (vLLM, Ollama, llm-infer server)
- **Sync, async, streaming** - All execution modes supported
- **Rate limiting** - Per-backend request throttling
- **Retry with backoff** - Configurable exponential backoff on transient errors
- **Multi-backend routing** - `LLMRouter` with pluggable `RoutingStrategy` and lazy model discovery
- **Cross-provider fallback** - `FallbackClient` with chained pairs and `model@backend` pinning;
  automatic escalation from exhausted 429 retries to a fallback model
- **Embeddings** - `EmbeddingClient` for OpenAI and Google (AI Studio + Vertex) with the same
  retry/callback contract
- **Structured callbacks** - Six lifecycle hooks (`on_request`, `on_response`, `on_retry`,
  `on_error`, `on_before_send`, `on_after_send`) for cost tracking, tracing, and metrics
- **Extensible** - Register custom backends via `Factory.register()`

```python
from appinfra.log import Logger
from llm_infer.client import Factory, FallbackClient

lg = Logger("my-app")
factory = Factory(lg)

with factory.openai(base_url="http://localhost:8000/v1") as client:
    response = client.chat(
        messages=[{"role": "user", "content": "Hello!"}],
        system="You are a helpful assistant.",
    )
    print(response.content)

# Streaming
with factory.openai(base_url="http://localhost:8000/v1") as client:
    messages = [{"role": "user", "content": "Hello!"}]
    for token in client.chat_stream(messages):
        print(token, end="", flush=True)

# Async
async with factory.openai(base_url="http://localhost:8000/v1") as client:
    messages = [{"role": "user", "content": "Hello!"}]
    response = await client.chat_async(messages)

# Multi-backend router with cross-provider fallback
router = factory.from_config(load_config())  # -> LLMRouter
client = FallbackClient(lg, router, fallbacks={"gpt-4o": "claude-sonnet-4-20250514"})
response = client.chat(messages, model="gpt-4o")
```

### Protocol Extensions

The server extends the OpenAI chat completions API:

**Request** - adds `think` and `adapter` fields:
```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "What is 15 * 23?"}],
  "think": true,
  "adapter": "my-lora-adapter"
}
```

**Response** - adds `thinking` in message and `adapter` metadata:
```json
{
  "id": "chatcmpl-123",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "345",
      "thinking": "Let me calculate step by step..."
    }
  }],
  "adapter": {
    "requested": "my-lora-adapter",
    "actual": "my-lora-adapter",
    "fallback": false
  }
}
```

The client library exposes these as keyword arguments:

```python
response = client.chat(messages, think=True, adapter="my-adapter")
print(response.thinking)  # Reasoning content
print(response.content)  # Final answer
```

### Multiple Backends

```python
# Anthropic
async with factory.anthropic(default_model="claude-sonnet-4-20250514") as client:
    response = await client.chat_async(messages)

# OpenAI
with factory.openai(base_url="https://api.openai.com/v1", api_key="sk-...") as client:
    response = client.chat(messages)

# Google Gemini (OpenAI-compatible endpoint; auto-selects GeminiBackend)
with factory.openai(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    api_key="AIza...",
    default_model="gemini-2.5-flash",
) as client:
    response = client.chat(messages)

# Vertex AI native REST (cachedContents + generateContent) via config
# config = {"backends": {"vertex": {"type": "vertex_native", "project": "...", ...}}}
vertex_backends = factory.vertex_natives_from_config(config)
```

## Engines

| Engine | Description | Install |
|--------|-------------|---------|
| `ollama` (default) | Wraps Ollama server | [ollama.com](https://ollama.com) |
| `vllm` | vLLM Python API | `pip install vllm` |
| `vllm-server` | vLLM HTTP subprocess | `pip install vllm` |
| `native` | Custom torch implementation | `pip install llm-infer[runtime]` |

```bash
llm-infer serve --model qwen2.5:7b                          # Ollama
llm-infer serve --engine vllm --model-path /path/to/model   # vLLM
llm-infer serve --engine native --model-path /path/to/model # Native
```

### Native Engine

The native engine is a from-scratch torch implementation with PagedAttention and FlashInfer. Useful
for learning how LLM inference works or experimenting with custom modifications.

```bash
pip install llm-infer[runtime]
llm-infer serve --engine native --model-path /path/to/model
```

## Configuration

```yaml
# etc/llm-infer.yaml
backends:
  engine: ollama

models:
  locations:
    - /path/to/models
  selection:
    generate:
      default: qwen2.5-7b
    embed:
      default: bge-small-en-v1.5

api:
  host: 0.0.0.0
  port: 8000
```

Per-model overrides in `etc/models.yaml` — optional; the shipped catalog is
empty and unlisted models still serve (Ollama passes through, vLLM/native
resolve via `--model-path` or `locations:`). Add entries when you want
per-model tuning:

```yaml
models:
  qwen2.5-7b:
    max_model_len: 8192
    vllm:
      enforce_eager: true

  qwen2.5:7b:
    ollama: qwen2.5:7b  # Ollama model name mapping
```

The bundled `etc/` ships inside the wheel; override individual settings
with `--etc-dir /path/to/custom/etc/` or `-o key=value`. See
[llm_infer/etc/README.md](llm_infer/etc/README.md).

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completion (OpenAI-compatible) |
| `POST /v1/completions` | Text completion (OpenAI-compatible) |
| `GET /v1/models` | List available models |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |

## Installation

```bash
pip install llm-infer              # Client only
pip install llm-infer[anthropic]   # With Anthropic support
pip install llm-infer[saia]        # With llm-saia integration
pip install llm-infer[runtime]     # With native engine (torch)
```

## License

Apache License 2.0

Maintained by [LLM Works LLC](https://llm-works.ai) and contributors.
