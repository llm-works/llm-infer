# llm-infer

[![PyPI](https://img.shields.io/pypi/v/llm-infer.svg)](https://pypi.org/project/llm-infer/)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
[![CI](https://github.com/serendip-ml/llm-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/serendip-ml/llm-infer/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

Unified CLI and client library for local LLM inference. Wraps Ollama, vLLM, and a native engine
behind a single interface.

**Components:**

- **CLI & Server** - Single command to serve models via Ollama, vLLM, or native torch engine
- **Client Package** - Standard interface to multiple LLM backends (OpenAI, Anthropic, local servers)
- **Native Engine** - Custom torch implementation for learning and experimentation

## Quick Start

```bash
pip install llm-infer

# With Ollama (https://ollama.ai)
ollama pull qwen2.5:0.5b
llm-infer serve --model qwen2.5:0.5b

# Query
llm-infer query "What is the capital of France?"
```

## Client Package

`llm_infer.client` is a Python client library for LLM inference with a unified interface across
backends. Built for autonomous agents and production use:

- **Multiple backends** - OpenAI, Anthropic, and any OpenAI-compatible API
- **Sync, async, streaming** - All execution modes supported
- **Rate limiting** - Per-backend request throttling
- **Retry with backoff** - Configurable exponential backoff on failures
- **Model routing** - Route requests to backends by model name
- **Extensible** - Register custom backends via `Factory.register()`

```python
from appinfra.log import Logger
from llm_infer.client import Factory

lg = Logger("my-app")
factory = Factory(lg)

with factory.openai(base_url="http://localhost:8000/v1") as client:
    response = client.chat(
        messages=[{"role": "user", "content": "Hello!"}],
        system="You are a helpful assistant.",
    )
    print(response.content)

# Streaming
with factory.openai() as client:
    for token in client.chat_stream(messages):
        print(token, end="", flush=True)

# Async
async with factory.openai() as client:
    response = await client.chat_async(messages)
```

### Protocol Extensions

The server extends the OpenAI chat completions API:

**Request** - adds `think` and `adapter_id` fields:
```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "What is 15 * 23?"}],
  "think": true,
  "adapter_id": "my-lora-adapter"
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
print(response.content)   # Final answer
```

### Multiple Backends

```python
# Anthropic
async with factory.anthropic(model="claude-sonnet-4-20250514") as client:
    response = await client.chat_async(messages)

# OpenAI
with factory.openai(base_url="https://api.openai.com/v1", api_key="sk-...") as client:
    response = client.chat(messages)
```

## Engines

| Engine | Description | Install |
|--------|-------------|---------|
| `ollama` (default) | Wraps Ollama server | [ollama.ai](https://ollama.ai) |
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

Per-model overrides in `etc/models.yaml`:

```yaml
models:
  qwen2.5-7b:
    max_model_len: 8192
    vllm:
      enforce_eager: true

  qwen2.5:7b:
    ollama: qwen2.5:7b  # Ollama model name mapping
```

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
pip install llm-infer[runtime]     # With native engine (torch)
```

## License

Apache License 2.0
