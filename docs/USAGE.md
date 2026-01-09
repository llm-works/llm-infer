# Usage Guide

This document covers configuration, CLI commands, and API usage.

## Quick Start

```bash
# Start the server
llm-infer serve --model-path /path/to/model

# Query in another terminal
llm-infer query "What is the capital of France?"
```

## Configuration

Configuration uses YAML files with the primary config at `etc/inference.yaml`.

### Configuration Priority

Model selection follows this priority:
1. CLI `--model-path` (absolute path)
2. CLI `--model` (model name, resolved from `models.location`)
3. Selection file (`models.selection.path`)
4. Default (`models.selection.default`)

### Main Configuration File

```yaml
# etc/inference.yaml

# Model location and selection
models:
  location: /path/to/models/directory
  selection:
    path: ~/ops/models/selected.yaml  # Optional ops-controlled selection
    default: qwen2.5-1.5b              # Fallback model name

# Backend selection
backends:
  engine: native      # native | vllm
  model: native       # native | gptqmodel (only if engine=native)
  linear: marlin      # pytorch | marlin (only if model=native)

# Engine-specific settings
engines:
  native:
    num_blocks: 1024
    block_size: 16
    max_batch_size: 4
    device: cuda
    dtype: float16
    attention_backend: flashinfer  # auto | flashinfer | naive
    torch_compile: false
    warmup: true

  vllm:
    gpu_memory_utilization: 0.9
    max_model_len: 16384
    tensor_parallel_size: 1
    max_num_seqs: 256
    enable_prefix_caching: true
    warmup: true

# Request dispatch
dispatch:
  handler: bounded       # sequential | bounded | batching
  max_pending: 10
  poll_timeout: 0.01
  batch_streaming: true

# API server
api:
  host: 0.0.0.0
  port: 8000
  response_timeout: 60.0
```

### Engine Selection

**Native Engine** - For learning and research:
- Full visibility into the inference pipeline
- Custom implementation using PyTorch and FlashInfer
- Best for understanding how inference works

**vLLM Engine** - For production:
- Production-grade PagedAttention and continuous batching
- Tensor parallelism for multi-GPU setups
- Prefix caching and speculative decoding
- Drop-in replacement via `backends.engine: vllm`

### Quantization

The system auto-detects quantization from model config. Supported formats:

| Format | Bits | Description |
|--------|------|-------------|
| AWQ | 4 | Activation-aware Weight Quantization |
| FP8 | 8 | 8-bit floating point |

For native engine, kernel selection:
- `marlin`: Fast Marlin kernels (recommended)
- `pytorch`: Reference PyTorch implementation

### Dispatch Handlers

| Handler | Behavior |
|---------|----------|
| `sequential` | One request at a time (simplest) |
| `bounded` | Bounded queue with backpressure (recommended) |
| `batching` | Full continuous batching |

### Model-Specific Configuration

Per-model settings in `etc/models.yaml`:

```yaml
models:
  qwen3-4b-instruct:
    system_prompt: "You are a helpful assistant."
    think:
      default: false
      enable_suffix: " /think"
      disable_suffix: " /no_think"
      system_prompt: "Reason step by step in <think> tags."
      tags:
        open: ["<think>"]
        close: ["</think>"]

defaults:
  system_prompt: null
  think:
    default: false
```

## CLI Commands

### `llm-infer serve`

Start the inference server.

```bash
# With config file
llm-infer serve --config etc/inference.yaml

# With model path
llm-infer serve --model-path /path/to/model

# With model name (resolved from models.location)
llm-infer serve --model qwen2.5-1.5b
```

Options:
- `--config, -c`: Config file path (default: `etc/inference.yaml`)
- `--model-path`: Direct path to model directory
- `--model, -m`: Model name to load
- `--port, -p`: Override server port
- `--host`: Override server host

### `llm-infer query`

Send queries to a running server.

```bash
# Simple query
llm-infer query "What is 2+2?"

# With streaming
llm-infer query --stream "Explain quantum computing"

# Custom parameters
llm-infer query --max-tokens 200 --temperature 0.7 "Write a poem"

# Target specific server
llm-infer query --url http://localhost:8000 "Hello"
```

Options:
- `--stream, -s`: Stream response tokens
- `--max-tokens`: Maximum tokens to generate
- `--temperature`: Sampling temperature (0-2)
- `--url`: Server URL (default: `http://localhost:8000`)

### `llm-infer metrics`

Display server metrics.

```bash
llm-infer metrics
```

### `llm-infer compat`

Compatibility specification tools.

```bash
# Generate compat spec
llm-infer compat generate

# Check existing spec
llm-infer compat check
```

## API Reference

The server provides OpenAI-compatible endpoints.

### Chat Completions

```http
POST /v1/chat/completions
```

Request:
```json
{
  "model": "default",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 1.0,
  "stream": false,
  "stop": ["\n\n"]
}
```

Response:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "qwen2.5-1.5b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20
  }
}
```

### Text Completions (Legacy)

```http
POST /v1/completions
```

Request:
```json
{
  "model": "default",
  "prompt": "The capital of France is",
  "max_tokens": 50,
  "temperature": 0.0
}
```

### Streaming

Set `stream: true` in request. Response uses Server-Sent Events:

```
data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"!"}}]}

data: [DONE]
```

### Model List

```http
GET /v1/models
```

Response:
```json
{
  "object": "list",
  "data": [{
    "id": "qwen2.5-1.5b",
    "object": "model",
    "created": 1234567890,
    "owned_by": "local"
  }]
}
```

### Health Check

```http
GET /health
```

Response:
```json
{"status": "healthy"}
```

### Metrics

```http
GET /metrics
```

Returns Prometheus-format metrics.

## Using with OpenAI SDK

The API is compatible with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # No auth required
)

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

Streaming:

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `HF_TOKEN` | HuggingFace token for gated models |

## Performance Tuning

### Native Engine

1. **Use FlashInfer**: Set `attention_backend: flashinfer` (default)
2. **Increase batch size**: Higher `max_batch_size` for throughput
3. **Adjust blocks**: More `num_blocks` for longer contexts
4. **Enable torch.compile**: For reduced CPU overhead (incompatible with FlashInfer)

### vLLM Engine

1. **Memory utilization**: Increase `gpu_memory_utilization` (default 0.9)
2. **Prefix caching**: Enable for repeated prefixes
3. **Tensor parallelism**: Use multiple GPUs with `tensor_parallel_size`
4. **Speculative decoding**: Configure draft model for faster decoding

## Debugging

### Naive Attention Backend

For debugging attention issues:
```yaml
engines:
  native:
    attention_backend: naive
```

This uses pure PyTorch implementation instead of FlashInfer.

### Verbose Logging

Configure logging levels in config:
```yaml
third_party_logging:
  torch: debug
  transformers: info

logging:
  level: debug
```

### Memory Stats

Check memory usage via metrics endpoint or in code:
```python
engine.memory_stats()
# {'allocated': ..., 'reserved': ..., 'peak': ..., ...}
```
