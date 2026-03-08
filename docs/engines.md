# Inference Engines

llm-infer supports multiple inference backends through a unified interface. Each engine has
different trade-offs for ease of use, performance, and features.

## Quick Reference

| Engine | How It Works | LoRA Support | Best For |
|--------|--------------|--------------|----------|
| `ollama` | HTTP to Ollama server | No | Getting started, CPU inference |
| `vllm-server` | HTTP to `vllm serve` subprocess | Pre-registered only | Production (full optimizations) |
| `vllm` | Python API (in-process) | Dynamic loading | Dynamic LoRA adapters |
| `native` | Custom torch implementation | No | Learning, experimentation |
| `peft` | HuggingFace Transformers + PEFT | PROMPT_TUNING only | Prompt tuning adapters |

## Ollama Engine (Default)

Connects to an [Ollama](https://ollama.com) server via HTTP. llm-infer can auto-start Ollama if not
running.

### How It Works

```text
┌─────────────┐       HTTP       ┌─────────────┐
│  llm-infer  │ ──────────────── │   Ollama    │
│   server    │    /api/chat     │   server    │
└─────────────┘                  └─────────────┘
```

- llm-infer starts its own HTTP server (OpenAI-compatible)
- Requests are proxied to Ollama's HTTP API
- Ollama manages model loading, GPU memory, and inference
- If `auto_start: true`, llm-infer starts Ollama as a subprocess

### Prerequisites

Install Ollama from [ollama.ai](https://ollama.com), then pull a model:

```bash
ollama pull qwen2.5:0.5b
```

### Usage

```bash
# Ollama is the default engine
llm-infer serve --model qwen2.5:0.5b

# Explicit
llm-infer serve --engine ollama --model qwen2.5:0.5b
```

### Configuration

```yaml
# etc/ollama.yaml
host: http://localhost:11434  # Ollama server URL
timeout: 300                   # Request timeout (seconds)
models_path: ~/.ollama/models  # Model storage (sets OLLAMA_MODELS)
auto_start: true               # Start Ollama if not running
keep_alive: 5m                 # How long to keep model loaded
num_ctx: null                  # Context window (null = model default)
num_gpu: null                  # GPU layers (null = auto, 0 = CPU only)
max_concurrent: 4              # Concurrent HTTP requests to Ollama
```

### Model Name Mapping

Ollama uses its own model naming. Map llm-infer names in `etc/models.yaml`:

```yaml
models:
  qwen2.5-0.5b:
    ollama: qwen2.5:0.5b  # Ollama model name
```

### When to Use

- Getting started quickly
- CPU-only machines
- Simple deployments without LoRA adapters

---

## vLLM Server Engine (HTTP API) - Recommended for Production

Runs `vllm serve` as a subprocess and connects via OpenAI-compatible HTTP API. This is the
**recommended production engine** as it includes all vLLM optimizations.

### How It Works

```
┌─────────────┐       HTTP      ┌─────────────┐
│  llm-infer  │ ─────────────── │ vllm serve  │
│   server    │  /v1/chat/...   │ subprocess  │
└─────────────┘                 └─────────────┘
```

- llm-infer starts `vllm serve` as a subprocess (if `auto_start: true`)
- Requests are proxied to vLLM's OpenAI-compatible API
- Process isolation: vLLM crash doesn't take down llm-infer
- Output streams to stderr (visible but separable)

### vLLM Server Optimizations

`vllm serve` runs the full `AsyncLLMEngine` with production optimizations not available in the
Python API:

**Inference optimizations:**
- **Continuous batching** - Dynamic batch formation for maximum throughput
- **Chunked prefill** - Interleave prefill and decode for better latency
- **Prefix caching** - Cache and reuse common prefixes across requests
- **Speculative decoding** - Use draft model for faster generation

**Post-processing (delegated to server):**
- **Tool call parsing** - Server extracts function calls from model output, returns structured
  `tool_calls` in the response (via `--enable-auto-tool-choice` and `--tool-call-parser`)
- **Guided decoding** - Server constrains generation to valid JSON via `response_format`
- **Chat templating** - Server applies the model's chat template

### Prerequisites

```bash
pip install vllm
```

### Usage

```bash
llm-infer serve --engine vllm-server --model-path /path/to/model
```

### Configuration

```yaml
# etc/vllm-server.yaml
host: http://localhost
port: 8001                     # vLLM server port (llm-infer uses 8000)
auto_start: true               # Start vllm serve subprocess
startup_timeout: 300           # Seconds to wait for vLLM to start
timeout: 120                   # Request timeout

# Engine options (use gpu_memory_gb OR gpu_memory_utilization)
gpu_memory_gb: null            # Absolute limit, e.g., 8.0 for 8GB
gpu_memory_utilization: 0.9    # Fraction of VRAM (used if gpu_memory_gb is null)
max_model_len: 16384
tensor_parallel_size: 1
max_num_seqs: 256
enable_prefix_caching: true
dtype: auto

# Dispatch layer concurrency
max_concurrent: 4              # Concurrent HTTP requests to vLLM server

# LoRA configuration
lora:
  enabled: true
  base_path: /path/to/adapters
```

### Parallel Requests

llm-infer sends multiple concurrent HTTP requests to vLLM server, allowing vLLM's continuous
batching to process them together. This achieves near-linear speedup:

- 4 requests completing in ~1.3s instead of ~5s sequential
- Configurable via `max_concurrent` (default: 4)
- Works together with vLLM's `max_num_seqs` for internal batching

### LoRA Limitation

`vllm-server` only supports **pre-registered adapters**:

- Adapters are registered at server startup via `--lora-modules`
- Adapters created after server startup require a server restart
- For dynamic adapter loading, use the `vllm` engine instead

This is a limitation of vLLM's HTTP API, which doesn't expose runtime adapter loading.

### When to Use

- Production deployments (recommended)
- Maximum throughput with all optimizations
- Process isolation needed
- Static adapter configurations

---

## vLLM Engine (Python API)

Uses vLLM's Python `LLM` class directly in the same process. Simpler but **lacks the full
optimizations** of `vllm serve`. Primary advantage: dynamic LoRA adapter loading.

### How It Works

```
┌─────────────────────────────────────┐
│            llm-infer process        │
│  ┌───────────┐    ┌──────────────┐  │
│  │  FastAPI  │────│  vLLM LLM()  │  │
│  │  server   │    │  (in-process)│  │
│  └───────────┘    └──────────────┘  │
└─────────────────────────────────────┘
```

- vLLM's `LLM` class runs in the same Python process
- Direct function calls (no HTTP overhead)
- Simpler sync API without full async engine optimizations

### What's Missing vs vllm-server

The Python `LLM` class is a simplified wrapper. Compared to `vllm serve`, it lacks:

**Inference:**
- Full continuous batching (limited batching via `LLM.generate()`)
- Chunked prefill scheduling
- Some speculative decoding optimizations

**Post-processing:**
- **No tool call support** - `tools`/`tool_choice` params are accepted but ignored; you must parse
  tool calls from the model's text output yourself
- Chat templating must be handled by llm-infer

### Why Use It

The main advantage is **dynamic LoRA adapter loading**:

```python
# Adapters can be loaded at request time
response = client.chat(messages, adapter="my-adapter")
```

This is useful when:
- Adapters are created dynamically (e.g., training pipeline)
- Running e2e tests that create temporary adapters
- Need to switch adapters without server restart

### Prerequisites

```bash
pip install vllm
```

### Usage

```bash
llm-infer serve --engine vllm --model-path /path/to/model
```

### Configuration

```yaml
# etc/vllm.yaml
gpu_memory_gb: null            # Absolute limit, e.g., 8.0 for 8GB
gpu_memory_utilization: 0.9    # Fraction of VRAM (used if gpu_memory_gb is null)
max_model_len: 16384
tensor_parallel_size: 1
max_num_seqs: 256
enable_prefix_caching: true
dtype: auto

# LoRA configuration
lora:
  enabled: true
  max_loras: 4
  max_lora_rank: 128
  base_path: /path/to/adapters
```

### When to Use

- Dynamic LoRA adapter loading required
- E2E tests with temporary adapters
- Simpler deployment (single process)

---

## Native Engine

Custom torch implementation with PagedAttention and FlashInfer. Full visibility into the inference
pipeline for learning and experimentation.

### How It Works

```
┌─────────────────────────────────────────────────┐
│               llm-infer process                 │
│  ┌─────────┐    ┌─────────────────────────────┐ │
│  │ FastAPI │────│     Native Engine           │ │
│  │ server  │    │  ┌─────────┐ ┌────────────┐ │ │
│  └─────────┘    │  │Scheduler│ │Transformer │ │ │
│                 │  └─────────┘ │  + KVCache │ │ │
│                 │              └────────────┘ │ │
│                 └─────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

- Pure Python/PyTorch implementation
- PagedAttention for memory efficiency
- FlashInfer or naive attention backends
- Supports AWQ and FP8 quantization

### Prerequisites

```bash
pip install llm-infer[runtime]
```

Requires CUDA GPU.

### Usage

```bash
llm-infer serve --engine native --model-path /path/to/model
```

### Configuration

```yaml
# etc/native.yaml
num_blocks: 1024              # KV cache blocks
block_size: 16                # Tokens per block
max_batch_size: 4             # Maximum batch size
device: cuda                  # Device (cuda only)
dtype: float16                # Model dtype
attention_backend: flashinfer # flashinfer | naive
torch_compile: false          # Enable torch.compile (incompatible with FlashInfer)
warmup: true                  # Warmup on startup
```

### When to Use

- Learning how LLM inference works
- Experimenting with custom modifications
- Debugging inference issues (naive attention backend)

---

## PEFT Engine

Uses HuggingFace Transformers + PEFT library for prompt-learning adapter types that vLLM's
`--enable-lora` doesn't support: PROMPT_TUNING, PREFIX_TUNING, and P_TUNING.

### How It Works

```
┌─────────────────────────────────────────────────┐
│               llm-infer process                 │
│  ┌─────────┐    ┌─────────────────────────────┐ │
│  │ FastAPI │────│       PEFT Engine           │ │
│  │ server  │    │  ┌───────────┐ ┌──────────┐ │ │
│  └─────────┘    │  │Transformers│ │  PEFT   │ │ │
│                 │  │ AutoModel  │ │ Adapter │ │ │
│                 │  └───────────┘ └──────────┘ │ │
│                 └─────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

- Lazy-loads base model on first adapter request (minimizes memory when coexisting with vLLM)
- LRU cache for loaded adapters (configurable size)
- Supports 4-bit quantization via bitsandbytes
- Automatically routes PROMPT_TUNING requests away from vLLM

### Why This Engine Exists

vLLM's `--enable-lora` only supports LoRA adapters. When loading a PROMPT_TUNING adapter, vLLM
fails with:

```text
ValueError: Missing required configuration fields: {'r', 'target_modules', 'lora_alpha'}
```

The PEFT engine handles these non-LoRA adapter types using the PEFT library directly.

### Supported Adapter Types

| Adapter Type | Engine | Notes |
|-------------|--------|-------|
| LoRA | vllm, vllm-server | Use vLLM for best performance |
| PROMPT_TUNING | peft | Soft prompts prepended to input |
| PREFIX_TUNING | peft | Learnable prefix activations |
| P_TUNING | peft | Continuous prompt embeddings |

### Prerequisites

```bash
pip install transformers peft

# Optional: 4-bit quantization
pip install bitsandbytes
```

### Usage

```bash
llm-infer serve --engine peft --model-path /path/to/model
```

### Configuration

```yaml
# etc/peft.yaml
device: cuda                  # Device to load model on
dtype: auto                   # Model dtype (auto, float16, bfloat16)
max_cached_adapters: 4        # LRU cache size for loaded adapters
warmup: true                  # Run warmup on first adapter load
load_in_4bit: false           # Use bitsandbytes 4-bit quantization
adapter_base_path: /path/to/adapters  # Base directory for adapters
```

### Adapter Type Detection

The PEFT engine automatically validates adapter types. When an adapter is requested:

1. Reads `peft_type` from `adapter_config.json`
2. Validates it's a supported prompt-learning type
3. Rejects LoRA adapters with a helpful error directing to vLLM

This prevents accidentally loading LoRA adapters through the slower PEFT engine.

### Memory Management

The engine uses lazy loading and LRU caching:

- **Lazy loading**: Base model only loads on first adapter request
- **LRU cache**: Keeps `max_cached_adapters` adapters in memory
- **Automatic eviction**: Least-recently-used adapter deleted when cache is full

### When to Use

- PROMPT_TUNING, PREFIX_TUNING, or P_TUNING adapters
- Non-LoRA adapter types that vLLM doesn't support
- Research/experimentation with prompt-learning methods

**Note:** For LoRA adapters, always use `vllm` or `vllm-server` for better performance.

---

## Engine Selection Guide

| Scenario | Recommended Engine |
|----------|-------------------|
| Just getting started | `ollama` |
| CPU-only machine | `ollama` |
| Production deployment | `vllm-server` |
| Maximum throughput | `vllm-server` |
| Dynamic LoRA adapters | `vllm` |
| E2E tests with temp adapters | `vllm` |
| Process isolation | `vllm-server` |
| Multi-GPU (tensor parallel) | `vllm-server` |
| PROMPT_TUNING adapters | `peft` |
| Non-LoRA adapter types | `peft` |
| Learning/experimentation | `native` |

## Overriding Engine at Runtime

```bash
# CLI flag overrides config
llm-infer serve --engine vllm-server --model-path /path/to/model

# Or via -o override
llm-infer serve -o backends.engine=vllm-server --model-path /path/to/model
```
