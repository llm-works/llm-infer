# LLM Inference Engine

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x-blue.svg)
![vLLM](https://img.shields.io/badge/backend-vLLM%20%7C%20native-blue.svg)
![Type Hints](https://img.shields.io/badge/type%20hints-100%25-brightgreen.svg)
[![Typed](https://img.shields.io/badge/typed-PEP%20561-brightgreen.svg)](https://peps.python.org/pep-0561/)
[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-brightgreen)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/serendip-ml/llm-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/serendip-ml/llm-infer/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

A readable LLM llm-infer server implementing paged attention and continuous batching.

## What This Is

A research and experimentation platform for LLM inference with a clean and modular architecture.

**For learning**: The codebase is structured to be understandable. If you want to learn how paged
attention, KV caching, continuous batching, or quantized inference actually work, this is a good
place to start. The code prioritizes clarity over micro-optimizations.

**For research**: Full control over the inference pipeline. Modular architecture with clean
interfaces (Python protocols) makes it easy to swap components, add instrumentation, or experiment
with new techniques without fighting the framework.

**For experimentation**: Streamlined developer experience through a CLI tool and YAML configuration.
Go from model weights to running server in minutes, not hours.

**For production**: When you're ready to scale, swap in vLLM as the inference backend. The
OpenAI-compatible API means your application code stays the same—only the backend changes.

## Why This Matters

There's growing evidence that many smaller, specialized models may outperform a single massive
one—at a fraction of the cost. This framework is a step in that direction: making it easy to
experiment with, understand, and deploy efficient models.

## What This Is Not

- **Not a vLLM/TGI replacement at scale**: For large-scale deployments with tensor parallelism across multiple nodes, use vLLM or TGI. This project focuses on single-GPU scenarios.

- **Not a training framework**: This is inference-only. For training or fine-tuning, look elsewhere.

- **Not maximally optimized**: While we use FlashInfer kernels and support quantization, the primary goal is maintainable code. We avoid complexity that yields marginal gains.

## Features

- **Paged Attention**: Efficient KV cache management using block-based memory allocation
- **Continuous Batching**: Dynamic request batching for optimal GPU utilization
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's `/v1/completions` and `/v1/chat/completions`
- **Multiple Quantization Formats**: Support for AWQ (4-bit) and FP8 (8-bit) quantized models
- **Streaming**: Real-time token streaming with SSE
- **FlashInfer Backend**: Optimized CUDA kernels for attention computation

## Supported Models

- Llama (1, 2, 3, 3.1, 3.2, 3.3)
- Mistral
- Qwen (1, 2, 3)
- Granite (3.x)

## Installation

```bash
git clone https://github.com/serendip-ml/inference.git
cd inference
pip install -e .
```

### Environment Setup

Create a local environment config with paths to your model directories:

```bash
cp .env.yaml.example .env.yaml
```

Edit `.env.yaml` to point to your local model storage:

```yaml
paths:
  models: !path ~/models/huggingface
  adapters: !path ~/models/adapters
```

### Requirements

- Python >= 3.11
- PyTorch >= 2.0
- CUDA-capable GPU (compute capability 8.0+ recommended)

## Quick Start

### Start the Server

```bash
# Using a model from HuggingFace
llm-infer serve --model-path /path/to/model

# Using a config file
llm-infer serve --config etc/llm-infer.yaml
```

### Query the Server

```bash
# Simple query
llm-infer query "What is the capital of France?"

# With streaming
llm-infer query --stream "Explain quantum computing"

# Using curl (OpenAI-compatible)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Configuration

Create `etc/llm-infer.yaml`:

```yaml
serve:
  host: "0.0.0.0"
  port: 8000
  models_dir: /path/to/models
  model: my-model
  handler: bounded  # or "sequential"

engine:
  max_batch_size: 32
  num_blocks: 2048
  block_size: 16
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/completions` | Text completion (OpenAI-compatible) |
| `POST /v1/chat/completions` | Chat completion (OpenAI-compatible) |
| `GET /v1/models` | List available models |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |

## Architecture

```
llm_infer/
├── backends/          # Quantization backends (AWQ, FP8)
│   └── linear/        # Linear layer implementations
├── cli/               # Command-line interface
├── pipelines/         # Model loading and execution
│   ├── model/         # Model architecture definitions
│   ├── engine.py      # Inference engine
│   └── scheduler.py   # Request scheduling
├── primitives/        # Core components
│   ├── attention/     # Attention backends (FlashInfer, naive)
│   ├── kv_cache/      # Paged KV cache
│   ├── sampler/       # Token sampling
│   └── tokenizer/     # Tokenizer wrappers
└── serving/           # HTTP server
    ├── api/           # FastAPI routes
    └── dispatch/      # Request handling
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
make test

# Run all checks (lint, type, test)
make check

# Format code
make fmt
```

## CLI Commands

```bash
llm-infer serve      # Start the llm-infer server
llm-infer query      # Send queries to a running server
llm-infer compat     # Generate/check compatibility specs
llm-infer metrics    # Display server metrics
```

## License

Apache License 2.0
