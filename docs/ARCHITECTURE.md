# Architecture

This document describes the system architecture of the LLM inference engine.

## Overview

The system is organized into four layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                               │
│                     (llm_infer/cli/)                            │
│  Commands: serve, query, metrics, compat                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Serving Layer                              │
│                   (llm_infer/serving/)                          │
│  FastAPI routes, OpenAI-compatible API, request dispatch        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Engines Layer                              │
│                   (llm_infer/engines/)                          │
│  Inference backends: native, vLLM, Ollama                       │
│  Protocol interfaces for engine abstraction                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Native Engine Internals                       │
│                (llm_infer/engines/native/)                      │
│  Transformer model, scheduler, KV cache, attention, sampler     │
│  Quantization backends (AWQ, FP8)                               │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Details

### CLI Layer (`llm_infer/cli/`)

Entry point for all operations. Built on the `appinfra` framework.

| Command | Description |
|---------|-------------|
| `llm-infer serve` | Start the inference server |
| `llm-infer query` | Send queries to a running server |
| `llm-infer metrics` | Display server metrics |
| `llm-infer compat` | Generate/check compatibility specs |

Configuration is loaded from YAML files (default: `etc/llm-infer.yaml`).

### Serving Layer (`llm_infer/serving/`)

HTTP server and request handling.

**API Routes** (`serving/api/`)
- OpenAI-compatible endpoints (`/v1/completions`, `/v1/chat/completions`)
- Health and metrics endpoints (`/health`, `/metrics`)
- Model listing (`/v1/models`)

**Dispatch** (`serving/dispatch/`)
- Request handlers that bridge HTTP requests to the inference engine
- Three handler types:
  - `sequential`: One request at a time (simplest)
  - `bounded`: Bounded queue with backpressure
  - `batching`: Full continuous batching

**Request Flow:**
```
HTTP Request → FastAPI Route → Dispatch Handler → Engine → Response
```

### Engines Layer (`llm_infer/engines/`)

Inference engine implementations with a common protocol interface.

**Protocol** (`engines/protocol.py`)
- `InferenceEngineProtocol`: Common interface for all engines
- `StreamingResultProtocol`: Streaming generation interface
- Enables engine-agnostic handlers and serving layer

**Engine Implementations:**
- `native/`: Custom implementation for learning/research
- `vllm.py`: vLLM wrapper for production
- `ollama.py`: Ollama wrapper for local models

### Native Engine (`llm_infer/engines/native/`)

Custom inference implementation with full visibility into the pipeline.

**InferenceEngine** (`native/engine.py`)
- Orchestrates model, scheduler, and KV cache
- Exposes `generate()` and `generate_stream()` methods
- Manages memory and lifecycle

**Scheduler** (`native/scheduler.py`)
- Manages request queues
- Forms batches of prefill and decode requests
- Tracks request state

**Model** (`native/model/`)
- `TransformerModel`: Main model class
- `architecture.py`: Model-specific configurations (Llama, Mistral, Qwen, Granite)
- `layers.py`: Layer implementations (attention, MLP, embeddings)
- `config.py`: Model configuration loading from HuggingFace

**Generation** (`native/generation.py`)
- `run_prefill()`: Process prompt tokens
- `run_decode()`: Generate one token
- `run_decode_batch()`: Batched decode for multiple requests

**Attention** (`native/attention/`)
- `AttentionBackend` protocol for swappable implementations
- `FlashInferBackend`: Optimized CUDA kernels (recommended)
- `NaiveBackend`: Pure PyTorch implementation (for debugging)
- `rope.py`: Rotary position embedding implementations

**KV Cache** (`native/kv_cache/`)
- `BlockPool`: Block-based memory allocator
- `SequenceKVCache`: Per-sequence cache management
- Implements paged attention memory model

**Tokenizer** (`native/tokenizer/`)
- `Tokenizer` protocol
- `HuggingFaceTokenizer`: Wraps HF tokenizers with chat template support

**Sampler** (`native/sampler/`)
- Token sampling with temperature, top-p, top-k
- Repetition penalty support

**Guards** (`native/guards/`)
- Generation guards that can modify or stop generation
- `RepetitionGuard`: Detects and handles repetitive output

**Quantization Backends** (`native/backends/linear/`)
- `QuantFormat`: AWQ (4-bit) and FP8 (8-bit) formats
- Kernel implementations:
  - `awq_pytorch.py`: Pure PyTorch AWQ (reference)
  - `awq_marlin.py`: Marlin kernels for AWQ (fast)
  - `fp8_pytorch.py`: Pure PyTorch FP8
  - `fp8_cutlass.py`: CUTLASS kernels for FP8

## Protocol-Based Design

The codebase uses Python protocols (`typing.Protocol`) to define interfaces between
components. This enables:

1. **Dependency injection**: Components accept protocols, not concrete classes
2. **Testability**: Mock implementations for unit testing
3. **Extensibility**: Add new implementations without changing consumers

Key protocols in `llm_infer/engines/protocol.py` and `llm_infer/engines/native/protocols.py`:

| Protocol | Purpose |
|----------|---------|
| `BlockAllocator` | Block-level memory allocation |
| `KVCache` | Per-sequence KV cache management |
| `AttentionBackend` | Attention computation |
| `Tokenizer` | Text tokenization |
| `SchedulerProtocol` | Request scheduling |
| `InferenceEngineProtocol` | Engine abstraction for handlers |

## Engine Backends

The system supports multiple engine backends:

**Native Engine** (default for learning/research)
- Custom implementation in pure Python/PyTorch
- Full visibility into inference pipeline
- FlashInfer for optimized attention

**vLLM Engine** (for production)
- Production-grade with PagedAttention
- Continuous batching
- Tensor parallelism support

**Ollama Engine** (for local development)
- Uses Ollama for model management and inference
- Requires Ollama installed on host; llm-infer can auto-start and manage the server if configured
- Simple setup: just `ollama pull <model>` to download models
- Good for quick experimentation without GPU configuration

Select via configuration:
```yaml
backends:
  engine: native  # or vllm, ollama
```

## Request Lifecycle

1. **HTTP Request** arrives at FastAPI endpoint
2. **Dispatch Handler** validates and queues the request
3. **Tokenization**: Text → token IDs (with optional chat template)
4. **Prefill Phase**: Process all prompt tokens, populate KV cache
5. **Decode Loop**: Generate tokens one at a time
   - Sample next token from logits
   - Append to output
   - Check stop conditions
6. **Response**: Stream tokens or return complete text

## Memory Management

### Paged Attention

KV cache is managed using paged attention:
- Memory divided into fixed-size blocks
- Sequences allocate blocks as needed
- Blocks returned to pool when sequence completes

Configuration:
```yaml
engines:
  native:
    num_blocks: 1024    # Total blocks in pool
    block_size: 16      # Tokens per block
```

### Block Allocation

```
┌─────────────────────────────────────────┐
│              Block Pool                 │
├─────┬─────┬─────┬─────┬─────┬─────┬────┤
│ B0  │ B1  │ B2  │ B3  │ B4  │ ... │ Bn │
└─────┴─────┴─────┴─────┴─────┴─────┴────┘
       ▲           ▲
       │           │
   ┌───┴───┐   ┌───┴───┐
   │ Seq A │   │ Seq B │
   └───────┘   └───────┘
```

## Continuous Batching

Multiple sequences can be processed together:

1. **Prefill requests**: New sequences needing prompt processing
2. **Decode requests**: Active sequences generating tokens

The scheduler forms batches that maximize GPU utilization while respecting
memory constraints.

## Adding New Model Architectures

1. Create entry in `llm_infer/engines/native/model/architecture.py`:
   ```python
   @register_architecture("my_model")
   class MyModelArchitecture(Architecture):
       def layer_config(self) -> LayerConfig:
           # Return layer dimensions and settings
           ...
   ```

2. Handle any model-specific quirks (attention patterns, normalization, etc.)

3. Test with a small model variant first

## File Organization

```
llm_infer/
├── __init__.py
├── context.py              # Request context for tracing
├── compat.py               # Compatibility spec generation
├── logging_setup.py        # Logging configuration
├── cli/                    # Command-line interface
│   ├── cli.py              # Main entry point
│   ├── config/             # Configuration loading
│   └── tools/              # Subcommand implementations
├── engines/                # Inference engines
│   ├── __init__.py         # Engine factory (create_engine)
│   ├── protocol.py         # InferenceEngineProtocol
│   ├── vllm.py             # vLLM engine wrapper
│   ├── ollama.py           # Ollama engine wrapper
│   └── native/             # Native engine implementation
│       ├── engine.py       # InferenceEngine
│       ├── scheduler.py    # Request scheduling
│       ├── generation.py   # Prefill/decode functions
│       ├── config.py       # Engine configuration
│       ├── protocols.py    # Internal protocol definitions
│       ├── model/          # Model implementation
│       │   ├── transformer.py  # TransformerModel
│       │   ├── architecture.py # Model architectures
│       │   ├── layers.py       # Layer implementations
│       │   └── config.py       # Model config loading
│       ├── attention/      # Attention implementations
│       ├── kv_cache/       # KV cache management
│       ├── sampler/        # Token sampling
│       ├── tokenizer/      # Tokenizer wrappers
│       ├── guards/         # Generation guards
│       └── backends/       # Quantization backends
│           └── linear/
│               ├── formats/    # AWQ, FP8 format definitions
│               └── kernels/    # Kernel implementations
└── serving/                # HTTP server
    ├── api/                # FastAPI routes
    │   └── openai/         # OpenAI-compatible endpoints
    └── dispatch/           # Request dispatch
        └── handlers/       # Handler implementations
```
