# Configuration

llm-infer uses YAML configuration files in the `etc/` directory. The main config file includes
engine-specific configs via `!include` directives.

## File Structure

```
etc/
├── llm-infer.yaml      # Main config (includes other files)
├── models.yaml         # Model-specific settings and templates
├── ollama.yaml         # Ollama engine settings
├── vllm.yaml           # vLLM Python API settings
├── vllm-server.yaml    # vLLM HTTP server settings
├── native.yaml         # Native engine settings
├── uvicorn.yaml        # HTTP server settings
└── infra.yaml          # Logging configuration
```

## Main Configuration

The main config (`etc/llm-infer.yaml`) selects the engine and includes engine-specific configs:

```yaml
# Backend selection
backends:
  engine: ollama        # ollama | vllm | vllm-server | native

# Engine configs (loaded via !include)
engines:
  native: !include './native.yaml'
  vllm: !include './vllm.yaml'
  vllm_server: !include './vllm-server.yaml'
  ollama: !include './ollama.yaml'

# API settings
api:
  host: 0.0.0.0
  port: 8000
  response_timeout: 180.0

# Request handling
dispatch:
  handler: bounded      # sequential | bounded | batching
  max_pending: 10
```

The `!include` directive is from [appinfra](https://github.com/serendip-ml/appinfra) and pulls in
the contents of the referenced file at that location.

## Engine Configuration

Each engine has its own config file. Common settings:

### Ollama (`etc/ollama.yaml`)

```yaml
host: http://localhost:11434
timeout: 300
auto_start: true        # Start Ollama server automatically
keep_alive: 5m          # Keep model loaded after request
num_ctx: null           # Context window (null = model default)
num_gpu: null           # GPU layers (null = auto, 0 = CPU only)
warmup: true            # Run warmup query on startup
```

### vLLM (`etc/vllm.yaml`)

```yaml
# Memory: use gpu_memory_gb (absolute) OR gpu_memory_utilization (fraction)
gpu_memory_gb: null             # e.g., 8.0 for 8GB limit (null = use utilization)
gpu_memory_utilization: 0.9     # fraction of total VRAM (0-1)
max_model_len: 16384
tensor_parallel_size: 1
max_num_seqs: 256
enable_prefix_caching: true
enforce_eager: false    # true = skip CUDA graphs (faster startup)
dtype: auto
warmup: true

lora:
  enabled: true
  max_loras: 4
  max_lora_rank: 128
  base_path: /path/to/adapters
```

### vLLM Server (`etc/vllm-server.yaml`)

```yaml
host: http://localhost
port: 8100              # Different from llm-infer port (8000)
auto_start: true        # Start `vllm serve` subprocess
startup_timeout: 300    # vLLM model loading is slow
timeout: 300

gpu_memory_gb: null             # e.g., 8.0 for 8GB limit
gpu_memory_utilization: 0.95    # used if gpu_memory_gb is null
max_model_len: null
enforce_eager: true
enable_prefix_caching: true
tool_call_parser: hermes  # Tool call extraction

lora:
  enabled: true
  base_path: /path/to/adapters
```

## Model Configuration

Per-model settings in `etc/models.yaml` override engine defaults:

```yaml
# Model search paths
locations:
  - /path/to/models

# Default model selection
selection:
  generate:
    default: qwen2.5-7b
  embed:
    default: bge-small-en-v1.5

# Model-specific settings
models:
  qwen2.5-0.5b-instruct:
    max_model_len: 4096
    vllm:
      enforce_eager: true
      gpu_memory_gb: 4.0        # 4GB absolute limit

  qwen2.5-7b:
    ollama: qwen2.5:7b   # Ollama model name mapping
```

### Templates (YAML Anchors)

Reusable settings via YAML anchors:

```yaml
templates:
  qwen_think: &qwen_think
    think:
      default: false
      enable_suffix: " /think"
      disable_suffix: " /no_think"
      tags:
        open: ["<think>"]
        close: ["</think>"]

  vllm_small: &vllm_small
    vllm:
      enforce_eager: true

models:
  qwen3-0.6b-instruct:
    <<: *qwen_think      # Merge template
    <<: *vllm_small
    max_model_len: 4096
```

## CLI Overrides

Override config values from command line with `-o`:

```bash
# Change engine
llm-infer serve --model qwen2.5-7b -o backends.engine=vllm

# Change vLLM settings
llm-infer serve --model qwen2.5-7b -o engines.vllm.gpu_memory_utilization=0.8

# Note: vllm_server uses underscore (matches YAML key)
llm-infer serve -o engines.vllm_server.port=8200
```

Override precedence (highest to lowest):
1. CLI flags (`--engine`, `--model-path`)
2. CLI overrides (`-o key=value`)
3. Config file values
4. Built-in defaults

## Include Syntax

The `!include` directive pulls in values from other files:

```yaml
# Include entire file
engines:
  vllm: !include './vllm.yaml'

# Include nested value (use # to select path)
models_path: !include "../.env.yaml#paths.models.ollama"
```

Paths can also be hardcoded directly:

```yaml
models_path: /data/models/ollama
```
