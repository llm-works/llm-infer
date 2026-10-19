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
├── peft.yaml           # PEFT engine settings (PROMPT_TUNING adapters)
├── uvicorn.yaml        # HTTP server settings
└── infra.yaml          # Logging configuration
```

## Main Configuration

The main config (`etc/llm-infer.yaml`) selects the engine and includes engine-specific configs:

```yaml
# Backend selection
backends:
  engine: ollama        # ollama | vllm | vllm-server | native | peft

# Engine configs (loaded via !include)
engines:
  native: !include './native.yaml'
  vllm: !include './vllm.yaml'
  vllm_server: !include './vllm-server.yaml'
  ollama: !include './ollama.yaml'
  peft: !include './peft.yaml'

# API settings
api:
  host: 0.0.0.0
  port: 8000
  response_timeout: 180.0

# Request handling
dispatch:
  handler:
    primary: concurrent_http  # Used for HTTP engines (vllm-server, ollama)
    fallback: bounded         # Used for in-process engines (native, vllm)
  max_pending: 10
```

The `!include` directive is from [appinfra](https://github.com/llm-works/appinfra) and pulls in
the contents of the referenced file at that location.

## Admin Listener

The admin listener releases the backend's GPU (e.g. for a training run) and reacquires it without
restarting serve. Supported with the `vllm-server` engine when serve starts vLLM itself
(`auto_start: true`): release stops the vLLM process, reacquire starts it again. With a vLLM
server serve didn't start, release is refused with 409.

It is a listener of its own in serve's main process, separate from the public API port. Bind it
to an intranet address only, never a public one. Off by default:

```yaml
admin:
  enabled: true
  host: 127.0.0.1         # An intranet address
  port: 8090
  tokens: !include? '/etc/llm-infer/secrets/admin-tokens.yaml'  # List of bearer tokens
  retry_after: 60         # Retry-After (s) on inference answered 503 while released
```

Every call needs `Authorization: Bearer <token>` with one of the tokens; without tokens, every
call is refused with 401.

| Call | Result |
|---|---|
| `POST /admin/release` | Stops the backend; returns 200 `{"state": "released"}` once its processes are gone and the GPU memory is free. Idempotent. |
| `POST /admin/reacquire` | Starts the backend again; returns 202 `{"state": "loading"}` (200 if already serving). |
| `GET /admin/status` | `{"state": ..., "last_error": ...}`; state is `initializing`, `serving`, `releasing`, `released` or `loading`. A failed reacquire leaves `released` with the error in `last_error`. |

While the backend is releasing, released or loading, the inference routes (`/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings`, `/generate`) answer 503 with `Retry-After`; `/health` and
`/v1/models` keep answering. Requests already queued or running when a release starts are
answered 503 too.

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8090/admin/release
curl -X POST -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8090/admin/reacquire
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8090/admin/status
```

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
max_concurrent: 4       # Concurrent HTTP requests to Ollama
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
max_concurrent: 4         # Concurrent HTTP requests to vLLM server

lora:
  enabled: true
  base_path: /path/to/adapters
```

### PEFT (`etc/peft.yaml`)

```yaml
device: cuda                  # Device to load model on
dtype: auto                   # Model dtype (auto, float16, bfloat16)
max_cached_adapters: 4        # LRU cache size for loaded adapters
warmup: false                 # Disabled: prompt-tuning KV cache bug causes ~30min warmup
load_in_4bit: false           # Use bitsandbytes 4-bit quantization
adapter_base_path: /path/to/adapters  # Base directory for adapters
```

The PEFT engine is for PROMPT_TUNING, PREFIX_TUNING, and P_TUNING adapters that vLLM's
`--enable-lora` doesn't support. For LoRA adapters, use `vllm` or `vllm-server` instead.

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
logging: !include "./infra.yaml#logging"
```

Use `!include?` for a soft include that returns `{}` when the file is missing
instead of raising — useful for optional per-machine overrides that may not
exist in every deployment.
