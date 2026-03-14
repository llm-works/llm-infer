# Adapters

This guide covers adapter setup, versioning, and usage with `llm-infer`. Supports both LoRA
adapters (via vLLM) and prompt-learning adapters like PROMPT_TUNING (via PEFT engine).

## Overview

Adapters allow fine-tuning large language models efficiently. `llm-infer` supports multiple adapter
types with automatic routing to the appropriate backend:

| Adapter Type | Backend | Notes |
|-------------|---------|-------|
| LoRA | vllm, vllm-server | Efficient low-rank adaptation |
| PROMPT_TUNING | peft | Soft prompts prepended to input |
| PREFIX_TUNING | peft | Learnable prefix activations |
| P_TUNING | peft | Continuous prompt embeddings |

The server automatically detects adapter type from `adapter_config.json` and routes to the correct
engine.

## Directory Structure

Adapters are organized in a base directory configured via `lora.base_path`:

```text
/path/to/adapters/
├── my-adapter/
│   ├── config.yaml          # Adapter configuration
│   ├── adapter_config.json  # LoRA parameters (from training)
│   └── adapter_model.safetensors
├── another-adapter/
│   ├── config.yaml
│   └── ...
```

### config.yaml

Each adapter directory must contain a `config.yaml` file:

```yaml
# Adapter is enabled by default if this field is omitted
enabled: true   # Set to false to disable without removing

description: "Optional human-readable description"
```

**Note:** Adapters default to enabled. To disable an adapter, explicitly set `enabled: false`.

## Versioned Adapters

Multiple versions of the same adapter can coexist using versioned symlinks. This is useful for
A/B testing or gradual rollouts.

### Naming Convention

Versioned adapters use the format `{name}-{md5}` where `md5` is the first 12 characters of the
weights file's MD5 hash:

```text
/path/to/adapters/
├── my-adapter-a1b2c3d4e5f6/   # Version 1 (older)
│   └── ...
├── my-adapter-f6e5d4c3b2a1/   # Version 2 (newer)
│   └── ...
```

### Resolution Behavior

When requesting an adapter:

1. **Exact match**: `my-adapter-a1b2c3d4e5f6` returns that specific version
2. **Name match**: `my-adapter` resolves to the latest version (by weights file mtime)

This allows clients to request either:
- A specific version for reproducibility
- The base name to always get the latest

### Example

```python
# Always use latest version
client.chat(messages, adapter="my-adapter")

# Pin to specific version
client.chat(messages, adapter="my-adapter-a1b2c3d4e5f6")
```

## Using Adapters

### In Requests

Use the `adapter` field in chat completion requests:

```python
from llm_infer import Client

client = Client.openai(base_url="http://localhost:8000/v1")
response = client.chat(
    messages=[{"role": "user", "content": "Hello"}],
    adapter="my-adapter",
)
```

### OpenAI Compatibility

For clients using the standard OpenAI SDK, the `model` field can select adapters:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="my-adapter",  # Selects the adapter
    messages=[{"role": "user", "content": "Hello"}],
)
```

Reserved model names (`auto`, `default`) always use the base model.

## API Endpoints

### List Adapters

```text
GET /v1/adapters
```

Returns all loaded adapters:

```json
{
  "adapters": [
    {
      "key": "my-adapter-a1b2c3d4e5f6",
      "name": "my-adapter",
      "description": "My fine-tuned adapter",
      "loaded_at": "2025-01-15T10:30:00+00:00",
      "md5": "a1b2c3d4e5f6",
      "mtime": "2025-01-14T08:00:00+00:00"
    }
  ],
  "count": 1
}
```

### Refresh Adapters

```text
POST /v1/adapters/refresh
POST /v1/adapters/refresh?key=my-adapter
```

Rescans the adapter directory and reloads enabled adapters. Use `key` parameter to refresh a
specific adapter.

```json
{
  "key": null,
  "adapters_loaded": 3,
  "status": "scanned"
}
```

## Engine Support

| Engine | Adapter Types | Dynamic Loading | Notes |
|--------|---------------|-----------------|-------|
| `vllm` | LoRA | Yes | Full dynamic loading at request time |
| `vllm-server` | LoRA | Pre-registered only | Adapters must exist at server startup |
| `peft` | PROMPT_TUNING, PREFIX_TUNING, P_TUNING | Yes | LRU cache for loaded adapters |
| `ollama` | None | No | Not supported |
| `native` | None | No | Not supported |

### Adapter Type Detection

llm-infer reads `peft_type` from `adapter_config.json` to determine which engine handles each
adapter:

```json
{
  "peft_type": "PROMPT_TUNING",
  "base_model_name_or_path": "/path/to/model",
  ...
}
```

- LoRA adapters → vLLM engine (optimized, fast)
- PROMPT_TUNING/PREFIX_TUNING/P_TUNING → PEFT engine

This happens automatically. You don't need to configure routing.

### vllm Engine

The `vllm` engine (Python API) supports full dynamic adapter loading:

```yaml
engine: vllm
lora:
  base_path: /path/to/adapters
```

- Adapters can be added/removed while the server is running
- Use `/v1/adapters/refresh` to pick up changes
- Best for development and testing workflows

### vllm-server Engine

The `vllm-server` engine runs `vllm serve` as a subprocess:

```yaml
engine: vllm-server
lora:
  base_path: /path/to/adapters
```

**Limitation:** Adapters are registered at server startup via `--lora-modules`. Adapters created
after startup require a server restart.

Use `vllm-server` for production with stable adapter sets. Use `vllm` if you need dynamic loading.

### peft Engine

The `peft` engine handles PROMPT_TUNING and other prompt-learning adapters:

```yaml
engine: peft
peft:
  adapter_base_path: /path/to/adapters
  max_cached_adapters: 4
```

Features:
- **Lazy loading**: Base model only loads on first adapter request
- **LRU cache**: Keeps recently-used adapters in memory
- **4-bit quantization**: Optional bitsandbytes support (`load_in_4bit: true`)

Use the `peft` engine when you have PROMPT_TUNING, PREFIX_TUNING, or P_TUNING adapters.

## Hot Reloading

To reload adapters without restarting the server (vllm engine only):

```bash
# Rescan all adapters
curl -X POST http://localhost:8000/v1/adapters/refresh

# Refresh specific adapter
curl -X POST "http://localhost:8000/v1/adapters/refresh?key=my-adapter"
```

After refreshing:
- New adapters in the directory become available
- Removed adapters are unloaded
- Modified `config.yaml` changes take effect (enabled/disabled)
- Updated weights are reloaded on next use
