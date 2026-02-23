# LoRA Adapters

This guide covers LoRA adapter setup, versioning, and usage with `llm-infer`.

## Overview

LoRA (Low-Rank Adaptation) adapters allow fine-tuning large language models efficiently. `llm-infer`
supports dynamic adapter loading, enabling you to switch between adapters at inference time without
restarting the server.

## Directory Structure

Adapters are organized in a base directory configured via `lora.base_path`:

```
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

```
/path/to/adapters/
├── my-adapter-abc123def456/   # Version 1 (older)
│   └── ...
├── my-adapter-789xyz012345/   # Version 2 (newer)
│   └── ...
```

### Resolution Behavior

When requesting an adapter:

1. **Exact match**: `my-adapter-abc123def456` returns that specific version
2. **Name match**: `my-adapter` resolves to the latest version (by weights file mtime)

This allows clients to request either:
- A specific version for reproducibility
- The base name to always get the latest

### Example

```python
# Always use latest version
client.chat(messages, adapter="my-adapter")

# Pin to specific version
client.chat(messages, adapter="my-adapter-abc123def456")
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

```
GET /v1/adapters
```

Returns all loaded adapters:

```json
{
  "adapters": [
    {
      "key": "my-adapter-abc123def456",
      "name": "my-adapter",
      "description": "My fine-tuned adapter",
      "loaded_at": "2025-01-15T10:30:00+00:00",
      "md5": "abc123def456",
      "mtime": "2025-01-14T08:00:00+00:00"
    }
  ],
  "count": 1
}
```

### Refresh Adapters

```
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

| Engine | Dynamic Loading | Notes |
|--------|-----------------|-------|
| `vllm` | Yes | Full dynamic loading at request time |
| `vllm-server` | Pre-registered only | Adapters must exist at server startup |
| `ollama` | No | Not supported |
| `native` | No | Not supported |

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
