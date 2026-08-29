# llm-infer configuration

Default configuration files shipped inside the `llm-infer` wheel. Used by
`llm-infer serve` when no `--etc-dir` is passed on the command line.

The bundled defaults are enough to run any engine out of the box (Ollama,
vLLM, native, PEFT). Per-deployment customization is done by pointing
`--etc-dir` at your own copy, or by using `-o key=value` overrides at the
CLI. See [docs/config.md](../../docs/config.md) for the schema and
[docs/usage.md](../../docs/usage.md) for engine walkthroughs.

## Files

| File | Purpose |
|------|---------|
| `llm-infer.yaml` | Main config. Selects engine, wires per-engine includes, sets API host/port/timeouts. |
| `models.yaml` | Optional model catalog (see below). |
| `ollama.yaml` | Ollama engine (server URL, binary path, auto-start, keep-alive). |
| `vllm.yaml` | vLLM Python API engine (GPU memory, batching, LoRA). |
| `vllm-server.yaml` | vLLM subprocess engine (same domain plus process management). |
| `native.yaml` | Native PyTorch engine (KV cache, attention backend). |
| `peft.yaml` | PEFT engine (PROMPT_TUNING / PREFIX_TUNING / P_TUNING adapters). |
| `uvicorn.yaml` | HTTP server (uvicorn) settings. |
| `infra.yaml` | Logging configuration. |
| `compat.yaml`, `compat_template.yaml` | Used by `llm-infer compat` for API-compatibility specs. |
| `pip.conf` | Dev-only, not consumed at runtime. Custom pip config for the flashinfer index; used by `make install`. |

## Overriding for a deployment

Copy the bundled directory into your working area and point `--etc-dir` at
the copy:

```bash
# Find where the bundled etc/ lives
python -c "import llm_infer, os; print(os.path.dirname(llm_infer.__file__) + '/etc')"

# Copy it and edit
cp -r "$(python -c 'import llm_infer, os; print(os.path.dirname(llm_infer.__file__) + "/etc")')" ./my-etc
$EDITOR ./my-etc/ollama.yaml

# Serve from your copy
llm-infer serve --etc-dir ./my-etc --model qwen2.5:0.5b
```

For one-off tweaks without a full copy, use `-o`:

```bash
llm-infer serve -o api.port=8123 -o engines.vllm.gpu_memory_utilization=0.8 ...
```

Overrides applied in order (lowest to highest precedence): bundled config →
`--etc-dir` files → `-o key=value` flags → explicit CLI args (`--port`,
`--host`, `--model`, `--model-path`, `--engine`).

---

# `models.yaml` — the optional model catalog

`models.yaml` is a *catalog* of models with per-model tuning. **It is
optional.** `llm-infer serve --model <name>` works whether or not the name
appears in the catalog:

- **Ollama backend** — an unlisted name is passed straight through to the
  Ollama server. A warning is logged (`no ollama field in model config,
  using as-is`) and the request proceeds. The Ollama daemon serves the
  model as long as it has been pulled (`ollama pull <name>`).
- **vLLM / native / PEFT** — model resolution is by path. `--model-path
  <dir>` points at a HuggingFace snapshot directory directly, or `--model
  <name>` searches `locations:` for a subdirectory matching that name. The
  catalog is not consulted for resolution.

So the shipped `models.yaml` is intentionally empty; add entries only when
you want the per-model overrides described below.

## What entries provide

Declaring a model in `models.yaml` lets you attach settings that would
otherwise have to be repeated on every `llm-infer serve` invocation or
edited into the global engine config:

- **Ollama name mapping** — `ollama: <registry-name>`. Point one canonical
  llm-infer name at any tag: `qwen2.5-7b: {ollama: qwen2.5:7b}` means
  `--model qwen2.5-7b` runs Ollama's `qwen2.5:7b`. Useful for pinning tags
  or exposing a stable name across deployments.
- **vLLM tuning** — `vllm: {gpu_memory_gb: 8.0, max_num_seqs: 4,
  enforce_eager: true, ...}`. Applied on top of `etc/vllm.yaml` and
  `etc/vllm-server.yaml` at engine startup. Lets you size each model
  independently on a shared box.
- **Thinking mode** — `think: {default, enable_suffix, disable_suffix,
  system_prompt, tags: {open, close}}`. Controls how the server enables /
  disables reasoning-mode for that model, and what XML tags to parse
  reasoning content out of. See [docs/client.md](../../docs/client.md) for
  the `think=True` request extension.
- **Custom system prompt** — `system_prompt: "..."`. Prepended to every
  request that doesn't set its own system prompt.
- **Task type** — `task: embed` (or `generate`) overrides the engine's
  default task type. Needed for models used as embedders.
- **Context length** — `max_model_len: 4096` overrides
  `engines.vllm.max_model_len` for that specific model.

## Where the settings are applied

The flow, so it's clear which override wins when:

1. `serve.py:_resolve_model_config(config, model_name)` looks up
   `model_name` in `config.models.models`. If present, returns its
   `ModelConfig`. If absent, returns `None` (no error).
2. `_apply_model_overrides` mutates `config.engines.vllm`,
   `config.engines.vllm_server`, and `config.engines.ollama` with the
   catalog entry's fields (task, max_model_len, vllm map).
3. `apply_cli_overrides` then applies `-o` flags and explicit CLI args
   (`--engine`, `--model-path`, `--port`, `--host`) — these always win over
   catalog entries.
4. At request time, `router.py` and `mappers.py` consult the catalog entry
   (or `defaults:` if the name is not in the catalog) for per-request
   settings like thinking-mode resolution and the effective system prompt.

## `defaults:` — fallback for undeclared models

The `defaults:` section at the bottom of `models.yaml` provides the request
-time settings used when a model isn't in the catalog: base thinking-mode
config, base system prompt, base tag list. Edit these to change behavior
for every undeclared model at once.

## `--model-template` — reuse an entry for a new model

For one-off deployments of a model you don't want to catalog, borrow
another entry's overrides:

```bash
llm-infer serve --model my-experimental-qwen \
                --model-template qwen2.5-1.5b-instruct \
                --model-path /data/models/my-fine-tune
```

Applies `qwen2.5-1.5b-instruct`'s vLLM tuning and think config to the new
model without editing `models.yaml`.

## Walkthrough — adding an Ollama model to the catalog

Situation: you want `llm-infer serve --model qwen-coder` to invoke
Ollama's `qwen2.5-coder:7b`, with thinking mode enabled by default.

```bash
# 1. Copy the bundled etc into your workspace
cp -r "$(python -c 'import llm_infer, os; print(os.path.dirname(llm_infer.__file__) + "/etc")')" ./etc

# 2. Add the model to ./etc/models.yaml
```

```yaml
# ./etc/models.yaml
models:
  qwen-coder:
    ollama: qwen2.5-coder:7b
    think:
      default: true
      tags:
        open: ["<think>"]
        close: ["</think>"]
```

```bash
# 3. Pull the model and serve
ollama pull qwen2.5-coder:7b
llm-infer serve --etc-dir ./etc --engine ollama --model qwen-coder
```

## Walkthrough — vLLM tuning for a specific model

Situation: your box has 24GB VRAM. You want `--model qwen2.5-7b` (a local
HF snapshot in `~/.cache/huggingface/hub/`) to reserve 20GB and cap context
at 16k tokens.

```yaml
# ./etc/models.yaml (uncomment the qwen_think and vllm_small templates
# at the top of the shipped file first)
models:
  qwen2.5-7b:
    <<: *qwen_think
    <<: !deep *vllm_small
    max_model_len: 16384
    vllm:
      gpu_memory_gb: 20.0
      max_num_seqs: 8
```

```bash
llm-infer serve --etc-dir ./etc --engine vllm-server --model qwen2.5-7b
```

## When to skip the catalog entirely

- One-shot experiments: `llm-infer serve --model-path /path/to/model` with
  a stock `models.yaml`. Engine defaults from `etc/vllm.yaml` etc. are
  used; no per-model tuning.
- Client-side users of an already-running llm-infer server: irrelevant.
  `models.yaml` is server-side only.
