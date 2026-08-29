# llm-infer

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Coverage](https://img.shields.io/badge/coverage-53%25-yellow.svg)
[![Typed](https://img.shields.io/badge/typed-PEP%20561-brightgreen.svg)](https://peps.python.org/pep-0561/)
[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-brightgreen)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/llm-works/llm-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/llm-works/llm-infer/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llm-infer.svg)](https://pypi.org/project/llm-infer/)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

`llm-infer` is a Python package for LLM inference: a production-grade
multi-backend client library, plus a thin wrapper for serving local models
behind an OpenAI-compatible endpoint.

Two components:

1. **`llm_infer.client`** — a production-grade multi-backend client library.
   Speaks OpenAI, Anthropic, Google (AI Studio + Vertex OpenAI-compat), Vertex
   AI native REST, and any OpenAI-compatible server through one interface, with
   routing, cross-provider fallback, retries, rate limiting, structured
   callbacks, and embeddings. **This is the main event.**
2. **`llm-infer serve`** — a thin devops wrapper around Ollama, vLLM, native
   torch, and PEFT that exposes them all behind one OpenAI-compatible HTTP
   endpoint. Useful when a team runs several models across mixed engines and
   wants a single operator contract. Not a replacement for real inference
   platforms (KServe, Ray Serve, NVIDIA Triton, vLLM's production stack) at
   GPU-farm scale.

The two are independent — the client works against any endpoint (cloud or
self-hosted), and the server can be consumed by any OpenAI-compatible client.

---

## `llm_infer.client` — Multi-Backend Client Library

Unified interface across cloud providers and self-hosted servers, with the
primitives production agent code actually needs: fallback across providers,
per-backend rate limits, exponential backoff, structured callbacks for cost
and tracing, embeddings, and async everywhere.

### Backends

| Backend | Notes |
|---------|-------|
| `openai` | OpenAI API |
| `openai_compatible` | Any OpenAI-compatible endpoint (vLLM, Ollama, `llm-infer serve`, ...) |
| `anthropic` | Anthropic Claude |
| Google | Gemini via AI Studio, Vertex OpenAI-compat, and native Vertex REST (`generateContent` + `cachedContents`) |

### Highlights

- **Multi-backend routing** — `LLMRouter` with pluggable `RoutingStrategy` and
  lazy model discovery.
- **Cross-provider fallback** — `FallbackClient` with chained pairs and
  `model@backend` pinning; 429s exhaust their retry budget on the primary
  before failing over.
- **Structured callbacks** — six lifecycle hooks split across retry-loop level
  (`on_request`, `on_response`, `on_retry`, `on_error`) and HTTP level
  (`on_before_send`, `on_after_send`) for cost tracking, tracing, and latency
  histograms.
- **Embeddings** — `EmbeddingClient` for OpenAI and Google (AI Studio +
  Vertex) with the same retry/callback contract.
- **Sync, async, streaming** — every execution mode.
- **Auth** — bearer tokens or GCP service accounts, resolved from the `auth:`
  block on any backend.
- **Extensible** — register custom backends via `Factory.register()`.

### Quick example

```python
from appinfra.log import Logger
from llm_infer.client import Factory, FallbackClient

lg = Logger("my-app")
factory = Factory(lg)

messages = [{"role": "user", "content": "Hello!"}]

# Single backend
with factory.openai(base_url="http://localhost:8000/v1") as client:
    response = client.chat(messages)

# Multi-backend router with cross-provider fallback
config = {
    "default": "primary",
    "backends": {
        "primary": {
            "type": "openai_compatible",
            "base_url": "http://localhost:8000/v1",
        },
        "fallback": {"type": "anthropic"},
    },
}  # see llm_infer/client/README.md for full schema
router = factory.from_config(config)
client = FallbackClient(lg, router, fallbacks={"gpt-4o": "claude-sonnet-4-20250514"})
response = client.chat(messages, model="gpt-4o")
```

**Full API, configuration schema, routing and fallback semantics, embeddings,
and observability hooks:** see
[`llm_infer/client/README.md`](llm_infer/client/README.md).

---

## `llm-infer serve` — Local Inference Wrapper

A single OpenAI-compatible HTTP server that dispatches to Ollama, vLLM,
native torch, or PEFT — selected by one yaml key. This is a
**devops wrapper**, not a serving platform: it unifies the *operator surface*
(CLI, health, shutdown, model catalog, OpenAI API, `think`/`adapter` protocol
extensions) across engines. Each engine keeps its own tuning knobs under a
shared outer envelope. Serves **one model per process**.

### Good fit

- **Several models with different settings** — one `models.yaml` entry per
  model, one CLI flag to switch which model the server hosts.
- **Mixed engines behind one contract** — Ollama for small models, vLLM for
  production serving (hot LoRA swap in-process, or pinned adapters as an HTTP
  subprocess), native for experimentation, PEFT for PROMPT_TUNING adapters
  that vLLM doesn't support.
- **Client code doesn't change when the backend changes** — same OpenAI
  surface, same `think`/`adapter` extensions across engines.
- **Dev↔prod parity** — the yaml (with `-o key=value` overrides) is the same
  shape everywhere.

### Not a fit

- **One model, laptop, casual use** — `ollama run` is simpler.
- **One model, single engine, production** — `vllm serve` or `ollama serve`
  alone gives every knob directly, no abstraction cost.
- **Multi-model in one process, dynamic KV cache sharing, engine-crash
  auto-restart** — this wrapper doesn't do those. Use a supervisor with one
  process per model, or a real inference platform (KServe, Ray Serve, NVIDIA
  Triton, vLLM's production stack) — see **Scale ceiling** below.

### Scale ceiling

At GPU-farm scale — autoscaling, disaggregated prefill/decode, cross-replica
batching, hot-swap under traffic — reach for a real inference platform:
**KServe, Ray Serve, NVIDIA Triton, or vLLM's production stack**. Not raw
engine CLIs. Because `llm-infer serve` preserves the OpenAI contract,
downstream client code doesn't have to change when migrating; the model
catalog does.

### Production deployment assumes

- **A reverse proxy in front** for auth, TLS, rate limiting, and per-model
  routing — the server has no built-in auth.
- **A supervisor** (systemd, k8s) for restart and multi-model fan-out — the
  server won't auto-restart a crashed engine subprocess.
- **Scrape-side handling of `/metrics`**, which returns structured JSON, not
  Prometheus text-exposition.

### Quick start

Serve on Ollama (the simplest path — CPU or GPU, no local weights to manage):

```bash
# 1. Install the Ollama binary (once per machine)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the model
ollama pull qwen2.5:0.5b

# 3. Install llm-infer and serve — llm-infer manages the ollama daemon
pip install 'llm-infer[runtime]'
llm-infer serve --model qwen2.5:0.5b

# 4. Query from another terminal
llm-infer query "What is the capital of France?"
```

See [docs/usage.md](docs/usage.md) for per-engine walkthroughs and
[`llm_infer/etc/README.md`](llm_infer/etc/README.md) for the bundled
configuration and override patterns.

### Engines

| Engine | Description | Install |
|--------|-------------|---------|
| `ollama` (default) | Wraps the Ollama server | [ollama.com](https://ollama.com) |
| `vllm` / `vllm-server` | vLLM — in-process (LoRA hot-swap) or as HTTP subprocess (LoRA pinned at boot) | `pip install vllm` |
| `native` | From-scratch torch implementation (PagedAttention + FlashInfer) | `pip install llm-infer[runtime]` |
| `peft` | HuggingFace PEFT, incl. PROMPT_TUNING adapters | `pip install llm-infer[runtime]` |

```bash
llm-infer serve --model qwen2.5:7b                          # Ollama
llm-infer serve --engine vllm --model-path /path/to/model   # vLLM (in-process)
llm-infer serve --engine native --model-path /path/to/model # Native
```

### Protocol extensions

The server extends OpenAI chat completions with `think` (reasoning content)
and `adapter` (LoRA selection) request fields, mirrored back as `thinking` in
the message and an `adapter` metadata block on the response. The client
library passes these through as keyword arguments on `client.chat()`.

### API endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completion (OpenAI-compatible) |
| `POST /v1/completions` | Text completion (OpenAI-compatible) |
| `POST /v1/embeddings` | Embeddings (OpenAI-compatible) |
| `GET /v1/models` | List available models |
| `GET /health` | Readiness gate — reports `initializing` until warmup completes |
| `GET /metrics` | Structured JSON metrics (not Prometheus text-exposition) |

---

## Installation

```bash
pip install llm-infer              # Client library only
pip install llm-infer[anthropic]   # + Anthropic support
pip install llm-infer[saia]        # + llm-saia integration
pip install llm-infer[runtime]     # + native engine and serve (torch)
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

Maintained by [LLM Works LLC](https://llm-works.ai) and contributors.
