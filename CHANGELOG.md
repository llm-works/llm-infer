# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SAIAAdapter.chat()` now accepts `temperature` parameter for sampling control

### Changed

- Update IPC calls for appinfra 0.6.0 API: request ID now passed via `request.id` attribute
  instead of separate argument
- Default `enforce_eager: false` in vllm-server.yaml (enables CUDA graphs)

## [0.2.0] - 2026-03-14

### Added

- **PEFT engine for PROMPT_TUNING adapters**: New `peft` engine type using HuggingFace Transformers
  + PEFT library for adapter types that vLLM's `--enable-lora` doesn't support (PROMPT_TUNING,
  PREFIX_TUNING, P_TUNING). Lazy-loads base model on first adapter request to minimize memory when
  coexisting with other engines.
- **Adapter peft_type detection**: AdapterManager reads `peft_type` from `adapter_config.json` and
  stores it in `LoadedAdapter.peft_type`. Enables routing to appropriate backend based on adapter
  type.
- **TraceMiddleware for API debugging**: Request/response logging at TRACE level for `/v1/`
  endpoints. Enable with `--log-level trace` to see full request bodies and responses.
- **Model warmup with EOS verification**: base model and LoRA adapters warmed up with token sweep
  (32, 128, 512, 2048 tokens) using prompts calibrated to each length. Adapters flagged if they
  hit max_tokens where base model produced EOS (indicates training issue). Note: EOS verification
  only works with vLLM-server engine.
- **Parallel HTTP requests** for vLLM server and Ollama engines: llm-infer now sends concurrent HTTP
  requests to leverage backend continuous batching, achieving near-linear speedup (4 requests in
  ~1.3s vs ~5s sequential)
- `max_concurrent` config option for HTTP engines: controls how many concurrent requests llm-infer
  sends to the backend (default: 4). Configure in `engines.vllm_server.max_concurrent` or
  `engines.ollama.max_concurrent`
- New `ConcurrentHttpHandler` for HTTP-based engines with thread pool execution
- Clean timeout error handling: request timeouts now log a single error line without stack traces
  and return proper 504 JSON response to clients
- `gpu_memory_gb` config option for vLLM engines: specify absolute GPU memory limit in GB instead
  of vLLM's fraction-based `gpu_memory_utilization` (e.g., `gpu_memory_gb: 8.0` for 8GB limit)
- vLLM added to `cuda` optional dependencies: `pip install llm-infer[cuda]` now includes vLLM
- `make setup` now installs `dev,runtime,cuda` extras by default
- `num_params_b` field in `ModelMetadata` for base model parameter count; used for adapter base
  model validation
- `--model-template` CLI flag: use another model's config as template for engine initialization
  (useful when model lacks proper HuggingFace config)

### Breaking Changes

- **Renamed `ConfigurationError` to `ConfigError`** for brevity

### Changed

- **Rate limiting moved to Backend layer**: Rate limiter is now injected into Backend and enforced
  on all HTTP requests (including `list_models()` for model discovery), not just chat calls. This
  ensures rate limiting applies uniformly to all external API calls.
- `dispatch.handler` config now supports primary/fallback structure: `handler.primary` for HTTP
  engines (vllm-server, ollama), `handler.fallback` for in-process engines (native, vllm). Legacy
  string format still supported for backward compatibility.
- `decoded` request lifecycle event moved from DEBUG to TRACE log level (reduces log noise)
- Refactored vLLM GPU memory resolution into shared `vllm_common.py` module
- Auto model resolution now retries on `BackendUnavailableError` if client has backoff configured,
  waiting for backend to come up before falling back. Logs expected errors without stack traces.
- **Internal**: Renamed `llm_infer.client.exceptions` module to `llm_infer.client.errors`. Public
  API unchanged (import from `llm_infer.client` as before)

### Fixed

- vLLM engines now correctly load PROMPT_TUNING adapters (previously failed with config mismatch)

## [0.1.1] - 2026-02-25

### Breaking Changes

- Wire protocol: `adapter_id` field renamed to `adapter` in request body
- Adapters now default to enabled (`enabled: true`) when config.yaml omits the field; previously
  required explicit `enabled: true`

### Added

- Versioned adapter resolution: multiple versions of an adapter can coexist using symlinks in
  format `{name}-{md5}` (12 hex chars); requests for base name resolve to latest version by mtime
- Model metadata extraction: `get_model_metadata()` detects quantization (BNB, GPTQ, AWQ, FP8) and
  precision from HuggingFace config.json for training parameter inference
- OpenAI compatibility: `model` field can select adapters (external clients can use standard
  `{"model": "my-adapter"}` pattern)
- Reserved model names (`auto`, `default`) always use base model, not adapter lookup
- Adapter metadata (md5, mtime) included in log messages and API responses
- Optional dependency `saia` for llm-saia integration (now available on PyPI)

### Changed

- Internal adapter representation uses `key` instead of `adapter_id`
- vllm-server engine: single source of truth for adapter scanning via AdapterManager (removed
  duplicate scanning)
- CI/CD: llm-saia installed via `[saia]` extra (single source of truth in pyproject.toml)

### Fixed

- Use relative imports consistently within package
- Entry point script (`llm-infer.py`) now prefers local source over installed package
- OpenAI backend: extract `extra_body` contents as top-level request keys (fixes `response_format`
  passed via `extra_body` being ignored)

## [0.1.0] - 2026-02-20

Initial public release.

### CLI & Server

- **Inference engines**: Ollama (default), vLLM, vLLM-server, and native torch
- **OpenAI-compatible API**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
- **Model support**: Llama, Mistral, Qwen, Granite architectures
- **Quantization**: AWQ (4-bit) and FP8 (8-bit)
- **Function calling**: OpenAI-compatible tool/function call support
- **Structured output**: `response_format` for JSON mode
- **LoRA adapters**: Dynamic adapter loading with fallback (vLLM engine)
- **Thinking mode**: `think` parameter for extended reasoning with `<think>` tags
- **CLI tools**: `serve`, `query`, `metrics`, `compat`

### Client Library

- **Multiple backends**: OpenAI-compatible APIs, Anthropic Claude
- **All execution modes**: Sync, async, and streaming
- **Rate limiting**: Per-backend request throttling
- **Retry with backoff**: Configurable exponential backoff on transient errors
- **Multi-backend routing**: Route requests by model name via `LLMRouter`
- **Factory pattern**: `Factory.openai()`, `Factory.anthropic()` for easy client creation
- **Lazy discovery**: Backends discovered on first use (no import-time failures)

### Documentation

- Engine comparison guide (`docs/engines.md`)
- Configuration reference (`docs/config.md`)
- Client library guide (`docs/client.md`)
- Contributing guide (`CONTRIBUTING.md`)

[Unreleased]: https://github.com/llm-works/llm-infer/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/llm-works/llm-infer/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/llm-works/llm-infer/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/llm-works/llm-infer/releases/tag/v0.1.0
