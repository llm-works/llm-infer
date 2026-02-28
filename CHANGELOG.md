# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `gpu_memory_gb` config option for vLLM engines: specify absolute GPU memory limit in GB instead
  of vLLM's fraction-based `gpu_memory_utilization` (e.g., `gpu_memory_gb: 8.0` for 8GB limit)
- vLLM added to `cuda` optional dependencies: `pip install llm-infer[cuda]` now includes vLLM
- `make setup` now installs `dev,runtime,cuda` extras by default

### Changed

- Refactored vLLM GPU memory resolution into shared `vllm_common.py` module

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

[Unreleased]: https://github.com/serendip-ml/llm-infer/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/serendip-ml/llm-infer/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/serendip-ml/llm-infer/releases/tag/v0.1.0
