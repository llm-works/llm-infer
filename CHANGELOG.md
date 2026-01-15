# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- Pin minimum versions for transitive dependencies to fix CVEs:
  - aiohttp >= 3.13.3 (CVE-2025-69223 through CVE-2025-69230)
  - filelock >= 3.20.1 (CVE-2025-68146)
  - urllib3 >= 2.6.3 (CVE-2026-21441)
  - werkzeug >= 3.1.5 (CVE-2026-21860)

### Fixed
- Circular import between `llm_infer.client` and `llm_infer.api` modules
- Circular import in `llm_infer.serving.api` preventing use of streaming utilities

### Changed
- Default logging level changed from debug to info
- Model configuration moved from `llm_infer.cli.config.models` to `llm_infer.models`
- Request handlers refactored to use Template Method pattern (common logic in base class)
- Streaming response generation refactored to use Template Method pattern
- Config overrides refactored to use Strategy pattern (env vars, CLI args)
- Request dispatch refactored to use Chain of Responsibility pattern
- Metrics response construction refactored to use Builder pattern
- Move OpenAI schemas to `llm_infer.schemas.openai` (leaf module with no dependencies)
- Remove `importlib.util` hack from `llm_infer.api` module

### Added
- `max_cudagraph_capture_size` config option for vLLM to limit CUDA graph batch sizes (reduces startup from ~2min to ~2sec)
- Generic `-o KEY=VALUE` CLI flag for `serve` command to override any config value at runtime
- `llm_infer.text` package with streaming text formatters (usable without CLI):
  - `ThinkFormatter` for styling `<think>`/`<thinking>` blocks
  - `LatexFormatter` for converting LaTeX math to Unicode
  - `Utf8StreamBuffer` for handling incomplete UTF-8 sequences
- `llm_infer.models` package for model configuration and path resolution (usable without CLI)
- `ModelResolver` class for unified model path resolution with priority chain
- OpenAI-compatible `/v1/embeddings` endpoint for vLLM backend with embedding models
- Model-specific config in `models.yaml`: `task` and `max_model_len` override vLLM settings per model
- `supports_embeddings()` method on engines for capability detection
- Embedding schemas: `EmbeddingRequest`, `EmbeddingResponse`, `EmbeddingObject`, `EmbeddingUsage`
- New public import path `llm_infer.schemas.openai` for schemas without client dependencies
- Regression tests for circular import issues (`TestCircularImportRegression`)
- Initial release
- Native inference engine with paged attention and continuous batching
- vLLM engine backend for production deployments
- OpenAI-compatible API (`/v1/completions`, `/v1/chat/completions`)
- Support for Llama, Mistral, Qwen, and Granite model architectures
- AWQ (4-bit) and FP8 (8-bit) quantization support
- FlashInfer attention backend
- CLI tools: `serve`, `query`, `metrics`, `compat`
- Streaming token generation with SSE
- Health and metrics endpoints
- Public API module (`llm_infer.api`) exporting OpenAI-compatible schemas for downstream use
- OpenAI-compatible async client (`llm_infer.client`) for consuming SSE streams from OpenAI-compatible
  APIs, enabling downstream packages to proxy streaming responses without reimplementing SSE parsing
