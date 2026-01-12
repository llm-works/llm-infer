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

### Added
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
