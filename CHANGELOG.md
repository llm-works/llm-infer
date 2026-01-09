# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
