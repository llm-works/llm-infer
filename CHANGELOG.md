# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- Engine comparison guide (`docs/ENGINES.md`)
- Configuration reference (`docs/CONFIG.md`)
- Client library guide (`docs/CLIENT.md`)
- Contributing guide (`CONTRIBUTING.md`)
