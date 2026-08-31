#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Client-library quickstart against an OpenAI-compatible endpoint.

Mirrors the leading example in README.md: constructs a Factory, opens a
single-backend LLMClient against an OpenAI-compatible server, sends one chat
completion, prints the response.

Environment overrides:
    LLM_INFER_BASE_URL    default: http://localhost:8000/v1
    LLM_INFER_MODEL       default: qwen2.5:0.5b
    LLM_INFER_PROMPT      default: "Say hello in exactly three words."
    LLM_INFER_MAX_TOKENS  default: 32

Exits non-zero on empty or missing response content. Used by the CI
wheel-smoke job to verify a fresh `pip install llm-infer` + a live
`llm-infer serve` endpoint round-trips end-to-end.
"""

import os
import sys

from appinfra.log import Logger

from llm_infer.client import Factory


def main() -> int:
    base_url = os.environ.get("LLM_INFER_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("LLM_INFER_MODEL", "qwen2.5:0.5b")
    prompt = os.environ.get("LLM_INFER_PROMPT", "Say hello in exactly three words.")
    max_tokens = int(os.environ.get("LLM_INFER_MAX_TOKENS", "32"))

    lg = Logger("quickstart")
    factory = Factory(lg)

    messages = [{"role": "user", "content": prompt}]

    with factory.openai(base_url=base_url, default_model=model) as client:
        response = client.chat(messages, max_tokens=max_tokens)

    content = (response.content or "").strip()
    if not content:
        print(f"ERROR: empty response from {base_url} (model={model})", file=sys.stderr)
        return 1

    print(f"prompt:   {prompt}")
    print(f"model:    {model}")
    print(f"base_url: {base_url}")
    print(f"response: {content}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
