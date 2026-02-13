#!/usr/bin/env python3
"""Interactive LLM chat client with multi-backend routing.

Usage:
    python client.py chat                              # Interactive chat
    python client.py chat "What is Python?"            # Single question
    python client.py chat --stream "Explain Python"   # Stream response
    python client.py chat --backend openai "Hello"     # Use specific backend
    python client.py chat --model gpt-4o "Hello"       # Route by model
    python client.py models                            # List available models
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from appinfra.app.builder import AppBuilder

from llm_infer.client import Factory, LLMRouter

# Config file path (same directory as this script)
CONFIG_FILE = Path(__file__).parent / "client.yaml"

builder = (
    AppBuilder("llm-chat")
    .with_description("Interactive LLM chat client")
    .with_config_file(str(CONFIG_FILE), from_etc_dir=False)
)


def get_config(tool: Any) -> dict[str, Any]:
    """Get config from app."""
    return cast(dict[str, Any], tool.app.config.to_dict())


def _print_response(
    router: LLMRouter,
    messages: list[dict[str, Any]],
    backend: str | None,
    model: str | None,
    stream: bool,
    prefix: str = "",
) -> str:
    """Print response from router, optionally streaming to stdout.

    Args:
        router: The LLMRouter to use.
        messages: Chat messages to send.
        backend: Backend to route to.
        model: Model to use.
        stream: Whether to stream the response.
        prefix: Optional prefix to print before response.

    Returns:
        The response text.
    """
    if prefix:
        print(prefix, end="", flush=True)
    if stream:
        response_text = ""
        for token in router.chat_stream(messages, backend=backend, model=model):
            print(token, end="", flush=True)
            response_text += token
        print()
        return response_text
    response_text = router.chat(messages, backend=backend, model=model)
    print(response_text)
    return response_text


def chat_interactive(
    router: LLMRouter, backend: str | None, model: str | None, stream: bool
) -> None:
    """Run interactive chat loop."""
    messages: list[dict[str, Any]] = []
    print("Chat started. Type 'quit' or 'exit' to end, 'clear' to reset history.")
    print(f"Backend: {backend or router.default}, Model: {model or 'default'}")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            messages.clear()
            print("History cleared.")
            continue

        messages.append({"role": "user", "content": user_input})
        response_text = _print_response(
            router, messages, backend, model, stream, prefix="\nAssistant: "
        )
        messages.append({"role": "assistant", "content": response_text})


@builder.tool(name="chat", help="Chat with an LLM")
@builder.argument(
    "question", nargs="?", help="Single question (interactive if omitted)"
)
@builder.argument("--backend", "-b", help="Backend to use")
@builder.argument("--model", "-m", help="Model to use (auto-routes to backend)")
@builder.argument("--stream", "-s", action="store_true", help="Stream responses")
def chat(self):
    config = get_config(self)

    with Factory(self.lg).from_config(config) as router:
        if self.args.question:
            messages = [{"role": "user", "content": self.args.question}]
            _print_response(
                router,
                messages,
                self.args.backend,
                self.args.model,
                self.args.stream,
            )
        else:
            chat_interactive(
                router, self.args.backend, self.args.model, self.args.stream
            )

    return 0


@builder.tool(name="models", help="List available models")
def models(self):
    config = get_config(self)

    with Factory(self.lg).from_config(config) as router:
        if router.models:
            for model_name, backend_name in sorted(router.models.items()):
                print(f"  {model_name} -> {backend_name}")
        else:
            print("  (no models discovered - backends may be offline)")

    return 0


app = builder.with_main_tool("chat").build()

if __name__ == "__main__":
    exit(app.main())
