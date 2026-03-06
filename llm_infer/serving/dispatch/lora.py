"""LoRA adapter warmup utilities."""

from typing import Any, Protocol

from appinfra.log import Logger
from appinfra.time import since, start

from ..adapters import AdapterManager


class LoraRequest(Protocol):
    """Protocol for LoRA request objects expected by vLLM engine."""

    @property
    def lora_name(self) -> str: ...


class _WarmupLoraRequest:
    """Simple LoRA request for warmup."""

    def __init__(self, name: str) -> None:
        self.lora_name = name


def warmup_adapters(
    lg: Logger,
    engine: Any,
    adapter_manager: AdapterManager | None,
) -> None:
    """Warmup registered LoRA adapters and verify EOS generation.

    Tests each adapter with a short generation to ensure:
    1. The adapter loads and generates correctly
    2. The adapter produces EOS tokens (doesn't generate infinitely)

    If an adapter produces finish_reason="length" instead of "stop",
    it likely has a training issue where EOS tokens weren't learned.
    """
    if adapter_manager is None:
        return

    adapters = adapter_manager.list()
    if not adapters:
        return

    lg.debug("warming up LoRA adapters...", extra={"count": len(adapters)})
    t0 = start()

    failed = 0
    for adapter in adapters:
        if not _warmup_single_adapter(lg, engine, adapter.key):
            failed += 1

    extra = {"after": since(t0), "count": len(adapters), "failed": failed}
    if failed:
        lg.warning("adapter warmup completed with failures", extra=extra)
    else:
        lg.info("all adapters warmed up", extra=extra)


# Warmup prompts with varying expected response lengths to catch partial EOS learning
_WARMUP_PROMPTS = [
    ("Say hello in one word.", 16),
    ("Explain why the sky is blue in 2-3 sentences.", 128),
]


def _warmup_single_adapter(lg: Logger, engine: Any, adapter_key: str) -> bool:
    """Warmup a single LoRA adapter with multiple prompts. Returns True on success."""
    t0 = start()
    lora_req = _WarmupLoraRequest(adapter_key)

    for prompt, max_tokens in _WARMUP_PROMPTS:
        if not _test_adapter_prompt(
            lg, engine, lora_req, adapter_key, prompt, max_tokens, t0
        ):
            return False

    lg.info(
        "adapter warmup complete", extra={"after": since(t0), "adapter": adapter_key}
    )
    return True


def _test_adapter_prompt(
    lg: Logger,
    engine: Any,
    lora_req: _WarmupLoraRequest,
    adapter_key: str,
    prompt: str,
    max_tokens: int,
    t0: float,
) -> bool:
    """Test a single prompt and return True if EOS was produced."""
    try:
        output = engine.generate(prompt, max_tokens=max_tokens, lora_request=lora_req)
        finish_reason = _extract_finish_reason(output)

        if finish_reason == "length":
            lg.warning(
                "adapter warmup: hit max_tokens without EOS - may generate "
                "infinitely. Check training data for proper EOS tokens.",
                extra={
                    "after": since(t0),
                    "adapter": adapter_key,
                    "prompt": prompt[:40],
                    "max_tokens": max_tokens,
                },
            )
            return False
        return True

    except Exception as e:
        lg.error(
            "adapter warmup failed",
            extra={"adapter": adapter_key, "prompt": prompt[:40], "exception": e},
        )
        return False


def _extract_finish_reason(output: str | dict[str, Any]) -> str:
    """Extract finish_reason from generate output."""
    if isinstance(output, dict):
        reason: str = output.get("finish_reason", "unknown")
        return reason
    return "unknown"
