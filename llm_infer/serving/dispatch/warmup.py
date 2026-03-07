"""Model and adapter warmup utilities."""

from typing import Any

from appinfra.log import Logger
from appinfra.time import since, start

from ..adapters import AdapterManager

# Default max_model_len when engine doesn't provide it
_DEFAULT_MAX_MODEL_LEN = 4096

# Open-ended prompt that encourages long generation (for EOS stress testing)
_STRESS_PROMPT = (
    "Write a detailed, comprehensive guide explaining how neural networks learn. "
    "Cover backpropagation, gradient descent, activation functions, and optimization."
)


class _WarmupLoraRequest:
    """Simple LoRA request for warmup."""

    def __init__(self, name: str) -> None:
        self.lora_name = name


# ---------------------------------------------------------------------------
# Base model warmup
# ---------------------------------------------------------------------------


def warmup_base_model(lg: Logger, engine: Any) -> None:
    """Warmup base model with a simple query."""
    lg.debug("warming up base model...")
    t0 = start()

    if getattr(engine, "supports_embeddings", lambda: False)():
        engine.embed(["warmup"])
        lg.info("base model warmed up", extra={"after": since(t0), "type": "embed"})
    else:
        output = engine.generate("Say hello", max_tokens=8)
        text = output["content"] if isinstance(output, dict) else output
        lg.info(
            "base model warmed up",
            extra={"after": since(t0), "words": len(text[:100].split())},
        )


# ---------------------------------------------------------------------------
# Adapter warmup
# ---------------------------------------------------------------------------


def warmup_adapters(
    lg: Logger,
    engine: Any,
    adapter_manager: AdapterManager | None,
) -> None:
    """Warmup registered LoRA adapters and verify EOS generation.

    Tests each adapter with geometrically increasing max_tokens (32, 128, 512, 2048, ...)
    up to max_model_len. Stops on first failure to detect EOS production issues early.

    If an adapter produces finish_reason="length" instead of "stop",
    it likely has a training issue where EOS tokens weren't learned.

    Note: EOS verification only works with vLLM-server engine, which returns finish_reason
    in generate() responses. The native vLLM engine returns plain strings, so EOS checks
    are skipped (finish_reason is "unknown").
    """
    if adapter_manager is None:
        return

    adapters = adapter_manager.list()
    if not adapters:
        return

    # Build token sweep: 32, 128, 512, 2048, ... up to max_model_len
    max_len = _get_max_model_len(engine)
    token_sweep = _build_token_sweep(max_len)

    lg.debug("warming up LoRA adapters...", extra={"count": len(adapters)})
    t0 = start()

    failed = 0
    for adapter in adapters:
        if not _warmup_single_adapter(lg, engine, adapter.key, token_sweep):
            failed += 1

    extra = {"after": since(t0), "count": len(adapters), "failed": failed}
    if failed:
        lg.warning("adapter warmup completed with failures", extra=extra)
    else:
        lg.info("all adapters warmed up", extra=extra)


def _get_max_model_len(engine: Any) -> int:
    """Get max_model_len from engine, with fallback default."""
    max_len = getattr(engine, "max_model_len", None)
    return max_len if max_len is not None else _DEFAULT_MAX_MODEL_LEN


def _build_token_sweep(max_model_len: int) -> list[int]:
    """Build geometric sweep: 32, 128, 512, 2048, ... up to max_model_len // 2.

    We cap at half of max_model_len to leave room for the prompt tokens.
    """
    sweep = []
    tokens = 32
    cap = max_model_len // 2
    while tokens <= cap:
        sweep.append(tokens)
        tokens *= 4
    return sweep


def _warmup_single_adapter(
    lg: Logger, engine: Any, adapter_key: str, token_sweep: list[int]
) -> bool:
    """Warmup a single LoRA adapter with geometric token sweep."""
    lora_req = _WarmupLoraRequest(adapter_key)

    for max_tokens in token_sweep:
        if not _test_adapter_eos(lg, engine, lora_req, adapter_key, max_tokens):
            return False

    lg.info("adapter verified", extra={"adapter": adapter_key})
    return True


def _test_adapter_eos(
    lg: Logger,
    engine: Any,
    lora_req: _WarmupLoraRequest,
    adapter_key: str,
    max_tokens: int,
) -> bool:
    """Test adapter EOS generation at given max_tokens. Returns True if EOS produced."""
    lg.debug(
        "warming up adapter...",
        extra={"adapter": adapter_key, "max_tokens": max_tokens},
    )
    t0 = start()

    try:
        output = engine.generate(
            _STRESS_PROMPT, max_tokens=max_tokens, lora_request=lora_req
        )
    except Exception as e:
        lg.error(
            "adapter warmup failed",
            extra={"adapter": adapter_key, "max_tokens": max_tokens, "exception": e},
        )
        return False

    return _verify_adapter_output(lg, output, adapter_key, max_tokens, t0)


def _verify_adapter_output(
    lg: Logger,
    output: str | dict[str, Any],
    adapter_key: str,
    max_tokens: int,
    t0: float,
) -> bool:
    """Verify adapter output: check fallback and finish_reason."""
    extra = {"adapter": adapter_key, "max_tokens": max_tokens}

    if _check_adapter_fallback(output):
        lg.warning("adapter warmup: vLLM fell back to base model", extra=extra)
        return False

    finish_reason = _extract_finish_reason(output)

    if finish_reason == "unknown":
        lg.info(
            "adapter warmed up (EOS verification skipped)",
            extra={"after": since(t0), **extra},
        )
        return True

    if finish_reason == "length":
        lg.warning(
            "adapter warmup: hit max_tokens without EOS - may generate infinitely. "
            "Check training data for proper EOS tokens.",
            extra=extra,
        )
        return False

    lg.info("adapter warmed up", extra={"after": since(t0), **extra})
    return True


def _check_adapter_fallback(output: str | dict[str, Any]) -> bool:
    """Check if vLLM fell back to base model instead of using requested adapter."""
    if not isinstance(output, dict):
        return False
    adapter_info = output.get("adapter_info")
    if not isinstance(adapter_info, dict):
        return False
    return bool(adapter_info.get("fallback", False))


def _extract_finish_reason(output: str | dict[str, Any]) -> str:
    """Extract finish_reason from generate output."""
    if isinstance(output, dict):
        reason: str = output.get("finish_reason", "unknown")
        return reason
    return "unknown"
