"""Model and adapter warmup utilities."""

from dataclasses import dataclass
from typing import Any

from appinfra.log import Logger
from appinfra.time import since, start

from ..adapters import AdapterManager

# Default max_model_len when engine doesn't provide it
_DEFAULT_MAX_MODEL_LEN = 4096

# Constrained prompt that should complete quickly with a definite answer
_WARMUP_PROMPT = "What is 2+2? Answer with just the number."


@dataclass
class WarmupResult:
    """Result of a single warmup test."""

    max_tokens: int
    finish_reason: str  # "stop", "length", "unknown"


class _WarmupLoraRequest:
    """Simple LoRA request for warmup."""

    def __init__(self, name: str) -> None:
        self.lora_name = name


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _get_max_model_len(engine: Any) -> int:
    """Get max_model_len from engine, with fallback default."""
    max_len = getattr(engine, "max_model_len", None)
    return max_len if max_len is not None else _DEFAULT_MAX_MODEL_LEN


def _build_token_sweep(max_model_len: int) -> list[int]:
    """Build geometric sweep: 128, 512, 2048, ... up to max_model_len // 2."""
    sweep = []
    tokens = 128
    cap = max_model_len // 2
    while tokens <= cap:
        sweep.append(tokens)
        tokens *= 4
    return sweep


def _extract_finish_reason(output: str | dict[str, Any]) -> str:
    """Extract finish_reason from generate output."""
    if isinstance(output, dict):
        reason: str = output.get("finish_reason", "unknown")
        return reason
    return "unknown"


def _check_adapter_fallback(output: str | dict[str, Any]) -> bool:
    """Check if vLLM fell back to base model instead of using requested adapter."""
    if not isinstance(output, dict):
        return False
    adapter_info = output.get("adapter_info")
    if not isinstance(adapter_info, dict):
        return False
    return bool(adapter_info.get("fallback", False))


def _run_warmup_test(
    engine: Any,
    max_tokens: int,
    lora_request: _WarmupLoraRequest | None = None,
) -> tuple[WarmupResult, str | dict[str, Any]]:
    """Run single warmup test, return result and raw output."""
    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    if lora_request is not None:
        kwargs["lora_request"] = lora_request

    output = engine.generate(_WARMUP_PROMPT, **kwargs)
    finish_reason = _extract_finish_reason(output)
    return WarmupResult(max_tokens=max_tokens, finish_reason=finish_reason), output


# ---------------------------------------------------------------------------
# Base model warmup
# ---------------------------------------------------------------------------


def warmup_base_model(lg: Logger, engine: Any) -> list[WarmupResult]:
    """Warmup base model with token sweep, return results as baseline.

    For embedding models, runs a simple embed and returns empty list.
    For generation models, runs token sweep and records finish_reason at each level.
    The results serve as baseline for adapter comparison.
    """
    if getattr(engine, "supports_embeddings", lambda: False)():
        lg.debug("warming up embedding model...")
        t0 = start()
        engine.embed(["warmup"])
        lg.info("embedding model warmed up", extra={"after": since(t0)})
        return []

    token_sweep = _build_token_sweep(_get_max_model_len(engine))
    lg.debug("warming up base model...", extra={"sweep": token_sweep})
    t0 = start()

    results: list[WarmupResult] = []
    for max_tokens in token_sweep:
        result = _warmup_base_step(lg, engine, max_tokens)
        if result is None:
            break
        results.append(result)

    lg.info("base model warmed up", extra={"after": since(t0), "steps": len(results)})
    return results


def _warmup_base_step(lg: Logger, engine: Any, max_tokens: int) -> WarmupResult | None:
    """Run single base model warmup step. Returns None on error."""
    lg.debug("warming up base...", extra={"max_tokens": max_tokens})
    t_step = start()
    try:
        result, _ = _run_warmup_test(engine, max_tokens)
        lg.info(
            "base warmed up", extra={"after": since(t_step), "max_tokens": max_tokens}
        )
        return result
    except Exception as e:
        lg.error(
            "base model warmup failed",
            extra={"max_tokens": max_tokens, "exception": e},
        )
        return None


# ---------------------------------------------------------------------------
# Adapter warmup
# ---------------------------------------------------------------------------


def warmup_adapters(
    lg: Logger,
    engine: Any,
    adapter_manager: AdapterManager | None,
    baseline: list[WarmupResult],
) -> None:
    """Warmup LoRA adapters and compare against baseline.

    Compares adapter behavior to base model baseline. Flags adapters that
    hit max_tokens where base model produced EOS - indicates potential
    training issue with EOS tokens.
    """
    if adapter_manager is None:
        return

    adapters = adapter_manager.list()
    if not adapters:
        return

    if not baseline:
        lg.debug("no baseline results, skipping adapter EOS verification")
        _warmup_adapters_simple(lg, engine, adapters)
        return

    lg.debug("warming up LoRA adapters...", extra={"count": len(adapters)})
    t0 = start()

    failed = 0
    for adapter in adapters:
        if not _warmup_single_adapter(lg, engine, adapter.key, baseline):
            failed += 1

    extra = {"after": since(t0), "count": len(adapters), "failed": failed}
    if failed:
        lg.warning("adapter warmup completed with issues", extra=extra)
    else:
        lg.info("all adapters warmed up", extra=extra)


def _warmup_adapters_simple(lg: Logger, engine: Any, adapters: list[Any]) -> None:
    """Simple adapter warmup without baseline comparison."""
    t0 = start()
    for adapter in adapters:
        lora_req = _WarmupLoraRequest(adapter.key)
        try:
            _run_warmup_test(engine, 128, lora_req)
            lg.info("adapter warmed up", extra={"adapter": adapter.key})
        except Exception as e:
            lg.error(
                "adapter warmup failed", extra={"adapter": adapter.key, "exception": e}
            )
    lg.info("adapters warmed up", extra={"after": since(t0), "count": len(adapters)})


def _warmup_single_adapter(
    lg: Logger,
    engine: Any,
    adapter_key: str,
    baseline: list[WarmupResult],
) -> bool:
    """Warmup single adapter and compare against baseline."""
    lora_req = _WarmupLoraRequest(adapter_key)
    mismatches: list[int] = []

    for base_result in baseline:
        step_result = _test_adapter_step(lg, engine, adapter_key, base_result, lora_req)
        if step_result is None:
            return False  # Fatal error (exception or fallback)
        if step_result:
            mismatches.append(base_result.max_tokens)

    if mismatches:
        lg.warning(
            "adapter did not produce EOS where base model did",
            extra={"adapter": adapter_key, "at_max_tokens": mismatches},
        )
        return False

    lg.info("adapter verified", extra={"adapter": adapter_key})
    return True


def _test_adapter_step(
    lg: Logger,
    engine: Any,
    adapter_key: str,
    base_result: WarmupResult,
    lora_req: _WarmupLoraRequest,
) -> bool | None:
    """Test single adapter warmup step. Returns True if EOS mismatch, False if ok, None on error."""
    max_tokens = base_result.max_tokens
    lg.debug(
        "warming up adapter...",
        extra={"adapter": adapter_key, "max_tokens": max_tokens},
    )
    t_step = start()

    test_result = _execute_adapter_test(lg, engine, adapter_key, max_tokens, lora_req)
    if test_result is None:
        return None

    result, output = test_result
    if _check_adapter_fallback(output):
        lg.warning(
            "adapter warmup: vLLM fell back to base model",
            extra={"adapter": adapter_key, "max_tokens": max_tokens},
        )
        return None

    lg.info(
        "adapter warmed up",
        extra={
            "after": since(t_step),
            "adapter": adapter_key,
            "max_tokens": max_tokens,
        },
    )
    return base_result.finish_reason == "stop" and result.finish_reason == "length"


def _execute_adapter_test(
    lg: Logger,
    engine: Any,
    adapter_key: str,
    max_tokens: int,
    lora_req: _WarmupLoraRequest,
) -> tuple[WarmupResult, str | dict[str, Any]] | None:
    """Execute adapter warmup test. Returns (result, output) or None on error."""
    try:
        return _run_warmup_test(engine, max_tokens, lora_req)
    except Exception as e:
        lg.error(
            "adapter warmup failed",
            extra={"adapter": adapter_key, "max_tokens": max_tokens, "exception": e},
        )
        return None
