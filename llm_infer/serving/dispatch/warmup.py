# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Model and adapter warmup utilities."""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

from appinfra.log import Logger
from appinfra.time import since, start

from ...schemas.openai import FinishReason
from ..adapters import AdapterManager

# Default max_model_len when engine doesn't provide it
_DEFAULT_MAX_MODEL_LEN = 4096

# Prompts calibrated to different response lengths for EOS stress testing.
# Each prompt is designed to naturally produce roughly that many tokens,
# allowing us to test EOS behavior at different generation lengths.
_WARMUP_PROMPTS: dict[int, str] = {
    32: "What is 2+2? Answer with just the number.",
    128: "Explain addition in one paragraph.",
    512: "Describe how neural networks learn through backpropagation.",
    2048: "Write a detailed guide explaining gradient descent optimization.",
}


@dataclass
class WarmupResult:
    """Result of a single warmup test."""

    max_tokens: int
    finish_reason: FinishReason | None  # None if the engine didn't report one


class _WarmupLoraRequest:
    """Simple LoRA request for warmup."""

    def __init__(self, name: str) -> None:
        self.lora_name = name


T = TypeVar("T")
R = TypeVar("R")


def _run_steps(
    fn: Callable[[T], R | None], items: list[T], concurrency: int
) -> list[R]:
    """Run fn over items, up to `concurrency` at a time.

    Returns the results in item order, up to (not including) the first None,
    which marks a failed step. With concurrency 1 the steps run one at a time
    and stop at the first failure. With more, they all run at once, which also
    warms the backend for batched requests, and the results are cut the same way.
    """
    outcomes: Iterable[R | None]
    if concurrency <= 1 or len(items) <= 1:
        outcomes = map(fn, items)  # Lazy: no step runs after a failed one
    else:
        workers = min(concurrency, len(items))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="warmup"
        ) as pool:
            outcomes = list(pool.map(fn, items))

    results: list[R] = []
    for outcome in outcomes:
        if outcome is None:
            break
        results.append(outcome)
    return results


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _get_max_model_len(engine: Any) -> int:
    """Get max_model_len from engine, with fallback default."""
    max_len: int | None = getattr(engine, "max_model_len", None)
    return max_len if max_len is not None else _DEFAULT_MAX_MODEL_LEN


def _build_token_sweep(max_model_len: int) -> list[int]:
    """Build sweep from available prompts, capped at max_model_len // 2."""
    cap = max_model_len // 2
    return [t for t in sorted(_WARMUP_PROMPTS.keys()) if t <= cap]


def _extract_finish_reason(output: str | dict[str, Any]) -> FinishReason | None:
    """Extract finish_reason from generate output (None if not reported)."""
    if isinstance(output, dict):
        reason: FinishReason | None = output.get("finish_reason")
        return reason
    return None


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
    prompt = _WARMUP_PROMPTS[max_tokens]
    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    if lora_request is not None:
        kwargs["lora_request"] = lora_request

    output = engine.generate(prompt, **kwargs)
    finish_reason = _extract_finish_reason(output)
    return WarmupResult(max_tokens=max_tokens, finish_reason=finish_reason), output


# ---------------------------------------------------------------------------
# Base model warmup
# ---------------------------------------------------------------------------


def warmup_base_model(lg: Logger, engine: Any, concurrency: int) -> list[WarmupResult]:
    """Warmup base model with token sweep, return results as baseline.

    For embedding models, runs a simple embed and returns empty list.
    For generation models, runs token sweep and records finish_reason at each level,
    up to `concurrency` requests at a time. The results serve as baseline for
    adapter comparison.
    """
    if getattr(engine, "supports_embeddings", lambda: False)():
        lg.debug("warming up embedding model...")
        t0 = start()
        engine.embed(["warmup"])
        lg.info("embedding model warmed up", extra={"after": since(t0)})
        return []

    token_sweep = _build_token_sweep(_get_max_model_len(engine))
    lg.debug(
        "warming up base model...",
        # Joined with "|": the logger renders a list as "32,128,...", which reads as one number
        extra={"sweep": "|".join(map(str, token_sweep)), "concurrency": concurrency},
    )
    t0 = start()

    results = _run_steps(
        lambda max_tokens: _warmup_base_step(lg, engine, max_tokens),
        token_sweep,
        concurrency,
    )

    lg.info(
        "base model warmed up",
        extra={"after": since(t0), "steps": len(results), "concurrency": concurrency},
    )
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
    concurrency: int,
) -> None:
    """Warmup LoRA adapters and compare against baseline.

    Compares adapter behavior to base model baseline. Flags adapters that
    hit max_tokens where base model produced EOS - indicates potential
    training issue with EOS tokens. Runs up to `concurrency` requests at a time.
    """
    if adapter_manager is None:
        return

    adapters = adapter_manager.list()
    if not adapters:
        return

    if not baseline:
        lg.debug("no baseline results, skipping adapter EOS verification")
        _warmup_adapters_simple(lg, engine, adapters, concurrency)
        return

    lg.debug("warming up LoRA adapters...", extra={"count": len(adapters)})
    t0 = start()

    failed = 0
    for adapter in adapters:
        if not _warmup_single_adapter(lg, engine, adapter.key, baseline, concurrency):
            failed += 1

    extra = {"after": since(t0), "count": len(adapters), "failed": failed}
    if failed:
        lg.warning("adapter warmup completed with issues", extra=extra)
    else:
        lg.info("all adapters warmed up", extra=extra)


def _warmup_adapters_simple(
    lg: Logger, engine: Any, adapters: list[Any], concurrency: int
) -> None:
    """Simple adapter warmup without baseline comparison."""
    t0 = start()
    min_tokens = min(_WARMUP_PROMPTS.keys())

    def warm(adapter: Any) -> bool:
        try:
            _run_warmup_test(engine, min_tokens, _WarmupLoraRequest(adapter.key))
            lg.info("adapter warmed up", extra={"adapter": adapter.key})
            return True
        except Exception as e:
            lg.error(
                "adapter warmup failed", extra={"adapter": adapter.key, "exception": e}
            )
            return False  # Not None: a failed adapter doesn't stop the others

    _run_steps(warm, adapters, concurrency)
    lg.info("adapters warmed up", extra={"after": since(t0), "count": len(adapters)})


def _warmup_single_adapter(
    lg: Logger,
    engine: Any,
    adapter_key: str,
    baseline: list[WarmupResult],
    concurrency: int,
) -> bool:
    """Warmup single adapter and compare against baseline."""
    lora_req = _WarmupLoraRequest(adapter_key)
    outcomes = _run_steps(
        lambda base: _test_adapter_step(lg, engine, adapter_key, base, lora_req),
        baseline,
        concurrency,
    )
    if len(outcomes) < len(baseline):
        return False  # Fatal error (exception or fallback) at some step
    mismatches = [b.max_tokens for b, mismatch in zip(baseline, outcomes) if mismatch]

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
    return _is_eos_mismatch(base_result, result)


def _is_eos_mismatch(base_result: WarmupResult, result: WarmupResult) -> bool:
    """True when the base model stopped on EOS but the adapter ran to max_tokens."""
    return (
        base_result.finish_reason == FinishReason.STOP
        and result.finish_reason == FinishReason.LENGTH
    )


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
