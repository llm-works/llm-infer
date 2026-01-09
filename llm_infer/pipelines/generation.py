"""Generation logic: prefill and decode steps."""

from collections.abc import Sequence

import torch
from torch import Tensor

from ..context import Event
from ..primitives.guards import GenerationGuard
from ..primitives.kv_cache import BlockPool
from ..primitives.sampler import sample
from .model import TransformerModel
from .scheduler import Request, RequestState


def _sample_next_token(
    logits: Tensor, request: Request, past_tokens: list[int] | None
) -> int:
    """Sample next token from logits using request parameters."""
    return int(
        sample(
            logits,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            past_tokens=past_tokens,
        ).item()
    )


def _run_guards(
    request: Request,
    guards: Sequence[GenerationGuard],
    logits: Tensor | None = None,
) -> bool:
    """Run all guards and handle their results.

    Args:
        request: The request being processed.
        guards: Sequence of guards to run.
        logits: Optional logits from the last forward pass.

    Returns:
        True if generation should continue, False if it should stop.
    """
    for guard in guards:
        result = guard.check(
            request.output_tokens,
            request.prompt_tokens,
            logits,
        )
        if result.action == "stop":
            request.finish("guard", result.message)
            return False
        elif result.action == "warn" and result.message:
            request.add_warning(result.message)
    return True


def _finalize_prefill(
    request: Request,
    next_token: int,
    block_pool: BlockPool,
    guards: Sequence[GenerationGuard],
    logits: Tensor,
) -> None:
    """Finalize prefill: update state, run guards, mark context."""
    request.output_tokens.append(next_token)
    request.kv_cache.append_token(block_pool)
    request.state = RequestState.DECODE
    if guards:
        _run_guards(request, guards, logits)
    if request.context:
        request.context.mark(Event.PREFILLED, first_token=next_token)


def run_prefill(
    request: Request,
    model: TransformerModel,
    block_pool: BlockPool,
    device: str,
    guards: Sequence[GenerationGuard] = (),
) -> int:
    """Run prefill phase: process prompt and generate first token."""
    with torch.inference_mode():
        prompt_len = len(request.prompt_tokens)
        token_ids = torch.tensor([request.prompt_tokens], device=device)
        positions = torch.arange(prompt_len, device=device).unsqueeze(0)
        logits = model.forward(token_ids, positions, [request.kv_cache], block_pool)

        past_tokens = (
            request.prompt_tokens if request.repetition_penalty != 1.0 else None
        )
        next_token = _sample_next_token(logits, request, past_tokens)
        _finalize_prefill(request, next_token, block_pool, guards, logits)
        return next_token


def _finalize_decode_step(
    request: Request,
    next_token: int,
    block_pool: BlockPool,
    guards: Sequence[GenerationGuard],
    logits: Tensor,
) -> None:
    """Finalize decode step: update request state and run guards."""
    request.output_tokens.append(next_token)
    request.kv_cache.append_token(block_pool)
    if request.context:
        request.context.mark(
            Event.DECODE, token_idx=len(request.output_tokens), token_id=next_token
        )
    if guards:
        _run_guards(request, guards, logits)
    if request.is_finished:
        request.mark_finished()


def run_decode_batch(
    requests: list[Request],
    model: TransformerModel,
    block_pool: BlockPool,
    device: str,
    guards: Sequence[GenerationGuard] = (),
) -> list[int | None]:
    """Run one batched decode step for multiple requests."""
    active = [(i, r) for i, r in enumerate(requests) if not r.is_finished]
    if not active:
        return [None] * len(requests)

    with torch.inference_mode():
        token_ids = torch.tensor(
            [[r.output_tokens[-1]] for _, r in active], device=device
        )
        positions = torch.tensor(
            [[r.kv_cache.num_tokens - 1] for _, r in active], device=device
        )
        logits = model.forward(
            token_ids, positions, [r.kv_cache for _, r in active], block_pool
        )

        results: list[int | None] = [None] * len(requests)
        for batch_idx, (orig_idx, request) in enumerate(active):
            past_tokens = (
                request.prompt_tokens + request.output_tokens
                if request.repetition_penalty != 1.0
                else None
            )
            batch_logits = logits[batch_idx : batch_idx + 1]
            next_token = _sample_next_token(batch_logits, request, past_tokens)
            _finalize_decode_step(request, next_token, block_pool, guards, batch_logits)
            results[orig_idx] = next_token
        return results


def _prepare_decode_tensors(
    request: Request, device: str, buffers: dict[str, Tensor] | None
) -> tuple[Tensor, Tensor]:
    """Prepare token_ids and positions tensors for decode step."""
    if buffers:
        buffers["token_ids"][0, 0] = request.output_tokens[-1]
        buffers["positions"][0, 0] = request.kv_cache.num_tokens - 1
        return buffers["token_ids"], buffers["positions"]
    return (
        torch.tensor([[request.output_tokens[-1]]], device=device),
        torch.tensor([[request.kv_cache.num_tokens - 1]], device=device),
    )


def run_decode(
    request: Request,
    model: TransformerModel,
    block_pool: BlockPool,
    device: str,
    guards: Sequence[GenerationGuard] = (),
    buffers: dict[str, Tensor] | None = None,
) -> int | None:
    """Run one decode step for a request."""
    if request.is_finished:
        return None

    with torch.inference_mode():
        token_ids, positions = _prepare_decode_tensors(request, device, buffers)
        logits = model.forward(token_ids, positions, [request.kv_cache], block_pool)

        past_tokens = (
            request.prompt_tokens + request.output_tokens
            if request.repetition_penalty != 1.0
            else None
        )
        next_token = _sample_next_token(logits, request, past_tokens)
        _finalize_decode_step(request, next_token, block_pool, guards, logits)
        return next_token
