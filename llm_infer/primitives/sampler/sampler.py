"""Token sampling strategies."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def sample(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    past_tokens: list[int] | None = None,
) -> Tensor:
    """
    Sample next tokens from logits.

    Args:
        logits: Logits tensor of shape [batch, vocab_size]
        temperature: Sampling temperature (0 = greedy, higher = more random)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        top_k: Top-k sampling (0 = disabled)
        repetition_penalty: Penalty for repeating tokens (1.0 = disabled, >1.0 = discourage)
        past_tokens: Previously generated tokens to apply penalty to

    Returns:
        Sampled token ids of shape [batch]
    """
    # Apply repetition penalty before temperature
    if repetition_penalty != 1.0 and past_tokens:
        logits = _apply_repetition_penalty(logits, past_tokens, repetition_penalty)

    if temperature == 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k > 0:
        logits = _apply_top_k(logits, top_k)

    if top_p < 1.0:
        logits = _apply_top_p(logits, top_p)

    probs = F.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _apply_repetition_penalty(
    logits: Tensor, past_tokens: list[int], penalty: float
) -> Tensor:
    """Apply repetition penalty to previously seen tokens.

    For tokens that appeared before:
    - If logit > 0: divide by penalty (reduce probability)
    - If logit < 0: multiply by penalty (reduce probability further)
    """
    logits = logits.clone()
    for token_id in set(past_tokens):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits


def _apply_top_k(logits: Tensor, top_k: int) -> Tensor:
    """Zero out logits below the top-k threshold."""
    top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: Tensor, top_p: float) -> Tensor:
    """Apply nucleus (top-p) sampling."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits.float(), dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_mask = cumulative_probs > top_p
    # Keep at least one token
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    # Scatter mask back to original order
    mask = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))
