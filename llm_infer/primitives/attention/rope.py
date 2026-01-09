"""Rotary Position Embedding (RoPE) functions."""

import torch
from torch import Tensor


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[Tensor, Tensor]:
    """Precompute RoPE sin/cos frequencies."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(positions, freqs)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims (used in LLaMA-style RoPE)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    positions: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to Q and K.

    Uses LLaMA-style half-split rotation (first/second half of head_dim).
    """
    # q, k: [batch, seq_len, num_heads, head_dim]
    cos = cos[positions]  # [batch, seq_len, head_dim/2]
    sin = sin[positions]  # [batch, seq_len, head_dim/2]

    # Expand cos/sin to full head_dim by repeating
    cos = torch.cat([cos, cos], dim=-1)  # [batch, seq_len, head_dim]
    sin = torch.cat([sin, sin], dim=-1)  # [batch, seq_len, head_dim]

    # Reshape for broadcasting: [batch, seq_len, 1, head_dim]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    # Apply rotation: q * cos + rotate_half(q) * sin
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)

    return q_out, k_out
