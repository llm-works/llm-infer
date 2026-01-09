"""FP8 (8-bit floating point) format definitions.

FP8 uses block-wise scaling for quantization:
- Weights stored as torch.float8_e4m3fn (4 exponent, 3 mantissa bits)
- Scale applied per-block (typically 128x128 blocks)
- No zero points (symmetric quantization)
- Popular for LLM inference due to good accuracy/speed tradeoff
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class FP8Weights:
    """Container for FP8-format quantized weights.

    All tensors are on the same device and ready for computation.

    Attributes:
        weight: FP8 weights [out_features, in_features], dtype float8_e4m3fn
        weight_scale_inv: Per-block inverse scales [out_blocks, in_blocks], dtype float16
            Multiply by this to dequantize (it stores 1/scale or just scale depending on format)
        block_size: Quantization block size (typically 128)

    Example shapes for a (4096, 4096) linear with block_size=128:
        weight: [4096, 4096]
        weight_scale_inv: [32, 32]  (4096 // 128 = 32 blocks per dim)
    """

    weight: Tensor  # [out_features, in_features], dtype float8_e4m3fn
    weight_scale_inv: Tensor  # [out_blocks, in_blocks], dtype float16
    block_size: int

    @property
    def in_features(self) -> int:
        """Input feature dimension."""
        return int(self.weight.shape[1])

    @property
    def out_features(self) -> int:
        """Output feature dimension."""
        return int(self.weight.shape[0])
