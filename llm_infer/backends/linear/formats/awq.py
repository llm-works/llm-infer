"""AWQ (Activation-aware Weight Quantization) format definitions.

AWQ uses 4-bit integer quantization with per-group scaling:
- Weights packed as 8 x INT4 per INT32 (pack_factor = 8)
- Group size typically 128 weights share one scale/zero pair
- Zero points are asymmetric (not assumed to be 0)
- Uses GEMM interleaved packing [0, 2, 4, 6, 1, 3, 5, 7]
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class AWQWeights:
    """Container for AWQ-format quantized weights.

    All tensors are on the same device and ready for computation.

    Attributes:
        qweight: Packed INT4 weights [in_features, out_features // 8], dtype int32
            Each int32 contains 8 INT4 values in GEMM interleaved order
        scales: Per-group scales [num_groups, out_features], dtype float16
        qzeros: Packed INT4 zero points [num_groups, out_features // 8], dtype int32
        group_size: Number of weights per quantization group
        bias: Optional bias [out_features], dtype float16

    Example shapes for a (4096, 4096) linear with group_size=128:
        qweight: [4096, 512]  (4096 // 8 = 512)
        scales: [32, 4096]    (4096 // 128 = 32 groups)
        qzeros: [32, 512]
    """

    qweight: Tensor  # [in_features, out_features // 8]
    scales: Tensor  # [num_groups, out_features]
    qzeros: Tensor  # [num_groups, out_features // 8]
    group_size: int
    bias: Tensor | None = None

    @property
    def in_features(self) -> int:
        """Input feature dimension."""
        return int(self.qweight.shape[0])

    @property
    def out_features(self) -> int:
        """Output feature dimension."""
        return int(self.qweight.shape[1] * 8)  # pack_factor = 8
