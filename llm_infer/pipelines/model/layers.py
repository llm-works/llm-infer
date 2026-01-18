"""Neural network layer implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ...backends.linear.formats import AWQWeights, FP8Weights, QuantFormat

if TYPE_CHECKING:
    from ...backends.linear.formats.base import QuantizedLinearBackend


class QuantizedLinear(nn.Module):
    """Unified quantized linear layer supporting multiple formats.

    This is the base implementation for quantized linear layers. It delegates
    the actual matmul to a format-specific backend (e.g., PyTorch, Marlin, CUTLASS).

    Supported formats:
    - AWQ: 4-bit with per-group scales (group_size typically 128)
    - FP8: 8-bit floating point with per-block scales (block_size typically 128)

    Backend must be provided at construction or set via the backend property
    before forward() is called. Use BackendRegistry to select the appropriate
    backend for your hardware.
    """

    # Type hints for registered buffers
    qweight: Tensor
    qzeros: Tensor
    scales: Tensor
    bias: Tensor | None
    weight: Tensor
    weight_scale_inv: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        format: QuantFormat,
        group_size: int = 128,  # For AWQ
        block_size: int = 128,  # For FP8
        bias: bool = False,
        backend: QuantizedLinearBackend | None = None,
    ):
        """Initialize quantized linear layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            format: Quantization format (AWQ, FP8)
            group_size: For AWQ - weights per quantization group
            block_size: For FP8 - quantization block size
            bias: Whether to include bias (typically False for quantized)
            backend: Backend for matmul. If None, must be set via backend property
                before forward() is called.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.format = format
        self.group_size = group_size
        self.block_size = block_size

        # Store backend (can be set later via backend property)
        self._backend = backend

        # Register format-specific buffers
        if format == QuantFormat.AWQ:
            self._init_awq_buffers(bias)
        elif format == QuantFormat.FP8:
            self._init_fp8_buffers()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _validate_awq_dims(self, pack_factor: int) -> None:
        """Validate AWQ dimension requirements."""
        if self.in_features % self.group_size != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be divisible by group_size ({self.group_size})"
            )
        if self.out_features % pack_factor != 0:
            raise ValueError(
                f"out_features ({self.out_features}) must be divisible by pack_factor ({pack_factor})"
            )

    def _init_awq_buffers(self, bias: bool) -> None:
        """Initialize AWQ-specific buffers."""
        pack_factor = 8  # 8 x INT4 per INT32
        self._validate_awq_dims(pack_factor)
        num_groups = self.in_features // self.group_size

        self.register_buffer(
            "qweight",
            torch.zeros(
                self.in_features, self.out_features // pack_factor, dtype=torch.int32
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                num_groups, self.out_features // pack_factor, dtype=torch.int32
            ),
        )
        self.register_buffer(
            "scales", torch.zeros(num_groups, self.out_features, dtype=torch.float16)
        )
        self.register_buffer(
            "bias",
            torch.zeros(self.out_features, dtype=torch.float16) if bias else None,
        )

    def _init_fp8_buffers(self) -> None:
        """Initialize FP8-specific buffers."""
        in_blocks = (self.in_features + self.block_size - 1) // self.block_size
        out_blocks = (self.out_features + self.block_size - 1) // self.block_size

        self.register_buffer(
            "weight",
            torch.empty(self.out_features, self.in_features, dtype=torch.float8_e4m3fn),
        )
        self.register_buffer(
            "weight_scale_inv",
            torch.empty(out_blocks, in_blocks, dtype=torch.float16),
        )

    @property
    def backend(self) -> QuantizedLinearBackend:
        """Get the backend."""
        if self._backend is None:
            raise RuntimeError(
                "Backend not set. Pass backend= to QuantizedLinear constructor."
            )
        return self._backend

    @backend.setter
    def backend(self, value: QuantizedLinearBackend) -> None:
        """Set the backend."""
        self._backend = value

    def _get_weights(self) -> AWQWeights | FP8Weights:
        """Get weights container for the current format."""
        if self.format == QuantFormat.AWQ:
            return AWQWeights(
                qweight=self.qweight,
                scales=self.scales,
                qzeros=self.qzeros,
                group_size=self.group_size,
                bias=self.bias,
            )
        else:  # FP8
            return FP8Weights(
                weight=self.weight,
                weight_scale_inv=self.weight_scale_inv,
                block_size=self.block_size,
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with backend-specific quantized matmul.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        return self.backend.forward(x, self._get_weights())

    def extra_repr(self) -> str:
        """String representation for printing."""
        backend_name = self._backend.name if self._backend else "not set"
        if self.format == QuantFormat.AWQ:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"format=AWQ, group_size={self.group_size}, "
                f"bias={self.bias is not None}, backend={backend_name}"
            )
        else:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"format=FP8, block_size={self.block_size}, backend={backend_name}"
            )


# Backward compatibility factory functions


def AWQLinear(  # noqa: N802
    in_features: int,
    out_features: int,
    group_size: int = 128,
    bias: bool = False,
    backend: QuantizedLinearBackend | None = None,
) -> QuantizedLinear:
    """Create an AWQ quantized linear layer.

    This is a factory function that creates a QuantizedLinear with AWQ format.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        group_size: Weights per quantization group (typically 128)
        bias: Whether to include bias
        backend: Backend for matmul (required before forward())

    Returns:
        QuantizedLinear configured for AWQ format
    """
    return QuantizedLinear(
        in_features=in_features,
        out_features=out_features,
        format=QuantFormat.AWQ,
        group_size=group_size,
        bias=bias,
        backend=backend,
    )


def Fp8Linear(  # noqa: N802
    in_features: int,
    out_features: int,
    block_size: int = 128,
    backend: QuantizedLinearBackend | None = None,
) -> QuantizedLinear:
    """Create an FP8 quantized linear layer.

    This is a factory function that creates a QuantizedLinear with FP8 format.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        block_size: Quantization block size (typically 128)
        backend: Backend for matmul (required before forward())

    Returns:
        QuantizedLinear configured for FP8 format
    """
    return QuantizedLinear(
        in_features=in_features,
        out_features=out_features,
        format=QuantFormat.FP8,
        block_size=block_size,
        backend=backend,
    )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Compute in float32 for numerical stability (large values overflow float16)
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)
