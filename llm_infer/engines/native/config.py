"""Engine configuration."""

from dataclasses import dataclass
from pathlib import Path

import torch

from .model import TransformerConfig


@dataclass
class EngineConfig:
    """Configuration for the inference engine."""

    model: TransformerConfig
    model_path: str
    max_batch_size: int = 32
    num_blocks: int = 1024
    block_size: int = 16  # tokens per block
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    attention_backend: str = "auto"  # auto, flashinfer, naive
    linear_backend: str = "pytorch"  # pytorch, marlin
    torch_compile: bool = False  # Use torch.compile for reduced CPU overhead
    warmup: bool = False  # Run warmup query on startup

    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length based on available blocks."""
        return self.num_blocks * self.block_size

    def validate(self) -> None:
        """Validate configuration."""
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if not Path(self.model_path).exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")
        if self.torch_compile and self.attention_backend in ("flashinfer", "auto"):
            raise ValueError(
                "torch_compile is incompatible with attention_backend='flashinfer' "
                "(or 'auto' which may resolve to flashinfer). "
                "Use attention_backend='naive' or disable torch_compile."
            )
