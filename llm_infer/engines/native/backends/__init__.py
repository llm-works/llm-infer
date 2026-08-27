# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Backend abstraction layer for inference components.

This module provides pluggable backends at three levels:
- linear: Kernel-level backends for quantized matmul (e.g., pytorch, marlin)
- model: Model-level backends (e.g., native TransformerModel, gptqmodel)
- engine: Full engine backends (e.g., native, vllm)

Higher levels override lower levels in the hierarchy.
"""

from .linear import BackendRegistry, QuantizedLinearBackend

__all__ = [
    "BackendRegistry",
    "QuantizedLinearBackend",
]
