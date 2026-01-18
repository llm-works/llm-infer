"""Transformer model implementation.

This module provides a Llama-style transformer model with paged attention,
compatible with HuggingFace model weights in safetensors format.

Supported architectures:
    - LLaMA (1, 2, 3.x)
    - Mistral
    - Qwen2
    - Any model using the same weight naming conventions

Memory efficiency:
    Weight loading uses a streaming approach to minimize peak GPU memory.
    Instead of loading all weights to GPU then converting dtype (2x peak memory),
    we load one tensor at a time to CPU, convert dtype, copy to GPU, then free.
    This allows loading models that would otherwise OOM during the dtype conversion.

    The naive approach for loading a 24B model (Mistral-22B):
        1. Load all weights to GPU: ~48GB (bfloat16 stored weights)
        2. Call model.to(dtype=float16): creates copies during conversion
        3. Peak memory: ~90-96GB (OOM on 95GB GPU)

    The streaming approach:
        1. Create empty model on GPU with target dtype: ~48GB
        2. Stream weights one tensor at a time via CPU: +1GB temporary
        3. Peak memory: ~50GB (fits comfortably on 95GB GPU)

    Key insight: dtype conversion on CPU is "free" (system RAM is abundant),
    while dtype conversion on GPU doubles memory usage temporarily.

Example:
    >>> from llm_infer.model import ModelConfig, TransformerModel
    >>> config = ModelConfig.from_hf_config("/path/to/model")
    >>> model = TransformerModel(config, "/path/to/model", device="cuda")
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors import safe_open
from torch import Tensor, nn

from ...backends.linear.formats.base import QuantizedLinearBackend
from ...primitives.attention import (
    apply_rope,
    get_attention_backend,
    precompute_rope_freqs,
    update_kv_cache,
)
from ...primitives.kv_cache import BlockPool, SequenceKVCache
from ...primitives.protocols import AttentionBackend
from .architecture import ModelArchitecture
from .config import ModelConfig
from .layers import AWQLinear, Fp8Linear, RMSNorm


class TransformerModel(nn.Module):
    """Llama-style transformer with paged attention.

    This implements a decoder-only transformer architecture with:
    - RMSNorm for layer normalization
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (GQA) via num_kv_heads
    - SwiGLU activation in the MLP
    - Paged KV cache for memory-efficient inference

    The model supports various configurations through ModelConfig, including
    models where head_dim != hidden_size // num_heads (e.g., Mistral-22B).

    Attributes:
        config: Model configuration (dimensions, layers, etc.)
        arch: Model architecture (forward-pass behavior)
        device: Target device for inference ("cuda" or "cpu")
        dtype: Target dtype for weights (torch.float16 or torch.bfloat16)
        embed_tokens: Token embedding layer
        layers: ModuleList of transformer blocks
        norm: Final RMSNorm before lm_head
        lm_head: Linear projection to vocabulary logits
        cos, sin: Precomputed RoPE frequency tensors
    """

    def __init__(
        self,
        config: ModelConfig,
        arch: ModelArchitecture,
        weights_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        on_progress: Callable[[str, int, int], None] | None = None,
        attention_backend: AttentionBackend | None = None,
        linear_backend: QuantizedLinearBackend | None = None,
    ):
        """Initialize the transformer model.

        Args:
            config: Model configuration from ModelConfig.from_hf_config()
            arch: Model architecture instance for forward-pass behavior
            weights_path: Path to model directory containing .safetensors files,
                or path to a single .safetensors file
            device: Target device ("cuda" or "cpu")
            dtype: Target dtype for model weights. Common choices:
                - torch.float16: Good balance of speed/memory on most GPUs
                - torch.bfloat16: Better numerical stability, requires Ampere+
            on_progress: Optional callback for loading progress.
                Called with (phase, current, total) where phase is one of:
                - "init": Layer initialization (0 to num_layers)
                - "alloc": GPU allocation (0 to 1)
                - "stream": Weight streaming (0 to num_tensors)
            attention_backend: Attention backend to use. If None, uses
                auto-detected backend (FlashInfer if available, else Naive).
            linear_backend: Backend for quantized linear layers (AWQ/FP8).

        Raises:
            ValueError: If no safetensors files found at weights_path
        """
        super().__init__()
        self.config = config
        self.arch = arch
        self.device = device
        self.dtype = dtype

        # Attention backend (auto-detect if not provided)
        self.attention_backend = attention_backend or get_attention_backend()

        # Linear backend for quantized layers
        self._linear_backend = linear_backend

        # Initialize layers with progress reporting
        self._init_layers(on_progress=on_progress)
        self._load_weights(weights_path, on_progress=on_progress)

        # Precompute RoPE frequencies for position embeddings
        self.cos, self.sin = precompute_rope_freqs(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
            device,
            dtype,
        )

    def _create_linear(
        self, in_features: int, out_features: int, bias: bool = False
    ) -> nn.Module:
        """Create Linear, AWQLinear, or Fp8Linear depending on quantization config."""
        if self.config.quant_method == "fp8":
            return Fp8Linear(
                in_features,
                out_features,
                self.config.quant_group_size,
                backend=self._linear_backend,
            )
        if self.config.quant_method == "awq":
            return AWQLinear(
                in_features,
                out_features,
                self.config.quant_group_size,
                bias=bias,
                backend=self._linear_backend,
            )
        return nn.Linear(in_features, out_features, bias=bias)

    def _create_attn_projs(self, cfg: Any) -> dict[str, nn.Module]:
        """Create attention projection layers."""
        kv_dim = cfg.num_kv_heads * cfg.head_dim
        return {
            "q_proj": self._create_linear(
                cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=cfg.attention_bias
            ),
            "k_proj": self._create_linear(
                cfg.hidden_size, kv_dim, bias=cfg.attention_bias
            ),
            "v_proj": self._create_linear(
                cfg.hidden_size, kv_dim, bias=cfg.attention_bias
            ),
            "o_proj": self._create_linear(
                cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False
            ),
        }

    def _create_layer_dict(self) -> dict[str, nn.Module]:
        """Create a single transformer layer's modules."""
        cfg = self.config
        layer_dict: dict[str, nn.Module] = {
            "input_layernorm": RMSNorm(cfg.hidden_size, cfg.rms_norm_eps),
            "post_attention_layernorm": RMSNorm(cfg.hidden_size, cfg.rms_norm_eps),
            **self._create_attn_projs(cfg),
            "gate_proj": self._create_linear(
                cfg.hidden_size, cfg.intermediate_size, bias=False
            ),
            "up_proj": self._create_linear(
                cfg.hidden_size, cfg.intermediate_size, bias=False
            ),
            "down_proj": self._create_linear(
                cfg.intermediate_size, cfg.hidden_size, bias=False
            ),
        }
        if cfg.qk_norm:
            layer_dict["q_norm"] = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
            layer_dict["k_norm"] = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        return layer_dict

    def _init_layers(
        self,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Initialize model architecture with empty weights."""
        cfg = self.config
        num_layers = cfg.num_layers

        if on_progress:
            on_progress("init", 0, num_layers)

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.ModuleDict(self._create_layer_dict()))
            if on_progress:
                on_progress("init", i + 1, num_layers)

        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def _get_safetensor_files(self, weights_path: str) -> list[Path]:
        """Get list of safetensor files from path (file or directory)."""
        path = Path(weights_path)
        if path.is_dir():
            files = sorted(path.glob("*.safetensors"))
            if not files:
                raise ValueError(f"No safetensors files found in {path}")
            return files
        return [path]

    def _stream_weights(
        self,
        files: list[Path],
        total: int,
        on_progress: Callable | None,
    ) -> None:
        """Stream weights from safetensor files directly to GPU.

        Weights are loaded one tensor at a time, converted to target dtype,
        moved to GPU, and assigned to model parameters. This enables loading
        quantized models without OOM from bulk allocation.
        """
        if on_progress:
            on_progress("stream", 0, total)
        loaded = 0
        for sf_path in files:
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    # Quantized tensors should not have dtype converted:
                    # - AWQ: qweight, qzeros are int32
                    # - FP8: weights are float8_e4m3fn, scales are float16
                    if not self._is_awq_int_tensor(key) and not self._is_fp8_tensor(
                        key
                    ):
                        tensor = tensor.to(dtype=self.dtype)
                    # Move to target device and assign (not copy) to parameter
                    tensor = tensor.to(device=self.device)
                    self._assign_weight(key, tensor)
                    del tensor
                    loaded += 1
                    if on_progress:
                        on_progress("stream", loaded, total)

    def _is_awq_int_tensor(self, key: str) -> bool:
        """Check if tensor is an AWQ integer tensor (qweight or qzeros)."""
        if self.config.quant_method != "awq":
            return False
        return ".qweight" in key or ".qzeros" in key

    def _is_fp8_tensor(self, key: str) -> bool:
        """Check if tensor is FP8 and should not have dtype converted.

        FP8 models store projection weights in float8_e4m3fn format with
        weight_scale_inv scales. These should be loaded directly without
        dtype conversion. Non-projection weights (embeddings, layer norms,
        lm_head) are still FP16/BF16 and should be converted.
        """
        if self.config.quant_method != "fp8":
            return False
        # Only projection weights and their scales are FP8
        fp8_projs = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        for proj in fp8_projs:
            if key.endswith(f".{proj}.weight") or key.endswith(
                f".{proj}.weight_scale_inv"
            ):
                return True
        return False

    def _load_weights(
        self, weights_path: str, on_progress: Callable | None = None
    ) -> None:
        """Load weights from safetensors with memory-efficient streaming."""
        files = self._get_safetensor_files(weights_path)

        # NOTE: We don't pre-allocate the model on GPU here.
        # Instead, weights are streamed directly to GPU one tensor at a time.
        # This enables loading quantized models (FP8, INT8) without OOM.

        # Count tensors for progress
        total_tensors = 0
        if on_progress:
            for sf_path in files:
                with safe_open(sf_path, framework="pt", device="cpu") as f:
                    total_tensors += len(f.keys())

        self._stream_weights(files, total_tensors, on_progress)

        # Handle weight tying
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _assign_attn_weight(
        self, key: str, layer: nn.ModuleDict, tensor: Tensor
    ) -> bool:
        """Assign attention weight. Returns True if handled."""
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if key.endswith(f".{proj}.weight"):
                layer[proj].weight.data = tensor  # type: ignore[operator]
                return True
            if key.endswith(f".{proj}.bias") and layer[proj].bias is not None:
                layer[proj].bias.data = tensor  # type: ignore[operator]
                return True
        for norm in ["q_norm", "k_norm"]:
            if key.endswith(f".{norm}.weight") and norm in layer:
                layer[norm].weight.data = tensor  # type: ignore[operator]
                return True
        return False

    def _assign_layer_weight(
        self, key: str, layer: nn.ModuleDict, tensor: Tensor
    ) -> bool:
        """Assign weight to layer component. Returns True if handled."""
        if self.config.quant_method == "awq" and self._assign_awq_layer_weight(
            key, layer, tensor
        ):
            return True
        if self.config.quant_method == "fp8" and self._assign_fp8_layer_weight(
            key, layer, tensor
        ):
            return True

        if "self_attn" in key and self._assign_attn_weight(key, layer, tensor):
            return True
        if "mlp" in key:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                if key.endswith(f".{proj}.weight"):
                    layer[proj].weight.data = tensor  # type: ignore[operator]
                    return True
        if key.endswith(".input_layernorm.weight"):
            layer["input_layernorm"].weight.data = tensor  # type: ignore[operator]
            return True
        if key.endswith(".post_attention_layernorm.weight"):
            layer["post_attention_layernorm"].weight.data = tensor  # type: ignore[operator]
            return True
        return False

    def _assign_awq_layer_weight(
        self, key: str, layer: nn.ModuleDict, tensor: Tensor
    ) -> bool:
        """Assign AWQ quantized weight tensors. Returns True if handled."""
        all_projs = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        for proj in all_projs:
            if key.endswith(f".{proj}.qweight"):
                layer[proj].qweight.data = tensor  # type: ignore[operator]
                return True
            if key.endswith(f".{proj}.qzeros"):
                layer[proj].qzeros.data = tensor  # type: ignore[operator]
                return True
            if key.endswith(f".{proj}.scales"):
                layer[proj].scales.data = tensor  # type: ignore[operator]
                return True

        return False

    def _assign_fp8_layer_weight(
        self, key: str, layer: nn.ModuleDict, tensor: Tensor
    ) -> bool:
        """Assign FP8 quantized weight tensors. Returns True if handled."""
        all_projs = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        for proj in all_projs:
            if key.endswith(f".{proj}.weight"):
                layer[proj].weight.data = tensor  # type: ignore[operator]
                return True
            if key.endswith(f".{proj}.weight_scale_inv"):
                layer[proj].weight_scale_inv.data = tensor  # type: ignore[operator]
                return True

        return False

    def _assign_weight(self, key: str, tensor: Tensor) -> None:
        """Assign a weight tensor directly to the model parameter.

        Uses direct assignment (.data = tensor) instead of copy to avoid
        allocating duplicate memory. Tensor should already be on target device.
        """
        prefix = "model."

        if key == f"{prefix}embed_tokens.weight":
            self.embed_tokens.weight.data = tensor
        elif key == f"{prefix}norm.weight":
            self.norm.weight.data = tensor
        elif key == "lm_head.weight":
            self.lm_head.weight.data = tensor
        elif key.startswith(f"{prefix}layers."):
            parts = key[len(prefix) :].split(".")
            layer_idx = int(parts[1])
            layer_dict: nn.ModuleDict = self.layers[layer_idx]  # type: ignore[assignment]
            self._assign_layer_weight(key, layer_dict, tensor)

    def forward(
        self,
        token_ids: Tensor,
        positions: Tensor,
        kv_caches: list[SequenceKVCache],
        block_pool: BlockPool,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            token_ids: Input token IDs [batch, seq_len]
            positions: Position indices [batch, seq_len]
            kv_caches: Per-sequence KV cache metadata
            block_pool: Paged KV cache storage

        Returns:
            Logits for next token [batch, vocab_size]
        """
        hidden = self.embed_tokens(token_ids)
        hidden = self.arch.scale_embeddings(hidden)

        for layer_idx, layer in enumerate(self.layers):
            layer_dict: nn.ModuleDict = layer  # type: ignore[assignment]
            hidden = self._transformer_block(
                hidden, positions, layer_idx, layer_dict, kv_caches, block_pool
            )

        hidden = self.norm(hidden)
        logits: Tensor = self.lm_head(hidden[:, -1, :])
        logits = self.arch.scale_logits(logits)
        return logits

    def _compute_qkv(
        self, hidden: Tensor, layer: nn.ModuleDict
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute Q, K, V projections from hidden states."""
        batch, seq_len, _ = hidden.shape
        q = layer["q_proj"](hidden).view(
            batch, seq_len, self.config.num_heads, self.config.head_dim
        )
        k = layer["k_proj"](hidden).view(
            batch, seq_len, self.config.num_kv_heads, self.config.head_dim
        )
        v = layer["v_proj"](hidden).view(
            batch, seq_len, self.config.num_kv_heads, self.config.head_dim
        )
        return q, k, v

    def _apply_mlp(self, hidden: Tensor, layer: nn.ModuleDict) -> Tensor:
        """Apply SwiGLU MLP: gate * up -> down."""
        hidden = layer["post_attention_layernorm"](hidden)
        gate = F.silu(layer["gate_proj"](hidden))
        up = layer["up_proj"](hidden)
        result: Tensor = layer["down_proj"](gate * up)
        return result

    def _transformer_block(
        self,
        hidden: Tensor,
        positions: Tensor,
        layer_idx: int,
        layer: nn.ModuleDict,
        kv_caches: list[SequenceKVCache],
        block_pool: BlockPool,
    ) -> Tensor:
        """Execute a single transformer block (attention + MLP)."""
        residual = hidden
        hidden = layer["input_layernorm"](hidden)

        q, k, v = self._compute_qkv(hidden, layer)
        q, k = self.arch.apply_qk_norm(q, k, layer)
        q, k = apply_rope(q, k, self.cos, self.sin, positions)
        update_kv_cache(k, v, layer_idx, kv_caches, block_pool)

        batch, seq_len = hidden.shape[:2]
        attn_out = self.attention_backend.forward(
            q,
            layer_idx,
            kv_caches,
            block_pool,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.arch.get_attention_scale(),
        )

        hidden = layer["o_proj"](attn_out.view(batch, seq_len, -1))
        hidden = self.arch.apply_residual(residual, hidden)

        residual = hidden
        hidden = self._apply_mlp(hidden, layer)
        return self.arch.apply_residual(residual, hidden)
