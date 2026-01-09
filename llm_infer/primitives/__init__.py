"""Primitives layer - building blocks with clean Protocol interfaces."""

# Protocols
# Attention
from .attention import (
    FLASHINFER_AVAILABLE,
    FlashInferBackend,
    NaiveAttentionBackend,
    apply_rope,
    get_attention_backend,
    precompute_rope_freqs,
    update_kv_cache,
)

# Guards
from .guards import GenerationGuard, GuardResult, RepetitionGuard

# KV Cache implementations
from .kv_cache import BlockPool, SequenceKVCache
from .protocols import (
    AttentionBackend,
    BlockAllocator,
    ExecutionBackend,
    KVCache,
    KVCacheStorage,
    RequestProtocol,
    SchedulerProtocol,
    Tokenizer,
)

# Sampler
from .sampler import sample

# Tokenizer implementations
from .tokenizer import HuggingFaceTokenizer, TokenizerConfig

__all__ = [
    # Protocols
    "AttentionBackend",
    "BlockAllocator",
    "ExecutionBackend",
    "KVCache",
    "KVCacheStorage",
    "RequestProtocol",
    "SchedulerProtocol",
    "Tokenizer",
    # KV Cache
    "BlockPool",
    "SequenceKVCache",
    # Tokenizer
    "HuggingFaceTokenizer",
    "TokenizerConfig",
    # Attention
    "FLASHINFER_AVAILABLE",
    "FlashInferBackend",
    "NaiveAttentionBackend",
    "apply_rope",
    "get_attention_backend",
    "precompute_rope_freqs",
    "update_kv_cache",
    # Sampler
    "sample",
    # Guards
    "GenerationGuard",
    "GuardResult",
    "RepetitionGuard",
]
