"""Public API schemas for llm-infer.

This module provides clean public exports of OpenAI-compatible schemas,
enabling downstream projects (proxies, clients) to import without reaching
into internal module paths.

Usage:
    from llm_infer.api import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
"""

import importlib.util
from pathlib import Path

# Load schemas module directly to avoid circular imports through __init__.py chain.
# The serving.api package has circular dependencies that get triggered when importing
# through the normal package hierarchy.
_schemas_path = (
    Path(__file__).parent.parent / "serving" / "api" / "openai" / "schemas.py"
)
_spec = importlib.util.spec_from_file_location("_openai_schemas", _schemas_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Failed to load schemas module from {_schemas_path}")
_schemas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schemas)

# Re-export all public schemas
Role = _schemas.Role
FinishReason = _schemas.FinishReason
ChatMessage = _schemas.ChatMessage
ChatCompletionChoice = _schemas.ChatCompletionChoice
ChatCompletionRequest = _schemas.ChatCompletionRequest
ChatCompletionResponse = _schemas.ChatCompletionResponse
ChatCompletionUsage = _schemas.ChatCompletionUsage
ChatCompletionChunk = _schemas.ChatCompletionChunk
ChatCompletionChunkChoice = _schemas.ChatCompletionChunkChoice
ChatCompletionChunkDelta = _schemas.ChatCompletionChunkDelta
CompletionChoice = _schemas.CompletionChoice
CompletionChunk = _schemas.CompletionChunk
CompletionChunkChoice = _schemas.CompletionChunkChoice
CompletionRequest = _schemas.CompletionRequest
CompletionResponse = _schemas.CompletionResponse
ModelInfo = _schemas.ModelInfo
ModelList = _schemas.ModelList

__all__ = [
    # Enums
    "Role",
    "FinishReason",
    # Chat messages
    "ChatMessage",
    # Chat completions (non-streaming)
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChoice",
    "ChatCompletionUsage",
    # Chat completions (streaming)
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunkDelta",
    # Legacy completions
    "CompletionRequest",
    "CompletionResponse",
    "CompletionChoice",
    "CompletionChunk",
    "CompletionChunkChoice",
    # Models
    "ModelInfo",
    "ModelList",
]
