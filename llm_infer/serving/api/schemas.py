"""Pydantic models for HTTP API requests and responses."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for text generation."""

    prompt: str = Field(..., description="Input text prompt")
    max_tokens: int = Field(100, ge=1, description="Maximum tokens to generate")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(0, ge=0, description="Top-k sampling (0 = disabled)")
    repetition_penalty: float = Field(
        1.1, ge=1.0, le=2.0, description="Repetition penalty (1.0 = disabled)"
    )
    use_chat_template: bool | None = Field(
        None,
        description="Apply chat template formatting. None = auto-detect from model name",
    )


class GenerateResponse(BaseModel):
    """Response body for text generation."""

    text: str = Field(..., description="Generated text")
    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    completion_tokens: int = Field(..., description="Number of generated tokens")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    pending_requests: int = 0
