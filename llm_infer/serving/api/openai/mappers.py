"""Map OpenAI request parameters to internal request format."""

from ...dispatch.types import Request as InternalRequest
from .schemas import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    FinishReason,
    Role,
)


def format_messages_as_prompt(messages: list[ChatMessage]) -> str:
    """
    Format chat messages as a prompt string.

    This is a simple fallback format. The actual formatting should happen
    in the tokenizer using the model's chat template (use_chat_template=True).
    """
    parts = []
    for msg in messages:
        parts.append(f"{msg.role.value}: {msg.content}")
    return "\n".join(parts)


def map_frequency_penalty(frequency_penalty: float) -> float:
    """
    Map OpenAI frequency_penalty to internal repetition_penalty.

    OpenAI: -2.0 to 2.0 (0 = no penalty, positive = discourage repetition)
    Internal: 1.0 to 2.0 (1.0 = no penalty, higher = discourage repetition)

    Mapping: 0 -> 1.0, 2.0 -> 2.0, negative values -> 1.0
    """
    if frequency_penalty <= 0:
        return 1.0
    # Linear map: 0 -> 1.0, 2.0 -> 2.0
    return 1.0 + (frequency_penalty / 2.0)


def normalize_stop_sequences(stop: str | list[str] | None) -> list[str] | None:
    """Convert stop parameter to list of strings."""
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return stop


def chat_request_to_internal(
    body: ChatCompletionRequest,
    request_id: str,
) -> InternalRequest:
    """Convert OpenAI chat completion request to internal request format."""
    # For single user message, pass content directly - the tokenizer's encode_chat
    # will wrap it with proper chat template (adding role markers).
    # For multi-turn or system messages, pass full messages list.
    messages: list[dict[str, str]] | None = None
    if len(body.messages) == 1 and body.messages[0].role == Role.USER:
        prompt = body.messages[0].content or ""
    else:
        # Multi-turn or system messages - pass full messages list for template
        messages = [
            {"role": msg.role.value, "content": msg.content or ""}
            for msg in body.messages
        ]
        # Use last user message as prompt fallback (for logging/display)
        prompt = body.messages[-1].content or ""

    return InternalRequest(
        id=request_id,
        prompt=prompt,
        max_tokens=body.max_tokens or 256,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=0,  # OpenAI doesn't expose top_k
        repetition_penalty=map_frequency_penalty(body.frequency_penalty),
        stream=body.stream,
        use_chat_template=None,  # Let engine auto-detect based on model type
        stop_sequences=normalize_stop_sequences(body.stop),
        messages=messages,
    )


def completion_request_to_internal(
    body: CompletionRequest,
    request_id: str,
) -> InternalRequest:
    """Convert OpenAI legacy completion request to internal request format."""
    # Handle prompt as string or list
    prompt = body.prompt if isinstance(body.prompt, str) else body.prompt[0]

    return InternalRequest(
        id=request_id,
        prompt=prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=0,
        repetition_penalty=map_frequency_penalty(body.frequency_penalty),
        stream=body.stream,
        use_chat_template=False,  # Raw completion, no chat template
        stop_sequences=normalize_stop_sequences(body.stop),
    )


def determine_finish_reason(
    is_eos: bool,
    max_tokens_reached: bool,
    guard_triggered: bool = False,
) -> FinishReason:
    """Determine OpenAI finish_reason from internal state."""
    if guard_triggered:
        return FinishReason.CONTENT_FILTER
    if max_tokens_reached:
        return FinishReason.LENGTH
    return FinishReason.STOP
