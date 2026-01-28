"""Map OpenAI request parameters to internal request format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm_infer.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    FinishReason,
    Role,
    Tool,
    ToolChoice,
)

from ...dispatch.types import Request as InternalRequest

if TYPE_CHECKING:
    from llm_infer.models.config import ModelConfig


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


def tools_to_dict(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    """Convert Tool objects to dict format for internal/backend use."""
    if tools is None:
        return None
    return [tool.model_dump(exclude_none=True) for tool in tools]


def tool_choice_to_dict(
    tool_choice: ToolChoice | None,
) -> str | dict[str, Any] | None:
    """Convert ToolChoice to dict/string format for internal/backend use."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    # ToolChoiceObject
    return tool_choice.model_dump(exclude_none=True)


def resolve_think_mode(think: bool | None, model_config: ModelConfig | None) -> bool:
    """Resolve effective think mode from request and model config default."""
    if think is not None:
        return think
    if model_config is not None:
        return model_config.think.default
    return False


def _get_think_suffix(think: bool | None, model_config: ModelConfig | None) -> str:
    """Get the appropriate think suffix based on request and model config."""
    if model_config is None:
        return ""
    effective_think = resolve_think_mode(think, model_config)
    think_config = model_config.think
    if effective_think and think_config.enable_suffix:
        return think_config.enable_suffix
    if not effective_think and think_config.disable_suffix:
        return think_config.disable_suffix
    return ""


def _get_system_prompt(
    think: bool | None, model_config: ModelConfig | None
) -> str | None:
    """Get the appropriate system prompt based on request and model config."""
    if model_config is None:
        return None
    # Use think-specific system prompt when think mode is enabled (explicit or default)
    effective_think = resolve_think_mode(think, model_config)
    if effective_think and model_config.think.system_prompt:
        return model_config.think.system_prompt
    # Fall back to model's default system prompt
    return model_config.system_prompt


def _has_system_message(body: ChatCompletionRequest) -> bool:
    """Check if request already has a system message."""
    return any(msg.role == Role.SYSTEM for msg in body.messages)


def _inject_think_suffix(messages: list[dict[str, str]], suffix: str) -> str | None:
    """Inject think suffix into last user message, returning updated content.

    If no user message exists (rare edge case), appends to last message anyway.
    Returns the content of the modified message, or None if no suffix provided.
    """
    if not suffix:
        return None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            messages[i]["content"] += suffix
            return messages[i]["content"]
    # No user message - append to last message (e.g., prefill scenarios)
    messages[-1]["content"] += suffix
    return messages[-1]["content"]


def _message_to_dict(msg: ChatMessage) -> dict[str, Any]:
    """Convert ChatMessage to dict format, including tool calling fields."""
    result: dict[str, Any] = {"role": msg.role.value, "content": msg.content or ""}

    # Include tool_calls for assistant messages
    if msg.tool_calls:
        result["tool_calls"] = [
            tc.model_dump(exclude_none=True) for tc in msg.tool_calls
        ]

    # Include tool_call_id for tool response messages
    if msg.tool_call_id:
        result["tool_call_id"] = msg.tool_call_id

    return result


def _has_tool_messages(body: ChatCompletionRequest) -> bool:
    """Check if request contains tool-related messages."""
    return any(
        msg.tool_calls or msg.tool_call_id or msg.role == Role.TOOL
        for msg in body.messages
    )


def _build_messages_with_injections(
    body: ChatCompletionRequest, think_suffix: str, system_prompt: str | None
) -> tuple[str, list[dict[str, Any]] | None]:
    """Build messages list, injecting system prompt and think suffix as needed.

    Returns:
        Tuple of (prompt, messages) where prompt is the last user message content
        (with think suffix if applied) for logging/display, and messages is the
        full message list for template processing (or None for single-message case).
    """
    # Single user message with no system prompt to inject and no tool messages:
    # pass content directly
    if (
        len(body.messages) == 1
        and body.messages[0].role == Role.USER
        and not system_prompt
        and not _has_tool_messages(body)
    ):
        prompt = (body.messages[0].content or "") + think_suffix
        return prompt, None

    # Build messages list for template, including tool calling fields
    messages = [_message_to_dict(msg) for msg in body.messages]

    # Inject model config system prompt only if request doesn't already have one
    if system_prompt and not _has_system_message(body):
        messages.insert(0, {"role": "system", "content": system_prompt})

    # Inject think suffix and get prompt for logging/display
    prompt = body.messages[-1].content or ""
    if injected_prompt := _inject_think_suffix(messages, think_suffix):
        prompt = injected_prompt

    return prompt, messages


def chat_request_to_internal(
    body: ChatCompletionRequest,
    request_id: str,
    model_config: ModelConfig | None = None,
) -> InternalRequest:
    """Convert OpenAI chat completion request to internal request format."""
    think_suffix = _get_think_suffix(body.think, model_config)
    system_prompt = _get_system_prompt(body.think, model_config)
    prompt, messages = _build_messages_with_injections(
        body, think_suffix, system_prompt
    )

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
        adapter_id=body.adapter_id,
        tools=tools_to_dict(body.tools),
        tool_choice=tool_choice_to_dict(body.tool_choice),
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
        adapter_id=body.adapter_id,  # LoRA adapter selection
    )


def determine_finish_reason(
    is_eos: bool,
    max_tokens_reached: bool,
    guard_triggered: bool = False,
    has_tool_calls: bool = False,
) -> FinishReason:
    """Determine OpenAI finish_reason from internal state."""
    if guard_triggered:
        return FinishReason.CONTENT_FILTER
    if has_tool_calls:
        return FinishReason.TOOL_CALLS
    if max_tokens_reached:
        return FinishReason.LENGTH
    return FinishReason.STOP
