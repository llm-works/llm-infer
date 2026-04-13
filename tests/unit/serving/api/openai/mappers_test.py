"""Unit tests for serving/api/openai/mappers.py.

Existing test_openai_backend.py covers some helpers; this file adds direct
coverage of mapper functions, focusing on the missing branches.
"""

from __future__ import annotations

import pytest

from llm_infer.models.config import ModelConfig, ThinkConfig
from llm_infer.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    FinishReason,
    Role,
    Tool,
)
from llm_infer.serving.api.openai.mappers import (
    _build_messages_with_injections,
    _get_chat_template_kwargs,
    _get_system_prompt,
    _get_think_suffix,
    _has_system_message,
    _has_tool_messages,
    _inject_think_suffix,
    _message_to_dict,
    chat_request_to_internal,
    completion_request_to_internal,
    determine_finish_reason,
    format_messages_as_prompt,
    map_frequency_penalty,
    normalize_arguments,
    normalize_stop_sequences,
    resolve_think_mode,
    response_format_to_dict,
    tool_choice_to_dict,
    tools_to_dict,
)

pytestmark = pytest.mark.unit


def _msg(role: str, content: str = "") -> ChatMessage:
    return ChatMessage(role=Role(role), content=content)


def _model_cfg(
    *,
    think_default: bool = False,
    enable_suffix: str | None = None,
    disable_suffix: str | None = None,
    system_prompt: str | None = None,
    think_system_prompt: str | None = None,
    chat_template_kwargs: dict | None = None,
) -> ModelConfig:
    return ModelConfig(
        name="test",
        system_prompt=system_prompt,
        think=ThinkConfig(
            default=think_default,
            enable_suffix=enable_suffix,
            disable_suffix=disable_suffix,
            system_prompt=think_system_prompt,
        ),
        vllm={"chat_template_kwargs": chat_template_kwargs}
        if chat_template_kwargs is not None
        else {},
    )


# ---------------------------------------------------------------------------
# Simple helpers
# ---------------------------------------------------------------------------


def test_format_messages_as_prompt() -> None:
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    result = format_messages_as_prompt(msgs)
    assert "user: hi" in result
    assert "assistant: hello" in result


class TestMapFrequencyPenalty:
    def test_zero(self) -> None:
        assert map_frequency_penalty(0.0) == 1.0

    def test_negative(self) -> None:
        assert map_frequency_penalty(-1.0) == 1.0

    def test_max(self) -> None:
        assert map_frequency_penalty(2.0) == 2.0

    def test_half(self) -> None:
        assert map_frequency_penalty(1.0) == 1.5


class TestNormalizeStopSequences:
    def test_none(self) -> None:
        assert normalize_stop_sequences(None) is None

    def test_string(self) -> None:
        assert normalize_stop_sequences("</s>") == ["</s>"]

    def test_list(self) -> None:
        assert normalize_stop_sequences(["</s>", "<eot>"]) == ["</s>", "<eot>"]


class TestToolsToDict:
    def test_none(self) -> None:
        assert tools_to_dict(None) is None

    def test_with_tools(self) -> None:
        tool = Tool(
            type="function",
            function={"name": "f", "description": "d", "parameters": {}},
        )
        result = tools_to_dict([tool])
        assert result is not None
        assert result[0]["function"]["name"] == "f"


class TestToolChoiceToDict:
    def test_none(self) -> None:
        assert tool_choice_to_dict(None) is None

    def test_string(self) -> None:
        assert tool_choice_to_dict("auto") == "auto"

    def test_object(self) -> None:
        from llm_infer.schemas.openai import ToolChoiceObject

        choice = ToolChoiceObject(type="function", function={"name": "f"})
        result = tool_choice_to_dict(choice)
        assert isinstance(result, dict)


class TestResponseFormatToDict:
    def test_none(self) -> None:
        assert response_format_to_dict(None) is None

    def test_text_returns_none(self) -> None:
        from llm_infer.schemas.openai import ResponseFormatText

        assert response_format_to_dict(ResponseFormatText(type="text")) is None

    def test_json_object(self) -> None:
        from llm_infer.schemas.openai import ResponseFormatJSONObject

        result = response_format_to_dict(ResponseFormatJSONObject(type="json_object"))
        assert result is not None
        assert result["type"] == "json_object"


class TestNormalizeArguments:
    def test_none(self) -> None:
        assert normalize_arguments(None) == "{}"

    def test_string(self) -> None:
        assert normalize_arguments('{"x": 1}') == '{"x": 1}'

    def test_dict(self) -> None:
        assert normalize_arguments({"x": 1}) == '{"x": 1}'


# ---------------------------------------------------------------------------
# Think mode helpers
# ---------------------------------------------------------------------------


class TestResolveThinkMode:
    def test_explicit_true(self) -> None:
        assert resolve_think_mode(True, None) is True

    def test_explicit_false(self) -> None:
        assert resolve_think_mode(False, None) is False

    def test_default_no_config(self) -> None:
        assert resolve_think_mode(None, None) is False

    def test_default_from_config(self) -> None:
        cfg = _model_cfg(think_default=True)
        assert resolve_think_mode(None, cfg) is True


class TestGetThinkSuffix:
    def test_no_config(self) -> None:
        assert _get_think_suffix(True, None) == ""

    def test_enable_suffix_when_thinking(self) -> None:
        cfg = _model_cfg(enable_suffix=" /think")
        assert _get_think_suffix(True, cfg) == " /think"

    def test_disable_suffix_when_not_thinking(self) -> None:
        cfg = _model_cfg(disable_suffix=" /no_think")
        assert _get_think_suffix(False, cfg) == " /no_think"

    def test_no_suffix_when_unset(self) -> None:
        cfg = _model_cfg()
        assert _get_think_suffix(True, cfg) == ""


class TestGetSystemPrompt:
    def test_no_config(self) -> None:
        assert _get_system_prompt(True, None) is None

    def test_think_system_prompt(self) -> None:
        cfg = _model_cfg(think_system_prompt="think hard")
        assert _get_system_prompt(True, cfg) == "think hard"

    def test_default_system_prompt(self) -> None:
        cfg = _model_cfg(system_prompt="default")
        assert _get_system_prompt(False, cfg) == "default"

    def test_falls_back_to_default(self) -> None:
        cfg = _model_cfg(system_prompt="default", think_system_prompt=None)
        assert _get_system_prompt(True, cfg) == "default"


class TestGetChatTemplateKwargs:
    def test_no_config(self) -> None:
        assert _get_chat_template_kwargs(True, None) is None

    def test_no_kwargs(self) -> None:
        cfg = _model_cfg()
        assert _get_chat_template_kwargs(True, cfg) is None

    def test_with_enable_thinking_override(self) -> None:
        cfg = _model_cfg(chat_template_kwargs={"enable_thinking": False})
        result = _get_chat_template_kwargs(True, cfg)
        assert result == {"enable_thinking": True}

    def test_other_kwargs_passed_through(self) -> None:
        cfg = _model_cfg(chat_template_kwargs={"some_key": "value"})
        result = _get_chat_template_kwargs(True, cfg)
        assert result == {"some_key": "value"}


class TestHasSystemMessage:
    def test_with_system(self) -> None:
        body = ChatCompletionRequest(
            model="m",
            messages=[_msg("system", "x"), _msg("user", "hi")],
        )
        assert _has_system_message(body) is True

    def test_with_developer(self) -> None:
        body = ChatCompletionRequest(
            model="m",
            messages=[_msg("developer", "x"), _msg("user", "hi")],
        )
        assert _has_system_message(body) is True

    def test_user_only(self) -> None:
        body = ChatCompletionRequest(model="m", messages=[_msg("user", "hi")])
        assert _has_system_message(body) is False


class TestInjectThinkSuffix:
    def test_no_suffix(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        assert _inject_think_suffix(msgs, "") is None

    def test_appends_to_last_user(self) -> None:
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        result = _inject_think_suffix(msgs, " /think")
        assert result == "hi /think"
        assert msgs[1]["content"] == "hi /think"

    def test_no_user_appends_to_last(self) -> None:
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "ack"},
        ]
        result = _inject_think_suffix(msgs, " /think")
        assert result == "ack /think"
        assert msgs[-1]["content"] == "ack /think"


class TestHasToolMessages:
    def test_no_tools(self) -> None:
        body = ChatCompletionRequest(model="m", messages=[_msg("user", "hi")])
        assert _has_tool_messages(body) is False

    def test_with_tool_role(self) -> None:
        body = ChatCompletionRequest(
            model="m",
            messages=[
                _msg("user", "hi"),
                ChatMessage(
                    role=Role.TOOL,
                    content="result",
                    tool_call_id="tc1",
                ),
            ],
        )
        assert _has_tool_messages(body) is True


class TestMessageToDict:
    def test_developer_becomes_system(self) -> None:
        msg = _msg("developer", "instruct")
        result = _message_to_dict(msg)
        assert result["role"] == "system"
        assert result["content"] == "instruct"

    def test_with_tool_calls(self) -> None:
        from llm_infer.schemas.openai import FunctionCall, ToolCall

        msg = ChatMessage(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    type="function",
                    function=FunctionCall(name="f", arguments="{}"),
                )
            ],
        )
        result = _message_to_dict(msg)
        assert "tool_calls" in result

    def test_with_tool_call_id(self) -> None:
        msg = ChatMessage(role=Role.TOOL, content="result", tool_call_id="tc1")
        result = _message_to_dict(msg)
        assert result["tool_call_id"] == "tc1"


# ---------------------------------------------------------------------------
# _build_messages_with_injections
# ---------------------------------------------------------------------------


class TestBuildMessagesWithInjections:
    def test_simple_single_user(self) -> None:
        body = ChatCompletionRequest(model="m", messages=[_msg("user", "hi")])
        prompt, messages = _build_messages_with_injections(body, "", None)
        assert prompt == "hi"
        assert messages is None  # Single-user shortcut

    def test_simple_single_user_with_suffix(self) -> None:
        body = ChatCompletionRequest(model="m", messages=[_msg("user", "hi")])
        prompt, messages = _build_messages_with_injections(body, " /think", None)
        assert prompt == "hi /think"
        assert messages is None

    def test_multi_message_no_system(self) -> None:
        body = ChatCompletionRequest(
            model="m",
            messages=[_msg("user", "hi"), _msg("assistant", "ok")],
        )
        prompt, messages = _build_messages_with_injections(body, "", None)
        assert messages is not None
        assert len(messages) == 2

    def test_with_system_prompt_injection(self) -> None:
        body = ChatCompletionRequest(model="m", messages=[_msg("user", "hi")])
        prompt, messages = _build_messages_with_injections(body, "", "you are X")
        # Single user, but system prompt forces multi-message path
        assert messages is not None
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "you are X"

    def test_existing_system_not_overridden(self) -> None:
        body = ChatCompletionRequest(
            model="m",
            messages=[_msg("system", "existing"), _msg("user", "hi")],
        )
        prompt, messages = _build_messages_with_injections(body, "", "should-not-add")
        assert messages is not None
        # Only one system message
        assert sum(1 for m in messages if m["role"] == "system") == 1
        assert messages[0]["content"] == "existing"


# ---------------------------------------------------------------------------
# chat_request_to_internal / completion_request_to_internal
# ---------------------------------------------------------------------------


class TestChatRequestToInternal:
    def test_basic(self) -> None:
        body = ChatCompletionRequest(model="m", messages=[_msg("user", "hi")])
        result = chat_request_to_internal(body, "r1")
        assert result.id == "r1"
        assert result.prompt == "hi"
        assert result.max_tokens == 256

    def test_with_max_completion_tokens(self) -> None:
        body = ChatCompletionRequest(
            model="m",
            messages=[_msg("user", "hi")],
            max_completion_tokens=50,
        )
        result = chat_request_to_internal(body, "r1")
        assert result.max_tokens == 50

    def test_with_model_config_think(self) -> None:
        cfg = _model_cfg(think_default=True, enable_suffix=" /think")
        body = ChatCompletionRequest(model="m", messages=[_msg("user", "hi")])
        result = chat_request_to_internal(body, "r1", cfg)
        assert "/think" in result.prompt


class TestCompletionRequestToInternal:
    def test_string_prompt(self) -> None:
        body = CompletionRequest(model="m", prompt="hello")
        result = completion_request_to_internal(body, "r1")
        assert result.prompt == "hello"

    def test_list_prompt(self) -> None:
        body = CompletionRequest(model="m", prompt=["hello", "world"])
        result = completion_request_to_internal(body, "r1")
        assert result.prompt == "hello"  # First element

    def test_no_chat_template(self) -> None:
        body = CompletionRequest(model="m", prompt="hi")
        result = completion_request_to_internal(body, "r1")
        assert result.use_chat_template is False


# ---------------------------------------------------------------------------
# determine_finish_reason
# ---------------------------------------------------------------------------


class TestDetermineFinishReason:
    def test_eos(self) -> None:
        assert (
            determine_finish_reason(is_eos=True, max_tokens_reached=False)
            == FinishReason.STOP
        )

    def test_max_tokens(self) -> None:
        assert (
            determine_finish_reason(is_eos=False, max_tokens_reached=True)
            == FinishReason.LENGTH
        )

    def test_tool_calls_takes_precedence(self) -> None:
        assert (
            determine_finish_reason(
                is_eos=False, max_tokens_reached=True, has_tool_calls=True
            )
            == FinishReason.TOOL_CALLS
        )

    def test_guard_triggered(self) -> None:
        assert (
            determine_finish_reason(
                is_eos=True, max_tokens_reached=False, guard_triggered=True
            )
            == FinishReason.CONTENT_FILTER
        )
