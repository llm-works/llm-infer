"""Unit tests for OpenAI o1/o3/GPT-5 reasoning model compatibility."""

import pytest

from llm_infer.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionUsage,
    ChatMessage,
    CompletionTokensDetails,
    ImageContentPart,
    ImageUrlDetail,
    PromptTokensDetails,
    Role,
    TextContentPart,
    _extract_text_from_content,
)
from llm_infer.serving.api.openai.mappers import (
    _has_system_message,
    _message_to_dict,
    chat_request_to_internal,
    format_messages_as_prompt,
)

pytestmark = pytest.mark.unit


class TestDeveloperRole:
    """Test developer role support for reasoning models."""

    def test_role_enum_has_developer(self) -> None:
        """Role enum includes developer value."""
        assert Role.DEVELOPER == "developer"
        assert Role.DEVELOPER.value == "developer"

    def test_message_with_developer_role(self) -> None:
        """ChatMessage accepts developer role."""
        msg = ChatMessage(role=Role.DEVELOPER, content="You are a helpful assistant")
        assert msg.role == Role.DEVELOPER

    def test_request_with_developer_role(self) -> None:
        """ChatCompletionRequest accepts developer role messages."""
        request = ChatCompletionRequest(
            model="default",
            messages=[
                ChatMessage(role=Role.DEVELOPER, content="You are helpful"),
                ChatMessage(role=Role.USER, content="Hello"),
            ],
        )
        assert request.messages[0].role == Role.DEVELOPER

    def test_mixed_system_developer_rejected(self) -> None:
        """Request with both system and developer roles is rejected."""
        with pytest.raises(
            ValueError, match="Cannot use both 'system' and 'developer' roles"
        ):
            ChatCompletionRequest(
                model="default",
                messages=[
                    ChatMessage(role=Role.SYSTEM, content="System prompt"),
                    ChatMessage(role=Role.DEVELOPER, content="Developer prompt"),
                    ChatMessage(role=Role.USER, content="Hello"),
                ],
            )

    def test_developer_converted_to_system_in_dict(self) -> None:
        """Developer role is converted to system for backend compatibility."""
        msg = ChatMessage(role=Role.DEVELOPER, content="You are helpful")
        result = _message_to_dict(msg)
        assert result["role"] == "system"
        assert result["content"] == "You are helpful"

    def test_has_system_message_detects_developer(self) -> None:
        """_has_system_message returns True for developer role."""
        request = ChatCompletionRequest(
            model="default",
            messages=[
                ChatMessage(role=Role.DEVELOPER, content="You are helpful"),
                ChatMessage(role=Role.USER, content="Hello"),
            ],
        )
        assert _has_system_message(request) is True


class TestMaxCompletionTokens:
    """Test max_completion_tokens parameter for reasoning models."""

    def test_max_completion_tokens_accepted(self) -> None:
        """Request accepts max_completion_tokens parameter."""
        request = ChatCompletionRequest(
            model="default",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
            max_completion_tokens=100,
        )
        assert request.max_completion_tokens == 100

    def test_max_tokens_takes_precedence(self) -> None:
        """max_tokens takes precedence over max_completion_tokens in mapping."""
        request = ChatCompletionRequest(
            model="default",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
            max_tokens=50,
            max_completion_tokens=100,
        )
        internal = chat_request_to_internal(request, "test-id")
        assert internal.max_tokens == 50

    def test_max_completion_tokens_fallback(self) -> None:
        """max_completion_tokens used when max_tokens not provided."""
        request = ChatCompletionRequest(
            model="default",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
            max_completion_tokens=100,
        )
        internal = chat_request_to_internal(request, "test-id")
        assert internal.max_tokens == 100

    def test_default_max_tokens_when_neither_set(self) -> None:
        """Default 256 used when neither max_tokens nor max_completion_tokens set."""
        request = ChatCompletionRequest(
            model="default",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
        )
        internal = chat_request_to_internal(request, "test-id")
        assert internal.max_tokens == 256


class TestReasoningModelParams:
    """Test acceptance of reasoning model parameters (ignored but not rejected)."""

    def test_reasoning_effort_accepted(self) -> None:
        """reasoning_effort parameter is accepted."""
        request = ChatCompletionRequest(
            model="default",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
            reasoning_effort="high",
        )
        assert request.reasoning_effort == "high"

    def test_store_accepted(self) -> None:
        """store parameter is accepted."""
        request = ChatCompletionRequest(
            model="default",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
            store=True,
        )
        assert request.store is True

    def test_metadata_accepted(self) -> None:
        """metadata parameter is accepted."""
        request = ChatCompletionRequest(
            model="default",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
            metadata={"key": "value"},
        )
        assert request.metadata == {"key": "value"}

    def test_service_tier_accepted(self) -> None:
        """service_tier parameter is accepted."""
        request = ChatCompletionRequest(
            model="default",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
            service_tier="auto",
        )
        assert request.service_tier == "auto"

    def test_stream_options_accepted(self) -> None:
        """stream_options parameter is accepted."""
        request = ChatCompletionRequest(
            model="default",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
            stream_options={"include_usage": True},
        )
        assert request.stream_options == {"include_usage": True}


class TestTokenDetailsSchemas:
    """Test token detail schemas for reasoning model responses."""

    def test_completion_tokens_details_default(self) -> None:
        """CompletionTokensDetails has correct default."""
        details = CompletionTokensDetails()
        assert details.reasoning_tokens == 0

    def test_prompt_tokens_details_default(self) -> None:
        """PromptTokensDetails has correct default."""
        details = PromptTokensDetails()
        assert details.cached_tokens == 0

    def test_usage_with_details(self) -> None:
        """ChatCompletionUsage accepts optional detail fields."""
        usage = ChatCompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=30),
            prompt_tokens_details=PromptTokensDetails(cached_tokens=20),
        )
        assert usage.completion_tokens_details is not None
        assert usage.completion_tokens_details.reasoning_tokens == 30
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 20

    def test_usage_without_details(self) -> None:
        """ChatCompletionUsage works without detail fields (backward compatible)."""
        usage = ChatCompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert usage.completion_tokens_details is None
        assert usage.prompt_tokens_details is None


class TestContentArrayFormat:
    """Test content array format support for multi-modal messages."""

    def test_text_content_part_schema(self) -> None:
        """TextContentPart has correct structure."""
        part = TextContentPart(text="Hello world")
        assert part.type == "text"
        assert part.text == "Hello world"

    def test_image_content_part_schema(self) -> None:
        """ImageContentPart has correct structure."""
        part = ImageContentPart(
            image_url=ImageUrlDetail(url="https://example.com/image.png", detail="high")
        )
        assert part.type == "image_url"
        assert part.image_url.url == "https://example.com/image.png"
        assert part.image_url.detail == "high"

    def test_extract_text_from_string(self) -> None:
        """String content is returned as-is."""
        assert _extract_text_from_content("Hello") == "Hello"

    def test_extract_text_from_none(self) -> None:
        """None content returns None."""
        assert _extract_text_from_content(None) is None

    def test_extract_text_from_array_single_text(self) -> None:
        """Single text part in array is extracted."""
        content = [{"type": "text", "text": "Hello world"}]
        assert _extract_text_from_content(content) == "Hello world"

    def test_extract_text_from_array_multiple_text(self) -> None:
        """Multiple text parts are joined."""
        content = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ]
        assert _extract_text_from_content(content) == "Hello world"

    def test_extract_text_ignores_image_parts(self) -> None:
        """Image parts are ignored, only text is extracted."""
        content = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        assert _extract_text_from_content(content) == "What's in this image?"

    def test_extract_text_empty_array(self) -> None:
        """Empty array returns None."""
        assert _extract_text_from_content([]) is None

    def test_extract_text_only_images(self) -> None:
        """Array with only images returns None."""
        content = [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        ]
        assert _extract_text_from_content(content) is None

    def test_message_with_array_content(self) -> None:
        """ChatMessage accepts array content format."""
        msg = ChatMessage(
            role=Role.USER,
            content=[TextContentPart(text="Hello")],  # type: ignore[arg-type]
        )
        assert isinstance(msg.content, list)

    def test_message_to_dict_extracts_array_content(self) -> None:
        """_message_to_dict extracts text from array content."""
        msg = ChatMessage(
            role=Role.USER,
            content=[TextContentPart(text="Hello world")],  # type: ignore[arg-type]
        )
        result = _message_to_dict(msg)
        assert result["content"] == "Hello world"

    def test_chat_request_with_array_content(self) -> None:
        """ChatCompletionRequest works with array content in messages."""
        request = ChatCompletionRequest(
            model="default",
            messages=[
                ChatMessage(
                    role=Role.USER,
                    content=[TextContentPart(text="Describe this")],  # type: ignore[arg-type]
                )
            ],
        )
        internal = chat_request_to_internal(request, "test-id")
        assert internal.prompt == "Describe this"

    def test_format_messages_handles_array_content(self) -> None:
        """format_messages_as_prompt handles array content."""
        messages = [
            ChatMessage(
                role=Role.USER,
                content=[TextContentPart(text="Hello")],  # type: ignore[arg-type]
            )
        ]
        result = format_messages_as_prompt(messages)
        assert "Hello" in result
