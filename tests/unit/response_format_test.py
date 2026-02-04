"""Unit tests for OpenAI-compatible response_format (structured output) support."""

from typing import Any

import pytest

from llm_infer.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
    JSONSchema,
    ResponseFormat,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
    ResponseFormatText,
    Role,
)
from llm_infer.serving.api.openai.mappers import response_format_to_dict
from llm_infer.serving.dispatch.types import Request

pytestmark = pytest.mark.unit


class TestResponseFormatSchemas:
    """Test Pydantic schemas for response_format types."""

    def test_response_format_text_default(self) -> None:
        """Test ResponseFormatText has correct type."""
        fmt = ResponseFormatText()
        assert fmt.type == "text"

    def test_response_format_json_object(self) -> None:
        """Test ResponseFormatJSONObject has correct type."""
        fmt = ResponseFormatJSONObject()
        assert fmt.type == "json_object"

    def test_json_schema_basic(self) -> None:
        """Test JSONSchema with required fields."""
        schema = JSONSchema(
            name="test",
            schema={"type": "object"},
        )
        assert schema.name == "test"
        assert schema.schema_ == {"type": "object"}
        assert schema.description is None
        assert schema.strict is None

    def test_json_schema_full(self) -> None:
        """Test JSONSchema with all fields."""
        schema = JSONSchema(
            name="person",
            description="A person object",
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name"],
            },
            strict=True,
        )
        assert schema.name == "person"
        assert schema.description == "A person object"
        assert schema.strict is True
        assert "properties" in schema.schema_

    def test_response_format_json_schema(self) -> None:
        """Test ResponseFormatJSONSchema with nested schema."""
        fmt = ResponseFormatJSONSchema(
            type="json_schema",
            json_schema=JSONSchema(
                name="critique",
                schema={
                    "type": "object",
                    "properties": {
                        "counter_argument": {"type": "string"},
                        "weaknesses": {"type": "array", "items": {"type": "string"}},
                        "strength": {"type": "number"},
                    },
                    "required": ["counter_argument", "weaknesses", "strength"],
                },
                strict=True,
            ),
        )
        assert fmt.type == "json_schema"
        assert fmt.json_schema.name == "critique"
        assert fmt.json_schema.strict is True

    def test_json_schema_serialization_uses_alias(self) -> None:
        """Test that schema field uses 'schema' alias in serialization."""
        schema = JSONSchema(name="test", schema={"type": "object"})
        data = schema.model_dump(by_alias=True)
        assert "schema" in data
        assert "schema_" not in data


class TestResponseFormatToDict:
    """Test response_format_to_dict conversion function."""

    def test_none_returns_none(self) -> None:
        """Test None input returns None."""
        assert response_format_to_dict(None) is None

    def test_text_returns_none(self) -> None:
        """Test text format returns None (default behavior)."""
        fmt = ResponseFormatText()
        assert response_format_to_dict(fmt) is None

    def test_json_object_conversion(self) -> None:
        """Test json_object conversion to dict."""
        fmt = ResponseFormatJSONObject()
        result = response_format_to_dict(fmt)
        assert result == {"type": "json_object"}

    def test_json_schema_conversion(self) -> None:
        """Test json_schema conversion to dict."""
        fmt = ResponseFormatJSONSchema(
            type="json_schema",
            json_schema=JSONSchema(
                name="test",
                schema={"type": "object", "properties": {"x": {"type": "string"}}},
            ),
        )
        result = response_format_to_dict(fmt)
        assert result is not None
        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "test"
        assert result["json_schema"]["schema"]["type"] == "object"

    def test_json_schema_excludes_none_fields(self) -> None:
        """Test that None fields are excluded from serialization."""
        fmt = ResponseFormatJSONSchema(
            type="json_schema",
            json_schema=JSONSchema(
                name="test",
                schema={"type": "object"},
                # description and strict are None
            ),
        )
        result = response_format_to_dict(fmt)
        assert result is not None
        assert "description" not in result["json_schema"]
        assert "strict" not in result["json_schema"]

    def test_json_schema_includes_strict_when_set(self) -> None:
        """Test that strict field is included when set."""
        fmt = ResponseFormatJSONSchema(
            type="json_schema",
            json_schema=JSONSchema(
                name="test",
                schema={"type": "object"},
                strict=True,
            ),
        )
        result = response_format_to_dict(fmt)
        assert result is not None
        assert result["json_schema"]["strict"] is True


class TestChatCompletionRequestWithResponseFormat:
    """Test ChatCompletionRequest with response_format field."""

    def test_request_without_response_format(self) -> None:
        """Test request without response_format defaults to None."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
        )
        assert request.response_format is None

    def test_request_with_json_object(self) -> None:
        """Test request with json_object response format."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role=Role.USER, content="Return JSON")],
            response_format=ResponseFormatJSONObject(),
        )
        assert request.response_format is not None
        assert request.response_format.type == "json_object"

    def test_request_with_json_schema(self) -> None:
        """Test request with json_schema response format."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role=Role.USER, content="Generate person")],
            response_format=ResponseFormatJSONSchema(
                type="json_schema",
                json_schema=JSONSchema(
                    name="person",
                    schema={
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    strict=True,
                ),
            ),
        )
        assert request.response_format is not None
        assert request.response_format.type == "json_schema"

    def test_request_with_text_format(self) -> None:
        """Test request with explicit text format."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
            response_format=ResponseFormatText(),
        )
        assert request.response_format is not None
        assert request.response_format.type == "text"


class TestDispatchTypesResponseFormat:
    """Test response_format field in dispatch types."""

    def test_request_default_none(self) -> None:
        """Test Request has response_format=None by default."""
        req = Request(id="req-1", prompt="test")
        assert req.response_format is None

    def test_request_with_json_object(self) -> None:
        """Test Request with json_object format."""
        req = Request(
            id="req-1",
            prompt="test",
            response_format={"type": "json_object"},
        )
        assert req.response_format == {"type": "json_object"}

    def test_request_with_json_schema(self) -> None:
        """Test Request with json_schema format."""
        req = Request(
            id="req-1",
            prompt="test",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {"type": "object"},
                },
            },
        )
        assert req.response_format is not None
        assert req.response_format["type"] == "json_schema"
        assert req.response_format["json_schema"]["name"] == "test"


class TestResponseFormatUnionType:
    """Test ResponseFormat union type discrimination."""

    def test_union_accepts_text(self) -> None:
        """Test ResponseFormat union accepts text type."""
        fmt: ResponseFormat = ResponseFormatText()
        assert fmt.type == "text"

    def test_union_accepts_json_object(self) -> None:
        """Test ResponseFormat union accepts json_object type."""
        fmt: ResponseFormat = ResponseFormatJSONObject()
        assert fmt.type == "json_object"

    def test_union_accepts_json_schema(self) -> None:
        """Test ResponseFormat union accepts json_schema type."""
        fmt: ResponseFormat = ResponseFormatJSONSchema(
            type="json_schema",
            json_schema=JSONSchema(name="test", schema={}),
        )
        assert fmt.type == "json_schema"


class TestOllamaFormatExtraction:
    """Test Ollama format extraction logic (mirrors _extract_ollama_format)."""

    def _extract_format(
        self, response_format: dict[str, Any] | None
    ) -> str | dict[str, Any] | None:
        """Mirror of OllamaInferenceEngine._extract_ollama_format for testing."""
        if response_format is None:
            return None
        fmt_type = response_format.get("type")
        if fmt_type == "json_object":
            return "json"
        elif fmt_type == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema", {})
            if schema:
                # Workaround: add additionalProperties=false for strict compliance
                schema = dict(schema)
                if (
                    schema.get("type") == "object"
                    and "additionalProperties" not in schema
                ):
                    schema["additionalProperties"] = False
                return schema
            return "json"
        return None

    def test_none_returns_none(self) -> None:
        """Test None input returns None."""
        assert self._extract_format(None) is None

    def test_json_object_returns_json_string(self) -> None:
        """Test json_object returns 'json' string."""
        assert self._extract_format({"type": "json_object"}) == "json"

    def test_json_schema_adds_additional_properties_false(self) -> None:
        """Test json_schema adds additionalProperties=false for strict compliance."""
        result = self._extract_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            }
        )
        assert result == {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }

    def test_json_schema_preserves_existing_additional_properties(self) -> None:
        """Test json_schema doesn't override explicit additionalProperties setting."""
        result = self._extract_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "additionalProperties": True,
                    },
                },
            }
        )
        assert result is not None
        assert result["additionalProperties"] is True

    def test_json_schema_non_object_type_unchanged(self) -> None:
        """Test json_schema with non-object type doesn't add additionalProperties."""
        result = self._extract_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {"type": "array", "items": {"type": "string"}},
                },
            }
        )
        assert result == {"type": "array", "items": {"type": "string"}}

    def test_json_schema_empty_schema_returns_json(self) -> None:
        """Test json_schema with empty schema returns 'json'."""
        result = self._extract_format(
            {
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": {}},
            }
        )
        assert result == "json"

    def test_text_returns_none(self) -> None:
        """Test text type returns None (use default behavior)."""
        assert self._extract_format({"type": "text"}) is None


class TestVLLMGuidedJsonExtraction:
    """Test vLLM guided_json extraction logic (mirrors _create_sampling_params)."""

    def _extract_guided_json(
        self, response_format: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Mirror of VLLMInferenceEngine guided_json extraction for testing."""
        if response_format is None:
            return None
        fmt_type = response_format.get("type")
        if fmt_type == "json_object":
            return {"type": "object"}
        elif fmt_type == "json_schema":
            json_schema = response_format.get("json_schema", {})
            schema = json_schema.get("schema", {})
            # Fall back to basic object schema for empty schema (consistent with Ollama)
            return schema if schema else {"type": "object"}
        return None

    def test_none_returns_none(self) -> None:
        """Test None input returns None."""
        assert self._extract_guided_json(None) is None

    def test_json_object_returns_object_schema(self) -> None:
        """Test json_object returns basic object schema."""
        assert self._extract_guided_json({"type": "json_object"}) == {"type": "object"}

    def test_json_schema_returns_schema(self) -> None:
        """Test json_schema returns the provided schema."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = self._extract_guided_json(
            {
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": schema},
            }
        )
        assert result == schema

    def test_json_schema_empty_falls_back_to_object(self) -> None:
        """Test json_schema with empty schema falls back to basic object schema."""
        result = self._extract_guided_json(
            {
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": {}},
            }
        )
        assert result == {"type": "object"}

    def test_text_returns_none(self) -> None:
        """Test text type returns None."""
        assert self._extract_guided_json({"type": "text"}) is None
