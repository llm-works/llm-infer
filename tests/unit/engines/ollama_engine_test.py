"""Unit tests for OllamaEngine class.

The existing tests/unit/ollama_engine_test.py covers OllamaConfig and
OllamaStreamingIterator. This file covers the OllamaEngine class itself.

Uses __new__() to bypass __init__ (which tries to start a real subprocess
or connect to a real server), and stubs the engine state directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from appinfra.log import Logger

from llm_infer.engines.ollama import (
    OllamaEngine,
    _build_tool_call_id_mapping,
    _convert_messages_to_ollama_format,
    _convert_single_message,
    _convert_tool_call_to_ollama,
)
from llm_infer.serving.dispatch.config import OllamaConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestConvertToolCallToOllama:
    def test_basic(self) -> None:
        result = _convert_tool_call_to_ollama(
            {"id": "tc1", "function": {"name": "f", "arguments": '{"x": 1}'}}
        )
        assert result == {"function": {"name": "f", "arguments": {"x": 1}}}

    def test_dict_arguments(self) -> None:
        result = _convert_tool_call_to_ollama(
            {"function": {"name": "f", "arguments": {"x": 1}}}
        )
        assert result == {"function": {"name": "f", "arguments": {"x": 1}}}

    def test_invalid_json_arguments(self) -> None:
        result = _convert_tool_call_to_ollama(
            {"function": {"name": "f", "arguments": "{invalid"}}
        )
        assert result == {"function": {"name": "f", "arguments": {}}}

    def test_empty_arguments(self) -> None:
        result = _convert_tool_call_to_ollama(
            {"function": {"name": "f", "arguments": ""}}
        )
        assert result == {"function": {"name": "f", "arguments": {}}}

    def test_non_dict_arguments(self) -> None:
        result = _convert_tool_call_to_ollama(
            {"function": {"name": "f", "arguments": 42}}
        )
        assert result == {"function": {"name": "f", "arguments": {}}}

    def test_no_function(self) -> None:
        result = _convert_tool_call_to_ollama({"id": "tc1"})
        assert result == {"function": {"name": "", "arguments": {}}}

    def test_function_not_dict(self) -> None:
        result = _convert_tool_call_to_ollama({"function": "not-dict"})
        assert result == {"function": {"name": "", "arguments": {}}}


class TestBuildToolCallIdMapping:
    def test_empty(self) -> None:
        assert _build_tool_call_id_mapping([]) == {}

    def test_single_assistant_with_tool_calls(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "func1"}},
                    {"id": "tc2", "function": {"name": "func2"}},
                ],
            }
        ]
        result = _build_tool_call_id_mapping(messages)
        assert result == {"tc1": "func1", "tc2": "func2"}

    def test_skips_non_assistant(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
        ]
        assert _build_tool_call_id_mapping(messages) == {}

    def test_skips_invalid_tool_call(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    "not-a-dict",
                    {"id": "tc1", "function": {"name": "func1"}},
                ],
            }
        ]
        result = _build_tool_call_id_mapping(messages)
        assert result == {"tc1": "func1"}

    def test_skips_missing_id_or_name(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "func1"}},  # no id
                    {"id": "tc1", "function": {}},  # no name
                ],
            }
        ]
        assert _build_tool_call_id_mapping(messages) == {}


class TestConvertSingleMessage:
    def test_user_message_passthrough(self) -> None:
        msg = {"role": "user", "content": "hello"}
        assert _convert_single_message(msg, {}) == msg

    def test_assistant_with_tool_calls(self) -> None:
        msg = {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [{"id": "tc1", "function": {"name": "f", "arguments": "{}"}}],
        }
        result = _convert_single_message(msg, {})
        assert result["role"] == "assistant"
        assert result["content"] == "calling tool"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1

    def test_tool_response_with_known_id(self) -> None:
        msg = {"role": "tool", "tool_call_id": "tc1", "content": "result"}
        result = _convert_single_message(msg, {"tc1": "func1"})
        assert result["role"] == "tool"
        assert result["content"] == "result"
        assert result["tool_name"] == "func1"

    def test_tool_response_unknown_id(self) -> None:
        msg = {"role": "tool", "tool_call_id": "unknown", "content": "result"}
        result = _convert_single_message(msg, {})
        assert "tool_name" not in result

    def test_assistant_without_tool_calls_passthrough(self) -> None:
        msg = {"role": "assistant", "content": "regular reply"}
        assert _convert_single_message(msg, {}) == msg


class TestConvertMessagesToOllamaFormat:
    def test_full_flow(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "function": {"name": "f", "arguments": '{"x": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "42"},
        ]
        result = _convert_messages_to_ollama_format(messages)
        assert len(result) == 3
        # User message preserved
        assert result[0] == {"role": "user", "content": "hi"}
        # Assistant tool calls converted
        assert result[1]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
        # Tool response has tool_name
        assert result[2]["tool_name"] == "f"


# ---------------------------------------------------------------------------
# OllamaEngine helper methods (using __new__ bypass)
# ---------------------------------------------------------------------------


def _make_engine(
    *,
    model: str = "test-model",
    eos_token_id: int | None = None,
    tokenize_available: bool | None = None,
    keep_alive: str | None = "5m",
    task: str = "generate",
) -> OllamaEngine:
    """Create an OllamaEngine bypassing __init__."""
    engine = OllamaEngine.__new__(OllamaEngine)
    engine._lg = MagicMock(spec=Logger)
    cfg = MagicMock(spec=OllamaConfig)
    cfg.model = model
    cfg.host = "http://localhost:11434"
    cfg.keep_alive = keep_alive
    cfg.task = task
    cfg.binary_path = "ollama"
    cfg.timeout = 300
    cfg.models_path = None
    cfg.num_ctx = None
    cfg.num_gpu = None
    engine._config = cfg
    engine._client = MagicMock(spec=httpx.Client)
    engine._process = None
    engine._owns_process = False
    engine._tokenize_available = tokenize_available
    engine._model_info = {}
    engine._eos_token_id = eos_token_id
    return engine


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_model_name(self) -> None:
        engine = _make_engine(model="qwen-7b")
        assert engine.model_name == "qwen-7b"

    def test_eos_token_id(self) -> None:
        engine = _make_engine(eos_token_id=42)
        assert engine.eos_token_id == 42

    def test_supports_embeddings_true(self) -> None:
        engine = _make_engine(task="embed")
        assert engine.supports_embeddings() is True

    def test_supports_embeddings_false(self) -> None:
        engine = _make_engine(task="generate")
        assert engine.supports_embeddings() is False

    def test_should_use_chat_template_instruct(self) -> None:
        engine = _make_engine(model="qwen-7b-instruct")
        assert engine.should_use_chat_template() is True

    def test_should_use_chat_template_chat(self) -> None:
        engine = _make_engine(model="qwen-7b-chat")
        assert engine.should_use_chat_template() is True

    def test_should_use_chat_template_base(self) -> None:
        engine = _make_engine(model="qwen-7b")
        assert engine.should_use_chat_template() is False


# ---------------------------------------------------------------------------
# tokenize / count_tokens / decode_tokens
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_tokenize_unavailable_returns_estimate(self) -> None:
        engine = _make_engine(tokenize_available=False)
        result = engine.tokenize("hello world")
        # 11 chars / 4 = 2 estimated tokens
        assert len(result) == 2

    def test_tokenize_success(self) -> None:
        engine = _make_engine()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"tokens": [1, 2, 3]}
        mock_resp.raise_for_status = MagicMock()
        engine._client.post.return_value = mock_resp
        result = engine.tokenize("hi")
        assert result == [1, 2, 3]
        assert engine._tokenize_available is True

    def test_tokenize_falls_back_on_http_error(self) -> None:
        engine = _make_engine()
        engine._client.post.side_effect = httpx.HTTPError("nope")
        result = engine.tokenize("hello world")
        assert engine._tokenize_available is False
        assert len(result) == 2  # estimate

    def test_count_tokens(self) -> None:
        engine = _make_engine(tokenize_available=False)
        assert engine.count_tokens("hello world") == 2

    def test_decode_tokens_success(self) -> None:
        engine = _make_engine()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "decoded"}
        mock_resp.raise_for_status = MagicMock()
        engine._client.post.return_value = mock_resp
        assert engine.decode_tokens([1, 2]) == "decoded"

    def test_decode_tokens_http_error(self) -> None:
        engine = _make_engine()
        engine._client.post.side_effect = httpx.HTTPError("nope")
        assert engine.decode_tokens([1, 2]) == ""


class TestBuildStopTokenIds:
    def test_no_eos_no_stop_sequences(self) -> None:
        engine = _make_engine(eos_token_id=None)
        assert engine.build_stop_token_ids(None) == set()

    def test_with_eos_only(self) -> None:
        engine = _make_engine(eos_token_id=99)
        assert engine.build_stop_token_ids(None) == {99}

    def test_skips_stop_sequences_when_no_real_tokenizer(self) -> None:
        """Without real tokenizer, stop sequences are not tokenized."""
        engine = _make_engine(eos_token_id=99, tokenize_available=False)
        result = engine.build_stop_token_ids(["</s>"])
        assert result == {99}


# ---------------------------------------------------------------------------
# _post_json
# ---------------------------------------------------------------------------


class TestPostJson:
    def test_success(self) -> None:
        engine = _make_engine()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": "ok"}
        mock_resp.raise_for_status = MagicMock()
        engine._client.post.return_value = mock_resp
        result = engine._post_json("/api/x", {"a": 1}, "test")
        assert result == {"result": "ok"}

    def test_http_error_raises_runtime(self) -> None:
        engine = _make_engine()
        engine._client.post.side_effect = httpx.HTTPError("boom")
        with pytest.raises(RuntimeError, match="Ollama test failed"):
            engine._post_json("/api/x", {}, "test")


# ---------------------------------------------------------------------------
# _build_options / _extract_ollama_format / _build_*_payload
# ---------------------------------------------------------------------------


class TestBuildOptions:
    def test_basic(self) -> None:
        engine = _make_engine()
        opts = engine._build_options(100, 0.7, 0.9, 40, 1.1)
        assert opts["num_predict"] == 100
        assert opts["temperature"] == 0.7
        assert opts["top_p"] == 0.9
        assert opts["top_k"] == 40
        assert opts["repeat_penalty"] == 1.1

    def test_with_num_ctx(self) -> None:
        engine = _make_engine()
        engine._config.num_ctx = 8192
        opts = engine._build_options(100, 1.0, 1.0, 0, 1.0)
        assert opts["num_ctx"] == 8192

    def test_with_num_gpu(self) -> None:
        engine = _make_engine()
        engine._config.num_gpu = 32
        opts = engine._build_options(100, 1.0, 1.0, 0, 1.0)
        assert opts["num_gpu"] == 32


class TestExtractOllamaFormat:
    def test_none(self) -> None:
        engine = _make_engine()
        assert engine._extract_ollama_format(None) is None

    def test_json_object(self) -> None:
        engine = _make_engine()
        assert engine._extract_ollama_format({"type": "json_object"}) == "json"

    def test_json_schema(self) -> None:
        engine = _make_engine()
        result = engine._extract_ollama_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    }
                },
            }
        )
        assert isinstance(result, dict)
        assert result["additionalProperties"] is False

    def test_json_schema_no_schema(self) -> None:
        engine = _make_engine()
        result = engine._extract_ollama_format(
            {"type": "json_schema", "json_schema": {}}
        )
        assert result == "json"

    def test_unknown_type(self) -> None:
        engine = _make_engine()
        assert engine._extract_ollama_format({"type": "text"}) is None

    def test_json_schema_preserves_existing_additional_properties(self) -> None:
        engine = _make_engine()
        result = engine._extract_ollama_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "schema": {"type": "object", "additionalProperties": True}
                },
            }
        )
        assert result["additionalProperties"] is True  # type: ignore[index]


class TestBuildPayloads:
    def test_build_chat_payload_minimal(self) -> None:
        engine = _make_engine()
        payload = engine._build_chat_payload(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.0,
            stop_sequences=None,
            tools=None,
            tool_choice=None,
            stream=False,
        )
        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert payload["keep_alive"] == "5m"
        assert "options" in payload

    def test_build_chat_payload_with_extras(self) -> None:
        engine = _make_engine()
        payload = engine._build_chat_payload(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.0,
            stop_sequences=["</s>"],
            tools=[{"name": "f"}],
            tool_choice="auto",
            stream=False,
            response_format={"type": "json_object"},
        )
        assert payload["options"]["stop"] == ["</s>"]
        assert payload["tools"] == [{"name": "f"}]
        assert payload["tool_choice"] == "auto"
        assert payload["format"] == "json"

    def test_build_chat_payload_no_keep_alive(self) -> None:
        engine = _make_engine(keep_alive=None)
        payload = engine._build_chat_payload(
            messages=[],
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.0,
            stop_sequences=None,
            tools=None,
            tool_choice=None,
            stream=False,
        )
        assert "keep_alive" not in payload

    def test_build_generate_payload(self) -> None:
        engine = _make_engine()
        payload = engine._build_generate_payload(
            prompt="hello",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.0,
            stop_sequences=["</s>"],
            stream=False,
            response_format={"type": "json_object"},
        )
        assert payload["prompt"] == "hello"
        assert payload["options"]["stop"] == ["</s>"]
        assert payload["format"] == "json"


# ---------------------------------------------------------------------------
# generate / _generate_chat
# ---------------------------------------------------------------------------


class TestGenerate:
    def _setup_post(self, engine: OllamaEngine, response: dict) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        mock_resp.raise_for_status = MagicMock()
        engine._client.post.return_value = mock_resp

    def test_generate_with_prompt(self) -> None:
        engine = _make_engine()
        self._setup_post(engine, {"response": "generated text"})
        result = engine.generate("hello", max_tokens=50)
        assert result == "generated text"

    def test_generate_with_messages(self) -> None:
        engine = _make_engine()
        self._setup_post(engine, {"message": {"content": "chat reply"}})
        result = engine.generate(
            "ignored", messages=[{"role": "user", "content": "hi"}]
        )
        assert result == "chat reply"

    def test_generate_chat_with_tool_calls(self) -> None:
        engine = _make_engine()
        self._setup_post(
            engine,
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}],
                }
            },
        )
        result = engine.generate(
            "ignored", messages=[{"role": "user", "content": "hi"}]
        )
        assert isinstance(result, dict)
        assert result["content"] == ""
        assert result["tool_calls"] is not None


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_embed_single(self) -> None:
        engine = _make_engine(task="embed")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_resp.raise_for_status = MagicMock()
        engine._client.post.return_value = mock_resp

        embeddings, total = engine.embed(["hello"])
        assert embeddings == [[0.1, 0.2, 0.3]]
        assert total == 1  # 5 chars / 4 = 1

    def test_embed_with_dimensions_truncation(self) -> None:
        engine = _make_engine(task="embed")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [3.0, 4.0, 0.0, 0.0]}
        mock_resp.raise_for_status = MagicMock()
        engine._client.post.return_value = mock_resp

        embeddings, _ = engine.embed(["x"], dimensions=2)
        # Truncated to 2 dims and renormalized: [3, 4] / 5 = [0.6, 0.8]
        assert embeddings[0] == [0.6, 0.8]

    def test_embed_zero_norm(self) -> None:
        engine = _make_engine(task="embed")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.0, 0.0, 0.0, 0.0]}
        mock_resp.raise_for_status = MagicMock()
        engine._client.post.return_value = mock_resp

        embeddings, _ = engine.embed(["x"], dimensions=2)
        # Zero norm: no renormalization, just truncated
        assert embeddings[0] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Memory stats / no-op methods
# ---------------------------------------------------------------------------


class TestMemoryStats:
    def test_default_stats(self) -> None:
        engine = _make_engine()
        stats = engine._default_memory_stats()
        assert stats["allocated"] == 0
        assert stats["kv_blocks_total"] == 0

    def test_memory_stats_success(self) -> None:
        engine = _make_engine(model="qwen-7b")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {"name": "qwen-7b", "size": 1024, "size_vram": 512},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        engine._client.get.return_value = mock_resp

        stats = engine.memory_stats()
        assert stats["model_memory"] == 1024
        assert stats["allocated"] == 1024
        assert stats["device_used"] == 512

    def test_memory_stats_other_model_ignored(self) -> None:
        engine = _make_engine(model="qwen-7b")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "other-model", "size": 1024}]
        }
        mock_resp.raise_for_status = MagicMock()
        engine._client.get.return_value = mock_resp
        stats = engine.memory_stats()
        assert stats["allocated"] == 0

    def test_memory_stats_http_error(self) -> None:
        engine = _make_engine()
        engine._client.get.side_effect = httpx.HTTPError("boom")
        stats = engine.memory_stats()
        # Returns defaults
        assert stats["allocated"] == 0


def test_no_op_methods() -> None:
    engine = _make_engine()
    # All return None or empty
    assert engine.prefill_request(MagicMock()) is None
    assert engine.step_decode([MagicMock(), MagicMock()]) == [None, None]
    assert engine.free_request(MagicMock()) is None
    assert engine.reset_peak_memory() is None


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_no_owned_process(self) -> None:
        engine = _make_engine()
        engine._owns_process = False
        engine.shutdown()
        engine._client.close.assert_called_once()

    def test_shutdown_with_owned_process(self) -> None:
        engine = _make_engine()
        engine._owns_process = True
        engine._process = MagicMock()
        engine._process.pid = 12345
        # Mock os.killpg, os.getpgid
        import os

        with pytest.MonkeyPatch().context() as m:
            m.setattr(os, "killpg", lambda pid, sig: None)
            m.setattr(os, "getpgid", lambda pid: pid)
            engine._process.wait.return_value = 0
            engine.shutdown()
        engine._client.close.assert_called_once()


# ---------------------------------------------------------------------------
# generate_stream_sync
# ---------------------------------------------------------------------------


class TestGenerateStreamSync:
    def test_with_messages_returns_iterator(self) -> None:
        engine = _make_engine()
        result = engine.generate_stream_sync(
            "ignored",
            messages=[{"role": "user", "content": "hi"}],
        )
        # Returns an OllamaStreamingIterator
        assert hasattr(result, "__iter__")

    def test_with_prompt_returns_iterator(self) -> None:
        engine = _make_engine()
        result = engine.generate_stream_sync("hello")
        assert hasattr(result, "__iter__")


# ---------------------------------------------------------------------------
# _verify_connection / _fetch_model_info
# ---------------------------------------------------------------------------


class TestVerifyConnection:
    def test_success(self) -> None:
        engine = _make_engine()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        engine._client.get.return_value = mock_resp
        engine._verify_connection()  # No raise

    def test_http_error_raises(self) -> None:
        engine = _make_engine()
        engine._client.get.side_effect = httpx.HTTPError("boom")
        with pytest.raises(ConnectionError):
            engine._verify_connection()


class TestFetchModelInfo:
    def test_success(self) -> None:
        engine = _make_engine()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "details": {"family": "qwen", "parameter_size": "7B"}
        }
        mock_resp.raise_for_status = MagicMock()
        engine._client.post.return_value = mock_resp
        result = engine._fetch_model_info()
        assert result["details"]["family"] == "qwen"

    def test_http_error_returns_empty(self) -> None:
        engine = _make_engine()
        engine._client.post.side_effect = httpx.HTTPError("boom")
        assert engine._fetch_model_info() == {}


class TestExtractEosTokenId:
    def test_eos_token_id_present(self) -> None:
        engine = _make_engine()
        engine._model_info = {"model_info": {"eos_token_id": 42}}
        assert engine._extract_eos_token_id() == 42

    def test_eos_id_alt_name(self) -> None:
        engine = _make_engine()
        engine._model_info = {"model_info": {"eos_id": 99}}
        assert engine._extract_eos_token_id() == 99

    def test_no_eos(self) -> None:
        engine = _make_engine()
        engine._model_info = {"model_info": {}}
        assert engine._extract_eos_token_id() is None


# ---------------------------------------------------------------------------
# _is_server_running
# ---------------------------------------------------------------------------


class TestIsServerRunning:
    def test_true(self) -> None:
        engine = _make_engine()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        engine._client.get.return_value = mock_resp
        assert engine._is_server_running() is True

    def test_false_on_http_error(self) -> None:
        engine = _make_engine()
        engine._client.get.side_effect = httpx.HTTPError("nope")
        assert engine._is_server_running() is False


# ---------------------------------------------------------------------------
# _build_server_env
# ---------------------------------------------------------------------------


def test_build_server_env() -> None:
    engine = _make_engine()
    engine._config.host = "http://localhost:11434"
    env = engine._build_server_env()
    assert env["OLLAMA_HOST"] == "http://localhost:11434"

    engine._config.models_path = "~/models"
    env = engine._build_server_env()
    assert "OLLAMA_MODELS" in env
