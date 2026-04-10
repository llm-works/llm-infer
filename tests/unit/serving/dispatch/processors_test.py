"""Unit tests for serving/dispatch/processors.py."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_infer.serving.dispatch.processors import (
    AdapterProcessor,
    EmbeddingProcessor,
    InferenceProcessor,
    MetricsProcessor,
    create_request_processor_chain,
)
from llm_infer.serving.dispatch.types import (
    AdapterListRequest,
    AdapterListResponse,
    AdapterRefreshRequest,
    AdapterRefreshResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    MetricsRequest,
    Request,
    RequestStatus,
    Response,
)

from ._helpers import ResponseQueueFake

pytestmark = pytest.mark.unit


def _adapter(key: str = "k1", name: str = "n1") -> MagicMock:
    a = MagicMock()
    a.key = key
    a.name = name
    a.description = "desc"
    a.loaded_at = datetime(2026, 1, 1, tzinfo=UTC)
    a.md5 = "abc123"
    a.mtime = "2026-01-01T00:00:00Z"
    return a


# ---------------------------------------------------------------------------
# MetricsProcessor
# ---------------------------------------------------------------------------


def test_metrics_processor_can_process() -> None:
    p = MetricsProcessor()
    assert p.can_process(MetricsRequest(id="x"))
    assert not p.can_process(AdapterListRequest(id="x"))


def test_metrics_processor_handles(monkeypatch: pytest.MonkeyPatch) -> None:
    """MetricsProcessor delegates to build_metrics_response."""
    fake_response = MagicMock(spec_set=["id", "status"])
    fake_response.id = "req1"
    captured: dict[str, Any] = {}

    def fake_build(request_id: str, handler: Any, reset_peak: bool) -> Any:
        captured["request_id"] = request_id
        captured["reset_peak"] = reset_peak
        return fake_response

    monkeypatch.setattr(
        "llm_infer.serving.dispatch.processors.build_metrics_response", fake_build
    )

    q = ResponseQueueFake()
    handler = MagicMock()
    p = MetricsProcessor()
    p.handle(MetricsRequest(id="req1", reset_peak=True), handler, q)  # type: ignore[arg-type]

    assert captured == {"request_id": "req1", "reset_peak": True}
    assert q.items == [fake_response]


# ---------------------------------------------------------------------------
# AdapterProcessor
# ---------------------------------------------------------------------------


def test_adapter_processor_can_process_list_and_refresh() -> None:
    p = AdapterProcessor()
    assert p.can_process(AdapterListRequest(id="x"))
    assert p.can_process(AdapterRefreshRequest(id="x"))
    assert not p.can_process(MetricsRequest(id="x"))


def test_adapter_list_no_manager() -> None:
    handler = MagicMock()
    handler.get_adapter_manager.return_value = None
    q = ResponseQueueFake()

    AdapterProcessor().handle(AdapterListRequest(id="r1"), handler, q)  # type: ignore[arg-type]

    assert len(q.items) == 1
    resp = q.items[0]
    assert isinstance(resp, AdapterListResponse)
    assert resp.adapters == []


def test_adapter_list_with_adapters() -> None:
    handler = MagicMock()
    manager = MagicMock()
    manager.list.return_value = [_adapter("a1", "n1"), _adapter("a2", "n2")]
    handler.get_adapter_manager.return_value = manager
    q = ResponseQueueFake()

    AdapterProcessor().handle(AdapterListRequest(id="r1"), handler, q)  # type: ignore[arg-type]

    resp = q.items[0]
    assert isinstance(resp, AdapterListResponse)
    assert len(resp.adapters) == 2
    assert resp.adapters[0].key == "a1"
    assert resp.adapters[0].name == "n1"
    assert resp.adapters[0].loaded_at == "2026-01-01T00:00:00+00:00"


def test_adapter_refresh_no_manager() -> None:
    handler = MagicMock()
    handler.get_adapter_manager.return_value = None
    q = ResponseQueueFake()

    AdapterProcessor().handle(AdapterRefreshRequest(id="r1", key="k"), handler, q)  # type: ignore[arg-type]

    resp = q.items[0]
    assert isinstance(resp, AdapterRefreshResponse)
    assert resp.status == "disabled"
    assert resp.adapters_loaded == 0
    assert resp.key == "k"  # Echoed back


def test_adapter_refresh_full_scan() -> None:
    handler = MagicMock()
    manager = MagicMock()
    manager.scan.return_value = 5
    handler.get_adapter_manager.return_value = manager
    q = ResponseQueueFake()

    AdapterProcessor().handle(AdapterRefreshRequest(id="r1"), handler, q)  # type: ignore[arg-type]

    manager.scan.assert_called_once()
    resp = q.items[0]
    assert isinstance(resp, AdapterRefreshResponse)
    assert resp.status == "scanned"
    assert resp.adapters_loaded == 5


def test_adapter_refresh_with_key_still_full_scans() -> None:
    """Confirms intentional behavior: key is echoed but ignored."""
    handler = MagicMock()
    manager = MagicMock()
    manager.scan.return_value = 3
    handler.get_adapter_manager.return_value = manager
    q = ResponseQueueFake()

    AdapterProcessor().handle(
        AdapterRefreshRequest(id="r1", key="my-adapter"), handler, q
    )  # type: ignore[arg-type]

    manager.scan.assert_called_once()
    resp = q.items[0]
    assert resp.key == "my-adapter"
    assert resp.adapters_loaded == 3


# ---------------------------------------------------------------------------
# EmbeddingProcessor
# ---------------------------------------------------------------------------


def test_embedding_processor_can_process() -> None:
    p = EmbeddingProcessor()
    assert p.can_process(EmbeddingRequest(id="x", inputs=["hi"]))
    assert not p.can_process(MetricsRequest(id="x"))


def test_embedding_engine_does_not_support() -> None:
    handler = MagicMock()
    handler.engine.supports_embeddings.return_value = False
    q = ResponseQueueFake()

    EmbeddingProcessor().handle(EmbeddingRequest(id="r1", inputs=["a"]), handler, q)  # type: ignore[arg-type]

    resp = q.items[0]
    assert isinstance(resp, EmbeddingResponse)
    assert resp.status == RequestStatus.FAILED
    assert "does not support embeddings" in resp.error


def test_embedding_engine_missing_supports_method() -> None:
    """Engines without supports_embeddings attribute default to False."""
    handler = MagicMock()
    # spec=[] means no attributes; getattr fallback returns the lambda default
    handler.engine = MagicMock(spec=[])
    q = ResponseQueueFake()

    EmbeddingProcessor().handle(EmbeddingRequest(id="r1", inputs=["a"]), handler, q)  # type: ignore[arg-type]

    resp = q.items[0]
    assert resp.status == RequestStatus.FAILED


def test_embedding_success() -> None:
    handler = MagicMock()
    handler.engine.supports_embeddings.return_value = True
    handler.engine.embed.return_value = ([[0.1, 0.2]], 5)
    q = ResponseQueueFake()

    EmbeddingProcessor().handle(
        EmbeddingRequest(id="r1", inputs=["a"], dimensions=2), handler, q
    )  # type: ignore[arg-type]

    handler.engine.embed.assert_called_once_with(["a"], 2)
    resp = q.items[0]
    assert resp.status == RequestStatus.COMPLETED
    assert resp.embeddings == [[0.1, 0.2]]
    assert resp.total_tokens == 5


def test_embedding_engine_raises() -> None:
    handler = MagicMock()
    handler.engine.supports_embeddings.return_value = True
    handler.engine.embed.side_effect = RuntimeError("boom")
    q = ResponseQueueFake()

    EmbeddingProcessor().handle(EmbeddingRequest(id="r1", inputs=["a"]), handler, q)  # type: ignore[arg-type]

    resp = q.items[0]
    assert resp.status == RequestStatus.FAILED
    assert "boom" in resp.error


# ---------------------------------------------------------------------------
# InferenceProcessor
# ---------------------------------------------------------------------------


def test_inference_processor_can_process() -> None:
    p = InferenceProcessor()
    assert p.can_process(Request(id="x", prompt="hi"))
    assert not p.can_process(MetricsRequest(id="x"))


def test_inference_submit_accepted() -> None:
    """Successful submit puts no immediate response (response comes via engine loop)."""
    handler = MagicMock()
    handler.submit.return_value = True
    q = ResponseQueueFake()

    InferenceProcessor().handle(Request(id="r1", prompt="hi"), handler, q)  # type: ignore[arg-type]

    handler.submit.assert_called_once()
    assert q.items == []


def test_inference_submit_rejected() -> None:
    """If handler rejects, processor emits a REJECTED response."""
    handler = MagicMock()
    handler.submit.return_value = False
    q = ResponseQueueFake()

    InferenceProcessor().handle(Request(id="r1", prompt="hi"), handler, q)  # type: ignore[arg-type]

    assert len(q.items) == 1
    resp = q.items[0]
    assert isinstance(resp, Response)
    assert resp.status == RequestStatus.REJECTED


# ---------------------------------------------------------------------------
# Chain dispatch
# ---------------------------------------------------------------------------


def test_chain_dispatches_metrics_to_metrics_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llm_infer.serving.dispatch.processors.build_metrics_response",
        lambda rid, h, rp: MagicMock(id=rid),
    )
    chain = create_request_processor_chain()
    handler = MagicMock()
    q = ResponseQueueFake()

    chain.process(MetricsRequest(id="r1"), handler, q)  # type: ignore[arg-type]

    assert len(q.items) == 1


def test_chain_dispatches_adapter_request() -> None:
    chain = create_request_processor_chain()
    handler = MagicMock()
    handler.get_adapter_manager.return_value = None
    q = ResponseQueueFake()

    chain.process(AdapterListRequest(id="r1"), handler, q)  # type: ignore[arg-type]

    assert isinstance(q.items[0], AdapterListResponse)


def test_chain_dispatches_inference_request() -> None:
    chain = create_request_processor_chain()
    handler = MagicMock()
    handler.submit.return_value = True
    q = ResponseQueueFake()

    chain.process(Request(id="r1", prompt="hi"), handler, q)  # type: ignore[arg-type]

    handler.submit.assert_called_once()


def test_chain_unhandled_request_emits_error() -> None:
    """A request type not matched by any processor emits a FAILED response."""

    class UnknownRequest:
        id = "r1"

    chain = create_request_processor_chain()
    handler = MagicMock()
    q = ResponseQueueFake()

    chain.process(UnknownRequest(), handler, q)  # type: ignore[arg-type]

    assert len(q.items) == 1
    resp = q.items[0]
    assert resp.status == RequestStatus.FAILED
    assert "Unhandled request type" in resp.error
