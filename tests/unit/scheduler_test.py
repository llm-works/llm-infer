"""Unit tests for scheduler and request management."""

import pytest

from llm_infer.engines.native.kv_cache.pool import BlockPool
from llm_infer.engines.native.scheduler import Request, RequestState, Scheduler

pytestmark = pytest.mark.unit


def make_pool() -> BlockPool:
    """Create a test BlockPool on CPU."""
    return BlockPool(
        num_blocks=10,
        block_size=4,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        device="cpu",
    )


class TestRequestCreate:
    """Test Request.create factory method."""

    def test_create_generates_uuid(self) -> None:
        """Test that create generates a UUID when no context."""
        request = Request.create(prompt_tokens=[1, 2, 3])
        assert request.id is not None
        assert len(request.id) > 0

    def test_create_with_params(self) -> None:
        """Test create with custom parameters."""
        request = Request.create(
            prompt_tokens=[1, 2, 3],
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            stop_token_ids={100, 101},
        )

        assert request.prompt_tokens == [1, 2, 3]
        assert request.max_tokens == 50
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.top_k == 40
        assert request.repetition_penalty == 1.1
        assert request.stop_token_ids == {100, 101}


class TestRequestProperties:
    """Test Request properties."""

    def test_total_tokens(self) -> None:
        """Test total_tokens calculation."""
        request = Request.create(prompt_tokens=[1, 2, 3])
        assert request.total_tokens == 3

        request.output_tokens = [4, 5]
        assert request.total_tokens == 5

    def test_is_finished_by_state(self) -> None:
        """Test is_finished when state is FINISHED."""
        request = Request.create(prompt_tokens=[1, 2, 3])
        assert not request.is_finished

        request.state = RequestState.FINISHED
        assert request.is_finished

    def test_is_finished_by_max_tokens(self) -> None:
        """Test is_finished when max_tokens reached."""
        request = Request.create(prompt_tokens=[1], max_tokens=3)
        request.output_tokens = [2, 3, 4]
        assert request.is_finished

    def test_is_finished_by_stop_token(self) -> None:
        """Test is_finished when stop token generated."""
        request = Request.create(prompt_tokens=[1], stop_token_ids={99})
        request.output_tokens = [2, 3, 99]
        assert request.is_finished

    def test_is_finished_by_finish_reason(self) -> None:
        """Test is_finished when finish_reason set."""
        request = Request.create(prompt_tokens=[1])
        request.finish_reason = "guard"
        assert request.is_finished


class TestRequestMethods:
    """Test Request methods."""

    def test_mark_finished(self) -> None:
        """Test mark_finished sets state."""
        request = Request.create(prompt_tokens=[1])
        request.mark_finished()
        assert request.state == RequestState.FINISHED

    def test_finish_with_reason(self) -> None:
        """Test finish sets reason and state."""
        request = Request.create(prompt_tokens=[1])
        request.finish("guard", "Token repetition detected")

        assert request.finish_reason == "guard"
        assert request.state == RequestState.FINISHED
        assert "Token repetition detected" in request.warnings

    def test_finish_without_message(self) -> None:
        """Test finish without message."""
        request = Request.create(prompt_tokens=[1])
        request.finish("max_tokens")

        assert request.finish_reason == "max_tokens"
        assert len(request.warnings) == 0

    def test_add_warning(self) -> None:
        """Test adding warnings."""
        request = Request.create(prompt_tokens=[1])
        request.add_warning("Warning 1")
        request.add_warning("Warning 2")

        assert request.warnings == ["Warning 1", "Warning 2"]


class TestSchedulerBasic:
    """Test basic Scheduler operations."""

    def test_empty_scheduler(self) -> None:
        """Test empty scheduler properties."""
        scheduler = Scheduler()
        assert scheduler.is_empty
        assert scheduler.num_waiting == 0
        assert scheduler.num_running == 0

    def test_add_request(self) -> None:
        """Test adding requests to scheduler."""
        scheduler = Scheduler()
        request = Request.create(prompt_tokens=[1, 2, 3])

        request_id = scheduler.add_request(request)

        assert request_id == request.id
        assert scheduler.num_waiting == 1
        assert not scheduler.is_empty


class TestSchedulerGetBatch:
    """Test Scheduler.get_batch."""

    def test_get_batch_promotes_waiting_to_prefill(self) -> None:
        """Test that get_batch promotes waiting requests."""
        scheduler = Scheduler(max_batch_size=2)
        req1 = Request.create(prompt_tokens=[1])
        req2 = Request.create(prompt_tokens=[2])
        scheduler.add_request(req1)
        scheduler.add_request(req2)

        prefill, decode = scheduler.get_batch()

        assert len(prefill) == 2
        assert len(decode) == 0
        assert req1.state == RequestState.PREFILL
        assert req2.state == RequestState.PREFILL
        assert scheduler.num_waiting == 0
        assert scheduler.num_running == 2

    def test_get_batch_respects_max_batch_size(self) -> None:
        """Test that get_batch respects max_batch_size."""
        scheduler = Scheduler(max_batch_size=1)
        req1 = Request.create(prompt_tokens=[1])
        req2 = Request.create(prompt_tokens=[2])
        scheduler.add_request(req1)
        scheduler.add_request(req2)

        prefill, decode = scheduler.get_batch()

        assert len(prefill) == 1
        assert scheduler.num_waiting == 1
        assert scheduler.num_running == 1

    def test_get_batch_returns_decode_requests(self) -> None:
        """Test that get_batch returns running decode requests."""
        scheduler = Scheduler(max_batch_size=2)
        req = Request.create(prompt_tokens=[1])
        scheduler.add_request(req)

        # First batch: prefill
        scheduler.get_batch()
        scheduler.update_after_step([])  # Transitions to decode

        # Second batch: decode
        prefill, decode = scheduler.get_batch()

        assert len(prefill) == 0
        assert len(decode) == 1
        assert decode[0] == req


class TestSchedulerUpdateAfterStep:
    """Test Scheduler.update_after_step."""

    def test_transitions_prefill_to_decode(self) -> None:
        """Test that prefill requests transition to decode."""
        scheduler = Scheduler()
        req = Request.create(prompt_tokens=[1])
        scheduler.add_request(req)
        scheduler.get_batch()

        assert req.state == RequestState.PREFILL

        scheduler.update_after_step([])

        assert req.state == RequestState.DECODE

    def test_removes_finished_requests(self) -> None:
        """Test that finished requests are removed."""
        scheduler = Scheduler()
        req = Request.create(prompt_tokens=[1])
        scheduler.add_request(req)
        scheduler.get_batch()

        scheduler.update_after_step([req.id])

        assert scheduler.num_running == 0
        assert scheduler.is_empty


class TestSchedulerGetRequest:
    """Test Scheduler.get_request."""

    def test_get_running_request(self) -> None:
        """Test getting a running request."""
        scheduler = Scheduler()
        req = Request.create(prompt_tokens=[1])
        scheduler.add_request(req)
        scheduler.get_batch()  # Move to running

        found = scheduler.get_request(req.id)
        assert found == req

    def test_get_waiting_request(self) -> None:
        """Test getting a waiting request."""
        scheduler = Scheduler(max_batch_size=1)
        req1 = Request.create(prompt_tokens=[1])
        req2 = Request.create(prompt_tokens=[2])
        scheduler.add_request(req1)
        scheduler.add_request(req2)
        scheduler.get_batch()  # Only req1 moves to running

        found = scheduler.get_request(req2.id)
        assert found == req2

    def test_get_nonexistent_request(self) -> None:
        """Test getting a request that doesn't exist."""
        scheduler = Scheduler()
        found = scheduler.get_request("nonexistent")
        assert found is None


class TestSchedulerCancelRequest:
    """Test Scheduler.cancel_request."""

    def test_cancel_running_request(self) -> None:
        """Test canceling a running request."""
        pool = make_pool()
        scheduler = Scheduler()
        req = Request.create(prompt_tokens=[1])
        req.kv_cache.allocate_for_prompt(pool, 5)  # Allocate some blocks
        scheduler.add_request(req)
        scheduler.get_batch()

        initial_free = pool.num_free_blocks

        result = scheduler.cancel_request(req.id, pool)

        assert result is True
        assert scheduler.num_running == 0
        assert pool.num_free_blocks > initial_free  # Blocks freed

    def test_cancel_waiting_request(self) -> None:
        """Test canceling a waiting request."""
        pool = make_pool()
        scheduler = Scheduler(max_batch_size=1)
        req1 = Request.create(prompt_tokens=[1])
        req2 = Request.create(prompt_tokens=[2])
        scheduler.add_request(req1)
        scheduler.add_request(req2)
        scheduler.get_batch()  # Only req1 moves to running

        result = scheduler.cancel_request(req2.id, pool)

        assert result is True
        assert scheduler.num_waiting == 0

    def test_cancel_nonexistent_request(self) -> None:
        """Test canceling a request that doesn't exist."""
        pool = make_pool()
        scheduler = Scheduler()

        result = scheduler.cancel_request("nonexistent", pool)

        assert result is False
