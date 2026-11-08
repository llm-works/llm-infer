# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for failure_response: an engine exception as a FAILED response."""

from llm_infer.engines.protocol import BackendRequestError
from llm_infer.serving.dispatch.types import RequestStatus, failure_response


class TestFailureResponse:
    def test_backend_client_error_keeps_its_status(self) -> None:
        response = failure_response("r1", BackendRequestError(400, "bad max_tokens"))

        assert response.status == RequestStatus.FAILED
        assert response.error == "bad max_tokens"
        assert response.http_status == 400

    def test_backend_server_error_has_no_status(self) -> None:
        """A 5xx from the backend is our failure, answered as 500."""
        response = failure_response("r1", BackendRequestError(503, "overloaded"))

        assert response.error == "overloaded"
        assert response.http_status is None

    def test_other_exception_has_no_status(self) -> None:
        response = failure_response("r1", RuntimeError("boom"))

        assert response.status == RequestStatus.FAILED
        assert response.error == "boom"
        assert response.http_status is None
