"""Request dispatch layer for inference server."""

from .config import InferenceConfig
from .handler import RequestHandler
from .handlers import BoundedQueueHandler, ContinuousBatchingHandler, SequentialHandler
from .loop import run_engine_loop
from .types import Request, RequestStatus, Response

# NOTE: run_server is lazily imported to avoid circular import.
# Import chain without lazy loading:
#   serving/api/__init__ → .routes → ..dispatch.types → ..dispatch.__init__
#     → .main → ..api.routes (circular!)

__all__ = [
    # Config
    "InferenceConfig",
    # Types
    "Request",
    "Response",
    "RequestStatus",
    # Handler interface
    "RequestHandler",
    # Handler implementations
    "SequentialHandler",
    "BoundedQueueHandler",
    "ContinuousBatchingHandler",
    # Functions
    "run_engine_loop",
    "run_server",
]


def __getattr__(name: str) -> object:
    """Lazy import for run_server to break circular import."""
    if name == "run_server":
        from .main import run_server

        return run_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
