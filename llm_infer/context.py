"""Request context for cross-cutting concerns."""

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from appinfra.log import Logger
from appinfra.time import delta_str, start


class Event(str, Enum):
    """Request lifecycle events."""

    # DEBUG level (~5 per request)
    REQUESTED = "requested"
    TOKENIZED = "tokenized"
    PREFILLED = "prefilled"
    DECODED = "decoded"
    COMPLETE = "complete"

    # TRACE level (~10-20 per request)
    DECODE = "decode"
    KV_ALLOC = "kv_alloc"
    KV_FREE = "kv_free"
    SAMPLED = "sampled"


# O(1) lookup for DEBUG-level events
_DEBUG_EVENTS = frozenset(
    {
        Event.REQUESTED,
        Event.TOKENIZED,
        Event.PREFILLED,
        Event.DECODED,
        Event.COMPLETE,
    }
)


@dataclass
class RequestContext:
    """Shared context for a request across all pipeline stages."""

    id: str
    lg: Logger
    start_time: float = field(default_factory=start)
    _last_mark: float = field(default_factory=start)

    # Timing accumulator for CSV export (raw seconds)
    timings: dict[str, float] = field(default_factory=dict)

    def mark(self, event: Event, **data: Any) -> None:
        """Record an event marker with automatic timing."""
        now = start()
        is_debug = event in _DEBUG_EVENTS
        elapsed = now - self._last_mark
        cumulative = now - self.start_time

        # Only DEBUG events update the marker (keeps DEBUG timing chain intact)
        if is_debug:
            self._last_mark = now

        # Store raw timing
        self.timings[event.value] = elapsed

        # Build log data with ordered fields (after and total first)
        extra: OrderedDict[str, object] = OrderedDict()
        extra["after"] = delta_str(elapsed)
        extra["total"] = delta_str(cumulative)
        extra["request_id"] = self.id
        extra.update(data)

        # Route to appropriate log level
        if is_debug:
            self.lg.debug(event.value, extra=extra)
        else:
            self.lg.trace(event.value, extra=extra)

    def get_timings_csv(self) -> str:
        """Export timings as CSV row."""
        return ",".join(f"{k}={v:.6f}" for k, v in self.timings.items())
