"""Stream events for response processing.

Defines the event types and data structures used to represent parsed content
from LLM response streams.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    """Types of events emitted by parsers."""

    TEXT = auto()
    THINK_START = auto()
    THINK_CONTENT = auto()
    THINK_END = auto()
    CODE_START = auto()
    CODE_CONTENT = auto()
    CODE_END = auto()


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """A single event from the response stream.

    Attributes:
        type: The type of event.
        content: Text content associated with the event.
        metadata: Additional event-specific data (e.g., {"language": "python"}).

    Note:
        Events are frozen (attributes cannot be reassigned) but contain a mutable
        metadata dict for practical reasons (avoiding MappingProxyType complexity).

        Important:
        - Events are NOT hashable due to the mutable metadata dict
        - Do NOT modify metadata after event creation - treat it as read-only
        - For hashable representation, use tuple: (event.type, event.content)

        The frozen+mutable pattern is intentional: events flow through streaming
        pipelines and are not used as dict keys or set members.
    """

    type: EventType
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
