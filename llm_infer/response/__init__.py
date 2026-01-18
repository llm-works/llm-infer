"""Response processing framework for LLM streams.

Provides abstract interfaces and implementations for parsing and resolving
LLM response streams. The framework supports customization at multiple levels:

1. Use defaults: ResponseProcessor() uses ThinkTagParser + TerminalResolver
2. Custom resolver: ResponseProcessor(resolver=MyResolver())
3. Custom parser: ResponseProcessor(parser=MyParser())
4. Fully custom: ResponseProcessor(parser=MyParser(), resolver=MyResolver())

Example:
    from llm_infer.response import ResponseProcessor

    processor = ResponseProcessor()
    for token in stream:
        processor.feed(token)
    processor.finish()

For custom behavior, subclass BaseResolver and override specific handlers:

    from llm_infer.response import ResponseProcessor, BaseResolver, StreamEvent

    class CodeExecutingResolver(BaseResolver):
        def on_code_end(self, event: StreamEvent) -> None:
            if self._code_language == "python":
                exec(self._code_buffer)
            super().on_code_end(event)

    processor = ResponseProcessor(resolver=CodeExecutingResolver())
"""

# Events
from .events import EventType, StreamEvent

# LaTeX converter
from .latex import LatexConverter

# Parsers
from .parsers import (
    BaseParser,
    CodeBlockParser,
    LatexTransformer,
    ThinkStreamSeparator,
    ThinkTagNormalizer,
    ThinkTagParser,
    extract_thinking,
)

# Processor
from .processor import ResponseProcessor, create_default_parser

# Protocols
from .protocols import Parser, Resolver

# Resolvers
from .resolvers import BaseResolver, TerminalResolver

# Utilities
from .utf8 import Utf8StreamBuffer

__all__ = [
    # Events
    "EventType",
    "StreamEvent",
    # Protocols
    "Parser",
    "Resolver",
    # Processor
    "ResponseProcessor",
    "create_default_parser",
    # Parsers
    "BaseParser",
    "CodeBlockParser",
    "LatexTransformer",
    "ThinkTagParser",
    "ThinkTagNormalizer",
    "ThinkStreamSeparator",
    "extract_thinking",
    # Resolvers
    "BaseResolver",
    "TerminalResolver",
    # Utilities
    "LatexConverter",
    "Utf8StreamBuffer",
]
