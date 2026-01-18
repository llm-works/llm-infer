"""Response processing framework for LLM streams.

Provides abstract interfaces and implementations for parsing and resolving
LLM response streams. The framework supports customization at multiple levels:

1. Use defaults: ResponseProcessor() uses ThinkTagParser + TerminalResolver
2. Custom resolver: ResponseProcessor(resolver=MyResolver())
3. Custom parser: ResponseProcessor(parser=MyParser())
4. Fully custom: ResponseProcessor(parser=MyParser(), resolver=MyResolver())

Core API (most users need only these):
    - ResponseProcessor: Main entry point for stream processing
    - TerminalResolver: Default resolver for terminal output with ANSI styling
    - EventType, StreamEvent: Event types emitted by parsers

Advanced API (for customization):
    - Parser, Resolver: Protocols for custom implementations
    - BaseParser: Composable parser chain (for combining parsers)
    - BaseResolver: Base class with hook methods for custom resolvers
    - ThinkTagParser: Parse <think>/<thinking> blocks (default parser)
    - CodeBlockParser: Parse markdown code fences
    - LatexTransformer: Transform LaTeX to Unicode in TEXT events

Utilities:
    - extract_thinking: Extract thinking content from complete text
    - ThinkTagNormalizer: Normalize think tag variants
    - ThinkStreamSeparator: Route tokens to thinking/content fields
    - LatexConverter: Low-level LaTeX to Unicode conversion
    - Utf8StreamBuffer: Handle incomplete UTF-8 sequences

Example:
    from llm_infer.response import ResponseProcessor

    processor = ResponseProcessor()
    for token in stream:
        processor.feed(token)
    processor.finish()

For custom behavior, subclass BaseResolver and override specific handlers:

    from llm_infer.response import ResponseProcessor, BaseResolver, StreamEvent

    class CodeCollectingResolver(BaseResolver):
        def __init__(self) -> None:
            super().__init__()
            self.python_snippets: list[str] = []

        def on_code_end(self, event: StreamEvent, code: str, language: str) -> None:
            if language == "python":
                self.python_snippets.append(code)
            super().on_code_end(event, code, language)

    processor = ResponseProcessor(resolver=CodeCollectingResolver())
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
    # --- Core API (most users need only these) ---
    "ResponseProcessor",
    "TerminalResolver",
    "EventType",
    "StreamEvent",
    # --- Advanced API (for customization) ---
    # Protocols
    "Parser",
    "Resolver",
    # Parser implementations
    "BaseParser",
    "ThinkTagParser",
    "CodeBlockParser",
    "LatexTransformer",
    # Resolver implementations
    "BaseResolver",
    # Factory
    "create_default_parser",
    # --- Utilities ---
    "extract_thinking",
    "ThinkTagNormalizer",
    "ThinkStreamSeparator",
    "LatexConverter",
    "Utf8StreamBuffer",
]
