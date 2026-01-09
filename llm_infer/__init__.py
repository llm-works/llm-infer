"""LLM inference server with paged attention and continuous batching."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-infer")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
