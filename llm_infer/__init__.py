"""LLM inference server with paged attention and continuous batching."""

from importlib.metadata import PackageNotFoundError, version

from .adapter_meta import AdapterMetadata, compute_adapter_metadata

try:
    __version__ = version("llm-infer")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = ["__version__", "AdapterMetadata", "compute_adapter_metadata"]
