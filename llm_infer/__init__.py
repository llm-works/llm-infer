# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors
#
# Maintained by LLM Works LLC (https://llm-works.ai) and contributors.

"""LLM inference server with paged attention and continuous batching."""

from importlib.metadata import PackageNotFoundError, version

from . import client, models
from .adapter_meta import AdapterMetadata, compute_adapter_metadata

try:
    __version__ = version("llm-infer")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "client",
    "models",
    "AdapterMetadata",
    "compute_adapter_metadata",
]
