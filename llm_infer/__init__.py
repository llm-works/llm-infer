# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors
#
# Maintained by LLM Works LLC (https://llm-works.ai) and contributors.

"""Multi-backend LLM client library, a from-scratch PagedAttention inference engine, and an OpenAI-compatible serve wrapper."""

from importlib.metadata import version as _meta_version

__version__ = _meta_version("llm-infer")

from . import client, models
from .adapter_meta import AdapterMetadata, compute_adapter_metadata

__all__ = [
    "__version__",
    "client",
    "models",
    "AdapterMetadata",
    "compute_adapter_metadata",
]
