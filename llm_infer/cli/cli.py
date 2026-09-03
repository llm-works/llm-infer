#!/home/ubuntu/.miniconda3/envs/ml/bin/python

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""CLI entry point."""

import os
from pathlib import Path

# Disable vLLM's dictConfig call BEFORE any vLLM imports
# vLLM's envs module caches env vars at import time, so this must be set early
# dictConfig closes existing FileHandler streams, breaking file logging
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

# Imports below must follow VLLM_CONFIGURE_LOGGING setup above.
from appinfra.app import AppBuilder  # noqa: E402
from appinfra.config import Config, xdg_candidates  # noqa: E402

from .tools import CompatTool, MetricsTool, QueryTool, ServeTool  # noqa: E402

# Bundled etc/ ships inside the wheel. Used as the default --etc-dir so
# `pip install llm-infer && llm-infer serve` works without local setup.
_BUNDLED_ETC_DIR = Path(__file__).parent.parent / "etc"


def main() -> int:
    """Main entry point for the CLI."""
    builder = AppBuilder("inference").with_description(
        "LLM inference server with paged attention"
    )
    builder = _configure_source(builder)
    app = (
        builder.tools.with_tool(CompatTool())
        .with_tool(MetricsTool())
        .with_tool(QueryTool())
        .with_tool(ServeTool())
        .done()
        .build()
    )
    result: int = app.main()
    return result


def _configure_source(builder: AppBuilder) -> AppBuilder:
    """Wire config-protocol v1 loading onto the builder.

    If an XDG overlay exists, load it directly with `project_root` pinned to
    the bundled etc dir — the tightest ancestor containing both the base and
    its `!include` siblings, so absolute-include-of-base plus relative sibling
    includes both resolve within the security boundary. Otherwise load the
    packaged base from --etc-dir.

    See `appinfra docs show config-protocol` (v1: one file per package load).
    """
    overlay = _find_xdg_overlay()
    if overlay is not None:
        return builder.with_config(Config(str(overlay), project_root=_BUNDLED_ETC_DIR))
    return builder.with_config_file("llm-infer.yaml").with_standard_arg(
        "etc_dir", default=str(_BUNDLED_ETC_DIR)
    )


def _find_xdg_overlay() -> Path | None:
    """Return the first existing XDG config candidate for llm-infer, or None."""
    for candidate in xdg_candidates("llm-works", "llm-infer"):
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    raise SystemExit(main())
