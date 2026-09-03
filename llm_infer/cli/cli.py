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

from .tools import CompatTool, MetricsTool, QueryTool, ServeTool  # noqa: E402

# Bundled base config; ships in the wheel. Anchor for XDG overlays and
# fallback when neither --etc-dir nor an XDG overlay is present.
_BUNDLED_CONFIG = Path(__file__).parent.parent / "etc" / "llm-infer.yaml"


def main() -> int:
    """Main entry point for the CLI."""
    # v1 config-protocol precedence handled by with_config_spec:
    # --etc-dir (if passed) > XDG overlay > packaged base. See
    # `appinfra docs show config-protocol`.
    app = (
        AppBuilder("inference")
        .with_description("LLM inference server with paged attention")
        .with_config_spec("llm-works", "llm-infer", _BUNDLED_CONFIG)
        .with_standard_args(etc_dir=True)
        .tools.with_tool(CompatTool())
        .with_tool(MetricsTool())
        .with_tool(QueryTool())
        .with_tool(ServeTool())
        .done()
        .build()
    )
    result: int = app.main()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
