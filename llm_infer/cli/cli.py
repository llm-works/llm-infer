#!/home/ubuntu/.miniconda3/envs/ml/bin/python

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""CLI entry point."""

import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Disable vLLM's dictConfig call BEFORE any vLLM imports
# vLLM's envs module caches env vars at import time, so this must be set early
# dictConfig closes existing FileHandler streams, breaking file logging
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

# Imports below must follow VLLM_CONFIGURE_LOGGING setup above.
from appinfra.app import AppBuilder  # noqa: E402

from .tools import CompatTool, MetricsTool, QueryTool, ServeTool  # noqa: E402


def _version() -> str:
    try:
        return _pkg_version("llm-infer")
    except PackageNotFoundError:
        return "0.0.0.dev0"


def main() -> int:
    """Main entry point for the CLI."""
    # ConfigSpec resolves the base file via AUTO origin (llm_infer package
    # dir + etc/llm-infer.yaml). Precedence: --etc-dir > XDG overlay >
    # packaged base. See `appinfra docs show config-protocol`.
    app = (
        AppBuilder("llm-infer")
        .with_description("LLM inference server with paged attention")
        .version.with_semver(_version())
        .done()
        .config.with_spec("llm-works", "llm-infer")
        .done()
        .cli.with_flags(
            etc_dir=True,
            config_file=True,
            log=True,
            version=True,
            quiet=True,
        )
        .done()
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
