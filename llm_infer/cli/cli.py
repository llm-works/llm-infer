#!/home/ubuntu/.miniconda3/envs/ml/bin/python
"""CLI entry point."""

import os

# Disable vLLM's dictConfig call BEFORE any vLLM imports
# vLLM's envs module caches env vars at import time, so this must be set early
# dictConfig closes existing FileHandler streams, breaking file logging
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

# Imports below must follow VLLM_CONFIGURE_LOGGING setup above.
from appinfra.app import AppBuilder  # noqa: E402

from .tools import CompatTool, MetricsTool, QueryTool, ServeTool  # noqa: E402


def main() -> int:
    """Main entry point for the CLI."""
    app = (
        AppBuilder("inference")
        .with_description("LLM inference server with paged attention")
        .with_config_file("llm-infer.yaml")
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
