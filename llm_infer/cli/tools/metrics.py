"""Metrics tool - query server metrics."""

import argparse
import json
import urllib.error
import urllib.request

from appinfra.app.tools import Tool, ToolConfig


class MetricsTool(Tool):
    """Query server metrics (GPU memory, KV cache, etc.)."""

    def __init__(self, parent=None):
        config = ToolConfig(
            name="metrics", aliases=["m"], help_text="Get server metrics"
        )
        super().__init__(parent, config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--host", default="localhost", help="Server host (default: localhost)"
        )
        parser.add_argument(
            "--port", "-p", type=int, default=8000, help="Server port (default: 8000)"
        )
        parser.add_argument(
            "--reset-peak", action="store_true", help="Reset peak memory after reading"
        )
        parser.add_argument("--json", "-j", action="store_true", help="Output raw JSON")

    def run(self, **kwargs) -> int:
        url = f"http://{self.args.host}:{self.args.port}/metrics"
        if self.args.reset_peak:
            url += "?reset_peak=true"

        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            if "Connection refused" in str(e):
                self.lg.error(
                    "server not running",
                    extra={"host": self.args.host, "port": self.args.port},
                )
            else:
                self.lg.error("request failed", extra={"error": str(e)})
            return 1
        except Exception as e:
            self.lg.error("request failed", extra={"error": str(e)})
            return 1

        if self.args.json:
            print(json.dumps(data, indent=2))
        else:
            self._print_formatted(data)
        return 0

    def _print_formatted(self, data: dict) -> None:
        """Print metrics in human-readable format."""
        gpu = data["gpu"]
        kv = data["kv_cache"]
        seq = data["sequences"]

        self.lg.info(
            "GPU memory",
            extra={
                "allocated_mb": f"{gpu['allocated_mb']:.1f}",
                "reserved_mb": f"{gpu['reserved_mb']:.1f}",
                "peak_mb": f"{gpu['peak_mb']:.1f}",
            },
        )
        self.lg.info(
            "KV cache",
            extra={
                "mb": f"{kv['mb']:.1f}",
                "blocks": f"{kv['blocks_used']}/{kv['blocks_total']}",
                "capacity_tokens": kv["capacity_tokens"],
            },
        )
        self.lg.info(
            "Sequences",
            extra={
                "active": seq["active"],
                "total_tokens": seq["total_tokens"],
                "pending": data["pending_requests"],
            },
        )
