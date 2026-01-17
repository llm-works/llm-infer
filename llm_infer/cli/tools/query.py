"""Query tool - send queries to a running inference server."""

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

from appinfra.app.tools import Tool, ToolConfig

from ...text import LatexFormatter, ThinkFormatter, Utf8StreamBuffer


class QueryTool(Tool):
    """Send a query to the running inference server."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="query", aliases=["q"], help_text="Query the running inference server"
        )
        super().__init__(parent, config)

    def _add_connection_args(self, parser: argparse.ArgumentParser) -> None:
        """Add server connection arguments."""
        parser.add_argument(
            "prompt", nargs="?", help="The prompt to send (or use --stdin)"
        )
        parser.add_argument(
            "--stdin", action="store_true", help="Read prompt from stdin"
        )
        parser.add_argument("--health", action="store_true", help="Check server health")
        parser.add_argument("--host", default="localhost", help="Server host")
        parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")

    def _add_sampling_args(self, parser: argparse.ArgumentParser) -> None:
        """Add sampling parameter arguments."""
        parser.add_argument(
            "--max-tokens", "-n", type=int, default=100, help="Max tokens"
        )
        parser.add_argument(
            "--temperature", "-t", type=float, default=1.0, help="Temperature"
        )
        parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling")
        parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
        parser.add_argument(
            "--repetition-penalty",
            "-r",
            type=float,
            default=1.1,
            help="Repetition penalty",
        )

    def _add_output_args(self, parser: argparse.ArgumentParser) -> None:
        """Add output format arguments."""
        parser.add_argument("--json", "-j", action="store_true", help="Output raw JSON")
        parser.add_argument(
            "--no-stream", action="store_true", help="Disable streaming"
        )
        parser.add_argument(
            "--raw",
            action="store_true",
            help="Raw output (no LaTeX conversion or think tag styling)",
        )

    def _add_prompt_args(self, parser: argparse.ArgumentParser) -> None:
        """Add prompt configuration arguments."""
        parser.add_argument("--system", "-s", type=str, help="System prompt to prepend")
        think_group = parser.add_mutually_exclusive_group()
        think_group.add_argument(
            "--think",
            action="store_true",
            help="Enable thinking mode (extended reasoning with <think> blocks)",
        )
        think_group.add_argument(
            "--no-think",
            action="store_true",
            help="Disable thinking mode",
        )

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        self._add_connection_args(parser)
        self._add_sampling_args(parser)
        self._add_output_args(parser)
        self._add_prompt_args(parser)

    def _get_prompt(self) -> str | None:
        """Get prompt from args or stdin. Returns None on error."""
        if self.args.stdin:
            prompt = sys.stdin.read().strip()
        elif self.args.prompt:
            prompt = self.args.prompt
        else:
            self.lg.error("no prompt provided, use positional argument or --stdin")
            return None
        if not prompt:
            self.lg.error("empty prompt")
            return None
        return prompt

    def _build_payload(self, prompt: str) -> dict:
        """Build request payload."""
        return {
            "prompt": prompt,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "repetition_penalty": self.args.repetition_penalty,
        }

    def _send_request(self, url: str, payload: dict) -> dict | None:
        """Send request to server. Returns response dict or None on error."""
        self.lg.debug("sending request", extra={"url": url, "payload": payload})
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result: dict = json.loads(resp.read().decode("utf-8"))
                return result
        except urllib.error.URLError as e:
            if "Connection refused" in str(e):
                self.lg.error(
                    f"cannot connect to server at {self.args.host}:{self.args.port}"
                )
                self.lg.info("is the server running? start with: inference serve")
            else:
                self.lg.error(f"request failed: {e}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            self.lg.error(f"server error {e.code}: {body}")
        except Exception as e:
            self.lg.error(f"request failed: {e}")
        return None

    def _get_system_prompt(self) -> str | None:
        """Get system prompt from CLI args. Server handles model-based prompts."""
        return str(self.args.system) if self.args.system else None

    def _get_think_flag(self) -> bool | None:
        """Get think flag value from args. Returns None if neither flag is set."""
        if self.args.think:
            return True
        if getattr(self.args, "no_think", False):
            return False
        return None

    def _build_streaming_payload(self, prompt: str) -> dict:
        """Build streaming chat completions payload."""
        messages = []
        system_prompt = self._get_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": "default",
            "messages": messages,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "frequency_penalty": (self.args.repetition_penalty - 1.0) * 2,
            "stream": True,
        }

        # Add think flag if set (server handles suffix injection and tag normalization)
        think = self._get_think_flag()
        if think is not None:
            payload["think"] = think

        return payload

    def _setup_stream_formatters(
        self,
    ) -> tuple[ThinkFormatter | None, LatexFormatter | None]:
        """Set up formatters for stream processing."""
        if self.args.raw:
            return None, None
        return ThinkFormatter(), LatexFormatter()

    def _process_sse_chunk(
        self,
        data: bytes,
        utf8_buffer: Utf8StreamBuffer,
        think_fmt: ThinkFormatter | None,
        latex_fmt: LatexFormatter | None,
    ) -> None:
        """Process a single SSE chunk and write to stdout."""
        try:
            chunk = json.loads(data)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content", "")
            if text:
                text = utf8_buffer.process(text)
                if think_fmt and latex_fmt:
                    text = think_fmt.process(text)
                    text = latex_fmt.process(text)
                if text:
                    sys.stdout.write(text)
                    sys.stdout.flush()
        except json.JSONDecodeError:
            pass

    def _flush_stream_formatters(
        self,
        utf8_buffer: Utf8StreamBuffer,
        think_fmt: ThinkFormatter | None,
        latex_fmt: LatexFormatter | None,
    ) -> None:
        """Flush remaining buffered content to stdout."""
        remaining = utf8_buffer.flush()
        if think_fmt and latex_fmt:
            remaining = think_fmt.process(remaining)
            remaining += think_fmt.flush()
            remaining = latex_fmt.process(remaining)
            remaining += latex_fmt.flush()
        if remaining:
            sys.stdout.write(remaining)

    def _process_sse_stream(self, resp: Any) -> None:
        """Process SSE stream and print tokens to stdout."""
        print()  # Start on new line

        utf8_buffer = Utf8StreamBuffer()
        think_fmt, latex_fmt = self._setup_stream_formatters()

        for line in resp:
            line = line.strip()
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            self._process_sse_chunk(data, utf8_buffer, think_fmt, latex_fmt)

        self._flush_stream_formatters(utf8_buffer, think_fmt, latex_fmt)
        print()  # End with newline

    def _stream_request(self, prompt: str) -> int:
        """Send streaming request and print tokens as they arrive."""
        url = f"http://{self.args.host}:{self.args.port}/v1/chat/completions"
        payload = self._build_streaming_payload(prompt)
        self.lg.debug("sending streaming request", extra={"url": url})

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                self._process_sse_stream(resp)
            return 0
        except urllib.error.URLError as e:
            if "Connection refused" in str(e):
                self.lg.error(
                    f"cannot connect to server at {self.args.host}:{self.args.port}"
                )
                self.lg.info("is the server running? start with: inference serve")
            else:
                self.lg.error(f"request failed: {e}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            self.lg.error(f"server error {e.code}: {body}")
        except Exception as e:
            self.lg.error(f"streaming request failed: {e}")
        return 1

    def run(self, **kwargs: Any) -> int:
        if self.args.health:
            return self._check_health()

        prompt = self._get_prompt()
        if not prompt:
            return 1

        # Use streaming by default (unless --no-stream or --json)
        if not self.args.no_stream and not self.args.json:
            return self._stream_request(prompt)

        # Non-streaming path
        url = f"http://{self.args.host}:{self.args.port}/generate"
        data = self._send_request(url, self._build_payload(prompt))
        if data is None:
            return 1

        if self.args.json:
            print(json.dumps(data, indent=2))
        else:
            self.lg.debug(
                "response received",
                extra={
                    "prompt_tokens": data.get("prompt_tokens", 0),
                    "completion_tokens": data.get("completion_tokens", 0),
                },
            )
            print(f"\n{data['text']}")
        return 0

    def _check_health(self) -> int:
        """Check server health."""
        url = f"http://{self.args.host}:{self.args.port}/health"
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
                self.lg.error("health check failed", extra={"error": str(e)})
            return 1
        except Exception as e:
            self.lg.error("health check failed", extra={"error": str(e)})
            return 1

        if self.args.json:
            print(json.dumps(data, indent=2))
        else:
            status = data.get("status", "unknown")
            if status == "ok":
                self.lg.info("server ready", extra={"status": status})
            else:
                self.lg.info("server initializing", extra={"status": status})
        return 0
