"""Query tool - send queries to a running inference server."""

import argparse
import codecs
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from appinfra.app.tools import Tool, ToolConfig

from ..config.models import ModelConfig, get_selected_model_name, load_models_config

# ANSI escape codes for think block formatting
_THINK_START = "\x1b[3;38;5;245m"  # Italic + grey (256-color: 245)
_THINK_END = "\x1b[0m"  # Reset all formatting

# Default config path relative to this file
_DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent / "etc" / "models.yaml"
)


class ThinkFormatter:
    """Formats thinking blocks with grey italic styling for terminal output.

    Supports configurable tag variants used by different models.
    """

    def __init__(
        self,
        open_tags: list[str] | None = None,
        close_tags: list[str] | None = None,
    ) -> None:
        self.open_tags = open_tags or ["<think>", "<thinking>"]
        self.close_tags = close_tags or ["</think>", "</thinking>"]
        self.max_tag_len = max(
            max((len(t) for t in self.open_tags), default=0),
            max((len(t) for t in self.close_tags), default=0),
        )
        self.in_think: bool = False
        self.buffer: str = ""

    def _find_open_tag(self, text: str) -> tuple[int, int]:
        """Find earliest opening tag. Returns (position, tag_length) or (-1, 0)."""
        best_pos, best_len = -1, 0
        for tag in self.open_tags:
            pos = text.find(tag)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos, best_len = pos, len(tag)
        return best_pos, best_len

    def _find_close_tag(self, text: str) -> tuple[int, int]:
        """Find earliest closing tag. Returns (position, tag_length) or (-1, 0)."""
        best_pos, best_len = -1, 0
        for tag in self.close_tags:
            pos = text.find(tag)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos, best_len = pos, len(tag)
        return best_pos, best_len

    def _process_in_think(self) -> tuple[str, bool]:
        """Process buffer when inside think block. Returns (output, should_break)."""
        end_idx, tag_len = self._find_close_tag(self.buffer)
        if end_idx == -1:
            safe_len = max(0, len(self.buffer) - self.max_tag_len)
            output = (
                _THINK_START + self.buffer[:safe_len] + _THINK_END
                if safe_len > 0
                else ""
            )
            self.buffer = self.buffer[safe_len:]
            return output, True
        output = _THINK_START + self.buffer[:end_idx] + _THINK_END
        self.buffer, self.in_think = self.buffer[end_idx + tag_len :], False
        return output, False

    def _process_outside_think(self) -> tuple[str, bool]:
        """Process buffer when outside think block. Returns (output, should_break)."""
        start_idx, tag_len = self._find_open_tag(self.buffer)
        if start_idx == -1:
            safe_len = max(0, len(self.buffer) - self.max_tag_len)
            output = self.buffer[:safe_len]
            self.buffer = self.buffer[safe_len:]
            return output, True
        output = self.buffer[:start_idx]
        self.buffer, self.in_think = self.buffer[start_idx + tag_len :], True
        return output, False

    def process(self, text: str) -> str:
        """Process text chunk, applying formatting to think blocks."""
        self.buffer += text
        output = ""
        while True:
            chunk, should_break = (
                self._process_in_think()
                if self.in_think
                else self._process_outside_think()
            )
            output += chunk
            if should_break:
                break
        return output

    def flush(self) -> str:
        """Flush remaining buffer at end of stream."""
        if self.buffer:
            if self.in_think:
                return _THINK_START + self.buffer + _THINK_END
            return self.buffer
        return ""


class LatexFormatter:
    """Convert LaTeX math notation to Unicode for terminal display."""

    # Simple command replacements (no arguments)
    # Keys use literal backslash (not raw strings) to match model output
    REPLACEMENTS: dict[str, str] = {
        "\\times": "×",
        "\\div": "÷",
        "\\pm": "±",
        "\\mp": "∓",
        "\\leq": "≤",
        "\\geq": "≥",
        "\\neq": "≠",
        "\\approx": "≈",
        "\\equiv": "≡",
        "\\infty": "∞",
        "\\pi": "π",
        "\\sum": "Σ",
        "\\prod": "Π",
        "\\int": "∫",
        "\\partial": "∂",
        "\\nabla": "∇",
        "\\cdot": "·",
        "\\ldots": "…",
        "\\cdots": "⋯",
        "\\to": "→",
        "\\rightarrow": "→",
        "\\leftarrow": "←",
        "\\Rightarrow": "⇒",
        "\\Leftarrow": "⇐",
        "\\iff": "⇔",
        "\\forall": "∀",
        "\\exists": "∃",
        "\\in": "∈",
        "\\notin": "∉",
        "\\subset": "⊂",
        "\\supset": "⊃",
        "\\cup": "∪",
        "\\cap": "∩",
        "\\emptyset": "∅",
        "\\neg": "¬",
        "\\land": "∧",
        "\\lor": "∨",
        "\\alpha": "α",
        "\\beta": "β",
        "\\gamma": "γ",
        "\\delta": "δ",
        "\\epsilon": "ε",
        "\\zeta": "ζ",
        "\\eta": "η",
        "\\theta": "θ",
        "\\lambda": "λ",
        "\\mu": "μ",
        "\\nu": "ν",
        "\\xi": "ξ",
        "\\rho": "ρ",
        "\\sigma": "σ",
        "\\tau": "τ",
        "\\phi": "φ",
        "\\chi": "χ",
        "\\psi": "ψ",
        "\\omega": "ω",
        "\\Delta": "Δ",
        "\\Gamma": "Γ",
        "\\Theta": "Θ",
        "\\Lambda": "Λ",
        "\\Sigma": "Σ",
        "\\Phi": "Φ",
        "\\Psi": "Ψ",
        "\\Omega": "Ω",
        "\\quad": " ",
        "\\qquad": "  ",
        "\\,": " ",
        "\\;": " ",
        "\\!": "",
    }

    def __init__(self) -> None:
        self.buffer: str = ""

    def _apply_replacements(self, text: str) -> str:
        """Apply simple command replacements."""
        for latex, unicode_char in self.REPLACEMENTS.items():
            text = text.replace(latex, unicode_char)
        return text

    def _process_frac(self, text: str) -> str:
        """Convert \\frac{a}{b} to a/b."""
        pattern = r"\\frac\{([^{}]*)\}\{([^{}]*)\}"
        return re.sub(pattern, r"\1/\2", text)

    def _process_sqrt(self, text: str) -> str:
        """Convert \\sqrt{x} to √x."""
        pattern = r"\\sqrt\{([^{}]*)\}"
        return re.sub(pattern, r"√\1", text)

    def _process_boxed(self, text: str) -> str:
        """Convert \\boxed{x} to [x]."""
        pattern = r"\\boxed\{([^{}]*)\}"
        return re.sub(pattern, r"[\1]", text)

    def _process_text(self, text: str) -> str:
        """Convert \\text{...} to just the text."""
        pattern = r"\\text\{([^{}]*)\}"
        return re.sub(pattern, r"\1", text)

    def _strip_delimiters(self, text: str) -> str:
        """Strip $$ and $ math delimiters."""
        # Replace $$ first (display math)
        text = text.replace("$$", "")
        # Strip standalone $ (math delimiters) - simple approach for streaming
        # This works because $ rarely appears alone in normal text
        text = text.replace(" $ ", " ")
        text = text.replace("$ ", " ")
        text = text.replace(" $", " ")
        # Handle $ at start/end of chunk
        if text.startswith("$"):
            text = text[1:]
        if text.endswith("$"):
            text = text[:-1]
        return text

    def _check_known_command(
        self, suffix: str, last_bs: int, text_len: int
    ) -> int | None:
        """Check if suffix matches a known command. Returns split position or None."""
        for cmd in sorted(self.REPLACEMENTS.keys(), key=len, reverse=True):
            if suffix.startswith(cmd):
                after_cmd = suffix[len(cmd) :]
                return (
                    text_len if not after_cmd or not after_cmd[0].isalpha() else last_bs
                )
        return None

    def _check_brace_pattern(
        self, suffix: str, last_bs: int, text_len: int
    ) -> int | None:
        """Check for patterns like \\frac{. Returns split position or None."""
        for pattern in ["\\frac{", "\\sqrt{", "\\boxed{", "\\text{"]:
            if pattern.startswith(suffix) and len(suffix) < len(pattern):
                return last_bs
            if suffix.startswith(pattern):
                return last_bs if suffix.count("{") > suffix.count("}") else text_len
        return None

    def _find_safe_split(self, text: str) -> int:
        """Find position where we can safely split for processing."""
        last_bs = text.rfind("\\")
        if last_bs == -1:
            return len(text)

        suffix = text[last_bs:]
        if (pos := self._check_known_command(suffix, last_bs, len(text))) is not None:
            return pos
        if (pos := self._check_brace_pattern(suffix, last_bs, len(text))) is not None:
            return pos

        # Backslash + letters only = potentially incomplete command
        if len(suffix) <= 15 and suffix[1:].replace(" ", "").isalpha():
            return last_bs
        return len(text)

    def process(self, text: str) -> str:
        """Process text chunk, converting LaTeX to Unicode.

        Buffers text to handle commands split across streaming chunks.
        """
        self.buffer += text

        split_pos = self._find_safe_split(self.buffer)

        if split_pos == 0:
            return ""  # Nothing safe to process yet

        to_process = self.buffer[:split_pos]
        self.buffer = self.buffer[split_pos:]

        return self._convert(to_process)

    def _convert(self, text: str) -> str:
        """Apply all LaTeX conversions."""
        text = self._strip_delimiters(text)
        text = self._process_frac(text)
        text = self._process_sqrt(text)
        text = self._process_boxed(text)
        text = self._process_text(text)
        text = self._apply_replacements(text)
        return text

    def flush(self) -> str:
        """Flush remaining buffer at end of stream."""
        if self.buffer:
            result = self._convert(self.buffer)
            self.buffer = ""
            return result
        return ""


class Utf8StreamBuffer:
    """Buffer for incomplete UTF-8 sequences in streaming output.

    Handles byte-level tokenizers (like Qwen) that may split multi-byte
    UTF-8 characters (e.g., emojis) across multiple tokens. Each partial
    byte sequence is buffered until complete characters can be output.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def process(self, text: str) -> str:
        """Process text, buffering incomplete UTF-8 sequences.

        Re-encodes text to bytes using surrogateescape to preserve any
        partial UTF-8 bytes, then decodes incrementally to output only
        complete characters.
        """
        # surrogateescape preserves invalid bytes as surrogates
        data = text.encode("utf-8", errors="surrogateescape")
        return self._decoder.decode(data, final=False)

    def flush(self) -> str:
        """Flush remaining buffered bytes at end of stream."""
        return self._decoder.decode(b"", final=True)


class QueryTool(Tool):
    """Send a query to the running inference server."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="query", aliases=["q"], help_text="Query the running inference server"
        )
        super().__init__(parent, config)
        self._models_config = load_models_config(_DEFAULT_CONFIG_PATH)
        self._model_config: ModelConfig | None = None

    def _get_model_config(self) -> ModelConfig:
        """Get config for the current model (lazy loaded)."""
        if self._model_config is None:
            model_name = get_selected_model_name()
            if model_name:
                self._model_config = self._models_config.get(model_name)
            else:
                self._model_config = self._models_config.defaults
        return self._model_config

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
            help="Enable thinking mode (model-specific suffix)",
        )
        think_group.add_argument(
            "--no-think",
            action="store_true",
            help="Disable thinking mode (model-specific suffix)",
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
        """Get system prompt from args or model config. Returns None if not set."""
        if self.args.system:
            return str(self.args.system)

        model_cfg = self._get_model_config()

        # Use think-specific system prompt when --think is active
        if self.args.think and model_cfg.think.system_prompt:
            return model_cfg.think.system_prompt

        # Fall back to model's default system prompt
        return model_cfg.system_prompt

    def _apply_think_suffix(self, prompt: str) -> str:
        """Apply think mode suffix to prompt based on flags and model config.

        Returns prompt with appropriate suffix appended, or unchanged if no action.
        """
        model_cfg = self._get_model_config()
        think_cfg = model_cfg.think

        if self.args.think and think_cfg.enable_suffix:
            return prompt + think_cfg.enable_suffix
        elif getattr(self.args, "no_think", False) and think_cfg.disable_suffix:
            return prompt + think_cfg.disable_suffix

        return prompt

    def _build_streaming_payload(self, prompt: str) -> dict:
        """Build streaming chat completions payload."""
        # Apply think mode suffix to prompt
        prompt = self._apply_think_suffix(prompt)

        messages = []
        system_prompt = self._get_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return {
            "model": "default",
            "messages": messages,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "frequency_penalty": (self.args.repetition_penalty - 1.0) * 2,
            "stream": True,
        }

    def _setup_stream_formatters(
        self,
    ) -> tuple[ThinkFormatter | None, LatexFormatter | None]:
        """Set up formatters for stream processing."""
        if self.args.raw:
            return None, None
        model_cfg = self._get_model_config()
        think = ThinkFormatter(
            open_tags=model_cfg.think.tags_open,
            close_tags=model_cfg.think.tags_close,
        )
        return think, LatexFormatter()

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
