"""vLLM server-backed inference engine.

This module provides an inference engine that connects to a `vllm serve` process
via its OpenAI-compatible HTTP API. Like the Ollama engine, it can auto-start the
server as a subprocess and stop it on shutdown.

The vllm-server engine delegates chat templating, tokenization, tool call parsing,
and structured output to the vLLM server, avoiding the need to reimplement these
in Python.

LoRA Limitation:
    Unlike the native vllm engine (Python API) which loads LoRA adapters dynamically
    at inference time, vllm-server requires adapters to be pre-registered at server
    startup via --lora-modules. Adapters created after server startup will fail with
    404 errors until the server is restarted. Use the native vllm engine for dynamic
    adapter use cases (e.g., e2e tests creating temporary adapters).
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from appinfra.log import Logger

from ..adapter_meta import compute_adapter_metadata

if TYPE_CHECKING:
    from ..context import RequestContext
    from ..serving.dispatch.config import VLLMServerConfig


class VLLMServerStreamingIterator:
    """Streaming iterator for vLLM server's OpenAI-compatible SSE stream.

    Parses Server-Sent Events (SSE) format from the /v1/chat/completions
    streaming endpoint. Yields text deltas and collects tool calls.
    """

    def __init__(
        self,
        lg: Logger,
        client: httpx.Client,
        url: str,
        payload: dict[str, Any],
        lora_request: Any | None = None,
        adapter_metadata: dict[str, str] | None = None,
    ) -> None:
        self._lg = lg
        self._client = client
        self._url = url
        self._payload = payload
        self._lora_request = lora_request
        self._adapter_metadata = adapter_metadata or {}
        self._init_state()

    def _init_state(self) -> None:
        """Initialize streaming state and tracking fields."""
        # Stream state
        self._stream_context: Any = None
        self._response: httpx.Response | None = None
        self._line_iter: Iterator[str] | None = None
        self._finished = False

        # Final stats (populated when generation completes)
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.finish_reason: str | None = None
        self._tool_call_chunks: dict[int, dict[str, Any]] = {}
        self.tool_calls: list[dict[str, Any]] | None = None

        # Adapter verification (populated from first chunk with model field)
        self._response_model: str | None = None
        self.adapter_mismatch: bool = False
        self.adapter_requested: str | None = None

    def _start_stream(self) -> None:
        """Start the streaming request."""
        self._stream_context = self._client.stream(
            "POST", self._url, json=self._payload
        )
        self._response = self._stream_context.__enter__()
        try:
            self._response.raise_for_status()
        except Exception:
            self._cleanup()
            raise
        self._line_iter = self._response.iter_lines()

    def __iter__(self) -> Iterator[str]:
        return self

    def __enter__(self) -> VLLMServerStreamingIterator:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up the streaming response."""
        if self._stream_context is not None:
            try:
                self._stream_context.__exit__(None, None, None)
            except Exception as e:
                self._lg.debug("cleanup exception suppressed", extra={"exception": e})
            self._stream_context = None
            self._response = None

    def __del__(self) -> None:
        self._cleanup()

    def _accumulate_tool_call_delta(self, delta_tc: dict[str, Any]) -> None:
        """Accumulate a tool call delta chunk.

        OpenAI streaming sends tool calls incrementally:
        - First chunk: {index, id, type, function: {name, arguments: ""}}
        - Subsequent chunks: {index, function: {arguments: "<partial>"}}
        """
        idx = delta_tc.get("index", 0)
        if idx not in self._tool_call_chunks:
            self._tool_call_chunks[idx] = {
                "id": delta_tc.get("id", ""),
                "type": delta_tc.get("type", "function"),
                "function": {
                    "name": delta_tc.get("function", {}).get("name", ""),
                    "arguments": "",
                },
            }
        # Append argument fragments
        func_delta = delta_tc.get("function", {})
        if args_fragment := func_delta.get("arguments", ""):
            self._tool_call_chunks[idx]["function"]["arguments"] += args_fragment

    def _finalize_tool_calls(self) -> None:
        """Finalize accumulated tool calls into the result."""
        if self._tool_call_chunks:
            self.tool_calls = [
                self._tool_call_chunks[idx]
                for idx in sorted(self._tool_call_chunks.keys())
            ]
            self.finish_reason = "tool_calls"

    def _handle_completion(self, data: dict[str, Any]) -> str | None:
        """Handle stream completion from the final SSE chunk."""
        self._finished = True

        # Extract usage stats if present
        usage = data.get("usage") or {}
        self.prompt_tokens = usage.get("prompt_tokens", self.prompt_tokens)
        self.completion_tokens = usage.get("completion_tokens", self.completion_tokens)

        # Finalize tool calls
        self._finalize_tool_calls()

        # Set finish reason from choices if not already set by tool calls
        if not self.finish_reason:
            choices = data.get("choices", [])
            if choices:
                self.finish_reason = choices[0].get("finish_reason", "stop")
            else:
                self.finish_reason = "stop"

        return None

    def _verify_adapter(self) -> None:
        """Verify vLLM used the requested adapter."""
        if self._lora_request is None or self._response_model is None:
            return

        requested = (
            self._lora_request.lora_name
            if hasattr(self._lora_request, "lora_name")
            else str(self._lora_request)
        )

        if self._response_model != requested:
            self._lg.warning(
                "adapter mismatch: vLLM used different model than requested",
                extra={
                    "requested": requested,
                    "actual": self._response_model,
                },
            )
            self.adapter_mismatch = True
            self.adapter_requested = requested
        else:
            self._lg.debug(
                "verified LoRA adapter",
                extra={
                    "adapter": requested,
                    "mtime": self._adapter_metadata.get("mtime", ""),
                    "md5": self._adapter_metadata.get("md5", ""),
                },
            )

    def _handle_done_sentinel(self) -> None:
        """Handle the [DONE] SSE sentinel, finalizing the stream."""
        if not self._finished:
            self._finished = True
            self._finalize_tool_calls()
            if not self.finish_reason:
                self.finish_reason = "stop"

    def _process_choice_delta(self, data: dict[str, Any]) -> str | None:
        """Process choice delta from SSE chunk, returning content if present."""
        choices = data.get("choices", [])
        if not choices:
            if data.get("usage"):
                self._handle_completion(data)
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        if tc_deltas := delta.get("tool_calls"):
            for tc_delta in tc_deltas:
                self._accumulate_tool_call_delta(tc_delta)
        if choice.get("finish_reason"):
            self._handle_completion(data)
        content: str = delta.get("content") or ""
        return content if content else None

    def _process_sse_line(self, line: str) -> str | None:
        """Process a single SSE line, returning text chunk if present."""
        if not line.startswith("data: "):
            return None

        data_str = line[6:]  # Strip "data: " prefix
        if data_str.strip() == "[DONE]":
            self._handle_done_sentinel()
            raise StopIteration

        data = json.loads(data_str)

        # Capture and verify model from first chunk
        if self._response_model is None and "model" in data:
            self._response_model = data["model"]
            self._verify_adapter()

        return self._process_choice_delta(data)

    def __next__(self) -> str:
        if self._finished:
            raise StopIteration
        if self._line_iter is None:
            self._start_stream()
            assert self._line_iter is not None
        line = ""  # Initialize before try block for exception handler
        try:
            while True:
                line = next(self._line_iter)
                if result := self._process_sse_line(line):
                    if self._finished:
                        self._cleanup()
                    return result
        except StopIteration:
            self._finished = True
            self._cleanup()
            raise
        except json.JSONDecodeError as e:
            self._lg.warning(
                "malformed JSON in SSE stream",
                extra={"line": line[:100], "exception": e},
            )
            self._finished = True
            self._cleanup()
            raise RuntimeError(f"vLLM server returned malformed JSON: {e}") from e


class VLLMServerEngine:
    """vLLM server-backed inference engine implementing InferenceEngineProtocol.

    Connects to a `vllm serve` process via its OpenAI-compatible HTTP API.
    If auto_start is enabled, the engine starts `vllm serve` as a subprocess
    with the configured model and settings, and stops it on shutdown.
    """

    def __init__(self, lg: Logger, config: VLLMServerConfig):
        self._lg = lg
        self._config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._owns_process = False
        self._base_url = self._parse_base_url(config)
        self._model_name = config.served_model_name or Path(config.model_path).name
        self._adapter_paths: dict[str, str] = {}
        self._adapter_metadata: dict[str, dict[str, str]] = {}

        if config.lora.enabled and config.lora.base_path:
            self._scan_adapters(Path(config.lora.base_path))

        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(config.timeout, connect=10.0),
        )
        self._initialize_connection()

    def _parse_base_url(self, config: VLLMServerConfig) -> str:
        """Parse and normalize host URL to avoid double-port issues."""
        from urllib.parse import urlparse

        parsed = urlparse(
            config.host if "://" in config.host else f"http://{config.host}"
        )
        scheme = parsed.scheme or "http"
        hostname = parsed.hostname or parsed.netloc.split(":")[0]
        return f"{scheme}://{hostname}:{config.port}"

    def _initialize_connection(self) -> None:
        """Initialize server connection, starting server if needed."""
        try:
            if self._config.auto_start and not self._is_server_running():
                self._start_server()

            self._verify_connection()
            self._verify_adapters()
        except Exception:
            self._client.close()
            if self._owns_process:
                try:
                    self._stop_server()
                except Exception as cleanup_err:
                    self._lg.warning(
                        "cleanup failed during init", extra={"exception": cleanup_err}
                    )
            raise

    # -------------------------------------------------------------------------
    # LoRA adapter scanning
    # -------------------------------------------------------------------------

    def _load_adapter_config(self, config_path: Path) -> dict[str, Any] | None:
        """Load and parse adapter config.yaml file.

        Returns config dict if successful, None if failed.
        """
        try:
            import yaml

            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self._lg.warning(
                "failed to load adapter config",
                extra={"path": str(config_path), "exception": e},
            )
            return None

    def _get_adapter_metadata(self, lora_request: Any | None) -> dict[str, str] | None:
        """Get cached adapter metadata for a LoRA request."""
        if lora_request and hasattr(lora_request, "lora_name"):
            return self._adapter_metadata.get(lora_request.lora_name)
        return None

    def _validate_adapter_base_path(self, base_path: Path) -> bool:
        """Validate adapter base path exists and is a directory.

        Returns True if valid, False otherwise (logs warnings).
        """
        if not base_path.exists():
            self._lg.warning(
                "LoRA adapter base_path does not exist",
                extra={"base_path": str(base_path)},
            )
            return False
        if not base_path.is_dir():
            self._lg.warning(
                "LoRA adapter base_path is not a directory",
                extra={"base_path": str(base_path)},
            )
            return False
        return True

    def _process_adapter_dir(self, entry: Path) -> None:
        """Process a single adapter directory and register if enabled."""
        config_path = entry / "config.yaml"
        if not config_path.exists():
            return

        config = self._load_adapter_config(config_path)
        if config is None:
            return

        enabled = config.get("enabled", False)
        self._lg.debug(
            "checked adapter config",
            extra={"adapter": entry.name, "enabled": enabled, "path": str(entry)},
        )
        if enabled:
            self._adapter_paths[entry.name] = str(entry)
            metadata = compute_adapter_metadata(entry).to_dict()
            self._adapter_metadata[entry.name] = metadata
            self._lg.info(
                "found LoRA adapter",
                extra={
                    "adapter": entry.name,
                    "mtime": metadata["mtime"],
                    "md5": metadata["md5"],
                },
            )

    def _scan_adapters(self, base_path: Path) -> None:
        """Scan for available LoRA adapters in base_path.

        Looks for subdirectories with config.yaml that have enabled: true.
        Stores adapter names and paths for use in --lora-modules flag.
        """
        self._lg.debug(
            "scanning for LoRA adapters", extra={"base_path": str(base_path)}
        )
        if not self._validate_adapter_base_path(base_path):
            return

        for entry in base_path.iterdir():
            if entry.is_dir():
                self._process_adapter_dir(entry)

        self._lg.info(
            "adapter scan complete (vllm-server engine)",
            extra={"count": len(self._adapter_paths)},
        )

    # -------------------------------------------------------------------------
    # Server lifecycle
    # -------------------------------------------------------------------------

    def _is_server_running(self) -> bool:
        """Check if vLLM server is already running."""
        try:
            response = self._client.get("/v1/models")
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def _add_engine_flags(self, cmd: list[str]) -> None:
        """Add vLLM engine configuration flags to command."""
        cfg = self._config
        cmd.extend(["--gpu-memory-utilization", str(cfg.gpu_memory_utilization)])
        cmd.extend(["--max-num-seqs", str(cfg.max_num_seqs)])
        cmd.extend(["--tensor-parallel-size", str(cfg.tensor_parallel_size)])
        cmd.extend(["--dtype", cfg.dtype])

        if cfg.max_model_len is not None:
            cmd.extend(["--max-model-len", str(cfg.max_model_len)])
        if cfg.quantization is not None:
            cmd.extend(["--quantization", cfg.quantization])
        if cfg.enforce_eager:
            cmd.append("--enforce-eager")
        if cfg.trust_remote_code:
            cmd.append("--trust-remote-code")
        if cfg.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

    def _add_lora_flags(self, cmd: list[str]) -> None:
        """Add LoRA configuration flags to command."""
        cfg = self._config
        if not (cfg.lora.enabled and cfg.task != "embed"):
            return

        cmd.append("--enable-lora")
        cmd.extend(["--max-loras", str(cfg.lora.max_loras)])
        cmd.extend(["--max-lora-rank", str(cfg.lora.max_lora_rank)])

        # Pre-register all scanned adapters
        if self._adapter_paths:
            for adapter_name, adapter_path in self._adapter_paths.items():
                cmd.extend(["--lora-modules", f"{adapter_name}={adapter_path}"])
            self._lg.info(
                "pre-registering LoRA adapters",
                extra={"count": len(self._adapter_paths)},
            )

    def _build_serve_command(self) -> list[str]:
        """Build the `vllm serve` command from config."""
        cfg = self._config
        cmd = [cfg.binary_path, "serve", cfg.model_path]

        cmd.extend(["--port", str(cfg.port)])
        cmd.extend(["--served-model-name", self._model_name])

        self._add_engine_flags(cmd)

        # Tool calling
        cmd.append("--enable-auto-tool-choice")
        cmd.extend(["--tool-call-parser", cfg.tool_call_parser])

        self._add_lora_flags(cmd)

        return cmd

    def _start_server(self) -> None:
        """Start the vLLM server process."""
        cmd = self._build_serve_command()
        self._lg.info(
            "starting vllm server",
            extra={"cmd": " ".join(cmd), "port": self._config.port},
        )

        self._process = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            stdout=sys.stderr,
            stderr=sys.stderr,
            start_new_session=True,
        )
        self._owns_process = True

        self._wait_for_server()

    def _wait_for_server(self) -> None:
        """Wait for the vLLM server to be ready."""
        timeout = self._config.startup_timeout
        start = time.monotonic()
        self._lg.info(
            "waiting for vllm server to start",
            extra={"timeout": timeout},
        )
        while time.monotonic() - start < timeout:
            # Check if process died
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"vllm serve process exited with code {self._process.returncode}. "
                    f"Check stderr output above for details."
                )
            if self._is_server_running():
                # Double-check process didn't die immediately after responding
                if self._process is not None and self._process.poll() is not None:
                    raise RuntimeError("vllm server died after startup")
                self._lg.info("vllm server is ready")
                return
            time.sleep(2.0)  # vllm startup is slow, no need to poll aggressively

        raise RuntimeError(
            f"vLLM server failed to start within {timeout}s. "
            f"Check that '{self._config.binary_path}' is installed and the model "
            f"at '{self._config.model_path}' is valid."
        )

    def _stop_server(self) -> None:
        """Stop the vLLM server process if we started it."""
        if self._process is None:
            return

        self._lg.info("stopping vllm server")

        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # Already dead, fine
        except PermissionError as e:
            self._lg.warning("failed to terminate vllm server", extra={"exception": e})

        try:
            self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._lg.warning("vllm server did not stop gracefully, force killing")
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                self._process.wait(timeout=5)  # Reap zombie process
            except ProcessLookupError:
                pass  # Already dead, fine
            except PermissionError as e:
                self._lg.error(
                    "failed to force kill vllm server", extra={"exception": e}
                )

        self._process = None

    def _verify_connection(self) -> None:
        """Verify vLLM server is reachable and serving our model."""
        try:
            response = self._client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            models = [m.get("id", "") for m in data.get("data", [])]
            self._lg.info(
                "connected to vllm server",
                extra={"url": self._base_url, "models": models},
            )
        except httpx.HTTPError as e:
            raise ConnectionError(
                f"Failed to connect to vLLM server at {self._base_url}. Error: {e}"
            ) from e

    def _verify_adapters(self) -> None:
        """Verify adapters were loaded by vLLM, remove any that failed.

        Queries vLLM's /v1/models endpoint and removes adapters from our
        available list that vLLM didn't actually load. This guarantees we
        only report adapters to customers that are confirmed working.
        """
        if not self._adapter_paths:
            return

        response = self._client.get("/v1/models")
        data = response.json()
        loaded_models = {m.get("id", "") for m in data.get("data", [])}

        # Find and remove adapters that vLLM didn't load
        failed: list[str] = []
        for adapter_id in list(self._adapter_paths.keys()):
            if adapter_id not in loaded_models:
                failed.append(adapter_id)
                self._lg.warning(
                    "adapter not loaded by vLLM, removing from available adapters",
                    extra={
                        "adapter_id": adapter_id,
                        "path": self._adapter_paths[adapter_id],
                    },
                )
                del self._adapter_paths[adapter_id]

        if failed:
            self._lg.warning(
                "adapter verification complete",
                extra={
                    "removed": failed,
                    "available": list(self._adapter_paths.keys()),
                },
            )

    # -------------------------------------------------------------------------
    # InferenceEngineProtocol: Properties
    # -------------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return model name used for API requests."""
        return self._model_name

    @property
    def eos_token_id(self) -> int | None:
        """EOS token ID (server handles stop conditions)."""
        return None

    def should_use_chat_template(self) -> bool:
        """Server handles chat templating."""
        return True

    # -------------------------------------------------------------------------
    # InferenceEngineProtocol: Tokenization (simplified, server handles it)
    # -------------------------------------------------------------------------

    def count_tokens(self, text: str, use_chat_template: bool | None = None) -> int:
        """Estimate token count (rough heuristic, ~4 chars per token)."""
        return len(text) // 4

    def tokenize(self, text: str, use_chat_template: bool | None = None) -> list[int]:
        """Return sequential ints for length estimation (not real token IDs)."""
        return list(range(len(text) // 4))

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode not implemented for server engine — return empty string."""
        return ""

    def build_stop_token_ids(self, stop_sequences: list[str] | None) -> set[int]:
        """Server handles stop sequences, return empty set."""
        return set()

    # -------------------------------------------------------------------------
    # InferenceEngineProtocol: Generation
    # -------------------------------------------------------------------------

    def _add_optional_params(
        self,
        payload: dict[str, Any],
        stop_sequences: list[str] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        response_format: dict[str, Any] | None,
        stream: bool,
    ) -> None:
        """Add optional parameters to payload."""
        if stop_sequences:
            payload["stop"] = stop_sequences
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        if stream:
            payload["stream_options"] = {"include_usage": True}

    def _build_payload(
        self,
        messages: list[dict[str, Any]] | None,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: list[str] | None,
        stream: bool,
        lora_request: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build OpenAI-compatible chat completions payload."""
        api_messages = (
            list(messages) if messages else [{"role": "user", "content": prompt}]
        )

        # For LoRA: use adapter name as model field (pre-registered at server startup)
        # NOTE: Unlike vllm (Python API) which loads adapters dynamically, vllm-server
        # requires adapters to be registered at startup via --lora-modules. Adapters
        # created after server startup will return 404 errors until server is restarted.
        model_name = (
            lora_request.lora_name
            if lora_request and hasattr(lora_request, "lora_name")
            else self._model_name
        )

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        self._add_optional_params(
            payload, stop_sequences, tools, tool_choice, response_format, stream
        )

        return payload

    def _extract_error_detail(self, response: httpx.Response) -> str:
        """Extract error message from vLLM error response body."""
        try:
            # vLLM returns {"object":"error","message":"...","type":"...","code":400}
            msg: str = response.json().get("message", response.text)
            return msg
        except Exception:
            return str(response.text)

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to /v1/chat/completions with error handling."""
        try:
            response = self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = self._extract_error_detail(e.response)
            self._lg.error("generate request failed", extra={"detail": detail})
            raise RuntimeError(f"vLLM server error: {detail}") from e
        except httpx.HTTPError as e:
            self._lg.error("generate request failed", extra={"exception": e})
            raise RuntimeError(f"vLLM server generate failed: {e}") from e
        result: dict[str, Any] = response.json()
        return result

    def _verify_adapter_response(
        self,
        response_model: str,
        lora_request: Any | None,
    ) -> tuple[bool, str | None]:
        """Verify vLLM used the requested adapter.

        Returns:
            Tuple of (mismatch, requested_adapter):
            - (False, None) if no adapter requested or adapter matches
            - (True, adapter_name) if adapter was requested but vLLM used different model
        """
        if lora_request is None:
            return False, None

        requested = (
            lora_request.lora_name
            if hasattr(lora_request, "lora_name")
            else str(lora_request)
        )

        if response_model != requested:
            self._lg.warning(
                "adapter mismatch: vLLM used different model than requested",
                extra={
                    "requested": requested,
                    "actual": response_model,
                },
            )
            return True, requested

        metadata = self._adapter_metadata.get(requested, {})
        self._lg.debug(
            "verified LoRA adapter",
            extra={
                "adapter": requested,
                "mtime": metadata.get("mtime", ""),
                "md5": metadata.get("md5", ""),
            },
        )
        return False, None

    def _parse_completion_response(
        self,
        data: dict[str, Any],
        lora_request: Any | None = None,
    ) -> str | dict[str, Any]:
        """Extract content, tool_calls, usage, and verify adapter from response."""
        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        message = choice.get("message", {})
        content: str = message.get("content") or ""
        tool_calls = message.get("tool_calls")
        usage = data.get("usage", {})

        # Verify adapter was actually used
        response_model = data.get("model", "")
        adapter_mismatch, adapter_requested = self._verify_adapter_response(
            response_model, lora_request
        )

        # Build response dict if we have extra info
        if tool_calls or usage or adapter_mismatch:
            result: dict[str, Any] = {"content": content}
            if tool_calls:
                result["tool_calls"] = tool_calls
            if usage:
                result["usage"] = usage
            if adapter_mismatch:
                result["adapter_mismatch"] = True
                result["adapter_requested"] = adapter_requested
            return result
        return content

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        use_chat_template: bool | None = None,
        stop_sequences: list[str] | None = None,
        context: RequestContext | None = None,
        messages: list[dict[str, Any]] | None = None,
        lora_request: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str | dict[str, Any]:
        """Generate text completion (blocking).

        Args:
            lora_request: LoRA adapter request (uses adapter name as model field).

        Returns:
            Generated text string, or dict with "content" and "tool_calls" when
            tools are provided and model returns tool calls.
        """
        payload = self._build_payload(
            messages=messages,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            stream=False,
            lora_request=lora_request,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )
        # Add optional params that vLLM supports via extra_body
        if repetition_penalty != 1.0:
            payload["repetition_penalty"] = repetition_penalty
        if top_k > 0:
            payload["top_k"] = top_k

        data = self._post_chat_completions(payload)
        return self._parse_completion_response(data, lora_request)

    def generate_stream_sync(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        use_chat_template: bool | None = None,
        stop_sequences: list[str] | None = None,
        context: RequestContext | None = None,
        messages: list[dict[str, Any]] | None = None,
        lora_request: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> VLLMServerStreamingIterator:
        """Generate text with streaming via SSE."""
        payload = self._build_payload(
            messages=messages,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            stream=True,
            lora_request=lora_request,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )
        if repetition_penalty != 1.0:
            payload["repetition_penalty"] = repetition_penalty
        if top_k > 0:
            payload["top_k"] = top_k

        return VLLMServerStreamingIterator(
            self._lg,
            self._client,
            "/v1/chat/completions",
            payload,
            lora_request,
            self._get_adapter_metadata(lora_request),
        )

    # -------------------------------------------------------------------------
    # Embedding methods
    # -------------------------------------------------------------------------

    def supports_embeddings(self) -> bool:
        """Check if engine supports embeddings."""
        return self._config.task == "embed"

    def embed(
        self,
        inputs: list[str],
        dimensions: int | None = None,
    ) -> tuple[list[list[float]], int]:
        """Generate embeddings via /v1/embeddings endpoint."""
        payload: dict[str, Any] = {
            "model": self._model_name,
            "input": inputs,
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions

        try:
            response = self._client.post("/v1/embeddings", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as e:
            self._lg.error("embedding request failed", extra={"exception": e})
            raise RuntimeError(f"vLLM server embedding failed: {e}") from e

        data = response.json()
        embeddings: list[list[float]] = []
        for item in data.get("data", []):
            embedding = item.get("embedding", [])
            if dimensions is not None and len(embedding) > dimensions:
                embedding = embedding[:dimensions]
                norm = math.sqrt(sum(x * x for x in embedding))
                if norm > 0:
                    embedding = [x / norm for x in embedding]
            embeddings.append(embedding)

        total_tokens = data.get("usage", {}).get("total_tokens", 0)
        return embeddings, total_tokens

    # -------------------------------------------------------------------------
    # InferenceEngineProtocol: Batched processing (no-ops, server handles it)
    # -------------------------------------------------------------------------

    def prefill_request(self, request: Any) -> None:
        """No-op — server handles prefill internally."""

    def step_decode(self, requests: list[Any]) -> list[int | None]:
        """No-op — server handles decoding internally."""
        return [None] * len(requests)

    def free_request(self, request: Any) -> None:
        """No-op — server manages resources internally."""

    # -------------------------------------------------------------------------
    # InferenceEngineProtocol: Utility methods
    # -------------------------------------------------------------------------

    def memory_stats(self) -> dict[str, int | float]:
        """Return empty memory statistics (server manages GPU memory)."""
        return {
            "allocated": 0,
            "reserved": 0,
            "peak": 0,
            "model_memory": 0,
            "kv_cache_bytes": 0,
            "kv_blocks_used": 0,
            "kv_blocks_total": 0,
            "kv_block_size": 0,
            "device_used": 0,
            "device_total": 0,
            "device_free": 0,
            "kv_cache_usage_perc": 0.0,
        }

    def reset_peak_memory(self) -> None:
        """No-op — server manages memory tracking."""

    def shutdown(self) -> None:
        """Shutdown the engine and stop the vLLM server if we started it."""
        self._client.close()

        if self._owns_process:
            self._stop_server()

        self._lg.info("vllm server engine shutdown complete")
