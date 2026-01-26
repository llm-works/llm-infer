"""Ollama-backed inference engine.

This module provides an inference engine that connects to an Ollama server,
enabling use of Ollama-managed models through llm-infer's unified interface.

The engine can automatically start/stop the Ollama server process with the
configured models_path, or connect to an already-running server.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import httpx
from appinfra.log import Logger

if TYPE_CHECKING:
    from ...context import RequestContext
    from ...serving.dispatch.config import OllamaConfig


class OllamaStreamingIterator:
    """True streaming iterator using Ollama's streaming API.

    Yields text chunks as they arrive from Ollama's newline-delimited
    JSON stream.
    """

    def __init__(
        self,
        lg: Logger,
        client: httpx.Client,
        url: str,
        payload: dict[str, Any],
    ) -> None:
        self._lg = lg
        self._client = client
        self._url = url
        self._payload = payload

        # State tracking
        self._stream_context: Any = None  # Must hold reference to prevent GC
        self._response: httpx.Response | None = None
        self._line_iter: Iterator[str] | None = None
        self._finished = False

        # Final stats (populated when generation completes)
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.finish_reason: str | None = None

    def _start_stream(self) -> None:
        """Start the streaming request."""
        # Must keep reference to context manager to prevent stream from closing
        self._stream_context = self._client.stream(
            "POST", self._url, json=self._payload
        )
        self._response = self._stream_context.__enter__()
        self._line_iter = self._response.iter_lines()

    def __iter__(self) -> Iterator[str]:
        return self

    def __enter__(self) -> OllamaStreamingIterator:
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

    def _handle_completion(self, data: dict[str, Any]) -> str | None:
        """Handle stream completion, returning final chunk if present."""
        self._finished = True
        self.prompt_tokens = data.get("prompt_eval_count", 0)
        self.completion_tokens = data.get("eval_count", 0)
        self.finish_reason = "length" if data.get("done_reason") == "length" else "stop"
        # Handle both generate and chat response formats
        response: str = data.get("response", "")
        if not response:
            message = data.get("message", {})
            response = message.get("content", "") if isinstance(message, dict) else ""
        return response if response else None

    def _process_stream_line(self, line: str) -> str | None:
        """Process a single stream line, returning text chunk if present.

        Handles both /api/generate and /api/chat response formats:
        - generate: {"response": "text", "done": false}
        - chat: {"message": {"content": "text"}, "done": false}

        Raises:
            json.JSONDecodeError: If line contains malformed JSON.
        """
        if not line.strip():
            return None
        data = json.loads(line)
        if data.get("done", False):
            result = self._handle_completion(data)
            if result:
                return result
            raise StopIteration
        # Handle both response formats
        response_text: str = data.get("response", "")
        if not response_text:
            message = data.get("message", {})
            response_text = (
                message.get("content", "") if isinstance(message, dict) else ""
            )
        return response_text if response_text else None

    def __next__(self) -> str:
        if self._finished:
            raise StopIteration
        if self._line_iter is None:
            self._start_stream()
            assert self._line_iter is not None
        try:
            while True:
                line = next(self._line_iter)
                if result := self._process_stream_line(line):
                    return result
        except StopIteration:
            self._finished = True
            self._cleanup()
            raise
        except json.JSONDecodeError as e:
            self._lg.warning(
                "malformed JSON in stream response",
                extra={"line": line[:100], "exception": e},
            )
            self._finished = True
            self._cleanup()
            raise RuntimeError(f"Ollama returned malformed JSON: {e}") from e


class OllamaEngine:
    """Ollama-backed inference engine implementing InferenceEngineProtocol.

    This engine connects to an Ollama server and proxies requests through
    its HTTP API. Ollama handles model loading, GPU memory management,
    and inference internally.

    If auto_start is enabled (default), the engine will automatically start
    the Ollama server process with the configured models_path and stop it
    on shutdown.
    """

    def __init__(self, lg: Logger, config: OllamaConfig):
        """Initialize Ollama engine.

        Args:
            lg: Logger instance.
            config: Ollama configuration.
        """
        self._lg = lg
        self._config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._owns_process = False  # True if we started the server
        self._tokenize_available: bool | None = (
            None  # None=unknown, False=not supported
        )

        # HTTP client with configured timeout
        self._client = httpx.Client(
            base_url=config.host,
            timeout=httpx.Timeout(config.timeout, connect=10.0),
        )

        # Initialize server and verify connection, cleaning up on failure
        try:
            if config.auto_start and not self._is_server_running():
                self._start_server()
                self._owns_process = True

            self._verify_connection()
            self._model_info = self._fetch_model_info()
            self._eos_token_id = self._extract_eos_token_id()
        except Exception:
            self._client.close()
            if self._owns_process:
                self._stop_server()
            raise

    def _is_server_running(self) -> bool:
        """Check if Ollama server is already running."""
        try:
            response = self._client.get("/api/tags")
            # Use raise_for_status() instead of checking status_code or is_success
            # to avoid mypy [no-any-return] when httpx types aren't available
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def _start_server(self) -> None:
        """Start the Ollama server process."""
        env = os.environ.copy()

        # Set models path if configured
        if self._config.models_path:
            env["OLLAMA_MODELS"] = os.path.expanduser(self._config.models_path)
            self._lg.info(
                "starting ollama server",
                extra={"models_path": self._config.models_path},
            )
        else:
            self._lg.info("starting ollama server")

        # Start the server process
        self._process = subprocess.Popen(
            [self._config.binary_path, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Detach from parent process group
        )

        # Wait for server to be ready
        self._wait_for_server()

    def _wait_for_server(self, timeout: float = 30.0) -> None:
        """Wait for the Ollama server to be ready."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self._is_server_running():
                self._lg.info("ollama server is ready")
                return
            time.sleep(0.5)

        raise RuntimeError(
            f"Ollama server failed to start within {timeout}s. "
            f"Check that '{self._config.binary_path}' is installed and working."
        )

    def _stop_server(self) -> None:
        """Stop the Ollama server process if we started it."""
        if self._process is None:
            return

        self._lg.info("stopping ollama server")

        # Send SIGTERM to the process group
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        # Wait for graceful shutdown
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop
            self._lg.warning("ollama server did not stop gracefully, force killing")
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        self._process = None

    def _verify_connection(self) -> None:
        """Verify Ollama server is reachable."""
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            self._lg.info(
                "connected to ollama",
                extra={"host": self._config.host},
            )
        except httpx.HTTPError as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self._config.host}. Error: {e}"
            ) from e

    def _fetch_model_info(self) -> dict[str, Any]:
        """Fetch model information from Ollama."""
        try:
            response = self._client.post(
                "/api/show",
                json={"name": self._config.model},
            )
            response.raise_for_status()
            info: dict[str, Any] = response.json()
            self._lg.info(
                "loaded model info",
                extra={
                    "model": self._config.model,
                    "family": info.get("details", {}).get("family", "unknown"),
                    "parameter_size": info.get("details", {}).get(
                        "parameter_size", "unknown"
                    ),
                },
            )
            return info
        except httpx.HTTPError as e:
            self._lg.warning(
                "failed to fetch model info",
                extra={"model": self._config.model, "exception": e},
            )
            return {}

    def _extract_eos_token_id(self) -> int | None:
        """Extract EOS token ID from model info if available."""
        # Ollama doesn't directly expose this, but we can try to get it
        # from the model parameters or use a common default
        model_info = self._model_info.get("model_info", {})
        # Check for common parameter names
        for key in ["eos_token_id", "eos_id"]:
            if key in model_info:
                return int(model_info[key])
        return None

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._config.model

    @property
    def eos_token_id(self) -> int | None:
        """End-of-sequence token ID."""
        return self._eos_token_id

    # -------------------------------------------------------------------------
    # InferenceEngineProtocol: Tokenization methods
    # -------------------------------------------------------------------------

    def count_tokens(self, text: str, use_chat_template: bool | None = None) -> int:
        """Count tokens in text using Ollama's tokenize API."""
        tokens = self.tokenize(text, use_chat_template)
        return len(tokens)

    def tokenize(self, text: str, use_chat_template: bool | None = None) -> list[int]:
        """Tokenize text using Ollama's tokenize API.

        Note: When Ollama's tokenize API is unavailable (common), returns an
        estimated token count as a list of sequential integers. These are NOT
        real token IDs - only use len() of the result for counting purposes.
        """
        # Skip if we already know tokenize is not supported (expected for Ollama)
        if self._tokenize_available is False:
            # Return sequential ints for length estimation only - not real token IDs
            return list(range(len(text) // 4))

        # Note: use_chat_template is handled by Ollama internally based on model
        try:
            response = self._client.post(
                "/api/tokenize",
                json={"model": self._config.model, "text": text},
            )
            response.raise_for_status()
            self._tokenize_available = True
            data = response.json()
            tokens_result: list[int] = data.get("tokens", [])
            return tokens_result
        except httpx.HTTPError:
            # Ollama doesn't support tokenize API - this is expected, use estimates
            if self._tokenize_available is None:
                self._lg.debug("tokenize not available, using estimates")
                self._tokenize_available = False
            return list(range(len(text) // 4))

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode token IDs to text using Ollama's detokenize API."""
        try:
            response = self._client.post(
                "/api/detokenize",
                json={"model": self._config.model, "tokens": tokens},
            )
            response.raise_for_status()
            data = response.json()
            text_result: str = data.get("text", "")
            return text_result
        except httpx.HTTPError as e:
            self._lg.warning(
                "detokenize failed",
                extra={"exception": e},
            )
            return ""

    def build_stop_token_ids(self, stop_sequences: list[str] | None) -> set[int]:
        """Build set of stop token IDs from EOS and stop sequences."""
        stop_ids: set[int] = set()

        if self._eos_token_id is not None:
            stop_ids.add(self._eos_token_id)

        if stop_sequences:
            for seq in stop_sequences:
                tokens = self.tokenize(seq, use_chat_template=False)
                if tokens:
                    stop_ids.add(tokens[0])

        return stop_ids

    def should_use_chat_template(self) -> bool:
        """Check if chat template should be used.

        Ollama handles chat templates internally, so we return True
        for chat/instruct models.
        """
        model_lower = self._config.model.lower()
        return "instruct" in model_lower or "chat" in model_lower

    def _post_json(self, endpoint: str, payload: dict[str, Any], operation: str) -> Any:
        """POST JSON to Ollama API with error handling.

        Args:
            endpoint: API endpoint (e.g., "/api/generate").
            payload: JSON payload to send.
            operation: Operation name for error messages (e.g., "generate").

        Returns:
            Parsed JSON response.

        Raises:
            RuntimeError: If the request fails.
        """
        try:
            response = self._client.post(endpoint, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as e:
            self._lg.error(f"{operation} request failed", extra={"exception": e})
            raise RuntimeError(f"Ollama {operation} failed: {e}") from e
        return response.json()

    # -------------------------------------------------------------------------
    # InferenceEngineProtocol: Generation methods
    # -------------------------------------------------------------------------

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
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate text completion (blocking).

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            repetition_penalty: Repetition penalty.
            use_chat_template: Whether to apply chat template (Ollama handles this).
            stop_sequences: Sequences that stop generation.
            context: Request context for logging.
            messages: Chat messages (alternative to prompt).

        Returns:
            Generated text.
        """
        if messages:
            return self._generate_chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop_sequences=stop_sequences,
            )

        payload = self._build_generate_payload(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
            stream=False,
        )
        data = self._post_json("/api/generate", payload, "generate")
        result: str = data.get("response", "")
        return result

    def _generate_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop_sequences: list[str] | None,
    ) -> str:
        """Generate using chat API."""
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "stream": False,
            "options": self._build_options(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            ),
        }

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        if self._config.keep_alive:
            payload["keep_alive"] = self._config.keep_alive

        data = self._post_json("/api/chat", payload, "chat")
        content: str = data.get("message", {}).get("content", "")
        return content

    def _build_generate_payload(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop_sequences: list[str] | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Build payload for /api/generate endpoint."""
        payload: dict[str, Any] = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": stream,
            "options": self._build_options(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            ),
        }

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        if self._config.keep_alive:
            payload["keep_alive"] = self._config.keep_alive

        return payload

    def _build_options(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> dict[str, Any]:
        """Build Ollama options dict."""
        options: dict[str, Any] = {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repetition_penalty,
        }

        # Ollama uses 0 to disable top_k, which matches our convention
        if top_k > 0:
            options["top_k"] = top_k

        # Apply config-level options
        if self._config.num_ctx:
            options["num_ctx"] = self._config.num_ctx
        if self._config.num_gpu is not None:
            options["num_gpu"] = self._config.num_gpu

        return options

    def _build_chat_stream_payload(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop_sequences: list[str] | None,
    ) -> dict[str, Any]:
        """Build payload for streaming chat endpoint."""
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "stream": True,
            "options": self._build_options(
                max_tokens, temperature, top_p, top_k, repetition_penalty
            ),
        }
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences
        if self._config.keep_alive:
            payload["keep_alive"] = self._config.keep_alive
        return payload

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
        messages: list[dict[str, str]] | None = None,
    ) -> OllamaStreamingIterator:
        """Generate text with streaming."""
        if messages:
            payload = self._build_chat_stream_payload(
                messages,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                stop_sequences,
            )
            return OllamaStreamingIterator(self._lg, self._client, "/api/chat", payload)

        payload = self._build_generate_payload(
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            stop_sequences,
            stream=True,
        )
        return OllamaStreamingIterator(self._lg, self._client, "/api/generate", payload)

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
        """Generate embeddings for input texts.

        Args:
            inputs: List of texts to embed.
            dimensions: Optional output dimensions (truncation for Matryoshka).

        Returns:
            Tuple of (embeddings list, total tokens).
        """
        embeddings: list[list[float]] = []
        total_tokens = 0

        for text in inputs:
            payload = {"model": self._config.model, "prompt": text}
            data = self._post_json("/api/embeddings", payload, "embedding")
            embedding = data.get("embedding", [])

            if dimensions is not None and len(embedding) > dimensions:
                # Truncate and renormalize for Matryoshka
                embedding = embedding[:dimensions]
                norm = math.sqrt(sum(x * x for x in embedding))
                if norm > 0:
                    embedding = [x / norm for x in embedding]

            embeddings.append(embedding)
            # Estimate tokens (Ollama doesn't return token count for embeddings)
            total_tokens += len(text) // 4

        return embeddings, total_tokens

    # -------------------------------------------------------------------------
    # InferenceEngineProtocol: Batched processing methods (no-ops for Ollama)
    # -------------------------------------------------------------------------

    def prefill_request(self, request: Any) -> None:
        """No-op - Ollama handles prefill internally."""
        pass

    def step_decode(self, requests: list[Any]) -> list[int | None]:
        """No-op - Ollama handles decoding internally."""
        return [None] * len(requests)

    def free_request(self, request: Any) -> None:
        """No-op - Ollama manages resources internally."""
        pass

    # -------------------------------------------------------------------------
    # InferenceEngineProtocol: Utility methods
    # -------------------------------------------------------------------------

    def _default_memory_stats(self) -> dict[str, int | float]:
        """Return default (zero) memory statistics."""
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

    def memory_stats(self) -> dict[str, int | float]:
        """Return memory statistics from Ollama."""
        stats = self._default_memory_stats()
        try:
            response = self._client.get("/api/ps")
            response.raise_for_status()
            for model in response.json().get("models", []):
                if model.get("name") == self._config.model:
                    stats["model_memory"] = model.get("size", 0)
                    stats["allocated"] = model.get("size", 0)
                    if vram := model.get("size_vram", 0):
                        stats["device_used"] = vram
        except httpx.HTTPError as e:
            self._lg.warning("failed to fetch memory stats", extra={"exception": e})
        return stats

    def reset_peak_memory(self) -> None:
        """No-op - Ollama doesn't expose peak memory tracking."""
        pass

    def shutdown(self) -> None:
        """Shutdown the engine and stop the Ollama server if we started it."""
        self._client.close()

        if self._owns_process:
            self._stop_server()

        self._lg.info("ollama engine shutdown complete")
