# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Process entry points for the serve integration tests.

Run as a script, never imported by pytest collection:

    _serve_procs.py vllm serve <model_path> --port N --served-model-name M ...
        Stand-in for `vllm serve`. Answers /v1/models. A chat completion whose
        prompt contains "quick" is answered at once; any other one is held open
        (a long generation) until the client closes the connection.
        Writes its pid to <model_path>/vllm.pid, touches
        <model_path>/request.received when a held request arrives and
        <model_path>/request.closed when its client closes it.

    _serve_procs.py serve <model_path> <api_port> <vllm_port> <vllm_binary> <options>
        Runs llm-infer serve with the vllm-server engine against the stand-in.
        <options> is JSON: auto_start (bool; false: connect to a running one),
        response_timeout (s), max_concurrent.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _arg_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def _completion(model_name: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


def run_fake_vllm(args: list[str]) -> None:
    model_dir = Path(args[1])
    port = int(_arg_after(args, "--port"))
    model_name = _arg_after(args, "--served-model-name")
    (model_dir / "vllm.pid").write_text(str(os.getpid()))
    handler = _handler_class(model_dir, model_name)
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


def _handler_class(model_dir: Path, model_name: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            models = {"data": [{"id": model_name, "max_model_len": 4096}]}
            self._send_json(json.dumps(models).encode())

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if b"quick" in body:
                self._send_json(_completion(model_name))
                return
            (model_dir / "request.received").touch()
            # Held until the client closes the connection (EOF), as vLLM
            # notices a disconnect and aborts the generation
            if self.connection.recv(1) == b"":
                (model_dir / "request.closed").touch()

        def log_message(self, *_: object) -> None:
            pass

    return Handler


def run_serve(
    model_dir: str, api_port: str, vllm_port: str, binary: str, options: str
) -> None:
    from appinfra.log import Logger

    from llm_infer.serving.dispatch import InferenceConfig, run_server

    opts = json.loads(options)
    config = InferenceConfig()
    config.backends.engine = "vllm-server"
    config.models.path = model_dir
    config.api.host = "127.0.0.1"
    config.api.port = int(api_port)
    config.api.response_timeout = opts.get("response_timeout", 300.0)
    server_cfg = config.engines.vllm_server
    server_cfg.host = "http://127.0.0.1"
    server_cfg.port = int(vllm_port)
    server_cfg.binary_path = binary
    server_cfg.auto_start = opts.get("auto_start", True)
    server_cfg.max_concurrent = opts.get("max_concurrent", 4)
    server_cfg.warmup = False
    server_cfg.startup_timeout = 30
    run_server(Logger("serve-test"), config)


if __name__ == "__main__":
    if sys.argv[1] == "vllm":
        run_fake_vllm(sys.argv[2:])
    elif sys.argv[1] == "serve":
        run_serve(*sys.argv[2:7])
    else:
        raise SystemExit(f"unknown role: {sys.argv[1]}")
