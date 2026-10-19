# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Process entry points for the serve shutdown integration test.

Run as a script, never imported by pytest collection:

    _shutdown_procs.py vllm serve <model_path> --port N --served-model-name M ...
        Stand-in for `vllm serve`: answers /v1/models, accepts chat completions
        but never responds (a long generation still running at stop time).
        Writes its pid to <model_path>/vllm.pid and touches
        <model_path>/request.received when a chat request arrives.

    _shutdown_procs.py serve <model_path> <api_port> <vllm_port> <vllm_binary> <0|1>
        Runs llm-infer serve with the vllm-server engine against the stand-in;
        the last argument is auto_start (0: connect to an already running one).
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _arg_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def run_fake_vllm(args: list[str]) -> None:
    model_dir = Path(args[1])
    port = int(_arg_after(args, "--port"))
    model_name = _arg_after(args, "--served-model-name")
    (model_dir / "vllm.pid").write_text(str(os.getpid()))
    never = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps(
                {"data": [{"id": model_name, "max_model_len": 4096}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            (model_dir / "request.received").touch()
            never.wait()

        def log_message(self, *_: object) -> None:
            pass

    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def run_serve(
    model_dir: str, api_port: str, vllm_port: str, binary: str, auto_start: str
) -> None:
    from appinfra.log import Logger

    from llm_infer.serving.dispatch import InferenceConfig, run_server

    config = InferenceConfig()
    config.backends.engine = "vllm-server"
    config.models.path = model_dir
    config.api.host = "127.0.0.1"
    config.api.port = int(api_port)
    server_cfg = config.engines.vllm_server
    server_cfg.host = "http://127.0.0.1"
    server_cfg.port = int(vllm_port)
    server_cfg.binary_path = binary
    server_cfg.auto_start = auto_start == "1"
    server_cfg.warmup = False
    server_cfg.startup_timeout = 30
    run_server(Logger("serve-shutdown-test"), config)


if __name__ == "__main__":
    if sys.argv[1] == "vllm":
        run_fake_vllm(sys.argv[2:])
    elif sys.argv[1] == "serve":
        run_serve(*sys.argv[2:7])
    else:
        raise SystemExit(f"unknown role: {sys.argv[1]}")
