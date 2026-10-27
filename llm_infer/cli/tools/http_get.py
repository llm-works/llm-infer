# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""GET a JSON endpoint of a running server, for the CLI tools."""

import json
import urllib.error
import urllib.request
from typing import Any

from appinfra.log import Logger


def get_json(
    lg: Logger, host: str, port: int, path: str, failure: str
) -> dict[str, Any] | None:
    """GET http://host:port<path> and parse the JSON body.

    Returns None after logging why the request failed: "server not running"
    when the connection is refused, else the failure message with the error.
    """
    url = f"http://{host}:{port}{path}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.URLError as e:
        if "Connection refused" in str(e):
            lg.error("server not running", extra={"host": host, "port": port})
        else:
            lg.error(failure, extra={"error": str(e)})
    except Exception as e:
        lg.error(failure, extra={"error": str(e)})
    return None
