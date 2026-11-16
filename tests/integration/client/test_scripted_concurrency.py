# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""The scripted backend hands each concurrent caller its own script entry."""

import threading
from unittest.mock import MagicMock

from appinfra.log import Logger

from llm_infer.client import ChatResponse, Factory


def test_concurrent_calls_take_distinct_entries() -> None:
    script = [ChatResponse(content=str(i)) for i in range(8)]
    client = Factory(MagicMock(spec=Logger)).scripted(script)
    answers: list[str] = []
    lock = threading.Lock()

    def call() -> None:
        content = client.chat([{"role": "user", "content": "Hi"}]).content
        with lock:
            answers.append(content)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(answers) == [str(i) for i in range(8)]
