# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""The health route factory resolves by qualname with its shared ready flag.

dispatch/main._build_routes wraps route factories in appinfra.subprocess.Lazy so
the child imports the factory by qualname. The health handler takes the
multiprocessing.Value the main process sets once warm; the other factories are
covered in tests/unit/serving/dispatch/lazy_routes_test.py.
"""

from __future__ import annotations

import multiprocessing as mp

from appinfra.subprocess import Lazy


def test_health_factory_qualname_resolves() -> None:
    ready = mp.Value("b", False)
    handler = Lazy(
        "llm_infer.serving.api.routes:create_health_handler", ready
    ).resolve()
    assert callable(handler)
