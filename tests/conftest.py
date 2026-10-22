# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Shared pytest setup for all test suites."""

# appinfra.testing destroys the loggers each test leaves registered, so the next
# test in the same worker can create the same names again.
pytest_plugins = ["appinfra.testing"]
