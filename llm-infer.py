#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Inference server entry point."""

import sys
from pathlib import Path

# Ensure local source takes precedence over installed package
sys.path.insert(0, str(Path(__file__).parent))

from llm_infer.cli import main

if __name__ == "__main__":
    main()
