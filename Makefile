# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

# Configuration
APPINFRA_DEV_PKG_NAME ?= llm_infer
APPINFRA_DEV_INSTALL_EXTRAS ?= dev,runtime,cuda

# Code quality strictness
# - true: Fail on any code quality violations (CI mode)
# - false: Report violations but don't fail (development mode)
APPINFRA_DEV_CQ_STRICT ?= true

# SPDX license header enforcement
APPINFRA_DEV_CQ_SPDX ?= true

# No ruff rules reported as warnings only: they are enforced in pyproject's select
APPINFRA_DEV_CQ_LINT_SELECT ?=

# Fail on duplicated code blocks and unused private definitions
APPINFRA_DEV_CQ_DUP ?= strict
APPINFRA_DEV_CQ_DEAD ?= strict

# Docstring coverage threshold (percentage)
APPINFRA_DEV_DOCSTRING_THRESHOLD ?= 90

# Test coverage threshold (percentage)
APPINFRA_PYTEST_COVERAGE_THRESHOLD ?= 70

# Report tests outside their suite and unit tests that leak process-wide state
APPINFRA_DEV_TEST_HEALTH ?= warn

# Custom pip config for flashinfer index
export PIP_CONFIG_FILE := etc/pip.conf

# Run example scripts as part of `make check` (per-file ci-* tags apply)
APPINFRA_DEV_CHECK_EXAMPLES ?= true

# Locate the framework; the environment and the command line still win
ifeq ($(origin APPINFRA_SCRIPTS),undefined)
APPINFRA_SCRIPTS := $(shell appinfra scripts-path)
endif

# Include framework (config first)
include $(APPINFRA_SCRIPTS)/make/Makefile.config
include $(APPINFRA_SCRIPTS)/make/Makefile.env
include $(APPINFRA_SCRIPTS)/make/Makefile.help
include $(APPINFRA_SCRIPTS)/make/Makefile.utils
include $(APPINFRA_SCRIPTS)/make/Makefile.dev
include $(APPINFRA_SCRIPTS)/make/Makefile.pytest
include $(APPINFRA_SCRIPTS)/make/Makefile.install
include $(APPINFRA_SCRIPTS)/make/Makefile.clean
