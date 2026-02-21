infra := $(shell appinfra scripts-path)

# Configuration
INFRA_DEV_PKG_NAME := llm_infer

# Code quality strictness
# - true: Fail on any code quality violations (CI mode)
# - false: Report violations but don't fail (development mode)
INFRA_DEV_CQ_STRICT := true

# Test coverage threshold (percentage)
INFRA_PYTEST_COVERAGE_THRESHOLD := 50

# Custom pip config for flashinfer index
export PIP_CONFIG_FILE := etc/pip.conf

# Include framework (config first)
include $(infra)/make/Makefile.config
include $(infra)/make/Makefile.env
include $(infra)/make/Makefile.help
include $(infra)/make/Makefile.utils
include $(infra)/make/Makefile.dev
include $(infra)/make/Makefile.pytest
include $(infra)/make/Makefile.install
include $(infra)/make/Makefile.clean
