"""Test configuration and utilities."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Configuration file paths
INFRA_CONFIG = PROJECT_ROOT / "etc" / "infra.yaml"
INFERENCE_CONFIG = PROJECT_ROOT / "etc" / "llm-infer.yaml"

# Test data directory
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "data"


# Test categories
class TestCategory:
    """Test category markers for pytest."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERF = "perf"
    SECURITY = "security"
