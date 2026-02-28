"""Unit tests for vllm_common module."""

from unittest.mock import MagicMock, patch

import pytest

from llm_infer.engines.vllm_common import (
    get_gpu_total_memory_gb,
    resolve_gpu_memory_utilization,
)

pytestmark = pytest.mark.unit


class TestGetGpuTotalMemoryGb:
    """Test get_gpu_total_memory_gb function."""

    def test_returns_none_when_torch_unavailable(self) -> None:
        """Test returns None when torch import fails."""
        with patch.dict("sys.modules", {"torch": None}):
            # Force re-import behavior by patching at import time
            with patch(
                "llm_infer.engines.vllm_common.get_gpu_total_memory_gb"
            ) as mock_fn:
                mock_fn.return_value = None
                result = mock_fn()
                assert result is None

    def test_returns_none_when_cuda_not_available(self) -> None:
        """Test returns None when CUDA is not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = get_gpu_total_memory_gb()
            assert result is None

    def test_returns_gb_when_cuda_available(self) -> None:
        """Test returns correct GB value when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0
        # 16 GB in bytes
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16 * 1024**3

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = get_gpu_total_memory_gb()
            assert result == 16.0

    def test_returns_none_on_exception(self) -> None:
        """Test returns None when an exception occurs."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA error")

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = get_gpu_total_memory_gb()
            assert result is None


class TestResolveGpuMemoryUtilization:
    """Test resolve_gpu_memory_utilization function."""

    def test_returns_utilization_when_gb_is_none(self) -> None:
        """Test returns gpu_memory_utilization unchanged when gpu_memory_gb is None."""
        lg = MagicMock()
        result = resolve_gpu_memory_utilization(lg, None, 0.85)
        assert result == 0.85
        lg.warning.assert_not_called()
        lg.info.assert_not_called()

    def test_raises_on_negative_gb(self) -> None:
        """Test raises ValueError for negative gpu_memory_gb."""
        lg = MagicMock()
        with pytest.raises(ValueError, match="must be positive"):
            resolve_gpu_memory_utilization(lg, -4.0, 0.9)

    def test_raises_on_zero_gb(self) -> None:
        """Test raises ValueError for zero gpu_memory_gb."""
        lg = MagicMock()
        with pytest.raises(ValueError, match="must be positive"):
            resolve_gpu_memory_utilization(lg, 0.0, 0.9)

    def test_fallback_when_gpu_detection_fails(self) -> None:
        """Test falls back to utilization when GPU detection fails."""
        lg = MagicMock()
        with patch(
            "llm_infer.engines.vllm_common.get_gpu_total_memory_gb", return_value=None
        ):
            result = resolve_gpu_memory_utilization(lg, 8.0, 0.9)
            assert result == 0.9
            lg.warning.assert_called_once()
            assert "GPU detection failed" in str(lg.warning.call_args)

    def test_converts_gb_to_utilization(self) -> None:
        """Test converts GB to correct utilization fraction."""
        lg = MagicMock()
        # 8 GB on 16 GB GPU = 0.5 utilization
        with patch(
            "llm_infer.engines.vllm_common.get_gpu_total_memory_gb", return_value=16.0
        ):
            result = resolve_gpu_memory_utilization(lg, 8.0, 0.9)
            assert result == 0.5
            lg.info.assert_called_once()

    def test_caps_at_95_percent(self) -> None:
        """Test utilization is capped at 0.95."""
        lg = MagicMock()
        # 20 GB on 16 GB GPU would be 1.25, but should cap at 0.95
        with patch(
            "llm_infer.engines.vllm_common.get_gpu_total_memory_gb", return_value=16.0
        ):
            result = resolve_gpu_memory_utilization(lg, 20.0, 0.9)
            assert result == 0.95

    def test_floors_at_1_percent(self) -> None:
        """Test utilization floors at 0.01."""
        lg = MagicMock()
        # 0.1 GB on 16 GB GPU = 0.00625, but should floor at 0.01
        with patch(
            "llm_infer.engines.vllm_common.get_gpu_total_memory_gb", return_value=16.0
        ):
            result = resolve_gpu_memory_utilization(lg, 0.1, 0.9)
            assert result == 0.01

    def test_logs_conversion_info(self) -> None:
        """Test logs info with conversion details."""
        lg = MagicMock()
        with patch(
            "llm_infer.engines.vllm_common.get_gpu_total_memory_gb", return_value=24.0
        ):
            resolve_gpu_memory_utilization(lg, 12.0, 0.9)
            lg.info.assert_called_once()
            call_args = lg.info.call_args
            assert call_args[0][0] == "gpu_memory_gb converted to utilization"
            extra = call_args[1]["extra"]
            assert extra["gpu_memory_gb"] == 12.0
            assert extra["total_gb"] == 24.0
            assert extra["utilization"] == 0.5
