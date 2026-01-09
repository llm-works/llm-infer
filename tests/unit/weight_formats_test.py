"""Unit tests for quantization weight formats."""

import pytest
import torch

from llm_infer.backends.linear.formats.awq import AWQWeights
from llm_infer.backends.linear.formats.fp8 import FP8Weights

pytestmark = pytest.mark.unit


class TestAWQWeights:
    """Test AWQWeights dataclass."""

    def test_in_features_property(self) -> None:
        """Test in_features is derived from qweight shape."""
        qweight = torch.zeros(1024, 64, dtype=torch.int32)  # in=1024, out=512
        scales = torch.zeros(8, 512, dtype=torch.float16)
        qzeros = torch.zeros(8, 64, dtype=torch.int32)

        weights = AWQWeights(
            qweight=qweight, scales=scales, qzeros=qzeros, group_size=128
        )

        assert weights.in_features == 1024

    def test_out_features_property(self) -> None:
        """Test out_features is derived from qweight shape * 8."""
        qweight = torch.zeros(1024, 64, dtype=torch.int32)  # 64 * 8 = 512
        scales = torch.zeros(8, 512, dtype=torch.float16)
        qzeros = torch.zeros(8, 64, dtype=torch.int32)

        weights = AWQWeights(
            qweight=qweight, scales=scales, qzeros=qzeros, group_size=128
        )

        assert weights.out_features == 512

    def test_optional_bias(self) -> None:
        """Test bias is optional."""
        qweight = torch.zeros(128, 16, dtype=torch.int32)
        scales = torch.zeros(1, 128, dtype=torch.float16)
        qzeros = torch.zeros(1, 16, dtype=torch.int32)

        weights_no_bias = AWQWeights(
            qweight=qweight, scales=scales, qzeros=qzeros, group_size=128
        )
        assert weights_no_bias.bias is None

        bias = torch.zeros(128, dtype=torch.float16)
        weights_with_bias = AWQWeights(
            qweight=qweight, scales=scales, qzeros=qzeros, group_size=128, bias=bias
        )
        assert weights_with_bias.bias is not None


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 not supported in this PyTorch version",
)
class TestFP8Weights:
    """Test FP8Weights dataclass."""

    def test_in_features_property(self) -> None:
        """Test in_features is derived from weight shape."""
        weight = torch.zeros(256, 128, dtype=torch.float8_e4m3fn)
        weight_scale_inv = torch.ones(2, 1, dtype=torch.float16)

        weights = FP8Weights(
            weight=weight, weight_scale_inv=weight_scale_inv, block_size=128
        )

        assert weights.in_features == 128

    def test_out_features_property(self) -> None:
        """Test out_features is derived from weight shape."""
        weight = torch.zeros(256, 128, dtype=torch.float8_e4m3fn)
        weight_scale_inv = torch.ones(2, 1, dtype=torch.float16)

        weights = FP8Weights(
            weight=weight, weight_scale_inv=weight_scale_inv, block_size=128
        )

        assert weights.out_features == 256
