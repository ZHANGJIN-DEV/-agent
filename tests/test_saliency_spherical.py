"""显著性的球面加权测试。

`latitude_weight_map` 与 `weighted_mean` 是纯 numpy 的，可以独立验证；
网络与损失函数需要 torch，未安装时自动跳过。
"""

from __future__ import annotations

import numpy as np
import pytest

from pano_agent.geometry.sphere import SPHERE_AREA_SR, erp_row_solid_angles
from pano_agent.perception.saliency import (
    latitude_weight_map,
    latitude_weight_map_2d,
    weighted_mean,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================== 权重图


class TestLatitudeWeightMap:
    def test_shapes(self):
        assert latitude_weight_map(64).shape == (64, 1)
        assert latitude_weight_map_2d(64, 128).shape == (64, 128)

    def test_peaked_at_equator(self):
        """赤道行权重最大，极点行最小 —— 这正是 cos φ 的形状。"""
        w = latitude_weight_map(180)[:, 0]
        assert w[90] > w[0], "赤道权重应大于北极"
        assert w[90] > w[-1], "赤道权重应大于南极"
        assert w[0] == pytest.approx(w[-1], rel=1e-6), "南北极应对称"

    def test_matches_solid_angle_exactly(self):
        """未归一化的权重必须精确等于各行立体角。"""
        height = 128
        expected = erp_row_solid_angles(height)
        actual = latitude_weight_map(height, normalize=False)[:, 0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_unnormalized_sums_to_4pi(self):
        """整个球面的立体角是 4π —— 这是权重正确性的硬性检验。"""
        for height in (64, 180, 512):
            total = latitude_weight_map(height, normalize=False).sum()
            assert total == pytest.approx(SPHERE_AREA_SR, rel=1e-6)

    def test_normalized_mean_is_one(self):
        for height in (64, 256):
            assert latitude_weight_map(height).mean() == pytest.approx(1.0, rel=1e-6)

    def test_pole_weight_is_near_zero(self):
        """极点行的权重应接近 0 —— 它们代表极小的一块球面。"""
        w = latitude_weight_map(180, normalize=False)[:, 0]
        assert w[0] < 0.02 * w[90]

    def test_rejects_bad_size(self):
        with pytest.raises(ValueError):
            latitude_weight_map(0)
        with pytest.raises(ValueError):
            latitude_weight_map_2d(64, 0)

    def test_2d_is_row_broadcast(self):
        w2 = latitude_weight_map_2d(32, 64)
        w1 = latitude_weight_map(32)
        np.testing.assert_allclose(w2, np.broadcast_to(w1, (32, 64)))


# ============================================================== 加权平均


class TestWeightedMean:
    def test_uniform_field_returns_that_value(self):
        """均匀场下加权平均退化为该常数，与朴素平均一致。"""
        field = np.full((64, 128), 3.0)
        assert weighted_mean(field, 64, 128) == pytest.approx(3.0, rel=1e-6)

    def test_polar_values_are_downweighted(self):
        """★ 极区的大值不应主导结果 —— 这是加权存在的全部意义。"""
        h, w = 128, 256
        field = np.zeros((h, w))
        field[:4, :] = 100.0  # 北极附近的巨大值
        mid = h // 2
        field[mid - 2 : mid + 2, :] = 1.0  # 赤道附近的轻微值

        weighted = weighted_mean(field, h, w)
        naive = float(field.mean())

        assert weighted < naive, "加权结果应低于朴素平均"
        assert weighted < 10.0, (
            f"极区 100 的值不应把加权均值拉到 {weighted:.2f} "
            f"（朴素平均为 {naive:.2f}）"
        )

    def test_equatorial_values_dominate(self):
        """赤道的值应当获得高于平均的权重。"""
        h, w = 128, 256
        field = np.zeros((h, w))
        field[h // 2, :] = 1.0  # 只在赤道行有值

        w_eq = float(latitude_weight_map(h)[h // 2, 0])

        # 归一化权重以 1 为平均，赤道行代表的球面面积大于平均，故 > 1
        assert w_eq > 1.0, f"赤道行归一化权重应大于 1，实际 {w_eq:.3f}"

        # 加权平均 = (1.0 × w_eq × W) / (H × W) = w_eq / H
        assert weighted_mean(field, h, w) == pytest.approx(w_eq / h, rel=1e-3)

    def test_polar_row_gets_less_weight_than_equatorial(self):
        """同样强度的信号，放在赤道应比放在极区产生更大的加权均值。"""
        h, w = 128, 256

        equatorial = np.zeros((h, w))
        equatorial[h // 2, :] = 1.0

        polar = np.zeros((h, w))
        polar[0, :] = 1.0

        assert weighted_mean(equatorial, h, w) > weighted_mean(polar, h, w)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="形状"):
            weighted_mean(np.zeros((10, 10)), 20, 20)

    def test_handles_all_zeros(self):
        assert weighted_mean(np.zeros((32, 64)), 32, 64) == pytest.approx(0.0)


# ============================================================== torch 部分


@pytest.mark.skipif(not HAS_TORCH, reason="需要 PyTorch（尚未安装）")
class TestTorchComponents:
    def test_unet_forward_shape_and_range(self):
        from pano_agent.perception.saliency import SaliencyUNet

        model = SaliencyUNet(base=8)
        x = torch.randn(1, 3, 64, 128)
        out = model(x)
        assert out.shape == (1, 1, 64, 128), "输出应与输入同尺寸（UNet 的跳跃连接）"

    def test_param_count_is_small(self):
        """模型必须小到能在 2.5GB 显存下工作。"""
        from pano_agent.perception.saliency import SaliencyUNet

        model = SaliencyUNet(base=32)
        n = model.num_parameters()
        assert n < 5_000_000, f"base=32 应有约 2M 参数，实际 {n}"

    def test_weighted_loss_prefers_correct_prediction(self):
        """正确的预测应得到更低的损失。"""
        from pano_agent.perception.saliency import weighted_bce_loss

        target = torch.zeros(1, 1, 32, 64)
        target[..., 14:18, :] = 1.0

        good = torch.where(target > 0.5, 5.0, -5.0)
        bad = -good

        loss_good = weighted_bce_loss(good, target).item()
        loss_bad = weighted_bce_loss(bad, target).item()
        assert loss_good < loss_bad

    def test_weighted_iou_perfect_prediction(self):
        from pano_agent.perception.saliency import weighted_iou

        target = torch.zeros(1, 1, 32, 64)
        target[..., 10:20, :] = 1.0
        logits = torch.where(target > 0.5, 10.0, -10.0)

        assert weighted_iou(logits, target) == pytest.approx(1.0, abs=1e-4)

    def test_weighted_iou_is_zero_for_disjoint(self):
        from pano_agent.perception.saliency import weighted_iou

        target = torch.zeros(1, 1, 32, 64)
        target[..., :5, :] = 1.0
        logits = torch.full_like(target, -10.0)  # 全预测为背景

        assert weighted_iou(logits, target) == pytest.approx(0.0, abs=1e-6)
