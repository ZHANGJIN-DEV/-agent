"""光流与球面运动能量的测试。

重点是**跨 ERP 接缝的光流连续性**：光流算法默认图像边界外"没有内容"，
而 ERP 的左右边界是同一条经线。若不做环形 padding，接缝附近的光流
会被边界约束污染，进而污染所有下游信号。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pano_agent.geometry.projection import circular_pad_erp
from pano_agent.perception.flow import (
    DisFlowEstimator,
    FlowConfig,
    spherical_motion_energy,
)

H, W = 128, 256


def textured_erp(h: int = H, w: int = W, seed: int = 0) -> np.ndarray:
    """生成有纹理的 ERP 测试图。

    纯噪声会让 DIS 难以找到对应（缺少可匹配的结构），
    纯色则完全无信息，因此用"模糊后的噪声"作为纹理。
    """
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w), dtype=np.uint8)
    return cv2.GaussianBlur(img, (7, 7), 2.0)


def translate(img: np.ndarray, dx: int, dy: int = 0) -> np.ndarray:
    """水平/垂直平移，水平方向**环绕**（模拟全景中的真实运动）。"""
    return np.roll(img, (dy, dx), axis=(0, 1))


# ============================================================== 基础正确性


class TestFlowCorrectness:
    def test_pure_horizontal_translation(self):
        """纯水平平移的光流应接近真实位移。"""
        prev = textured_erp()
        dx = 8
        curr = translate(prev, dx)

        est = DisFlowEstimator(FlowConfig(seam_pad_px=32, work_scale=1.0))
        flow = est.compute(prev, curr, use_seam_padding=False)

        # 只检查中央区域 —— 边界附近任何光流算法都不可靠
        cx = flow[32:-32, 64:-64]
        assert np.median(cx[..., 0]) == pytest.approx(dx, abs=2.0)
        assert np.median(np.abs(cx[..., 1])) < 2.0, "垂直方向不应有显著位移"

    def test_zero_motion_gives_near_zero_flow(self):
        img = textured_erp()
        est = DisFlowEstimator(FlowConfig(work_scale=1.0))
        flow = est.compute(img, img.copy(), use_seam_padding=False)
        assert np.median(np.hypot(flow[..., 0], flow[..., 1])) < 1.0

    def test_output_shape_matches_input(self):
        est = DisFlowEstimator(FlowConfig(work_scale=1.0))
        flow = est.compute(textured_erp(), textured_erp(seed=1))
        assert flow.shape == (H, W, 2)
        assert flow.dtype == np.float32

    def test_work_scale_changes_resolution(self):
        est = DisFlowEstimator(FlowConfig(work_scale=0.5))
        flow = est.compute(textured_erp(), textured_erp(seed=1))
        assert flow.shape == (H // 2, W // 2, 2)

    def test_mismatched_sizes_raise(self):
        est = DisFlowEstimator()
        with pytest.raises(ValueError, match="尺寸不一致"):
            est.compute(textured_erp(), textured_erp(h=64, w=128))

    def test_accepts_color_input(self):
        rgb = np.stack([textured_erp(seed=i) for i in range(3)], axis=-1)
        est = DisFlowEstimator(FlowConfig(work_scale=1.0))
        flow = est.compute(rgb, translate(rgb, 5))
        assert flow.shape == (H, W, 2)


# ============================================================== ★ 跨接缝


class TestSeamContinuity:
    """本模块存在的理由：让接缝处的光流与其他位置同样可靠。"""

    def test_seam_region_flow_with_padding(self):
        """★ 核心：跨越接缝的平移，接缝附近的光流应接近真值。"""
        prev = textured_erp()
        dx = 6
        curr = translate(prev, dx)  # np.roll 是环绕平移，自然跨越接缝

        est = DisFlowEstimator(FlowConfig(seam_pad_px=64, work_scale=1.0))
        flow = est.compute(prev, curr, use_seam_padding=True)

        # 接缝附近的列（x=0 和 x=W-1 两侧）
        seam_cols = np.concatenate([flow[32:-32, :8], flow[32:-32, -8:]], axis=1)
        seam_median = float(np.median(seam_cols[..., 0]))

        assert abs(seam_median - dx) < 2.5, (
            f"接缝处光流应接近真值 {dx}，实际中位数 {seam_median:.2f}"
        )

    def test_padding_improves_seam_over_no_padding(self):
        """对照：不做环形 padding 时，接缝处的误差应明显更大。

        这个测试证明环形 padding 不是可有可无的装饰，而是必要的修正。
        """
        prev = textured_erp()
        dx = 12
        curr = translate(prev, dx)

        cfg = FlowConfig(seam_pad_px=64, work_scale=1.0)
        est = DisFlowEstimator(cfg)

        with_pad = est.compute(prev, curr, use_seam_padding=True)
        without_pad = est.compute(prev, curr, use_seam_padding=False)

        def seam_error(flow: np.ndarray) -> float:
            band = np.concatenate([flow[32:-32, :6], flow[32:-32, -6:]], axis=1)
            return float(np.mean(np.abs(band[..., 0] - dx)))

        err_with = seam_error(with_pad)
        err_without = seam_error(without_pad)

        assert err_with < err_without, (
            f"环形 padding 应降低接缝误差：开启 {err_with:.2f} "
            f"vs 关闭 {err_without:.2f}"
        )

    def test_flow_is_finite_everywhere(self):
        """任何位置都不应出现 NaN / inf。"""
        prev = textured_erp()
        est = DisFlowEstimator(FlowConfig(seam_pad_px=32, work_scale=1.0))
        flow = est.compute(prev, translate(prev, 10), use_seam_padding=True)
        assert np.isfinite(flow).all()


# ============================================================== 运动能量


class TestSphericalMotionEnergy:
    def test_uniform_flow_weighted_equals_naive(self):
        """全场均匀时，加权与朴素平均应当一致。"""
        flow = np.zeros((64, 128, 2), dtype=np.float64)
        flow[..., 0] = 3.0

        energy = spherical_motion_energy(flow)
        assert energy.mean_weighted == pytest.approx(3.0, rel=1e-6)
        assert energy.mean_naive == pytest.approx(3.0, rel=1e-6)

    def test_high_latitude_motion_is_downweighted(self):
        """★ 极点附近的大幅噪声不应主导指标 —— 这正是 cos φ 加权的作用。"""
        height, width = 128, 256
        flow = np.zeros((height, width, 2), dtype=np.float64)

        # 只在最上面的几行（北极附近，纬度接近 +90°）制造巨大位移
        flow[:4, :, 0] = 100.0
        # 赤道附近加入轻微运动
        mid = height // 2
        flow[mid - 2 : mid + 2, :, 0] = 1.0

        energy = spherical_motion_energy(flow)

        # 加权后，极区的 100px 位移只贡献很小的权重；
        # 均值应在个位数，而不是被 100 拉高
        assert energy.mean_weighted < 10.0, (
            f"极区噪声不应主导加权均值，实际 {energy.mean_weighted:.2f}"
        )
        # 朴素平均则会被极区主导
        assert energy.mean_naive > 2.0

    def test_weighted_less_than_naive_for_polar_motion(self):
        """极区运动时，加权均值必然小于朴素均值。"""
        flow = np.zeros((128, 256, 2), dtype=np.float64)
        flow[:8, :, 0] = 50.0

        energy = spherical_motion_energy(flow)
        assert energy.mean_weighted < energy.mean_naive

    def test_band_energies_separate_regions(self):
        """分带统计应能区分赤道与高纬的运动。"""
        height, width = 180, 360
        flow = np.zeros((height, width, 2), dtype=np.float64)

        # 赤道带（行 75-105 对应纬度 ±15° 左右）
        flow[75:105, :, 0] = 5.0
        energy = spherical_motion_energy(flow)

        assert energy.band_energies["equatorial"] > 0.5
        assert energy.band_energies["high_latitude"] < 0.1, "高纬带应几乎无运动"
        assert energy.band_energies["mid_latitude"] < 0.1

    def test_band_energies_present(self):
        flow = np.zeros((64, 128, 2), dtype=np.float64)
        energy = spherical_motion_energy(flow)
        assert set(energy.band_energies) == {
            "equatorial",
            "mid_latitude",
            "high_latitude",
        }

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="形状"):
            spherical_motion_energy(np.zeros((10, 10)))

    def test_total_is_weighted_sum(self):
        """total 应为加权和，而非像素和。"""
        flow = np.zeros((64, 128, 2), dtype=np.float64)
        flow[..., 0] = 1.0

        energy = spherical_motion_energy(flow)
        # 加权和 = 每行权重之和 × 每行像素数 × 幅值
        # 行权重总和为 4π，故 total ≈ 4π × W
        expected = 4.0 * np.pi * 128
        assert energy.total == pytest.approx(expected, rel=1e-6)

    def test_to_dict_serializable(self):
        flow = np.random.default_rng(0).random((32, 64, 2)).astype(np.float64)
        d = spherical_motion_energy(flow).to_dict()
        assert isinstance(d, dict)
        assert all(isinstance(v, float) for k, v in d.items() if k != "band_energies")


class TestEstimatorIntegration:
    def test_motion_energy_from_frames(self):
        prev = textured_erp()
        curr = translate(prev, 5)

        est = DisFlowEstimator(FlowConfig(seam_pad_px=32, work_scale=1.0))
        energy = est.motion_energy(prev, curr)

        assert energy.mean_weighted > 0.5, "有平移时运动能量应为正"
        # 工作分辨率下的位移近似等于真实像素位移
        assert energy.mean_weighted == pytest.approx(5.0, abs=2.0)

    def test_static_scene_has_low_energy(self):
        img = textured_erp()
        est = DisFlowEstimator(FlowConfig(work_scale=1.0))
        energy = est.motion_energy(img, img.copy())
        assert energy.mean_weighted < 1.0


class TestCircularPadHelper:
    def test_padding_preserves_continuity(self):
        """确认 geometry 层的 padding 在光流场景下也保持接缝连续。"""
        img = textured_erp(h=8, w=16)
        padded = circular_pad_erp(img, 4)

        assert padded.shape == (8, 24)
        # 左补的部分应等于原图右侧
        np.testing.assert_array_equal(padded[:, :4], img[:, -4:])
        # 右补的部分应等于原图左侧
        np.testing.assert_array_equal(padded[:, -4:], img[:, :4])
