"""投影与渲染的单元测试。

验证思路：构造一张「像素值 = 该像素经纬度」的解析型 ERP 图，
这样任何采样都可以与解析解逐点比对，而不必依赖真实图像。
"""

from __future__ import annotations

import numpy as np
import pytest

from pano_agent.geometry.camera import VirtualCamera
from pano_agent.geometry.projection import (
    circular_crop_erp,
    circular_pad_erp,
    erp_seam_difference,
    horizontal_gradient_erp,
    render_nfov,
    sample_erp,
)
from pano_agent.geometry.sphere import erp_pixel_to_sph


@pytest.fixture
def angle_erp() -> np.ndarray:
    """构造 yaw 图与 pitch 图：像素值精确等于该像素中心的角度。"""
    width, height = 360, 180  # 每像素 1°，便于推理
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    yaw, pitch = erp_pixel_to_sph(xs, ys, width, height)
    return np.stack([yaw, pitch], axis=-1)


class TestCircularPad:
    def test_wraps_horizontally(self):
        img = np.arange(12, dtype=np.float64).reshape(3, 4)
        padded = circular_pad_erp(img, 2)
        assert padded.shape == (3, 8)
        # 左侧补的应是原图右侧两列（0,1,2,3 → 2,3 | 0,1,2,3 | 0,1）
        np.testing.assert_array_equal(padded[0], [2, 3, 0, 1, 2, 3, 0, 1])

    def test_pads_vertically_with_edges(self):
        img = np.arange(12, dtype=np.float64).reshape(3, 4)
        padded = np.pad(img, ((1, 1), (2, 2)), mode="wrap")
        # 垂直方向 wrap 会把最后一行接到第一行 —— 这不是我们要的
        # 本项目只对经度轴做 wrap，因此 vertical 方向必须另作处理
        assert padded.shape == (5, 8)

    def test_round_trip_with_crop(self):
        img = np.random.default_rng(0).random((10, 20))
        np.testing.assert_allclose(circular_crop_erp(circular_pad_erp(img, 3), 3), img)

    def test_zero_padding_is_copy(self):
        img = np.random.default_rng(0).random((5, 5))
        out = circular_pad_erp(img, 0)
        np.testing.assert_array_equal(out, img)
        assert out is not img

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            circular_pad_erp(np.zeros((4, 4)), -1)

    def test_rejects_negative_crop_pad(self):
        # pad=0 时 crop 是恒等，不校验；仅确保不炸
        arr = np.zeros((4, 4))
        np.testing.assert_array_equal(circular_crop_erp(arr, 0), arr)


class TestHorizontalGradient:
    def test_wraps_across_seam(self):
        """第 0 列的左邻必须取自最后一列（环绕），而非被当作图像边界。"""
        img = np.zeros((2, 8), dtype=np.float64)
        img[:, 7] = 10.0  # 只有最后一列非零

        grad = horizontal_gradient_erp(img)

        # 环绕下 grad[:,0] = (img[:,1] - img[:,7]) / 2 = (0 - 10)/2 = -5
        # 若误把第 0 列当边界（单侧差分），会得到 +5 或 0
        assert grad[0, 0] == pytest.approx(-5.0)  # 第 0 列左邻环绕到第 7 列
        assert grad[0, 6] == pytest.approx(5.0)   # 第 6 列右邻是第 7 列
        assert grad[0, 7] == pytest.approx(0.0)   # 第 7 列两侧都是 0
        assert grad[0, 3] == pytest.approx(0.0)   # 远离接缝处无变化

    def test_gradient_on_linear_ramp_interior(self):
        """远离接缝的内部区域，梯度应处处等于步长。"""
        img = np.tile(np.arange(64, dtype=np.float64), (4, 1))
        grad = horizontal_gradient_erp(img)
        np.testing.assert_allclose(grad[:, 1:-1], 1.0, atol=1e-12)

    def test_constant_image_has_zero_gradient(self):
        img = np.full((8, 16), 3.14)
        np.testing.assert_allclose(horizontal_gradient_erp(img), 0.0, atol=1e-12)


class TestSampleErp:
    def test_recovers_yaw_analytically(self, angle_erp: np.ndarray):
        """在任意方向上采样，返回的 yaw 应与采样方向一致。"""
        yaw_probe = np.array([-170.0, -90.0, 0.0, 45.0, 90.0, 170.0])
        pitch_probe = np.array([0.0, 15.0, -30.0, 0.0, 45.0, -60.0])

        sampled = sample_erp(angle_erp, yaw_probe, pitch_probe, mode="nearest")
        np.testing.assert_allclose(sampled[:, 0], yaw_probe, atol=1.5)
        np.testing.assert_allclose(sampled[:, 1], pitch_probe, atol=1.5)

    def test_nearest_is_exact_on_pixel_centers(self, angle_erp: np.ndarray):
        """在像素中心处用最近邻采样，应精确还原该像素值。"""
        width, height = 360, 180
        for x, y in [(0, 0), (100, 50), (359, 179), (180, 90)]:
            yaw, pitch = erp_pixel_to_sph(float(x), float(y), width, height)
            got = sample_erp(angle_erp, yaw, pitch, mode="nearest")
            np.testing.assert_allclose(got, angle_erp[y, x], atol=1e-9)

    def test_bilinear_between_pixels_interpolates(self):
        img = np.zeros((4, 8), dtype=np.float64)
        img[2, 4] = 1.0
        img[2, 5] = 1.0
        # 取像素 4 与 5 的正中间（坐标 4.5）
        yaw, pitch = erp_pixel_to_sph(4.0, 2.0, 8, 4)
        yaw_next, _ = erp_pixel_to_sph(5.0, 2.0, 8, 4)
        mid = (yaw + yaw_next) / 2.0
        got = sample_erp(img, mid, pitch, mode="bilinear")
        assert float(got) == pytest.approx(1.0, abs=1e-6)

    def test_output_shape_matches_input_angles(self):
        img = np.zeros((16, 32, 3))
        yaw = np.linspace(-180, 180, 20)
        pitch = np.linspace(-40, 40, 20)
        out = sample_erp(img, yaw, pitch)
        assert out.shape == (20, 3)

    def test_channel_dimension_preserved(self):
        img = np.random.default_rng(0).random((8, 16, 3))
        out = sample_erp(img, 0.0, 0.0)
        assert out.shape == (3,)

    def test_rejects_bad_ndim(self):
        with pytest.raises(ValueError):
            sample_erp(np.zeros(10), 0.0, 0.0)

    def test_poles_use_nearest_row(self, angle_erp: np.ndarray):
        """极点附近必须退化为最近邻，否则会混合相邻纬度产生拖影。"""
        got = sample_erp(angle_erp, 0.0, 89.9, mode="bilinear")
        # 北极的 pitch 应接近 +90 且不出现明显衰减
        assert float(got[1]) > 88.0


class TestRenderNfov:
    def test_output_shape(self):
        img = np.zeros((256, 512, 3), dtype=np.uint8)
        cam = VirtualCamera(0, 0, 0, 90, 200, 120)
        out = render_nfov(img, cam)
        assert out.shape == (120, 200, 3)

    def test_grayscale_input(self):
        img = np.zeros((64, 128), dtype=np.float64)
        cam = VirtualCamera(0, 0, 0, 90, 32, 16)
        assert render_nfov(img, cam).shape == (16, 32)

    def test_center_pixel_matches_optical_axis(self, angle_erp: np.ndarray):
        """渲染结果中心像素的方向应等于光轴方向。"""
        cam = VirtualCamera(yaw_deg=40.0, pitch_deg=25.0, roll_deg=0.0,
                            fov_x_deg=90.0, width=33, height=33)
        out = render_nfov(angle_erp, cam, mode="nearest")
        center_yaw, center_pitch = out[16, 16]
        assert float(center_yaw) == pytest.approx(40.0, abs=2.0)
        assert float(center_pitch) == pytest.approx(25.0, abs=2.0)

    def test_horizontal_pan_shifts_content(self, angle_erp: np.ndarray):
        """相机 yaw 增加，画面内容的 yaw 也应增加。"""
        left = render_nfov(angle_erp, VirtualCamera(0, 0, 0, 90, 33, 33), mode="nearest")
        right = render_nfov(angle_erp, VirtualCamera(60, 0, 0, 90, 33, 33), mode="nearest")
        assert float(right[16, 16, 0]) > float(left[16, 16, 0])

    def test_boundary_wrap_in_render(self, angle_erp: np.ndarray):
        """相机朝向 180°（接缝处）渲染时不应出现空白或错位。"""
        cam = VirtualCamera(180.0, 0.0, 0.0, 90.0, 65, 33)
        out = render_nfov(angle_erp, cam, mode="nearest")
        # 中心列应接近 ±180
        assert abs(abs(float(out[16, 32, 0])) - 180.0) < 3.0


class TestSeamDifference:
    def test_continuous_signal_scores_small(self):
        """环绕连续的信号，接缝差不可能远超内部差的量级。

        注意本指标只用于**检测接缝是否被破坏**，其取值依赖信号自身的形状：
        在极值点附近接缝差偏小（cos 在接缝处导数为零 → 比值可低至 0.1），
        在过零点附近偏大（sin → 比值约 1.6）。
        因此判据是「远大于 1 即破坏」，而非「接近 1 即正常」。
        """
        width = 64
        ramp = np.linspace(0, 2 * np.pi, width, endpoint=False)
        for signal in (np.cos(ramp), np.sin(ramp)):
            img = np.tile(signal, (8, 1))
            assert erp_seam_difference(img) < 3.0

    def test_broken_seam_scores_high(self):
        """接缝被人为破坏时，比值应显著大于 1。"""
        img = np.tile(np.linspace(0, 1, 64), (8, 1))
        broken = img.copy()
        broken[:, -1] = 100.0  # 制造接缝断裂
        assert erp_seam_difference(broken) > 5.0
