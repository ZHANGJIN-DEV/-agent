"""球面区域与球面 IoU 的单元测试。

重点验证「经度平移不变性」：一个跨 ERP 边界的目标与它平移 360° 后的
副本必须是**同一个球面区域**。任何在像素空间实现的 IoU 都会在这里失败。
"""

from __future__ import annotations

import numpy as np
import pytest

from pano_agent.geometry.region import SphericalBox, angular_offsets
from pano_agent.geometry.sphere import sph_to_vec, vec_to_sph


class TestSphericalBoxBasics:
    def test_rejects_bad_extent(self):
        with pytest.raises(ValueError):
            SphericalBox(0, 0, 0.0, 10.0)
        with pytest.raises(ValueError):
            SphericalBox(0, 0, 10.0, -5.0)
        with pytest.raises(ValueError):
            SphericalBox(0, 0, 400.0, 10.0)

    def test_center_vec_matches_angles(self):
        box = SphericalBox(-120, 35, 20, 10)
        yaw, pitch = vec_to_sph(box.center_vec)
        assert float(yaw) == pytest.approx(-120.0, abs=1e-9)
        assert float(pitch) == pytest.approx(35.0, abs=1e-9)

    def test_contains_center(self):
        box = SphericalBox(45, 20, 30, 20)
        assert bool(box.contains(45.0, 20.0))

    def test_excludes_far_points(self):
        box = SphericalBox(0, 0, 20, 20)
        assert not bool(box.contains(90.0, 0.0))
        assert not bool(box.contains(0.0, 60.0))

    def test_contains_is_broadcast(self):
        box = SphericalBox(0, 0, 40, 40)
        inside = box.contains(np.array([0.0, 10.0, 100.0]), np.array([0.0, 5.0, 0.0]))
        np.testing.assert_array_equal(inside, [True, True, False])


class TestSphericalIoU:
    def test_self_iou_is_one(self):
        box = SphericalBox(30, -10, 25, 15)
        assert box.iou(box, n_samples=20_000) == pytest.approx(1.0, abs=1e-9)

    def test_disjoint_iou_is_zero(self):
        a = SphericalBox(0, 0, 20, 20)
        b = SphericalBox(90, 0, 20, 20)
        assert a.iou(b, n_samples=20_000) == pytest.approx(0.0, abs=1e-9)

    def test_symmetry(self):
        a = SphericalBox(10, 5, 30, 20)
        b = SphericalBox(20, 8, 25, 18)
        assert a.iou(b, n_samples=20_000) == pytest.approx(
            b.iou(a, n_samples=20_000), abs=1e-9
        )

    def test_identical_but_shifted_longitude(self):
        """★ 核心测试：中心相差 360° 的两个框是同一个球面区域。

        这是球面表示相对像素表示的根本优势 —— 在 ERP 像素上，
        这两个框位于图像两端，像素 IoU 会得到 0。
        """
        a = SphericalBox(-175, 10, 40, 30)
        b = SphericalBox(185, 10, 40, 30)  # 相差 360°
        assert a.iou(b, n_samples=20_000) == pytest.approx(1.0, abs=1e-9)

    def test_cross_seam_box_is_one_connected_region(self):
        """跨 ERP 接缝的框应覆盖接缝两侧，而不是被切成两段。"""
        box = SphericalBox(180, 0, 60, 30)  # 中心恰在接缝上
        # 接缝两侧 ±20° 都应被包含
        assert bool(box.contains(179.0, 0.0))
        assert bool(box.contains(-179.0, 0.0))
        assert bool(box.contains(180.0, 0.0))
        # 对面（0°）不应被包含
        assert not bool(box.contains(0.0, 0.0))

    def test_partial_overlap_is_between_zero_and_one(self):
        a = SphericalBox(0, 0, 40, 40)
        b = SphericalBox(20, 0, 40, 40)
        iou = a.iou(b, n_samples=40_000)
        assert 0.05 < iou < 0.6, f"部分重叠的 IoU 应在 (0,1) 内，实际 {iou}"

    def test_larger_box_lower_iou_with_small_one(self):
        """小框完全落在不同大小的大框中时，大框越大 IoU 越小。"""
        small = SphericalBox(0, 0, 10, 10)
        tight = SphericalBox(0, 0, 12, 12)
        loose = SphericalBox(0, 0, 60, 60)
        assert small.iou(tight, n_samples=40_000) > small.iou(loose, n_samples=40_000)


class TestAngularOffsets:
    def test_zero_at_center(self):
        alpha, beta = angular_offsets(30.0, 20.0, 30.0, 20.0)
        assert float(alpha) == pytest.approx(0.0, abs=1e-9)
        assert float(beta) == pytest.approx(0.0, abs=1e-9)

    def test_differs_from_naive_latlon_difference(self):
        """切平面角偏移 ≠ 经纬度差 —— 在高纬处差异显著。

        这是「不能用 Δyaw/Δpitch 直接当角度距离」的具体证据。
        """
        center_pitch, probe_pitch = 70.0, 75.0
        center_yaw, probe_yaw = 0.0, 40.0

        naive = np.hypot(probe_yaw - center_yaw, probe_pitch - center_pitch)
        alpha, beta = angular_offsets(probe_yaw, probe_pitch, center_yaw, center_pitch)
        true_angular = np.hypot(float(alpha), float(beta))

        # 两种算法结果应有明显差异（>10%）
        assert abs(true_angular - naive) / naive > 0.10


class TestBoxSerialization:
    def test_bfov_round_trip(self):
        box = SphericalBox(12.5, -34.0, 20.0, 15.0)
        values = box.to_bfov_values()
        assert values.shape == (4,)
        back = SphericalBox.from_bfov_values(values)
        assert back.yaw_deg == pytest.approx(12.5)
        assert back.width_deg == pytest.approx(20.0)

    def test_rbfov_roll_unit_is_radians(self):
        """360VOT 的 rBFoV 中 roll 以弧度给出，构造时需转成度。"""
        import math

        values = [10.0, 20.0, 30.0, 40.0, math.radians(45.0)]
        box = SphericalBox.from_bfov_values(values, is_rotated=True)
        assert box.roll_deg == pytest.approx(45.0, abs=1e-9)

    def test_rejects_wrong_arity(self):
        with pytest.raises(ValueError):
            SphericalBox.from_bfov_values([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            SphericalBox.from_bfov_values([1.0, 2.0, 3.0, 4.0, 5.0], is_rotated=False)


class TestFromErpPixels:
    def test_center_of_erp_maps_to_origin(self):
        box = SphericalBox.from_erp_pixels(900, 460, 1100, 560, 2048, 1024)
        assert abs(box.yaw_deg) < 5.0
        assert abs(box.pitch_deg) < 5.0

    def test_positive_width_and_height(self):
        box = SphericalBox.from_erp_pixels(100, 100, 300, 200, 2048, 1024)
        assert box.width_deg > 0
        assert box.height_deg > 0

    def test_high_latitude_box_is_narrower_in_angle(self):
        """高纬处同样的像素宽度对应更小的角度 —— 这正是 ERP 的非均匀性。"""
        wide_mid = SphericalBox.from_erp_pixels(900, 500, 1100, 520, 2048, 1024)
        wide_pole = SphericalBox.from_erp_pixels(900, 20, 1100, 40, 2048, 1024)
        assert wide_pole.width_deg < wide_mid.width_deg

    def test_iou_with_self_after_pixel_conversion(self):
        box = SphericalBox.from_erp_pixels(500, 400, 700, 500, 2048, 1024)
        assert box.iou(box, n_samples=10_000) == pytest.approx(1.0, abs=1e-9)
