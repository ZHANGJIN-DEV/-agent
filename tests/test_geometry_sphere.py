"""球面基础模块的单元测试。

这些测试的价值不在于覆盖率，而在于：**球面几何里一个符号错误会让后面
十周的跟踪、优化、渲染全部错位，而且几乎无法从输出上察觉。**
因此每一步换算都必须有往返测试与恒等性测试。
"""

from __future__ import annotations

import numpy as np
import pytest

from pano_agent.geometry.sphere import (
    SPHERE_AREA_SR,
    erp_pixel_to_sph,
    erp_row_solid_angles,
    fibonacci_sphere,
    great_circle_distance,
    sph_to_erp_pixel,
    sph_to_vec,
    solid_angle_weights,
    unwrap_yaw,
    vec_to_sph,
    wrap_pitch,
    wrap_yaw,
)


class TestYawWrapping:
    def test_wrap_yaw_known_values(self):
        assert wrap_yaw(0.0) == pytest.approx(0.0)
        assert wrap_yaw(181.0) == pytest.approx(-179.0)
        assert wrap_yaw(-181.0) == pytest.approx(179.0)
        assert wrap_yaw(360.0) == pytest.approx(0.0)
        assert wrap_yaw(-360.0) == pytest.approx(0.0)

    def test_wrap_yaw_boundary_is_half_open(self):
        """[-180, 180) 是半开区间：-180 保留，180 折回 -180。"""
        assert wrap_yaw(-180.0) == pytest.approx(-180.0)
        assert wrap_yaw(180.0) == pytest.approx(-180.0)

    def test_wrap_yaw_array(self):
        result = wrap_yaw(np.array([0.0, 90.0, 270.0, -270.0]))
        np.testing.assert_allclose(result, [0.0, 90.0, -90.0, 90.0])

    def test_wrap_pitch_clips(self):
        assert wrap_pitch(100.0) == pytest.approx(90.0)
        assert wrap_pitch(-100.0) == pytest.approx(-90.0)
        assert wrap_pitch(45.0) == pytest.approx(45.0)


class TestYawUnwrap:
    def test_crossing_seam_is_continuous(self):
        """跨 ±180° 的序列提升后应无跳变。这是本项目最关键的一条不变量。"""
        raw = np.array([170.0, 175.0, -179.0, -175.0, -170.0])
        lifted = unwrap_yaw(raw)

        # 提升后相邻差应很小（真实转动只有 5°）
        diffs = np.diff(lifted)
        assert np.all(np.abs(diffs) < 10.0), f"出现跳变: {diffs}"

        # 提升后应单调递增，且首元素不变
        assert lifted[0] == pytest.approx(170.0)
        np.testing.assert_allclose(lifted, [170.0, 175.0, 181.0, 185.0, 190.0])

    def test_round_trip_recovers_original(self):
        raw = np.array([10.0, 50.0, -170.0, 179.0, -179.0, -170.0])
        lifted = unwrap_yaw(raw)
        np.testing.assert_allclose(wrap_yaw(lifted), raw, atol=1e-10)

    def test_scalar_input(self):
        assert unwrap_yaw(45.0).shape == (1,)

    def test_empty_input(self):
        assert unwrap_yaw(np.array([])).size == 0


class TestSphereVectorConversion:
    @pytest.mark.parametrize("yaw", [-180.0, -90.0, 0.0, 45.0, 90.0, 179.9])
    @pytest.mark.parametrize("pitch", [-89.0, -45.0, 0.0, 30.0, 89.0])
    def test_round_trip(self, yaw: float, pitch: float):
        vec = sph_to_vec(yaw, pitch)
        yaw_back, pitch_back = vec_to_sph(vec)
        assert float(yaw_back) == pytest.approx(yaw, abs=1e-9)
        assert float(pitch_back) == pytest.approx(pitch, abs=1e-9)

    def test_axis_convention(self):
        """钉死坐标约定：0° 指向 +Z，+90° 指向 +X，天顶指向 +Y。"""
        np.testing.assert_allclose(sph_to_vec(0.0, 0.0), [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(sph_to_vec(90.0, 0.0), [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(sph_to_vec(0.0, 90.0), [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(sph_to_vec(0.0, -90.0), [0.0, -1.0, 0.0], atol=1e-12)

    def test_unit_length(self):
        rng = np.random.default_rng(0)
        yaw = rng.uniform(-180, 180, 500)
        pitch = rng.uniform(-90, 90, 500)
        norms = np.linalg.norm(sph_to_vec(yaw, pitch), axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)


class TestGreatCircleDistance:
    def test_known_distances(self):
        # 赤道上相隔 90°
        assert great_circle_distance(0, 0, 90, 0) == pytest.approx(90.0, abs=1e-9)
        # 赤道到天顶
        assert great_circle_distance(0, 0, 0, 90) == pytest.approx(90.0, abs=1e-9)
        # 对跖点
        assert great_circle_distance(0, 0, 180, 0) == pytest.approx(180.0, abs=1e-9)
        # 同一点
        assert great_circle_distance(30, 40, 30, 40) == pytest.approx(0.0, abs=1e-9)

    def test_symmetry(self):
        assert great_circle_distance(10, 20, -50, -30) == pytest.approx(
            great_circle_distance(-50, -30, 10, 20), abs=1e-12
        )

    def test_cross_seam_is_short(self):
        """经度 ±180 两侧的邻近点距离应很小 —— 大圆距离天然处理环绕。"""
        d = great_circle_distance(179.0, 0.0, -179.0, 0.0)
        assert d == pytest.approx(2.0, abs=1e-9)

    def test_nearby_points_precision(self):
        """haversine 形式在极小距离下不应退化为 0（arccos(dot) 会）。"""
        d = great_circle_distance(0.0, 0.0, 1e-5, 0.0)
        assert d == pytest.approx(1e-5, rel=1e-6)

    def test_broadcasting(self):
        d = great_circle_distance(np.array([0.0, 90.0]), 0.0, 0.0, 0.0)
        np.testing.assert_allclose(d, [0.0, 90.0], atol=1e-9)


class TestFibonacciSphere:
    def test_unit_length(self):
        vecs = fibonacci_sphere(1000)
        np.testing.assert_allclose(np.linalg.norm(vecs, axis=-1), 1.0, atol=1e-12)

    def test_centroid_near_origin(self):
        """均匀采样 → 质心接近原点。用经纬网格采样会明显偏离。"""
        vecs = fibonacci_sphere(10_000)
        assert np.linalg.norm(vecs.mean(axis=0)) < 0.02

    def test_uniform_latitude_distribution(self):
        """纬度应近似均匀分布（而非按面积集中在赤道）。

        这是 Fibonacci 采样相对经纬网格的核心优势：y 轴分层等间距，
        因此 y 分量本身近似均匀 —— 无需再做 cos φ 加权。
        """
        vecs = fibonacci_sphere(20_000)
        y = vecs[:, 1]
        # y 在 [-1, 1] 上均匀，故任一长度为 L 的 y 区间包含 L/2 比例的点。
        # |y| < 0.1 的宽度为 0.2 ⇒ 占 10%
        equatorial = np.count_nonzero(np.abs(y) < 0.1)
        # y > 0.9 的宽度为 0.1 ⇒ 占 5%
        high_lat = np.count_nonzero(y > 0.9)
        assert equatorial / len(y) == pytest.approx(0.10, abs=0.01)
        assert high_lat / len(y) == pytest.approx(0.05, abs=0.01)

    def test_areas_sum_to_sphere(self):
        _, areas = fibonacci_sphere(1000, return_areas=True)
        assert areas.sum() == pytest.approx(SPHERE_AREA_SR, rel=1e-12)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            fibonacci_sphere(0)


class TestSolidAngle:
    def test_cos_weight_at_poles(self):
        assert float(solid_angle_weights(90.0)) == pytest.approx(0.0, abs=1e-12)
        assert float(solid_angle_weights(0.0)) == pytest.approx(1.0, abs=1e-12)
        assert float(solid_angle_weights(60.0)) == pytest.approx(0.5, abs=1e-12)

    def test_erp_row_angles_sum_to_4pi(self):
        """ERP 每行立体角之和必须是整个球面 —— 这是显著性评分的归一化基础。"""
        for height in (64, 256, 1024):
            rows = erp_row_solid_angles(height)
            assert len(rows) == height
            assert rows.sum() == pytest.approx(SPHERE_AREA_SR, rel=1e-9)

    def test_erp_row_angles_peaked_at_equator(self):
        """赤道行立体角最大，极点行最小。"""
        rows = erp_row_solid_angles(180)
        assert rows[90] > rows[0]
        assert rows[90] > rows[-1]


class TestErpPixelConversion:
    @pytest.mark.parametrize("size", [(2048, 1024), (512, 256), (3840, 1920)])
    def test_round_trip(self, size: tuple[int, int]):
        width, height = size
        rng = np.random.default_rng(1)
        x = rng.uniform(0, width, 200)
        y = rng.uniform(0, height, 200)
        yaw, pitch = erp_pixel_to_sph(x, y, width, height)
        x_back, y_back = sph_to_erp_pixel(yaw, pitch, width, height)
        # x 可能因 wrap 相差一个周期，模运算后应一致
        np.testing.assert_allclose(np.mod(x_back, width), np.mod(x, width), atol=1e-9)
        np.testing.assert_allclose(y_back, y, atol=1e-9)

    def test_corners_map_to_expected_angles(self):
        """左上角像素中心应接近 (yaw=-180+Δ/2, pitch=90-Δ/2)。"""
        width, height = 360, 180
        yaw, pitch = erp_pixel_to_sph(0.0, 0.0, width, height)
        assert float(yaw) == pytest.approx(-179.5)
        assert float(pitch) == pytest.approx(89.5)

    def test_center_maps_to_origin(self):
        width, height = 360, 180
        yaw, pitch = erp_pixel_to_sph(width / 2 - 0.5, height / 2 - 0.5, width, height)
        assert float(yaw) == pytest.approx(0.0, abs=1e-9)
        assert float(pitch) == pytest.approx(0.0, abs=1e-9)
