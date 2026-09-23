"""虚拟相机模型的单元测试。

相机层是「球面 ↔ 图像平面」的桥梁，pixel_rays 与 project 必须严格互逆，
否则构图约束（三分法、主体居中）计算出的目标位置会与渲染结果不符。
"""

from __future__ import annotations

import numpy as np
import pytest

from pano_agent.geometry.camera import VirtualCamera, rotation_x, rotation_y, rotation_z
from pano_agent.geometry.sphere import vec_to_sph


class TestRotationMatrices:
    def test_are_orthonormal(self):
        for rot in (rotation_x(37.0), rotation_y(-112.0), rotation_z(88.0)):
            np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
            assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-12)

    def test_identity_at_zero(self):
        for rot in (rotation_x(0.0), rotation_y(0.0), rotation_z(0.0)):
            np.testing.assert_allclose(rot, np.eye(3), atol=1e-12)


class TestCameraGeometry:
    def test_rotation_sends_axis_to_expected_direction(self):
        """相机旋转矩阵作用在局部光轴 (0,0,1) 上，必须得到 sph_to_vec(yaw,pitch)。"""
        for yaw, pitch in [(0, 0), (90, 0), (-45, 30), (179, -60)]:
            cam = VirtualCamera(yaw, pitch, 0.0, 90.0, 64, 64)
            axis = cam.rotation_matrix() @ np.array([0.0, 0.0, 1.0])
            back_yaw, back_pitch = vec_to_sph(axis)
            assert float(back_yaw) == pytest.approx(yaw, abs=1e-9)
            assert float(back_pitch) == pytest.approx(pitch, abs=1e-9)

    def test_focal_length_from_fov(self):
        """端点采样约定下 f = (W-1)/(2·tan(fov/2))，不是 W/(2·tan(fov/2))。"""
        cam = VirtualCamera(fov_x_deg=90.0, width=100, height=50)
        assert cam.focal_px == pytest.approx(99.0 / 2.0)

    def test_fov_y_is_not_linear_scaling(self):
        """垂直 fov 不是水平 fov 按宽高比的线性缩放 —— 透视投影是非线性的。"""
        cam = VirtualCamera(fov_x_deg=90.0, width=1920, height=1080)
        linear_wrong = cam.fov_x_deg / cam.aspect
        assert cam.fov_y_deg != pytest.approx(linear_wrong, rel=1e-3)
        # 端点采样约定：方形像素要求 half_extent 按 (N-1) 等比缩放
        expected = np.degrees(
            2 * np.arctan(np.tan(np.radians(90.0) / 2) * (cam.height - 1) / (cam.width - 1))
        )
        assert cam.fov_y_deg == pytest.approx(float(expected))

    def test_pixel_spacing_is_square(self):
        """两方向的切平面像素间距必须相等，否则画面会各向异性拉伸。"""
        cam = VirtualCamera(fov_x_deg=75.0, width=1920, height=1080)
        dx = 2 * cam.half_extent_x / (cam.width - 1)
        dy = 2 * cam.half_extent_y / (cam.height - 1)
        assert dx == pytest.approx(dy, rel=1e-12)

    def test_pixel_spacing_uniform_in_tangent_plane(self):
        """**切平面坐标**等距 —— 这是端点采样的直接结果。

        注意不能拿 yaw 本身去测均匀性：切平面横坐标 x 等距时，
        yaw = atan(x) 是**非线性**的，步长必然随位置变化。
        把它误当成 bug 是透视投影中最常见的误解之一。
        """
        cam = VirtualCamera(0, 0, 0, 90.0, 129, 73)
        rays = cam.pixel_rays()
        tx = rays[36, :, 0] / rays[36, :, 2]  # 反投影回切平面横坐标
        steps = np.diff(tx)
        assert steps.std() < 1e-12, f"切平面步长不均匀，std={steps.std():.2e}"

        # 反过来确认 yaw 确实是非均匀的（证明上面的区分有必要）
        yaw_row, _ = vec_to_sph(rays[36, :])
        yaw_steps = np.abs(np.diff(yaw_row))
        assert yaw_steps.std() > 1e-3

    def test_image_edges_match_fov_boundary(self):
        """端点采样约定：第 0 列与最后一列应精确对应 ±fov_x/2。"""
        cam = VirtualCamera(0, 0, 0, 60.0, 101, 57)
        rays = cam.pixel_rays()
        from pano_agent.geometry.sphere import great_circle_distance

        d_left = great_circle_distance(
            *np.array([vec_to_sph(rays[28, 0])]).flatten(), 0.0, 0.0
        )
        assert float(d_left) == pytest.approx(30.0, abs=1e-6)

    def test_rejects_invalid_fov(self):
        with pytest.raises(ValueError):
            VirtualCamera(fov_x_deg=0.0)
        with pytest.raises(ValueError):
            VirtualCamera(fov_x_deg=200.0)


class TestPixelRays:
    def test_are_unit_length(self):
        cam = VirtualCamera(30, 20, 5, 90, 64, 48)
        rays = cam.pixel_rays()
        assert rays.shape == (48, 64, 3)
        np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1.0, atol=1e-12)

    def test_center_pixel_is_optical_axis_odd_size(self):
        """奇数尺寸时中心像素精确存在，其光线必须等于光轴方向。"""
        cam = VirtualCamera(yaw_deg=35, pitch_deg=20, roll_deg=0, fov_x_deg=90, width=65, height=37)
        rays = cam.pixel_rays()
        center = rays[18, 32]  # (H-1)/2, (W-1)/2
        yaw, pitch = vec_to_sph(center)
        assert float(yaw) == pytest.approx(35.0, abs=1e-9)
        assert float(pitch) == pytest.approx(20.0, abs=1e-9)

    def test_roll_rotates_image_not_axis(self):
        """roll 不应改变光轴方向，只旋转画面内容。"""
        plain = VirtualCamera(10, 10, 0.0, 90, 65, 37)
        rolled = VirtualCamera(10, 10, 45.0, 90, 65, 37)
        np.testing.assert_allclose(
            plain.pixel_rays()[18, 32], rolled.pixel_rays()[18, 32], atol=1e-12
        )

    def test_x_increases_with_yaw(self):
        """yaw 增大时，画面右侧内容应来自更大的经度。"""
        cam = VirtualCamera(0, 0, 0, 90, 65, 37)
        rays = cam.pixel_rays()
        yaw_left, _ = vec_to_sph(rays[18, 0])
        yaw_right, _ = vec_to_sph(rays[18, 64])
        assert float(yaw_right) > float(yaw_left)


class TestProject:
    def test_round_trip_with_pixel_rays(self):
        """pixel_rays 与 project 必须互逆 —— 往返误差应远小于一个像素。"""
        cam = VirtualCamera(25, -15, 12, 80, 96, 54)
        rays = cam.pixel_rays()
        yaw, pitch = vec_to_sph(rays)

        x_px, y_px, visible = cam.project(yaw, pitch)

        rows, cols = np.mgrid[0 : cam.height, 0 : cam.width]
        np.testing.assert_allclose(x_px[visible], cols[visible], atol=1e-6)
        np.testing.assert_allclose(y_px[visible], rows[visible], atol=1e-6)
        assert visible.all(), "画面内的光线不应被判为不可见"

    def test_points_behind_camera_are_not_visible(self):
        """相机背后的点经透视除法后会得到看似合理的坐标 —— 必须被 visible 拦下。"""
        cam = VirtualCamera(0, 0, 0, 90, 64, 64)
        # 正后方
        _, _, visible = cam.project(180.0, 0.0)
        assert not bool(visible)

    def test_points_outside_frustum_not_visible(self):
        cam = VirtualCamera(0, 0, 0, 60, 64, 64)
        # 光轴正前方可见
        _, _, vis_center = cam.project(0.0, 0.0)
        assert bool(vis_center)
        # 侧向 90° 超出 60° 视场
        _, _, vis_side = cam.project(90.0, 0.0)
        assert not bool(vis_side)

    def test_angular_radius_brackets_corners(self):
        cam = VirtualCamera(0, 0, 0, 90, 1920, 1080)
        radius = cam.angular_radius_deg()
        # 四角方向与光轴的夹角应接近但不超过外接角半径
        yaw_corner, pitch_corner = cam.pixel_angles()
        from pano_agent.geometry.sphere import great_circle_distance

        d = great_circle_distance(yaw_corner[0, 0], pitch_corner[0, 0], 0.0, 0.0)
        assert float(d) <= radius + 1e-6
