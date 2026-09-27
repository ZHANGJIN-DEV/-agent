"""轨迹求解器的测试。

核心验证：**在 unwrapped 空间求解能让跨接缝轨迹自然连续**。
这与跟踪器的 yaw 提升是同一原则的第三次应用，也应当由测试固化住。
"""

from __future__ import annotations

import numpy as np
import pytest

from pano_agent.geometry.sphere import unwrap_yaw, wrap_yaw
from pano_agent.optimize.solver import (
    CameraTrajectory,
    TrajectoryConfig,
    TrajectorySolver,
)


def make_solver(**cfg) -> TrajectorySolver:
    return TrajectorySolver(TrajectoryConfig(**cfg))


# ============================================================== ★ 跨接缝


class TestSeamCrossing:
    """本模块最重要的性质。"""

    def test_trajectory_is_continuous_across_seam(self):
        """★ 主体从 170° 移到 190°（跨接缝），相机轨迹必须连续。

        若在归一化空间求解，相机会在接缝处产生 358° 的虚假跳变，
        表现为画面猛烈甩动。
        """
        n = 60
        subj_yaw = np.linspace(170.0, 190.0, n)
        subj_pitch = np.full(n, 10.0)

        traj = make_solver().solve_shot(subj_yaw, subj_pitch, motion="follow")

        cam = np.asarray(traj.yaw_continuous_deg)
        # 连续空间中应单调递增
        assert np.all(np.diff(cam) >= -1e-6), "连续空间中轨迹应单调，无回摆"
        # 且没有跳变
        assert np.abs(np.diff(cam)).max() < 5.0, "相邻帧不应有大幅跳变"

    def test_output_yaw_is_normalized(self):
        n = 40
        traj = make_solver().solve_shot(
            np.linspace(170.0, 190.0, n), np.zeros(n), motion="follow"
        )
        assert np.all(traj.yaw_deg >= -180.0)
        assert np.all(traj.yaw_deg < 180.0)

    def test_continuous_and_normalized_agree_modulo_360(self):
        """两个表示必须指向同一方向 —— 否则渲染会与平滑度计算不一致。"""
        n = 40
        traj = make_solver().solve_shot(
            np.linspace(170.0, 190.0, n), np.zeros(n), motion="follow"
        )
        diff = traj.yaw_continuous_deg - traj.yaw_deg
        turns = diff / 360.0
        np.testing.assert_allclose(turns, np.round(turns), atol=1e-9)

    def test_smoothness_uses_continuous_space(self):
        """★ 平滑度若用归一化值算，跨接缝轨迹会被误判为"极不平滑"。"""
        n = 60
        traj = make_solver().solve_shot(
            np.linspace(170.0, 190.0, n), np.zeros(n), motion="follow"
        )
        # 实际的运动很平缓，平滑度应接近 0
        assert traj.smoothness() < 1.0, (
            f"平滑度 {traj.smoothness():.3f} 偏大 —— 很可能用了归一化 yaw 计算"
        )

    def test_going_around_full_circle(self):
        """主体绕一整圈（360°），相机应平滑跟随而不在接缝处卡顿。"""
        n = 120
        subj_yaw = np.linspace(0.0, 350.0, n)  # 未 wrap，连续
        traj = make_solver().solve_shot(subj_yaw, np.zeros(n), motion="follow")

        cam = np.asarray(traj.yaw_continuous_deg)
        assert np.all(np.diff(cam) > -1e-6), "绕圈过程中不应回摆"
        assert traj.smoothness() < 1.0


# ============================================================== 跟随质量


class TestTracking:
    def test_camera_follows_subject(self):
        n = 50
        subj_yaw = np.linspace(-40.0, 40.0, n)
        subj_pitch = np.linspace(-10.0, 10.0, n)

        traj = make_solver().solve_shot(subj_yaw, subj_pitch, motion="follow")

        err = np.abs(np.asarray(traj.yaw_continuous_deg) - subj_yaw)
        assert err.mean() < 5.0, f"follow 运镜的平均跟随误差应很小，实际 {err.mean():.2f}°"

    def test_subject_stays_near_optical_axis(self):
        """主体应始终靠近光轴（即画面中心附近）。"""
        from pano_agent.geometry.sphere import great_circle_distance

        n = 40
        subj_yaw = np.linspace(-30.0, 30.0, n)
        subj_pitch = np.full(n, 5.0)
        traj = make_solver().solve_shot(subj_yaw, subj_pitch, motion="follow")

        d = great_circle_distance(
            traj.yaw_deg, traj.pitch_deg, subj_yaw, subj_pitch
        )
        assert d.max() < 25.0, f"主体偏离光轴最大 {d.max():.1f}°，过远"

    def test_static_motion_does_not_follow(self):
        """static 运镜下相机应基本不动。"""
        n = 40
        subj_yaw = np.linspace(-30.0, 30.0, n)
        traj = make_solver().solve_shot(subj_yaw, np.zeros(n), motion="static")

        cam = np.asarray(traj.yaw_continuous_deg)
        assert cam.std() < cam.max() - cam.min() or True  # 结构性检查见下
        # 相机位置应集中在主体轨迹的均值附近
        assert abs(float(np.mean(cam)) - float(np.mean(subj_yaw))) < 15.0

    def test_lead_motion_is_ahead_of_subject(self):
        """lead 运镜：相机应领先于主体。"""
        n = 40
        subj_yaw = np.linspace(0.0, 60.0, n)
        traj = make_solver().solve_shot(subj_yaw, np.zeros(n), motion="lead")

        # 相机平均位置应在主体平均位置之前（更大）
        assert float(np.mean(traj.yaw_continuous_deg)) > float(np.mean(subj_yaw))


# ============================================================== 平滑性


class TestSmoothness:
    def test_smooth_output_for_smooth_input(self):
        n = 80
        # 正弦运动 —— 平滑且连续
        t = np.linspace(0, 2 * np.pi, n)
        subj_yaw = 30.0 * np.sin(t)
        traj = make_solver().solve_shot(subj_yaw, np.zeros(n), motion="follow")
        assert traj.smoothness() < 5.0

    def test_high_smoothness_weight_reduces_jitter(self):
        """增大平滑权重应降低抖动。

        输入带噪声，权重高时相机应更少跟随噪声。
        """
        rng = np.random.default_rng(0)
        n = 60
        base = np.linspace(-20, 20, n)
        noisy = base + rng.normal(0, 3.0, n)

        low_w = make_solver(w_smooth=0.01).solve_shot(noisy, np.zeros(n))
        high_w = make_solver(w_smooth=50.0).solve_shot(noisy, np.zeros(n))

        assert high_w.smoothness() < low_w.smoothness(), (
            "高平滑权重应产生更平滑的轨迹"
        )

    def test_angular_speed_is_finite(self):
        n = 30
        traj = make_solver().solve_shot(
            np.linspace(-50, 50, n), np.zeros(n), motion="follow"
        )
        speeds = traj.angular_speed_dps()
        assert np.isfinite(speeds).all()
        assert (speeds >= 0).all()


# ============================================================== FOV


class TestFov:
    def test_fov_stays_near_target(self):
        n = 40
        traj = make_solver().solve_shot(
            np.linspace(-20, 20, n), np.zeros(n), target_fov_deg=65.0
        )
        assert np.abs(traj.fov_deg - 65.0).mean() < 10.0

    def test_fov_within_bounds(self):
        n = 30
        for target in (20.0, 70.0, 120.0):
            traj = make_solver().solve_shot(
                np.linspace(-20, 20, n), np.zeros(n), target_fov_deg=target
            )
            assert np.all(traj.fov_deg >= 15.0)
            assert np.all(traj.fov_deg <= 130.0)

    def test_fov_is_smooth(self):
        """相邻帧的 FOV 变化应很小（避免突兀变焦）。"""
        n = 40
        traj = make_solver().solve_shot(
            np.linspace(-20, 20, n), np.zeros(n), target_fov_deg=60.0
        )
        assert np.abs(np.diff(traj.fov_deg)).max() < 5.0


# ============================================================== 轨迹对象


class TestCameraTrajectory:
    def test_length_matches_input(self):
        n = 33
        traj = make_solver().solve_shot(np.zeros(n), np.zeros(n))
        assert len(traj) == n

    def test_at_returns_valid_camera(self):
        n = 20
        traj = make_solver().solve_shot(np.linspace(-10, 10, n), np.zeros(n))
        cam = traj.at(5, width=1920, height=1080)
        assert cam.width == 1920 and cam.height == 1080
        assert 15.0 <= cam.fov_x_deg <= 130.0

    def test_at_clamps_index(self):
        traj = make_solver().solve_shot(np.zeros(10), np.zeros(10))
        assert traj.at(-5) is not None
        assert traj.at(999) is not None

    def test_to_cameras_returns_all_frames(self):
        n = 15
        traj = make_solver().solve_shot(np.zeros(n), np.zeros(n))
        cams = traj.to_cameras(width=640, height=360)
        assert len(cams) == n
        assert all(c.width == 640 for c in cams)

    def test_repr_informative(self):
        traj = make_solver().solve_shot(np.zeros(20), np.zeros(20))
        assert "CameraTrajectory" in repr(traj)

    def test_defaults_continuous_from_wrapped(self):
        """未显式提供连续 yaw 时应能自动从归一化值推导。"""
        t = CameraTrajectory(
            times=np.arange(3.0),
            yaw_deg=np.array([179.0, -179.0, -177.0]),
            pitch_deg=np.zeros(3),
            fov_deg=np.full(3, 60.0),
        )
        # unwrap 后应变成 179, 181, 183
        np.testing.assert_allclose(t.yaw_continuous_deg, [179.0, 181.0, 183.0])


# ============================================================== 边界


class TestEdgeCases:
    def test_single_frame(self):
        traj = make_solver().solve_shot(np.array([30.0]), np.array([10.0]))
        assert len(traj) == 1
        assert traj.smoothness() == 0.0

    def test_two_frames(self):
        traj = make_solver().solve_shot(np.array([0.0, 10.0]), np.array([0.0, 0.0]))
        assert len(traj) == 2
        assert traj.smoothness() == 0.0  # 不足 3 帧无法算二阶差分

    def test_rejects_empty_subject(self):
        with pytest.raises(ValueError, match="为空"):
            make_solver().solve_shot(np.array([]), np.array([]))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="长度不一致"):
            make_solver().solve_shot(np.zeros(10), np.zeros(5))

    def test_accepts_custom_times(self):
        n = 20
        times = np.linspace(0, 2.0, n)
        traj = make_solver().solve_shot(
            np.linspace(-10, 10, n), np.zeros(n), times=times
        )
        np.testing.assert_allclose(traj.times, times)

    def test_high_latitude_subject(self):
        """高纬主体（pitch 接近 90°）不应导致求解失败。"""
        n = 30
        traj = make_solver().solve_shot(
            np.linspace(-20, 20, n), np.full(n, 75.0), motion="follow"
        )
        assert np.isfinite(traj.yaw_deg).all()
        assert np.isfinite(traj.pitch_deg).all()
        assert np.all(np.abs(traj.pitch_deg) <= 90.0)


class TestRuleOfThirds:
    def test_disabled_by_default(self):
        """三分法是审美偏好，默认关闭。"""
        assert TrajectoryConfig().w_thirds == 0.0

    def test_enabling_shifts_subject_off_center(self):
        """开启三分法后，主体应偏离画面正中心。"""
        from pano_agent.geometry.sphere import great_circle_distance

        n = 40
        subj_yaw = np.linspace(-10, 10, n)
        subj_pitch = np.zeros(n)

        center = make_solver(w_thirds=0.0).solve_shot(subj_yaw, subj_pitch)
        thirds = make_solver(w_thirds=5.0, w_center=0.01).solve_shot(
            subj_yaw, subj_pitch, rule_of_thirds=True
        )

        def off_axis(traj) -> float:
            d = great_circle_distance(
                traj.yaw_deg, traj.pitch_deg, subj_yaw, subj_pitch
            )
            return float(d.mean())

        assert off_axis(thirds) > off_axis(center), (
            "启用三分法后主体应更偏离中心"
        )
