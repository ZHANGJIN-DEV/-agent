"""镜头边界排除的测试。

==============================================================================
为什么需要这组测试
==============================================================================

一段成片由多个**独立求解**的镜头拼接而成，镜头之间相机会跳跃 ——
这是剪辑的正常行为（"切"本来就是不连续的）。

但早期版本把整条拼接轨迹当连续轨迹算平滑度与角速度，导致：

    实测：峰值 2195°/s，而配置上限只有 45°/s —— 超限 49 倍
    而各镜头内部的实际峰值是 46°/s，刚好贴着上限

这个 49 倍的差距不是系统的问题，是**指标算错了**。若不加区分地
把它写进报告，会得出"系统完全失控"的错误结论。

因此这组测试守住一条性质：**切镜处的跳变不得计入平滑度与角速度**。
"""

from __future__ import annotations

import numpy as np
import pytest

from pano_agent.eval import (
    angular_velocity_compliance,
    evaluate_trajectory,
    trajectory_smoothness,
)


def make_two_shot_trajectory(
    *,
    n_per_shot: int = 20,
    fps: float = 15.0,
    jump_deg: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造一段由两个镜头组成的轨迹。

    镜头 A：yaw 从 0 平滑走到 20
    镜头 B：yaw 从 120 平滑走到 140（与 A 之间跳 100° —— 模拟切镜）

    Returns:
        (yaw, pitch, times)
    """
    t = np.arange(2 * n_per_shot, dtype=np.float64) / fps
    yaw = np.concatenate([
        np.linspace(0.0, 20.0, n_per_shot),
        np.linspace(jump_deg, jump_deg + 20.0, n_per_shot),
    ])
    pitch = np.zeros(2 * n_per_shot)
    return yaw, pitch, t


# ============================================================== 角速度


class TestAngularVelocityCompliance:
    def test_cut_jump_inflates_peak_without_cut_frames(self):
        """★ 不传 cut_frames 时，切镜的跳变会被算成剧烈运动。

        这是"缺陷复现"测试 —— 它记录的是**错误行为的样貌**，
        以便说明为什么必须传 cut_frames。
        """
        yaw, pitch, t = make_two_shot_trajectory()
        _, peak = angular_velocity_compliance(yaw, pitch, t, max_dps=45.0)

        # 跳 100°，dt = 1/15 s → 约 1500°/s
        assert peak > 1000.0, f"切镜跳变应产生极大的虚假峰值，实际 {peak:.0f}"

    def test_excluding_cut_restores_sane_peak(self):
        """★ 传入 cut_frames 后峰值应恢复正常。

        各镜头内部是 20°/19 帧的平滑运动 ≈ 16°/s。
        """
        yaw, pitch, t = make_two_shot_trajectory()
        _, peak = angular_velocity_compliance(
            yaw, pitch, t, max_dps=45.0, cut_frames=[20]
        )

        assert peak < 45.0, f"排除切镜后峰值应正常，实际 {peak:.1f}°/s"
        assert peak > 5.0, "不应把正常运动也算掉"

    def test_compliance_rate_improves(self):
        """排除切镜后合规率应提升。"""
        yaw, pitch, t = make_two_shot_trajectory()

        rate_without, _ = angular_velocity_compliance(yaw, pitch, t, max_dps=45.0)
        rate_with, _ = angular_velocity_compliance(
            yaw, pitch, t, max_dps=45.0, cut_frames=[20]
        )

        assert rate_with > rate_without

    def test_all_excluded_returns_neutral(self):
        """若所有帧都是边界（退化情形），不应崩溃。"""
        yaw = np.array([0.0, 100.0])
        rate, peak = angular_velocity_compliance(
            yaw, np.zeros(2), np.array([0.0, 1.0]), cut_frames=[0, 1]
        )
        assert rate == 1.0

    def test_cut_frames_none_matches_legacy_behavior(self):
        """不传 cut_frames 时行为与旧版一致（向后兼容）。"""
        yaw = np.array([0.0, 10.0, 20.0])
        t = np.array([0.0, 1.0, 2.0])
        rate_a, peak_a = angular_velocity_compliance(yaw, np.zeros(3), t)
        rate_b, peak_b = angular_velocity_compliance(
            yaw, np.zeros(3), t, cut_frames=None
        )
        assert rate_a == rate_b
        assert peak_a == peak_b


# ============================================================== 平滑度


class TestSmoothnessWithCuts:
    def test_cut_inflates_smoothness(self):
        """切镜会让平滑度爆表。"""
        yaw, pitch, t = make_two_shot_trajectory()
        s = trajectory_smoothness(yaw, pitch, t)
        assert s > 1000.0, f"切镜应让平滑度爆表，实际 {s:.0f}"

    def test_excluding_cut_gives_small_smoothness(self):
        """★ 排除后应是正常量级。

        镜头内部是线性运动，二阶差分接近 0，故平滑度应很小。
        """
        yaw, pitch, t = make_two_shot_trajectory()
        s = trajectory_smoothness(yaw, pitch, t, cut_frames=[20])
        assert s < 10.0, f"排除切镜后平滑度应很小，实际 {s:.1f}"

    def test_magnitude_ratio_is_dramatic(self):
        """两种算法的差距应有数量级 —— 说明这个修正是必要的而非微调。"""
        yaw, pitch, t = make_two_shot_trajectory()
        raw = trajectory_smoothness(yaw, pitch, t)
        fixed = trajectory_smoothness(yaw, pitch, t, cut_frames=[20])
        assert raw / max(fixed, 1e-6) > 100, (
            f"差距应超过两个数量级，实际 {raw:.0f} vs {fixed:.1f}"
        )


# ============================================================== 集成


class TestEvaluateTrajectoryWithCuts:
    def test_end_to_end_with_cut_frames(self):
        yaw, pitch, t = make_two_shot_trajectory()
        m = evaluate_trajectory(
            yaw, pitch, np.full(len(yaw), 70.0), t,
            yaw, pitch,
            cut_frames=[20],
        )
        assert m.peak_angular_velocity_dps < 45.0
        assert m.smoothness < 10.0
        assert m.num_frames == len(yaw)

    def test_without_cut_frames_shows_problem(self):
        """不传时指标会很难看 —— 这正是需要被修正的现象。"""
        yaw, pitch, t = make_two_shot_trajectory()
        m = evaluate_trajectory(
            yaw, pitch, np.full(len(yaw), 70.0), t,
            yaw, pitch,
        )
        assert m.peak_angular_velocity_dps > 1000.0
        assert m.smoothness > 1000.0

    def test_single_shot_needs_no_cut_frames(self):
        """单镜头无需传 cut_frames，行为应正常。"""
        t = np.arange(30, dtype=np.float64) / 15.0
        yaw = np.sin(t * 2) * 10.0
        pitch = np.zeros(30)

        m = evaluate_trajectory(
            yaw, pitch, np.full(30, 70.0), t, yaw, pitch
        )
        assert m.smoothness < 100.0
        assert np.isfinite(m.peak_angular_velocity_dps)

    def test_multiple_cuts(self):
        """三个镜头、两处切镜。"""
        yaw = np.concatenate([
            np.linspace(0, 10, 15),
            np.linspace(100, 110, 15),
            np.linspace(200, 210, 15),
        ])
        t = np.arange(45, dtype=np.float64) / 15.0
        pitch = np.zeros(45)

        m = evaluate_trajectory(
            yaw, pitch, np.full(45, 70.0), t, yaw, pitch,
            cut_frames=[15, 30],
        )
        assert m.peak_angular_velocity_dps < 45.0


# ============================================================== 回归说明


class TestRegressionDocumentation:
    def test_documents_the_real_magnitude(self):
        """把这组测试背后的真实数据固化成断言。

        好让日后有人质疑"为什么要这么麻烦"时，能一眼看到差距有多大。
        """
        # 复现真实场景的量级：三个镜头、每个 10 帧、帧间跳 ~100°
        yaws = [np.linspace(0, 10, 10), np.linspace(120, 130, 10),
                np.linspace(250, 260, 10)]
        yaw = np.concatenate(yaws)
        t = np.arange(30, dtype=np.float64) / 15.0
        pitch = np.zeros(30)

        raw_s = trajectory_smoothness(yaw, pitch, t)
        fixed_s = trajectory_smoothness(yaw, pitch, t, cut_frames=[10, 20])

        # 两者差距应在 2 个数量级以上
        assert raw_s > 100 * max(fixed_s, 1e-6), (
            f"真实差距应很显著：{raw_s:.0f} vs {fixed_s:.1f}"
        )
