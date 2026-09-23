"""球面跟踪器的单元测试。

测试重点完全围绕**跨 ERP 接缝的 ID 保持** —— 这是本模块存在的理由。
朴素跟踪器在普通视频上工作得很好，只在全景视频的接缝处失败；
因此如果测试不专门覆盖这个场景，缺陷会被完全掩盖。
"""

from __future__ import annotations

import numpy as np
import pytest

from pano_agent.geometry.sphere import great_circle_distance, wrap_yaw
from pano_agent.perception.track_spherical import SphericalTracker, lift_yaw
from pano_agent.perception.types import SphericalDetection


def make_detection(
    yaw: float, pitch: float = 0.0, *, w: float = 10.0, h: float = 10.0, conf: float = 0.9
) -> SphericalDetection:
    return SphericalDetection(
        yaw_deg=yaw, pitch_deg=pitch, width_deg=w, height_deg=h, confidence=conf
    )


# ============================================================== lift_yaw


class TestLiftYaw:
    """核心操作的正确性 —— 整个模块建立在这三行之上。"""

    def test_lift_across_seam(self):
        assert lift_yaw(-179.0, 179.0) == pytest.approx(181.0)
        assert lift_yaw(179.0, -179.0) == pytest.approx(-181.0)

    def test_identity_when_already_close(self):
        assert lift_yaw(10.0, 12.0) == pytest.approx(10.0)
        assert lift_yaw(0.0, 0.0) == pytest.approx(0.0)

    def test_result_is_equivalent_angle(self):
        """提升只能改变周期代表元，不能改变实际方向。"""
        rng = np.random.default_rng(0)
        for _ in range(200):
            obs = float(rng.uniform(-180, 180))
            ref = float(rng.uniform(-720, 720))
            lifted = lift_yaw(obs, ref)
            # 与观测的差必须是 360 的整数倍。
            # 注意不能用 (lifted-obs) % 360 == 0 —— 浮点下可能得到 359.999…，
            # 因此改为检验「圈数是否为整数」。
            turns = (lifted - obs) / 360.0
            assert abs(turns - round(turns)) < 1e-9
            # 且折叠后与观测相同
            assert wrap_yaw(lifted) == pytest.approx(wrap_yaw(obs), abs=1e-9)

    def test_is_nearest_representative(self):
        """提升结果必须是所有等价角度中离参考最近的那个。"""
        rng = np.random.default_rng(1)
        for _ in range(200):
            obs = float(rng.uniform(-180, 180))
            ref = float(rng.uniform(-720, 720))
            lifted = lift_yaw(obs, ref)
            alternatives = [lifted + 360 * k for k in (-2, -1, 1, 2)]
            assert abs(lifted - ref) <= min(abs(a - ref) for a in alternatives) + 1e-9

    def test_multi_turn_reference(self):
        """参考值可能已经连续转动多圈（长视频），提升仍应正确。"""
        assert lift_yaw(5.0, 725.0) == pytest.approx(725.0)
        assert lift_yaw(-5.0, -725.0) == pytest.approx(-725.0)


# ============================================================== 基础跟踪


class TestBasicTracking:
    def test_single_target_maintains_id(self):
        tracker = SphericalTracker(dt=1 / 30, min_hits=1, gate_deg=30)
        ids = []
        for i in range(20):
            tracks = tracker.update([make_detection(yaw=i * 2.0)])
            assert len(tracks) == 1
            ids.append(tracks[0].track_id)
        assert len(set(ids)) == 1, f"ID 应保持稳定，实际 {ids}"

    def test_velocity_is_estimated(self):
        """匀速运动下速度估计应收敛到真值附近。"""
        tracker = SphericalTracker(dt=1.0, min_hits=1, gate_deg=30, accel_std=50)
        for i in range(30):
            tracker.update([make_detection(yaw=i * 5.0)])  # 5°/帧 = 5°/s
        tracks = tracker.update([make_detection(yaw=150.0)])
        assert tracks[0].v_yaw_dps == pytest.approx(5.0, abs=1.0)

    def test_track_removed_after_max_age(self):
        tracker = SphericalTracker(dt=1 / 30, max_age=5, min_hits=1)
        tracker.update([make_detection(yaw=0.0)])
        for _ in range(6):
            tracks = tracker.update([])
        assert len(tracks) == 0, "超过 max_age 后轨迹应被删除"

    def test_track_survives_brief_occlusion(self):
        """短暂丢失（如被遮挡）不应立即丢失 ID。"""
        tracker = SphericalTracker(dt=1 / 30, max_age=10, min_hits=1, gate_deg=30)
        first = tracker.update([make_detection(yaw=0.0)])
        tid = first[0].track_id

        for _ in range(3):  # 丢失 3 帧
            tracker.update([])

        tracks = tracker.update([make_detection(yaw=1.0)])
        assert tracks[0].track_id == tid, "短暂遮挡后应恢复同一 ID"

    def test_min_hits_gates_confirmation(self):
        tracker = SphericalTracker(dt=1 / 30, min_hits=3, gate_deg=30)
        t1 = tracker.update([make_detection(yaw=0.0)])
        assert not t1[0].is_confirmed, "首次出现的轨迹不应被确认"

        tracker.update([make_detection(yaw=0.5)])
        t3 = tracker.update([make_detection(yaw=1.0)])
        assert t3[0].is_confirmed, "累计 3 次匹配后应被确认"


# ============================================================== ★ 跨接缝


class TestSeamCrossing:
    """本模块存在的理由。"""

    def test_id_preserved_across_seam(self):
        """★ 核心测试：目标从 +179° 绕到 -179°，ID 必须保持。

        朴素跟踪器在此处必然失败：它看到的 yaw 从 +179 跳到 -179，
        像素距离达半个图像宽度，远超任何合理的关联门限。
        """
        tracker = SphericalTracker(dt=1 / 30, min_hits=1, gate_deg=30)
        yaws = [170.0, 175.0, 179.0, -179.0, -175.0, -170.0, -165.0]

        ids = []
        for y in yaws:
            tracks = tracker.update([make_detection(yaw=y)])
            assert len(tracks) == 1, f"yaw={y} 时轨迹数异常"
            ids.append(tracks[0].track_id)

        assert len(set(ids)) == 1, (
            f"跨接缝时 ID 丢失！序列 {yaws} 得到 IDs {ids}"
        )

    def test_naive_pixel_distance_would_reject(self):
        """对照组：证明朴素做法确实会失败，而非我们在解决伪问题。

        朴素做法直接拿 wrap 后的 yaw 相减当距离。在接缝处会得到
        约 358° 的"距离"，远超门限。
        """
        naive_distance = abs(wrap_yaw(-179.0) - wrap_yaw(179.0))
        assert naive_distance > 300.0, "wrap 后直接相减应得到接近 360 的假距离"

        # 而正确的大圆距离极小
        true_distance = great_circle_distance(179.0, 0.0, -179.0, 0.0)
        assert true_distance == pytest.approx(2.0, abs=1e-6)

        # 两者相差两个数量级 —— 这就是朴素做法失败的量化原因
        assert naive_distance / true_distance > 100

    def test_crosses_seam_flag_is_set(self):
        tracker = SphericalTracker(dt=1 / 30, min_hits=1, gate_deg=30)
        for y in [175.0, 179.0, -179.0, -175.0]:
            tracks = tracker.update([make_detection(yaw=y)])

        assert tracks[0].crosses_seam, "跨接缝标志应被置位"
        assert tracker.total_seam_crossings == 1, "应累计一次跨接缝"

    def test_no_flag_when_not_crossing(self):
        tracker = SphericalTracker(dt=1 / 30, min_hits=1, gate_deg=30)
        for y in [0.0, 5.0, 10.0, 15.0]:
            tracks = tracker.update([make_detection(yaw=y)])
        assert not tracks[0].crosses_seam

    def test_multiple_crossings(self):
        """连续绕圈（如无人机环绕飞行）应稳定保持 ID。"""
        tracker = SphericalTracker(dt=1 / 30, min_hits=1, gate_deg=40)
        ids = []
        for k in range(200):
            yaw = wrap_yaw(k * 10.0)  # 每帧转 10°，绕 5 圈多
            tracks = tracker.update([make_detection(yaw=yaw)])
            if tracks:
                ids.append(tracks[0].track_id)

        assert len(set(ids)) == 1, f"连续绕圈时 ID 丢失，出现 {len(set(ids))} 个 ID"
        assert tracker.tracks[0].crosses_seam

    def test_wrapped_output_is_in_range(self):
        """对外输出的 yaw 必须归一化回 [-180, 180)。"""
        tracker = SphericalTracker(dt=1 / 30, min_hits=1, gate_deg=30)
        for y in [175.0, 179.0, -179.0, -175.0]:
            tracks = tracker.update([make_detection(yaw=y)])
        for t in tracks:
            assert -180.0 <= t.yaw_deg < 180.0


# ============================================================== 关联


class TestAssociation:
    def test_two_targets_keep_separate_ids(self):
        tracker = SphericalTracker(dt=1 / 30, min_hits=1, gate_deg=30)
        for _ in range(10):
            tracks = tracker.update(
                [make_detection(yaw=-60.0), make_detection(yaw=60.0)]
            )
        assert len(tracks) == 2
        assert len({t.track_id for t in tracks}) == 2

    def test_two_objects_crossing_paths(self):
        """两个目标在 yaw 上交错，但在 pitch 上可分 —— ID 不应互换。

        注意：**如果两个目标在球面上完全重合，任何跟踪器都无法区分它们**，
        这不是缺陷而是观测的固有歧义。因此这里给它们不同的 pitch，
        测试的是「关联是否用上了完整的球面信息」而非要求跟踪器
        完成物理上不可能的事。
        """
        tracker = SphericalTracker(dt=1 / 30, min_hits=1, gate_deg=25)
        a_id = b_id = None

        for i in range(21):
            yaw_a, pitch_a = -30.0 + i * 3.0, 25.0    # 从左向右，在正 pitch
            yaw_b, pitch_b = 30.0 - i * 3.0, -25.0    # 从右向左，在负 pitch

            tracks = tracker.update(
                [
                    make_detection(yaw=yaw_a, pitch=pitch_a),
                    make_detection(yaw=yaw_b, pitch=pitch_b),
                ]
            )

            if i == 0:
                by_pitch = sorted(tracks, key=lambda t: t.pitch_deg)
                b_id, a_id = by_pitch[0].track_id, by_pitch[1].track_id
            else:
                by_id = {t.track_id: t for t in tracks}
                assert by_id[a_id].pitch_deg > 0, f"第 {i} 帧 A 跑到了负 pitch"
                assert by_id[b_id].pitch_deg < 0, f"第 {i} 帧 B 跑到了正 pitch"

    def test_gate_rejects_distant_detection(self):
        """超出关联门限的检测应被视为新目标，而非错误关联到旧轨迹。"""
        tracker = SphericalTracker(dt=1 / 30, min_hits=1, gate_deg=10)
        first = tracker.update([make_detection(yaw=0.0)])
        old_id = first[0].track_id

        # 突然出现在 90° 外 —— 不可能是同一个目标
        tracks = tracker.update([make_detection(yaw=90.0)])

        # 用 time_since_update 判断"本帧被匹配"，而不是 hits ——
        # 未被匹配的旧轨迹 hits 仍然是 1，用 hits 过滤会把它也算进来
        matched = {t.track_id for t in tracks if t.time_since_update == 0}
        assert old_id not in matched, "旧轨迹不应匹配到 90° 外的检测"
        assert len(matched) == 1, "应当只有新检测被匹配"
        # 旧轨迹此时应处于"丢失但未超龄"的状态，仍在列表中
        assert old_id in {t.track_id for t in tracks}

    def test_high_latitude_tracking(self):
        """高纬处跟踪仍应正常 —— 这正是用球面角度而非像素的价值。"""
        tracker = SphericalTracker(dt=1 / 30, min_hits=1, gate_deg=30)
        ids = []
        for i in range(15):
            tracks = tracker.update([make_detection(yaw=i * 3.0, pitch=80.0)])
            ids.append(tracks[0].track_id)
        assert len(set(ids)) == 1


# ============================================================== 复位


class TestReset:
    def test_reset_clears_everything(self):
        tracker = SphericalTracker(dt=1 / 30, min_hits=1, gate_deg=30)
        tracker.update([make_detection(yaw=0.0)])
        assert len(tracker.tracks) == 1

        tracker.reset()
        assert len(tracker.tracks) == 0
        assert tracker.frame_index == 0
        assert tracker.total_seam_crossings == 0

    def test_ids_restart_after_reset(self):
        tracker = SphericalTracker(dt=1 / 30, min_hits=1, gate_deg=30)
        tracker.update([make_detection(yaw=0.0)])
        tracker.reset()
        tracks = tracker.update([make_detection(yaw=0.0)])
        assert tracks[0].track_id == 1


class TestValidation:
    def test_rejects_bad_parameters(self):
        with pytest.raises(ValueError):
            SphericalTracker(max_age=-1)
        with pytest.raises(ValueError):
            SphericalTracker(min_hits=0)
        with pytest.raises(ValueError):
            SphericalTracker(gate_deg=0.0)

    def test_empty_detections_is_safe(self):
        tracker = SphericalTracker()
        assert tracker.update([]) == []
