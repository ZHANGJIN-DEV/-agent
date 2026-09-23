"""PSG 构建的测试 —— 从逐帧感知输出聚合成场景图。

这些测试用**合成数据**（不需要 torch、不需要真实视频）验证聚合逻辑。
`FrameTracks` / `TrackState` 是纯数据结构，因此构建链路可以独立于
感知模型开发，等模型就绪再切真实输出。
"""

from __future__ import annotations

import numpy as np
import pytest

from pano_agent.perception.types import FrameTracks, TrackState
from pano_agent.sir import (
    BuildConfig,
    PanoramaSceneGraph,
    VideoMeta,
    build_psg,
    load_numeric_layer,
)


def make_state(
    track_id: int,
    yaw: float,
    pitch: float = 0.0,
    *,
    w: float = 10.0,
    h: float = 10.0,
    crosses_seam: bool = False,
    confirmed: bool = True,
) -> TrackState:
    return TrackState(
        track_id=track_id,
        yaw_deg=yaw,
        pitch_deg=pitch,
        width_deg=w,
        height_deg=h,
        confidence=0.9,
        category="person",
        age=10,
        hits=10,
        time_since_update=0,
        is_confirmed=confirmed,
        crosses_seam=crosses_seam,
    )


def make_frames(
    track_specs: dict[int, list[tuple[float, float]]],
    *,
    fps: float = 30.0,
    confirmed: bool = True,
) -> list[FrameTracks]:
    """构造逐帧轨迹。

    Args:
        track_specs: {track_id: [(yaw, pitch), ...]} —— 每条轨迹的逐帧角度。
    """
    n_frames = max(len(v) for v in track_specs.values())
    frames = []
    for i in range(n_frames):
        states = []
        for tid, seq in track_specs.items():
            if i < len(seq):
                yaw, pitch = seq[i]
                states.append(
                    make_state(tid, yaw, pitch, confirmed=confirmed)
                )
        frames.append(FrameTracks(frame_index=i, time_s=i / fps, tracks=states))
    return frames


def default_meta(duration_s: float = 5.0) -> VideoMeta:
    return VideoMeta(fps=30.0, duration_s=duration_s, erp_width=2048, erp_height=1024)


# ============================================================== 聚合


class TestAggregation:
    def test_single_track_becomes_one_node(self):
        frames = make_frames({1: [(float(i), 0.0) for i in range(30)]})
        psg = build_psg(frames, default_meta())

        assert len(psg.tracks) == 1
        assert 1 in psg.tracks

    def test_multiple_tracks_kept_separate(self):
        frames = make_frames(
            {
                1: [(0.0, 0.0)] * 20,
                2: [(90.0, 0.0)] * 20,
                3: [(-90.0, 0.0)] * 20,
            }
        )
        psg = build_psg(frames, default_meta())
        assert set(psg.tracks) == {1, 2, 3}

    def test_short_tracks_filtered(self):
        """过短的轨迹应被丢弃 —— 抑制检测抖动产生的碎片。"""
        frames = make_frames(
            {
                1: [(0.0, 0.0)] * 30,   # 长轨迹，保留
                2: [(90.0, 0.0)] * 3,   # 太短，丢弃
            }
        )
        psg = build_psg(frames, default_meta(), config=BuildConfig(min_track_frames=5))
        assert set(psg.tracks) == {1}

    def test_unconfirmed_tracks_filtered(self):
        frames = make_frames({1: [(float(i), 0.0) for i in range(20)]}, confirmed=False)
        psg = build_psg(frames, default_meta(), config=BuildConfig(only_confirmed=True))
        assert len(psg.tracks) == 0

    def test_no_frames_gives_empty_psg(self):
        psg = build_psg([], default_meta())
        assert len(psg.tracks) == 0
        assert psg.events == []

    def test_frame_range_recorded(self):
        frames = make_frames({1: [(float(i), 0.0) for i in range(10, 40)]})
        psg = build_psg(frames, default_meta())

        # 该轨迹出现在第 0..29 帧（每帧一个状态）
        node = psg.tracks[1]
        assert node.stats.frame_range[0] == 0


# ============================================================== 统计量


class TestStats:
    def test_duration_computed_from_fps(self):
        frames = make_frames({1: [(0.0, 0.0)] * 31})  # 31 帧
        psg = build_psg(frames, default_meta())
        assert psg.tracks[1].stats.duration_s == pytest.approx(1.0, abs=0.05)

    def test_angular_velocity_measured_in_degrees_per_second(self):
        """★ 角速度必须是球面角度/秒，不是像素/秒。

        目标每帧转 2°、fps=30 ⇒ 60°/秒。
        """
        frames = make_frames(  # 注意：这里只造 3 帧，确保速度可解析
            {1: [(float(i * 2), 0.0) for i in range(3)]}
        )
        psg = build_psg(frames, default_meta(), config=BuildConfig(min_track_frames=3))
        assert psg.tracks[1].stats.angular_velocity_mean_dps == pytest.approx(
            60.0, rel=1e-6
        )

    def test_latitude_span_recorded(self):
        frames = make_frames(
            {1: [(0.0, float(p)) for p in (-30, -10, 10, 30, 50)]}
        )
        psg = build_psg(frames, default_meta())
        lo, hi = psg.tracks[1].stats.latitude_span_deg
        assert lo == pytest.approx(-30.0)
        assert hi == pytest.approx(50.0)

    def test_pole_proximity(self):
        frames = make_frames({1: [(0.0, 70.0)] * 10})
        psg = build_psg(frames, default_meta())
        assert psg.tracks[1].stats.pole_proximity_min_deg == pytest.approx(20.0)

    def test_crosses_seam_flag_propagates(self):
        frames = [
            FrameTracks(
                frame_index=i,
                time_s=i / 30.0,
                tracks=[make_state(1, float(i), crosses_seam=True)],
            )
            for i in range(10)
        ]
        psg = build_psg(frames, default_meta())
        assert psg.tracks[1].stats.crosses_seam is True
        assert "crosses_seam" in psg.tracks[1].attributes


# ============================================================== 高光分数


class TestHighlightScore:
    def test_fast_target_scores_higher_than_static(self):
        """运动快的目标高光分数应更高 —— 这是启发式的核心假设。"""
        fast = make_frames({1: [(float(i * 5), 0.0) for i in range(30)]})
        static = make_frames({1: [(0.0, 0.0)] * 30})

        psg_fast = build_psg(fast, default_meta())
        psg_static = build_psg(static, default_meta())

        assert (
            psg_fast.tracks[1].stats.highlight_score
            > psg_static.tracks[1].stats.highlight_score
        )

    def test_score_within_unit_range(self):
        frames = make_frames({1: [(float(i * 10), float(i)) for i in range(30)]})
        psg = build_psg(frames, default_meta())
        score = psg.tracks[1].stats.highlight_score
        assert 0.0 <= score <= 1.0

    def test_saliency_contributes(self):
        frames = make_frames({1: [(0.0, 0.0)] * 20})
        low = build_psg(frames, default_meta(), saliency={1: 0.0})
        high = build_psg(frames, default_meta(), saliency={1: 1.0})

        assert (
            high.tracks[1].stats.highlight_score
            > low.tracks[1].stats.highlight_score
        )

    def test_bigger_target_scores_higher(self):
        small_frames = [
            FrameTracks(
                frame_index=i, time_s=i / 30.0, tracks=[make_state(1, 0.0, w=2.0, h=2.0)]
            )
            for i in range(20)
        ]
        big_frames = [
            FrameTracks(
                frame_index=i, time_s=i / 30.0, tracks=[make_state(1, 0.0, w=30.0, h=30.0)]
            )
            for i in range(20)
        ]

        psg_small = build_psg(small_frames, default_meta())
        psg_big = build_psg(big_frames, default_meta())
        assert (
            psg_big.tracks[1].stats.highlight_score
            > psg_small.tracks[1].stats.highlight_score
        )


# ============================================================== 数值层


class TestNumericLayer:
    def test_writes_npz_and_sets_ref(self, tmp_path):
        frames = make_frames({1: [(float(i), float(i % 5)) for i in range(20)]})
        psg = build_psg(frames, default_meta(), numeric_dir=tmp_path)

        node = psg.tracks[1]
        assert node.numeric_ref is not None
        assert node.numeric_ref.endswith(".npz")
        assert (tmp_path / "0001.npz").is_file()

    def test_numeric_layer_is_mmap_loadable(self, tmp_path):
        """★ 数值层必须支持 mmap 零拷贝加载 —— 这是分层存储的意义。"""
        frames = make_frames({1: [(float(i), 0.0) for i in range(20)]})
        psg = build_psg(frames, default_meta(), numeric_dir=tmp_path)

        data = load_numeric_layer(tmp_path / "0001.npz")

        for key in ("yaw_deg", "pitch_deg", "width_deg", "height_deg", "confidence"):
            assert key in data, f"缺少字段 {key}"
            assert len(data[key]) == 20

        # 数值应与输入一致
        np.testing.assert_allclose(data["yaw_deg"], np.arange(20), atol=1e-5)

    def test_no_numeric_dir_means_no_ref(self):
        frames = make_frames({1: [(0.0, 0.0)] * 20})
        psg = build_psg(frames, default_meta(), numeric_dir=None)
        assert psg.tracks[1].numeric_ref is None

    def test_multiple_tracks_get_separate_files(self, tmp_path):
        frames = make_frames({1: [(0.0, 0.0)] * 20, 2: [(90.0, 0.0)] * 20})
        build_psg(frames, default_meta(), numeric_dir=tmp_path)

        assert (tmp_path / "0001.npz").is_file()
        assert (tmp_path / "0002.npz").is_file()

    def test_llm_view_never_exposes_numeric_ref(self, tmp_path):
        """★ 即使写了数值层，LLM 视图也不能泄漏它的路径。"""
        import json

        frames = make_frames({1: [(0.0, 0.0)] * 20})
        psg = build_psg(frames, default_meta(), numeric_dir=tmp_path)

        text = json.dumps(psg.to_llm_view(), ensure_ascii=False)
        assert ".npz" not in text
        assert "numeric_ref" not in text


# ============================================================== 事件


class TestEventDetection:
    def test_motion_burst_detected(self):
        """速度峰值应产生一个 motion_burst 事件。"""
        # 前 10 帧静止，中间 5 帧快速，之后静止
        seq = [(0.0, 0.0)] * 10 + [(float(i * 8), 0.0) for i in range(5)] + [(32.0, 0.0)] * 10
        frames = make_frames({1: seq})

        psg = build_psg(frames, default_meta())
        kinds = {e.kind for e in psg.events}
        assert "motion_burst" in kinds, f"应检测到运动爆发，实际事件 {psg.events}"

    def test_static_target_produces_no_event(self):
        frames = make_frames({1: [(0.0, 0.0)] * 30})
        psg = build_psg(frames, default_meta())
        assert psg.events == []

    def test_events_sorted_by_score(self):
        seq_fast = [(float(i * 10), 0.0) for i in range(20)]
        seq_slow = [(float(i * 2), 90.0) for i in range(20)]
        frames = make_frames({1: seq_fast, 2: seq_slow})

        psg = build_psg(frames, default_meta())
        scores = [e.score for e in psg.events]
        assert scores == sorted(scores, reverse=True)

    def test_event_references_valid_track_ids(self):
        """事件引用的 track_id 必须在 PSG 中存在 —— 否则 LLM 会引用到空。"""
        seq = [(0.0, 0.0)] * 5 + [(float(i * 8), 0.0) for i in range(10)]
        frames = make_frames({7: seq})

        psg = build_psg(frames, default_meta())
        for event in psg.events:
            for tid in event.involved_tracks:
                assert tid in psg.tracks


# ============================================================== 端到端


class TestEndToEnd:
    def test_builds_valid_psg_from_synthetic_tracks(self):
        frames = make_frames(
            {
                1: [(float(i), 0.0) for i in range(40)],       # 匀速水平移动
                2: [(-90.0, 20.0)] * 40,                        # 静止
            }
        )
        psg = build_psg(frames, default_meta(duration_s=2.0))

        assert isinstance(psg, PanoramaSceneGraph)
        assert len(psg.tracks) == 2
        assert psg.meta.erp_width == 2048

        # LLM 视图应可用
        view = psg.to_llm_view()
        assert len(view["tracks"]) == 2

    def test_shots_and_audio_passed_through(self):
        from pano_agent.sir import AudioInfo, ShotNode

        frames = make_frames({1: [(0.0, 0.0)] * 20})
        psg = build_psg(
            frames,
            default_meta(),
            shots=[ShotNode(shot_id=0, t_start=0.0, t_end=2.0)],
            audio=AudioInfo(duration_s=5.0, beat_times=[0.5, 1.0, 1.5]),
        )

        assert len(psg.shots) == 1
        assert psg.audio is not None
        assert psg.audio.nearest_beat(0.9) == pytest.approx(1.0)
