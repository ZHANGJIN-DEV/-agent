"""Panorama Scene Graph 的 schema 测试。

核心验证的是「双视图」设计的不变量：

    LLM 视图里**不能出现任何浮点坐标**，也不能出现 numeric_ref。

这不是风格问题 —— 一旦 LLM 能看到逐帧数值，它就会试图去生成坐标，
而 LLM 输出的坐标既无物理意义又不可校验。这个不变量一旦破坏，
整个 Agent 架构的立论基础就没了。
"""

from __future__ import annotations

import json

import pytest

from pano_agent.sir import (
    AudioInfo,
    EventNode,
    PanoramaSceneGraph,
    ShotNode,
    TrackNode,
    TrackStats,
    VideoMeta,
)

# ============================================================== 构造辅助

# LLM 视图中绝对不允许出现的键名（几何数值 / 数值层引用）
FORBIDDEN_KEYS = {
    "yaw_deg",
    "pitch_deg",
    "width_deg",
    "height_deg",
    "roll_deg",
    "numeric_ref",
    "v_yaw_dps",
    "v_pitch_dps",
}


def make_meta(*, duration_s: float = 30.0) -> VideoMeta:
    return VideoMeta(fps=30.0, duration_s=duration_s, erp_width=2048, erp_height=1024)


def make_stats(
    *,
    frame_range: tuple[int, int] = (0, 300),
    duration_s: float = 10.0,
    **kw,
) -> TrackStats:
    base = dict(
        frame_range=frame_range,
        duration_s=duration_s,
        angular_size_mean_deg=12.0,
        angular_velocity_mean_dps=20.0,
        angular_velocity_max_dps=45.0,
        latitude_span_deg=(-10.0, 10.0),
        saliency_mean=0.6,
        highlight_score=0.7,
    )
    base.update(kw)
    return TrackStats(**base)


def make_track(track_id: int, **kw) -> TrackNode:
    stats = kw.pop("stats", None) or make_stats()
    return TrackNode(
        track_id=track_id,
        category=kw.pop("category", "person"),
        caption=kw.pop("caption", f"轨道 {track_id}"),
        numeric_ref=kw.pop("numeric_ref", f"tracks/{track_id:04d}.npz"),
        stats=stats,
        **kw,
    )


# ============================================================== VideoMeta


class TestVideoMeta:
    def test_rejects_non_erp_aspect(self):
        """非 2:1 的输入必须在构造时就被拒绝 —— 这是 ERP 的硬性不变量。"""
        with pytest.raises(ValueError, match="2:1"):
            VideoMeta(fps=30, duration_s=10, erp_width=1920, erp_height=1080)

    def test_accepts_standard_erp_sizes(self):
        for w, h in [(2048, 1024), (3840, 1920), (512, 256), (4096, 2048)]:
            meta = VideoMeta(fps=30, duration_s=10, erp_width=w, erp_height=h)
            assert meta.projection == "ERP"

    def test_num_frames(self):
        meta = VideoMeta(fps=30, duration_s=10, erp_width=2048, erp_height=1024)
        assert meta.num_frames == 300

    def test_is_frozen(self):
        """元信息不应被就地修改。"""
        meta = make_meta()
        with pytest.raises(Exception):
            meta.fps = 60  # type: ignore[misc]

    def test_rejects_bad_fps(self):
        with pytest.raises(ValueError):
            VideoMeta(fps=0, duration_s=10, erp_width=2048, erp_height=1024)


# ============================================================== TrackStats


class TestTrackStats:
    def test_rejects_inverted_frame_range(self):
        with pytest.raises(ValueError, match="起点不能大于终点"):
            make_stats(frame_range=(100, 50))

    def test_rejects_bad_latitude_span(self):
        with pytest.raises(ValueError, match="纬度范围非法"):
            make_stats(latitude_span_deg=(50.0, -50.0))
        with pytest.raises(ValueError, match="纬度范围非法"):
            make_stats(latitude_span_deg=(-100.0, 50.0))

    def test_high_latitude_flag(self):
        """对应 360VOT 的 HIGH_LATITUDE 挑战 —— 用于分属性评测。"""
        assert make_stats(latitude_span_deg=(-70.0, -65.0)).is_high_latitude
        assert make_stats(latitude_span_deg=(65.0, 80.0)).is_high_latitude
        assert not make_stats(latitude_span_deg=(-20.0, 20.0)).is_high_latitude

    def test_seam_flag_defaults_false(self):
        assert make_stats().crosses_seam is False

    def test_rejects_negative_speed(self):
        with pytest.raises(ValueError):
            make_stats(angular_velocity_mean_dps=-1.0)


# ============================================================== ★ LLM 视图


class TestLLMView:
    """「双视图」设计的核心不变量。"""

    def test_no_numeric_ref_in_llm_view(self):
        """★ numeric_ref 绝不能出现在 LLM 视图中。

        它是数值层的入口。一旦 LLM 看到这个路径，就可能试图
        构造文件内容或引用它，而 LLM 不该接触逐帧数组。
        """
        psg = PanoramaSceneGraph(meta=make_meta(), tracks={1: make_track(1)})
        view = psg.to_llm_view()

        flat = json.dumps(view)
        assert "numeric_ref" not in flat
        assert ".npz" not in flat, "不应泄漏数值文件的路径"

    def test_no_geometric_coordinates_in_llm_view(self):
        """★ LLM 视图中不得出现任何几何坐标字段。"""
        psg = PanoramaSceneGraph(
            meta=make_meta(), tracks={i: make_track(i) for i in range(5)}
        )
        view = psg.to_llm_view()

        def all_keys(obj) -> set[str]:
            keys: set[str] = set()
            if isinstance(obj, dict):
                keys |= set(obj)
                for v in obj.values():
                    keys |= all_keys(v)
            elif isinstance(obj, list):
                for item in obj:
                    keys |= all_keys(item)
            return keys

        found = all_keys(view) & FORBIDDEN_KEYS
        assert not found, f"LLM 视图中出现了几何坐标字段: {found}"

    def test_view_is_json_serializable(self):
        """必须能直接序列化 —— 要送到 API 去的。"""
        psg = PanoramaSceneGraph(
            meta=make_meta(),
            tracks={1: make_track(1, attributes=["fast", "crosses_seam"])},
            events=[
                EventNode(
                    event_id=0,
                    t_start=1.0,
                    t_end=3.0,
                    kind="action_peak",
                    involved_tracks=[1],
                    score=0.9,
                )
            ],
            shots=[ShotNode(shot_id=0, t_start=0.0, t_end=5.0)],
        )
        text = json.dumps(psg.to_llm_view(), ensure_ascii=False)
        assert "action_peak" in text
        assert len(text) > 0

    def test_view_is_compact(self):
        """LLM 视图必须紧凑 —— 逐条轨迹的开销要有明确上界。

        判据用「每条轨迹的字符数」而非总字符数：总长随轨迹数线性增长，
        固定阈值在轨迹多时必然失败、在轨迹少时又毫无约束力。
        """
        n_tracks = 20
        psg = PanoramaSceneGraph(
            meta=make_meta(), tracks={i: make_track(i) for i in range(n_tracks)}
        )
        text = json.dumps(psg.to_llm_view(), ensure_ascii=False)

        per_track = len(text) / n_tracks
        assert per_track < 300, (
            f"每条轨迹占用 {per_track:.0f} 字符，超出 token 预算"
            f"（总长 {len(text)} 字符 / {n_tracks} 条）"
        )

    def test_max_tracks_truncation_is_disclosed(self):
        """截断必须显式告知 —— 让 LLM 知道自己看到的不是全部。"""
        psg = PanoramaSceneGraph(
            meta=make_meta(), tracks={i: make_track(i) for i in range(10)}
        )
        view = psg.to_llm_view(max_tracks=3)

        assert len(view["tracks"]) == 3
        assert "note" in view, "截断时必须有说明"
        assert "10" in view["note"], "说明中应包含完整轨迹数"

    def test_no_truncation_note_when_not_truncated(self):
        psg = PanoramaSceneGraph(meta=make_meta(), tracks={1: make_track(1)})
        assert "note" not in psg.to_llm_view(max_tracks=10)

    def test_time_window_filter(self):
        """时间窗过滤应只返回该区间内可见的轨迹。"""
        fps = 30.0
        early = make_track(1, stats=make_stats(frame_range=(0, 60)))
        late = make_track(2, stats=make_stats(frame_range=(600, 660)))
        psg = PanoramaSceneGraph(meta=make_meta(), tracks={1: early, 2: late})

        view = psg.to_llm_view(t_start=0.0, t_end=3.0)
        ids = {t["track_id"] for t in view["tracks"]}
        assert ids == {1}, f"只应返回早期轨迹，实际 {ids}"

    def test_tracks_sorted_by_duration(self):
        short = make_track(1, stats=make_stats(duration_s=1.0))
        long = make_track(2, stats=make_stats(duration_s=10.0))
        psg = PanoramaSceneGraph(meta=make_meta(), tracks={1: short, 2: long})

        view = psg.to_llm_view()
        durations = [t["duration_s"] for t in view["tracks"]]
        assert durations == sorted(durations, reverse=True)

    def test_video_section_present(self):
        psg = PanoramaSceneGraph(meta=make_meta(duration_s=42.0), tracks={})
        view = psg.to_llm_view()
        assert view["video"]["duration_s"] == pytest.approx(42.0)
        assert view["video"]["projection"] == "ERP"


# ============================================================== ID 校验


class TestTrackIdValidation:
    def test_filters_hallucinated_ids(self):
        """LLM 可能幻觉出不存在的 track_id —— 必须能过滤掉。"""
        psg = PanoramaSceneGraph(meta=make_meta(), tracks={1: make_track(1), 2: make_track(2)})
        assert psg.validate_track_ids([1, 2, 99, -1]) == [1, 2]

    def test_track_ids_property_is_whitelist(self):
        psg = PanoramaSceneGraph(meta=make_meta(), tracks={5: make_track(5)})
        assert psg.track_ids == {5}

    def test_get_track_raises_for_unknown(self):
        psg = PanoramaSceneGraph(meta=make_meta(), tracks={1: make_track(1)})
        with pytest.raises(KeyError, match="不存在"):
            psg.get_track(42)


# ============================================================== 其他节点


class TestEventNode:
    def test_rejects_inverted_time(self):
        with pytest.raises(ValueError, match="起止时间非法"):
            EventNode(event_id=0, t_start=5.0, t_end=2.0, kind="audio_beat")

    def test_duration(self):
        e = EventNode(event_id=0, t_start=1.0, t_end=4.5, kind="action_peak")
        assert e.duration_s == pytest.approx(3.5)

    def test_rejects_unknown_kind(self):
        """kind 是闭集 —— 防止 LLM 随意发明新的事件类型。"""
        with pytest.raises(ValueError):
            EventNode(event_id=0, t_start=0.0, t_end=1.0, kind="unknown_kind")  # type: ignore[arg-type]


class TestShotNode:
    def test_rejects_zero_or_inverted_range(self):
        with pytest.raises(ValueError):
            ShotNode(shot_id=0, t_start=5.0, t_end=5.0)
        with pytest.raises(ValueError):
            ShotNode(shot_id=0, t_start=5.0, t_end=1.0)


class TestAudioInfo:
    def test_nearest_beat(self):
        audio = AudioInfo(duration_s=10.0, beat_times=[0.0, 0.5, 1.0, 1.5, 2.0])
        assert audio.nearest_beat(0.9) == pytest.approx(1.0)
        assert audio.nearest_beat(1.3) == pytest.approx(1.5)

    def test_nearest_beat_handles_edges(self):
        audio = AudioInfo(duration_s=10.0, beat_times=[1.0, 2.0])
        assert audio.nearest_beat(0.0) == pytest.approx(1.0)
        assert audio.nearest_beat(100.0) == pytest.approx(2.0)

    def test_nearest_beat_without_beats(self):
        assert AudioInfo(duration_s=10.0).nearest_beat(1.0) is None


class TestHighlights:
    def test_filters_and_sorts(self):
        psg = PanoramaSceneGraph(
            meta=make_meta(),
            tracks={1: make_track(1)},
            events=[
                EventNode(event_id=0, t_start=0, t_end=1, kind="action_peak", score=0.3),
                EventNode(event_id=1, t_start=1, t_end=2, kind="action_peak", score=0.9),
                EventNode(event_id=2, t_start=2, t_end=3, kind="action_peak", score=0.7),
            ],
        )
        top = psg.highlights(min_score=0.5)
        assert [e.event_id for e in top] == [1, 2]

    def test_limit(self):
        psg = PanoramaSceneGraph(
            meta=make_meta(),
            tracks={},
            events=[
                EventNode(event_id=i, t_start=i, t_end=i + 1, kind="action_peak", score=0.9)
                for i in range(10)
            ],
        )
        assert len(psg.highlights(limit=3)) == 3


class TestRepr:
    def test_repr_is_informative(self):
        psg = PanoramaSceneGraph(meta=make_meta(), tracks={1: make_track(1)})
        text = repr(psg)
        assert "PSG" in text and "2048×1024" in text and "tracks=1" in text
