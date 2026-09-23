"""从感知层的逐帧输出构建 Panorama Scene Graph。

==============================================================================
为什么同一份数据要落成两种形态
==============================================================================

感知层每帧输出一次 `TrackState`。一段 30 秒 30fps 的视频有 900 帧、
可能 20 条轨迹 —— 逐帧数据是上万条记录，直接塞给 LLM 既浪费 token
又毫无必要（LLM 关心的是"有哪些主体、各自多长、快不快"，
不是第 412 帧的 yaw 是多少）。

因此本模块把同一份数据落成两种形态：

    数值层  tracks/0001.npz    逐帧数组，供轨迹优化器 mmap 零拷贝读取
    符号层  scene_graph.json   统计量与语义，供 LLM 阅读

两者通过 `track_id` 对齐，且**符号层里不含任何坐标**。

==============================================================================
高光分数的诚实定位
==============================================================================

`highlight_score` 是**启发式的弱监督信号**：由运动能量、角尺寸、
显著度按经验权重合成。它不是学出来的，也没有公开 GT 可供校准。

因此：
  - 它只作为 Agent 规划的一个**输入信号**，不是决策本身；
  - 技术报告中必须明确声明这一点，不能当作贡献点；
  - 权重与阈值都是经验值，换数据集需要重新标定。

把启发式规则包装成"算法创新"是这类项目最常见的失真，这里主动避免。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..geometry.sphere import great_circle_distance
from ..perception.types import FrameTracks, TrackState
from .schema import (
    AudioInfo,
    EventNode,
    PanoramaSceneGraph,
    ShotNode,
    TrackNode,
    TrackStats,
    VideoMeta,
)


@dataclass
class BuildConfig:
    """PSG 构建配置。"""

    min_track_frames: int = 5
    """少于这么多帧的轨迹会被丢弃 —— 抑制检测抖动产生的碎片轨迹。"""

    only_confirmed: bool = True
    """只使用已确认的轨迹（`TrackState.is_confirmed`）。"""

    highlight_weights: dict[str, float] = field(
        default_factory=lambda: {
            "motion": 0.40,
            "size": 0.30,
            "saliency": 0.30,
        }
    )
    """高光分数的合成权重（启发式，见模块文档）。"""

    motion_ref_dps: float = 60.0
    """角速度的参考值：达到此速度时运动项得满分。"""

    size_ref_deg: float = 25.0
    """角尺寸的参考值：达到此尺寸时尺寸项得满分。"""

    high_latitude_deg: float = 60.0
    """纬度阈值，超过即标记为高纬（对应 360VOT 的 HIGH_LATITUDE）。"""


# ============================================================== 数值层


def write_numeric_layer(
    track_id: int,
    samples: list[TrackState],
    frame_indices: list[int],
    out_dir: Path,
    *,
    prefix: str = "tracks",
) -> str:
    """把一条轨迹的逐帧数组写入 .npz，返回相对路径。

    存为 .npz（未压缩）而非 .npz 压缩包，是为了让优化器能用
    `np.load(..., mmap_mode="r")` **零拷贝**读取 —— 压缩会迫使全量解压。

    数组单位一律是**角度**，文件名与字段名都体现这一点。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(samples)
    yaw = np.empty(n, dtype=np.float32)
    pitch = np.empty(n, dtype=np.float32)
    width = np.empty(n, dtype=np.float32)
    height = np.empty(n, dtype=np.float32)
    conf = np.empty(n, dtype=np.float32)

    for i, s in enumerate(samples):
        yaw[i] = s.yaw_deg
        pitch[i] = s.pitch_deg
        width[i] = s.width_deg
        height[i] = s.height_deg
        conf[i] = s.confidence

    path = out_dir / f"{track_id:04d}.npz"
    np.savez(
        path,
        yaw_deg=yaw,
        pitch_deg=pitch,
        width_deg=width,
        height_deg=height,
        confidence=conf,
        frame_index=np.asarray(frame_indices, dtype=np.int32),
    )

    return f"{prefix}/{path.name}"


def load_numeric_layer(path: Path | str) -> dict[str, NDArray]:
    """mmap 加载数值层（零拷贝，不占内存）。

    这是「符号层给 LLM、数值层给优化器」的落点：
    优化器只需要读，不需要把整个数组载入内存。
    """
    with np.load(Path(path), mmap_mode="r") as data:
        return {k: data[k] for k in data.files}


# ============================================================== 统计量


def _compute_stats(
    samples: list[TrackState],
    frame_indices: list[int],
    fps: float,
    config: BuildConfig,
    *,
    saliency_mean: float = 0.0,
) -> TrackStats:
    """从逐帧状态聚合出统计量。"""
    n = len(samples)
    duration_s = (frame_indices[-1] - frame_indices[0]) / fps if fps > 0 else 0.0

    pitches = np.array([s.pitch_deg for s in samples], dtype=np.float64)
    sizes = np.array(
        [
            np.hypot(s.width_deg, s.height_deg) / 2.0  # 半对角线作为"尺寸"
            for s in samples
        ],
        dtype=np.float64,
    )

    # ---- 角速度：相邻帧的大圆距离 / dt ----
    speeds: list[float] = []
    for i in range(1, n):
        d = great_circle_distance(
            samples[i - 1].yaw_deg,
            samples[i - 1].pitch_deg,
            samples[i].yaw_deg,
            samples[i].pitch_deg,
        )
        dt = (frame_indices[i] - frame_indices[i - 1]) / fps if fps > 0 else 1.0
        if dt > 0:
            speeds.append(float(d) / dt)

    vel_mean = float(np.mean(speeds)) if speeds else 0.0
    vel_max = float(np.max(speeds)) if speeds else 0.0

    lat_lo, lat_hi = float(pitches.min()), float(pitches.max())
    pole_proximity = float(90.0 - max(abs(lat_lo), abs(lat_hi)))

    # ---- 高光分数：启发式加权（见模块文档的诚实声明）----
    w = config.highlight_weights
    motion_term = min(vel_mean / max(config.motion_ref_dps, 1e-6), 1.0)
    size_term = min(float(sizes.mean()) / max(config.size_ref_deg, 1e-6), 1.0)
    highlight = (
        w.get("motion", 0.0) * motion_term
        + w.get("size", 0.0) * size_term
        + w.get("saliency", 0.0) * saliency_mean
    )

    return TrackStats(
        frame_range=(frame_indices[0], frame_indices[-1]),
        duration_s=duration_s,
        angular_size_mean_deg=float(sizes.mean()),
        angular_velocity_mean_dps=vel_mean,
        angular_velocity_max_dps=vel_max,
        latitude_span_deg=(lat_lo, lat_hi),
        crosses_seam=any(s.crosses_seam for s in samples),
        pole_proximity_min_deg=pole_proximity,
        saliency_mean=float(np.clip(saliency_mean, 0.0, 1.0)),
        highlight_score=float(np.clip(highlight, 0.0, 1.0)),
    )


# ============================================================== 事件


def _detect_events(
    per_track: dict[int, tuple[list[int], list[TrackState]]],
    fps: float,
    *,
    motion_ref_dps: float = 60.0,
) -> list[EventNode]:
    """从轨迹中提取高光事件。

    目前只用「运动爆发」一种类型 —— 它是可从轨迹直接观测到的信号。
    音频节拍、场景切换等需要额外的输入源，在对应的模块就绪前不臆造。
    """
    events: list[EventNode] = []
    event_id = 0

    for track_id, (frames, samples) in per_track.items():
        speeds: list[tuple[int, float]] = []
        for i in range(1, len(samples)):
            d = great_circle_distance(
                samples[i - 1].yaw_deg,
                samples[i - 1].pitch_deg,
                samples[i].yaw_deg,
                samples[i].pitch_deg,
            )
            dt = (frames[i] - frames[i - 1]) / fps if fps > 0 else 1.0
            if dt > 0:
                speeds.append((i, float(d) / dt))

        if not speeds:
            continue

        # 找出速度峰值，并向前后扩展到速度降到峰值一半的位置
        peak_idx, peak_speed = max(speeds, key=lambda p: p[1])
        if peak_speed < motion_ref_dps * 0.5:
            continue

        half = peak_speed / 2.0
        lo = peak_idx
        while lo > 0 and speeds[lo - 1][1] > half:
            lo -= 1
        hi = peak_idx
        while hi < len(speeds) - 1 and speeds[hi + 1][1] > half:
            hi += 1

        t_start = frames[lo] / fps
        t_end = frames[min(hi + 1, len(frames) - 1)] / fps
        if t_end <= t_start:
            continue

        events.append(
            EventNode(
                event_id=event_id,
                t_start=t_start,
                t_end=t_end,
                kind="motion_burst",
                involved_tracks=[track_id],
                score=float(np.clip(peak_speed / motion_ref_dps, 0.0, 1.0)),
            )
        )
        event_id += 1

    return sorted(events, key=lambda e: -e.score)


# ============================================================== 主入口


def build_psg(
    frames: list[FrameTracks],
    meta: VideoMeta,
    *,
    config: BuildConfig | None = None,
    saliency: dict[int, float] | None = None,
    numeric_dir: Path | None = None,
    shots: list[ShotNode] | None = None,
    audio: AudioInfo | None = None,
) -> PanoramaSceneGraph:
    """从逐帧跟踪结果构建场景图。

    Args:
        frames: 感知层输出的逐帧轨迹，需按帧序排列。
        meta: 视频元信息。
        config: 构建配置。
        saliency: `{track_id: 平均显著度}`，来自显著性模块。
            缺失时高光分数的显著度项按 0 计。
        numeric_dir: 若提供，则把逐帧数组写入该目录并填充 `numeric_ref`；
            None 则只构建符号层（便于单元测试与快速预览）。
        shots: 镜头切分结果（可选）。
        audio: 音频信息（可选）。

    Returns:
        PanoramaSceneGraph。
    """
    config = config or BuildConfig()
    saliency = saliency or {}
    fps = meta.fps

    # ---- 1. 按 track_id 聚合逐帧状态 ----
    per_track: dict[int, tuple[list[int], list[TrackState]]] = {}
    grouped: dict[int, list[tuple[int, TrackState]]] = defaultdict(list)

    for frame in frames:
        for state in frame.tracks:
            if config.only_confirmed and not state.is_confirmed:
                continue
            grouped[state.track_id].append((frame.frame_index, state))

    for track_id, entries in grouped.items():
        if len(entries) < config.min_track_frames:
            continue
        idxs = [e[0] for e in entries]
        states = [e[1] for e in entries]
        per_track[track_id] = (idxs, states)

    # ---- 2. 逐条构建 TrackNode ----
    tracks: dict[int, TrackNode] = {}
    for track_id, (idxs, states) in per_track.items():
        stats = _compute_stats(
            states,
            idxs,
            fps,
            config,
            saliency_mean=saliency.get(track_id, 0.0),
        )

        numeric_ref = None
        if numeric_dir is not None:
            numeric_ref = write_numeric_layer(track_id, states, idxs, Path(numeric_dir))

        attributes: list[str] = []
        if stats.crosses_seam:
            attributes.append("crosses_seam")
        if stats.is_high_latitude:
            attributes.append("high_latitude")
        if stats.angular_velocity_max_dps > config.motion_ref_dps:
            attributes.append("fast_motion")

        tracks[track_id] = TrackNode(
            track_id=track_id,
            category=states[0].category,
            caption="",  # 由 VLM 在后续阶段填充
            attributes=attributes,
            stats=stats,
            numeric_ref=numeric_ref,
        )

    # ---- 3. 事件 ----
    events = _detect_events(per_track, fps) if per_track else []

    return PanoramaSceneGraph(
        meta=meta,
        shots=shots or [],
        tracks=tracks,
        events=events,
        audio=audio,
    )
