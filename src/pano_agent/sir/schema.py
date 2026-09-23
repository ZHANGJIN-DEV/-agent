"""Panorama Scene Graph —— 全项目的技术核心。

==============================================================================
一个图，两种视图
==============================================================================

LLM 需要**少 token、语义化、可读**；优化器需要**全精度、逐帧、可微**。
这两个需求是矛盾的 —— 前者要抽象，后者要精确。

本模块通过「一个图、两种视图、`track_id` 严格对齐」同时满足：

    符号层 Symbolic                                   数值层 Numeric
    （to_llm_view → JSON）        ⟷ track_id ⟷        （numeric_ref → .npz）
    ─────────────────────                              ─────────────────────
    track_id                                            yaw[]      逐帧
    category                                            pitch[]
    caption     语义描述                                 width[] / height[]
    attributes  属性标签                                 confidence[]
    stats       球面统计量                               rot[]
    keyframe_refs 代表帧路径

    给 LLM：30 秒视频约 2-4k token，**不含任何浮点坐标**
    给优化器：mmap 零拷贝加载，全精度逐帧

**对齐机制是强制的**：LLM 只能通过 `track_id` 引用轨迹，
从来不能写坐标。这不是疏漏而是设计 —— 详见 `agent/dsl.py`。

==============================================================================
为什么不让 LLM 直接输出轨迹
==============================================================================

1. **LLM 没有度量几何的概念**。它输出的角度值没有物理意义，
   也无法判断对错。
2. **token 成本与一致性**。30fps × 15s = 450 个连续值，
   让 LLM 逐个输出既贵又容易前后矛盾。
3. **职责应当分离**。LLM 做语义决策（拍谁、什么景别、什么节奏），
   优化器做几何求解（满足约束的连续轨迹）。
4. **可校验性**。LLM 输出的是**约束**（可校验、可拒绝、可兜底），
   而不是**解**（不可校验、错了也不知道）。

==============================================================================
统计量的单位约定
==============================================================================

`TrackStats` 中的所有量都以**球面角度**或**秒**为单位，**与分辨率无关**。
这是刻意的设计：它使得
  - 30 秒 2K 视频与 5 分钟 4K 视频的统计量可以直接比较；
  - LLM 读到的是"角速度 45°/秒"这类有物理含义的量，而非像素数；
  - 同一套阈值可以跨数据集复用。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# 事件类型的闭集 —— 用 Literal 而非 str，让 LLM 只能从这个集合里选
EventKind = Literal[
    "action_peak",     # 动作峰值（运动能量局部极大）
    "audio_beat",      # 音频节拍
    "scene_change",    # 镜头切分点
    "face_appear",     # 人脸/主体出现
    "motion_burst",    # 运动爆发
    "subject_enter",   # 主体进入视野
]


class VideoMeta(BaseModel):
    """视频级元信息。"""

    model_config = ConfigDict(frozen=True)

    fps: float = Field(gt=0, le=240)
    duration_s: float = Field(gt=0)
    erp_width: int = Field(gt=0)
    erp_height: int = Field(gt=0)
    projection: Literal["ERP"] = "ERP"

    @model_validator(mode="after")
    def _check_erp_aspect(self) -> VideoMeta:
        """ERP 必须是 2:1 —— 这个不变量值得在构造时就检查。"""
        ratio = self.erp_width / self.erp_height
        if abs(ratio - 2.0) > 0.01:
            raise ValueError(
                f"ERP 宽高比应为 2:1，实际 {self.erp_width}×{self.erp_height} "
                f"(比例 {ratio:.3f})。若数据源不是 ERP，请先转换。"
            )
        return self

    @property
    def num_frames(self) -> int:
        return int(round(self.fps * self.duration_s))


class TrackStats(BaseModel):
    """单条轨迹的球面/时间统计量 —— **与分辨率无关**。

    这些量是 LLM 判断"哪条轨迹值得拍"的主要依据，
    也是筛选候选镜头的量化基础。
    """

    frame_range: tuple[int, int] = Field(description="[起始帧, 结束帧]（含）")
    duration_s: float = Field(ge=0)

    angular_size_mean_deg: float = Field(ge=0, description="平均角尺寸（度）")
    angular_velocity_mean_dps: float = Field(ge=0, description="平均角速度（度/秒）")
    angular_velocity_max_dps: float = Field(ge=0, description="峰值角速度（度/秒）")

    latitude_span_deg: tuple[float, float] = Field(
        description="纬度覆盖范围 [min, max]（度）。跨度大意味着目标在极区附近活动"
    )

    crosses_seam: bool = Field(
        default=False,
        description="是否跨越 ERP ±180° 接缝 —— 360VOT 定义的 360° 特有挑战之一",
    )
    pole_proximity_min_deg: float = Field(
        default=90.0, ge=0, le=90, description="距最近极点的最小角距离（度）"
    )

    saliency_mean: float = Field(default=0.0, ge=0, le=1, description="平均显著度")
    highlight_score: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="高光分数。**启发式弱监督**，仅作规划输入信号，不是主要贡献点",
    )

    @field_validator("frame_range")
    @classmethod
    def _check_frame_range(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] > v[1]:
            raise ValueError(f"frame_range 起点不能大于终点: {v}")
        return v

    @field_validator("latitude_span_deg")
    @classmethod
    def _check_latitude(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not (-90.0 <= lo <= hi <= 90.0):
            raise ValueError(f"纬度范围非法: {v}（应在 [-90, 90] 内且 lo <= hi）")
        return v

    @property
    def is_high_latitude(self) -> bool:
        """是否属于 360VOT 定义的 HIGH_LATITUDE 挑战场景。"""
        _, hi = self.latitude_span_deg
        lo, _ = self.latitude_span_deg
        return max(abs(lo), abs(hi)) > 60.0


class TrackNode(BaseModel):
    """一条轨迹节点 —— PSG 的基本单元。"""

    track_id: int = Field(ge=0)
    category: str

    # ---------------- 符号层：LLM 只读这一部分 ----------------
    caption: str = Field(
        default="", description="VLM 生成的语义描述，如「一个穿红衣服的滑板者」"
    )
    attributes: list[str] = Field(
        default_factory=list, description="属性标签（对应 360VOT 的 20 个属性）"
    )
    stats: TrackStats
    keyframe_refs: list[str] = Field(
        default_factory=list, description="代表帧图像路径，供 VLM 复审时使用"
    )

    # ---------------- 数值层：LLM 不读，优化器读 ----------------
    numeric_ref: str | None = Field(
        default=None,
        description="逐帧数值数组的路径（.npz），供优化器 mmap 加载。"
        "包含 yaw/pitch/width/height/rot/confidence，全为角度单位",
    )

    def to_llm_dict(self) -> dict[str, Any]:
        """转成给 LLM 的字典 —— **刻意不含 numeric_ref**。

        这是「同一份数据，两种视图」的落点：LLM 看不到逐帧数组的路径，
        也就不会试图去读它。
        """
        return {
            "track_id": self.track_id,
            "category": self.category,
            "caption": self.caption,
            "attributes": list(self.attributes),
            "duration_s": round(self.stats.duration_s, 2),
            "angular_size_mean_deg": round(self.stats.angular_size_mean_deg, 2),
            "angular_velocity_mean_dps": round(self.stats.angular_velocity_mean_dps, 1),
            "crosses_seam": self.stats.crosses_seam,
            "saliency_mean": round(self.stats.saliency_mean, 3),
            "highlight_score": round(self.stats.highlight_score, 3),
        }


class EventNode(BaseModel):
    """高光事件 —— 跨越单条轨迹的时间段。

    例如"第 3 秒到第 6 秒期间发生了一次运动爆发，涉及轨道 2 和 5"。
    事件是剪辑决策的天然锚点：一个镜头通常围绕一个事件展开。
    """

    event_id: int = Field(ge=0)
    t_start: float = Field(ge=0)
    t_end: float = Field(ge=0)
    kind: EventKind
    involved_tracks: list[int] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def _check_time_range(self) -> EventNode:
        if self.t_start > self.t_end:
            raise ValueError(f"事件起止时间非法: [{self.t_start}, {self.t_end}]")
        return self

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


class ShotNode(BaseModel):
    """镜头切分后的连续片段。"""

    shot_id: int = Field(ge=0)
    t_start: float = Field(ge=0)
    t_end: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_range(self) -> ShotNode:
        if self.t_start >= self.t_end:
            raise ValueError(f"镜头区间非法: [{self.t_start}, {self.t_end}]")
        return self

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


class AudioInfo(BaseModel):
    """音频侧信息，用于节拍对齐剪辑。"""

    duration_s: float = Field(ge=0)
    beat_times: list[float] = Field(default_factory=list, description="节拍时刻（秒）")
    tempo_bpm: float | None = Field(default=None, gt=0)

    def nearest_beat(self, t: float) -> float | None:
        """返回距 t 最近的节拍时刻。用于把剪辑点吸附到节拍上。"""
        if not self.beat_times:
            return None
        import bisect

        idx = bisect.bisect_left(self.beat_times, t)
        candidates = [
            self.beat_times[i]
            for i in (idx - 1, idx, idx + 1)
            if 0 <= i < len(self.beat_times)
        ]
        return min(candidates, key=lambda b: abs(b - t)) if candidates else None


class PanoramaSceneGraph(BaseModel):
    """完整的全景场景图。

    这是感知层与 Agent 层之间的**接口契约** ——
    上游按它组织输出，下游按它提取信息。任何字段变更都是破坏性的。
    """

    meta: VideoMeta
    shots: list[ShotNode] = Field(default_factory=list)
    tracks: dict[int, TrackNode] = Field(default_factory=dict)
    events: list[EventNode] = Field(default_factory=list)
    audio: AudioInfo | None = None

    # ------------------------------------------------------------ 查询

    @property
    def track_ids(self) -> set[int]:
        """所有合法 track_id —— 用作 LLM 输出的**白名单校验**。"""
        return set(self.tracks)

    def get_track(self, track_id: int) -> TrackNode:
        if track_id not in self.tracks:
            raise KeyError(f"不存在 track_id={track_id}，可用: {sorted(self.track_ids)}")
        return self.tracks[track_id]

    def tracks_in_range(self, t_start: float, t_end: float) -> list[TrackNode]:
        """返回在给定时间区间内有可见帧的轨迹。"""
        fps = self.meta.fps
        f0, f1 = int(t_start * fps), int(t_end * fps)
        result = []
        for node in self.tracks.values():
            a, b = node.stats.frame_range
            if b >= f0 and a <= f1:
                result.append(node)
        return sorted(result, key=lambda n: -n.stats.highlight_score)

    def highlights(self, *, min_score: float = 0.5, limit: int = 10) -> list[EventNode]:
        """按分数返回高光事件。"""
        return sorted(
            (e for e in self.events if e.score >= min_score),
            key=lambda e: -e.score,
        )[:limit]

    # ------------------------------------------------------------ LLM 视图

    def to_llm_view(
        self,
        *,
        t_start: float | None = None,
        t_end: float | None = None,
        max_tracks: int | None = None,
    ) -> dict[str, Any]:
        """生成给 LLM 的紧凑 JSON 视图。

        **这是整个「双视图」设计的对外出口。** 产出中不含任何
        逐帧数值数组的引用，也不含浮点坐标 —— 只有语义信息与
        统计量，让 LLM 能在几千 token 内理解整段视频的结构。

        Args:
            t_start / t_end: 只看某个时间窗内的轨迹。
            max_tracks: 最多返回多少条轨迹（按时长排序取前 N），
                用于控制 token 预算。

        Returns:
            可直接 `json.dumps` 的字典。
        """
        nodes = list(self.tracks.values())
        if t_start is not None or t_end is not None:
            lo = t_start if t_start is not None else 0.0
            hi = t_end if t_end is not None else self.meta.duration_s
            nodes = self.tracks_in_range(lo, hi)

        nodes = sorted(nodes, key=lambda n: -n.stats.duration_s)
        truncated = False
        if max_tracks is not None and len(nodes) > max_tracks:
            nodes = nodes[:max_tracks]
            truncated = True

        view: dict[str, Any] = {
            "video": {
                "duration_s": round(self.meta.duration_s, 2),
                "fps": self.meta.fps,
                "projection": self.meta.projection,
            },
            "tracks": [n.to_llm_dict() for n in nodes],
        }

        if self.shots:
            view["shots"] = [
                {
                    "shot_id": s.shot_id,
                    "t_start": round(s.t_start, 2),
                    "t_end": round(s.t_end, 2),
                }
                for s in self.shots
            ]

        if self.events:
            view["events"] = [
                {
                    "event_id": e.event_id,
                    "t_start": round(e.t_start, 2),
                    "t_end": round(e.t_end, 2),
                    "kind": e.kind,
                    "involved_tracks": e.involved_tracks,
                    "score": round(e.score, 3),
                }
                for e in self.events
            ]

        if truncated:
            # 显式告知截断 —— 让 LLM 知道自己看到的不是全部
            view["note"] = (
                f"仅显示时长最长的 {len(nodes)} 条轨迹，"
                f"完整轨迹数为 {len(self.tracks)}"
            )

        return view

    # ------------------------------------------------------------ 校验

    def validate_track_ids(self, ids: list[int]) -> list[int]:
        """过滤出在白名单内的 track_id。

        LLM 可能幻觉出不存在的 id。**这是防御的第一道关卡**：
        调用方应当只使用本方法返回的 id，并根据被丢弃的数量决定
        是否要求 LLM 重新生成。
        """
        return [i for i in ids if i in self.tracks]

    def __repr__(self) -> str:
        return (
            f"<PSG {self.meta.duration_s:.1f}s {self.meta.erp_width}×{self.meta.erp_height} "
            f"tracks={len(self.tracks)} shots={len(self.shots)} events={len(self.events)}>"
        )
