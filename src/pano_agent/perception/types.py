"""感知层的公共数据结构。

设计约束（贯穿本包）:
    所有几何量一律使用**球面角度（度）**，不使用像素。
    ERP 是非均匀采样 —— 赤道处 10 像素约 0.3°，高纬处 10 像素约 3°，
    因此任何以像素为单位的距离、面积、速度都缺乏统一的物理含义。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..geometry.region import SphericalBox


@dataclass
class SphericalDetection:
    """单帧中的一个检测结果（球面表示）。

    这是检测器与跟踪器之间的接口。检测器负责把 ERP 上的像素框
    （或直接回归的球面框）转换为本结构，之后所有模块只见球面角度。
    """

    yaw_deg: float
    pitch_deg: float
    width_deg: float
    height_deg: float
    confidence: float = 1.0
    category: str = "object"
    roll_deg: float = 0.0

    embedding: NDArray[np.float32] | None = None
    """ReID 外观特征（可选）。在关联代价中与几何代价加权组合，
    用于缓解"目标交叉后 ID 互换"这一经典问题。"""

    def to_box(self) -> SphericalBox:
        return SphericalBox(
            self.yaw_deg, self.pitch_deg, self.width_deg, self.height_deg, self.roll_deg
        )

    @property
    def solid_angle_sr(self) -> float:
        """该目标占据的立体角（球面度）—— 与分辨率无关的"大小"度量。"""
        return self.to_box().solid_angle_sr

    def __repr__(self) -> str:
        return (
            f"Det({self.category} yaw={self.yaw_deg:.1f}° pitch={self.pitch_deg:.1f}° "
            f"{self.width_deg:.1f}°×{self.height_deg:.1f}° conf={self.confidence:.2f})"
        )


@dataclass
class TrackState:
    """一条轨迹在某一帧的状态快照。

    对外输出为不可变语义：调用方拿到的应是拷贝，不应修改。
    """

    track_id: int
    yaw_deg: float
    pitch_deg: float
    width_deg: float
    height_deg: float

    v_yaw_dps: float = 0.0
    """方位角速度（度/秒）。注意这是**球面角速度**，不是像素速度。"""

    v_pitch_dps: float = 0.0

    confidence: float = 1.0
    category: str = "object"
    age: int = 0
    """轨迹存活的总帧数（含未匹配的帧）。"""

    hits: int = 0
    """累计匹配成功的次数。用于判定轨迹是否已"确认"。"""

    time_since_update: int = 0
    is_confirmed: bool = False

    crosses_seam: bool = False
    """该轨迹是否跨越过 ERP ±180° 接缝。

    这是 360VOT 所定义的 7 个 360° 特有挑战之一，也是本项目刻意解决的场景：
    朴素实现会在此处丢失 ID（因为同一目标在图像上从最右跑到最左）。
    本跟踪器通过在 unwrapped 空间滤波天然规避该问题 ——
    详细说明见 `track_spherical` 模块。
    """

    @property
    def angular_speed_dps(self) -> float:
        """合角速度（度/秒）。"""
        return float(np.hypot(self.v_yaw_dps, self.v_pitch_dps))

    def to_box(self) -> SphericalBox:
        return SphericalBox(self.yaw_deg, self.pitch_deg, self.width_deg, self.height_deg)

    def __repr__(self) -> str:
        seam = " 跨接缝" if self.crosses_seam else ""
        return (
            f"Track#{self.track_id}(yaw={self.yaw_deg:7.1f}° pitch={self.pitch_deg:6.1f}° "
            f"v={self.angular_speed_dps:5.1f}°/s hits={self.hits}{seam})"
        )


@dataclass
class FrameTracks:
    """一帧的跟踪结果。"""

    frame_index: int
    time_s: float
    tracks: list[TrackState] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self):
        return iter(self.tracks)

    def confirmed(self) -> list[TrackState]:
        """仅返回已确认的轨迹（过滤掉刚出现的候选，抑制虚警）。"""
        return [t for t in self.tracks if t.is_confirmed]
