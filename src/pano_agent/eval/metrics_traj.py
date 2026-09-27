"""轨迹与构图质量指标。

==============================================================================
为什么所有"距离"都必须用大圆距离
==============================================================================

    赤道处 10 像素 ≈ 0.3°
    高纬处 10 像素 ≈ 3°

用像素距离衡量构图质量，会**系统性高估高纬目标的构图表现** ——
一个在高纬"偏离画面中心 100 像素"的主体，实际偏离的视角可能是
低纬同样像素数的十倍。而观众感知到的是**角度**，不是像素。

==============================================================================
指标清单与各自回答的问题
==============================================================================

**主体居中率** subject_centering_rate
    "观众看得清主体吗？"
    主体中心与光轴的大圆距离 < 阈值的帧占比。

**轨迹平滑度** trajectory_smoothness
    "运镜晃不晃？"
    角加加速度（jerk）的均值。人眼对加加速度敏感：匀速转动看起来平稳，
    而加速度突变会让人明显感到"顿挫"。
    注意算在**连续 yaw** 上 —— 归一化后跨接缝会得到虚假巨值。

**角速度符合率** angular_velocity_compliance
    "运镜是不是太快了？"
    角速度不超过配置上限的帧占比。

**主体可见率** subject_visibility_rate
    "主体有没有跑到画面外？"
    主体落在相机视锥内的帧占比。这个指标比"居中率"宽松，
    衡量的是硬性失败（主体完全丢失）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..geometry.camera import VirtualCamera
from ..geometry.sphere import great_circle_distance


@dataclass
class TrajectoryMetrics:
    """一条轨迹的评测结果。"""

    centering_rate: float
    """主体落在画面中央区域的帧占比。"""

    mean_offset_deg: float
    """主体与光轴的平均角距离。"""

    max_offset_deg: float
    """最大角距离（最差的一帧偏离多远）。"""

    smoothness: float
    """角加加速度均值，越小越平稳。"""

    angular_velocity_compliance: float
    """角速度不超限的帧占比。"""

    peak_angular_velocity_dps: float
    visibility_rate: float
    num_frames: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "centering_rate": round(self.centering_rate, 4),
            "mean_offset_deg": round(self.mean_offset_deg, 3),
            "max_offset_deg": round(self.max_offset_deg, 3),
            "smoothness": round(self.smoothness, 4),
            "angular_velocity_compliance": round(self.angular_velocity_compliance, 4),
            "peak_angular_velocity_dps": round(self.peak_angular_velocity_dps, 2),
            "visibility_rate": round(self.visibility_rate, 4),
            "num_frames": self.num_frames,
        }

    def __repr__(self) -> str:
        return (
            f"<TrajMetrics 居中 {self.centering_rate:.0%} "
            f"偏移 {self.mean_offset_deg:.1f}° "
            f"平滑 {self.smoothness:.2f} "
            f"合规 {self.angular_velocity_compliance:.0%}>"
        )


def subject_centering_rate(
    cam_yaw_deg: ArrayLike,
    cam_pitch_deg: ArrayLike,
    subj_yaw_deg: ArrayLike,
    subj_pitch_deg: ArrayLike,
    *,
    fov_deg: ArrayLike | float = 70.0,
    center_ratio: float = 1.0 / 3.0,
) -> float:
    """主体落在画面中央区域的帧占比。

    "中央区域"定义为画面中心的一个矩形，其半宽为 `center_ratio × fov/2`。
    默认 1/3 —— 即主体在画面中央三分之一的范围内。

    Args:
        cam_*: 相机逐帧朝向。
        subj_*: 主体逐帧朝向。
        fov_deg: 相机视场角（可逐帧变化）。
        center_ratio: 中央区域占半视场的比例。

    Returns:
        [0, 1] 的帧占比。
    """
    cam_yaw = np.asarray(cam_yaw_deg, dtype=np.float64)
    cam_pitch = np.asarray(cam_pitch_deg, dtype=np.float64)
    subj_yaw = np.asarray(subj_yaw_deg, dtype=np.float64)
    subj_pitch = np.asarray(subj_pitch_deg, dtype=np.float64)
    fov = np.broadcast_to(np.asarray(fov_deg, dtype=np.float64), cam_yaw.shape)

    if cam_yaw.size == 0:
        return 0.0

    offset = great_circle_distance(cam_yaw, cam_pitch, subj_yaw, subj_pitch)
    threshold = fov * center_ratio / 2.0
    return float(np.mean(offset <= threshold))


def subject_visibility_rate(
    cam_yaw_deg: ArrayLike,
    cam_pitch_deg: ArrayLike,
    subj_yaw_deg: ArrayLike,
    subj_pitch_deg: ArrayLike,
    *,
    fov_deg: ArrayLike | float = 70.0,
    aspect: float = 16.0 / 9.0,
) -> float:
    """主体落在相机视锥内的帧占比（硬性可用性指标）。

    用视锥的外接角半径做保守判断：主体在视锥内 ⟹ 角距离 < 外接角半径。
    这个判据偏保守（可能把边缘可见的主体判为不可见），
    但对"主体完全跑出画面"这类硬失败足够敏感。
    """
    cam_yaw = np.asarray(cam_yaw_deg, dtype=np.float64)
    if cam_yaw.size == 0:
        return 0.0

    fov = np.broadcast_to(np.asarray(fov_deg, dtype=np.float64), cam_yaw.shape)
    # 外接角半径 ≈ 对角视场的一半
    diag_fov = fov * np.sqrt(1.0 + 1.0 / (aspect**2)) / 2.0

    offset = great_circle_distance(
        cam_yaw,
        np.asarray(cam_pitch_deg, dtype=np.float64),
        np.asarray(subj_yaw_deg, dtype=np.float64),
        np.asarray(subj_pitch_deg, dtype=np.float64),
    )
    return float(np.mean(offset <= diag_fov))


def trajectory_smoothness(
    yaw_continuous_deg: ArrayLike,
    pitch_deg: ArrayLike,
    times_s: ArrayLike,
) -> float:
    """角加速度 RMS，单位为 度/秒²。

    **必须传入连续的 yaw**（unwrapped）—— 用归一化值算会在跨接缝处
    得到 358° 的虚假差分，让最平滑的轨迹看起来最糟。

    ------------------------------------------------------------------
    为什么用二阶差分（角加速度）而不是三阶（jerk）
    ------------------------------------------------------------------

    早期版本用的是 jerk（Δ³θ/Δt³），实测发现两个问题：

    1. **与优化目标不一致**
       轨迹求解器优化的是二阶差分。用三阶差分衡量它，等于拿一把
       和目标不对应的尺子量 —— 会出现"优化器把二阶差分降了 17 倍，
       而指标几乎不动"的怪象。

    2. **量纲不可跨帧率比较**
       jerk 除以 Δt³，15fps 与 30fps 下同一条轨迹会得到相差 8 倍的
       数值。这使得跨数据集的对比失去意义。

    角加速度（除以 Δt²）在两方面都更稳健：物理含义明确
    （人眼对加速度突变敏感），且与求解者优化的是同一个量。
    """
    y = np.asarray(yaw_continuous_deg, dtype=np.float64)
    p = np.asarray(pitch_deg, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)

    if len(y) < 3:
        return 0.0

    dt = np.diff(t)
    dt = np.where(dt > 1e-9, dt, 1e-9)

    # 二阶差分除以对应的时间间隔乘积，得到角加速度
    accel_yaw = np.diff(y, n=2) / (dt[:-1] * dt[1:])
    accel_pitch = np.diff(p, n=2) / (dt[:-1] * dt[1:])

    return float(np.sqrt(np.mean(accel_yaw**2 + accel_pitch**2)))


def angular_velocity_compliance(
    yaw_continuous_deg: ArrayLike,
    pitch_deg: ArrayLike,
    times_s: ArrayLike,
    *,
    max_dps: float = 45.0,
) -> tuple[float, float]:
    """角速度符合率。

    Returns:
        (符合率, 峰值角速度)
    """
    y = np.asarray(yaw_continuous_deg, dtype=np.float64)
    p = np.asarray(pitch_deg, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)

    if len(y) < 2:
        return 1.0, 0.0

    dt = np.diff(t)
    dt = np.where(dt > 1e-9, dt, 1e-9)
    speed = np.sqrt(np.diff(y) ** 2 + np.diff(p) ** 2) / dt

    return float(np.mean(speed <= max_dps)), float(speed.max())


def evaluate_trajectory(
    cam_yaw_deg: ArrayLike,
    cam_pitch_deg: ArrayLike,
    cam_fov_deg: ArrayLike,
    times_s: ArrayLike,
    subj_yaw_deg: ArrayLike,
    subj_pitch_deg: ArrayLike,
    *,
    max_angular_vel_dps: float = 45.0,
    center_ratio: float = 1.0 / 3.0,
    yaw_continuous_deg: ArrayLike | None = None,
) -> TrajectoryMetrics:
    """一次性算出一条轨迹的全部指标。

    Args:
        yaw_continuous_deg: 连续 yaw。None 则内部 unwrap cam_yaw_deg。
            若调用方已有连续值（求解器会保留），传入可避免重复计算。
    """
    cam_yaw = np.asarray(cam_yaw_deg, dtype=np.float64)
    cam_pitch = np.asarray(cam_pitch_deg, dtype=np.float64)
    cam_fov = np.broadcast_to(
        np.asarray(cam_fov_deg, dtype=np.float64), cam_yaw.shape
    )
    times = np.asarray(times_s, dtype=np.float64)

    if yaw_continuous_deg is None:
        from ..geometry.sphere import unwrap_yaw

        yaw_c = np.asarray(unwrap_yaw(cam_yaw), dtype=np.float64)
    else:
        yaw_c = np.asarray(yaw_continuous_deg, dtype=np.float64)

    offset = great_circle_distance(
        cam_yaw,
        cam_pitch,
        np.asarray(subj_yaw_deg, dtype=np.float64),
        np.asarray(subj_pitch_deg, dtype=np.float64),
    )

    compliance, peak = angular_velocity_compliance(
        yaw_c, cam_pitch, times, max_dps=max_angular_vel_dps
    )

    return TrajectoryMetrics(
        centering_rate=subject_centering_rate(
            cam_yaw, cam_pitch, subj_yaw_deg, subj_pitch_deg,
            fov_deg=cam_fov, center_ratio=center_ratio,
        ),
        mean_offset_deg=float(np.mean(offset)) if offset.size else 0.0,
        max_offset_deg=float(np.max(offset)) if offset.size else 0.0,
        smoothness=trajectory_smoothness(yaw_c, cam_pitch, times),
        angular_velocity_compliance=compliance,
        peak_angular_velocity_dps=peak,
        visibility_rate=subject_visibility_rate(
            cam_yaw, cam_pitch, subj_yaw_deg, subj_pitch_deg, fov_deg=cam_fov
        ),
        num_frames=len(cam_yaw),
    )


# ============================================================== 跟踪指标


def tracking_id_switches(track_ids_per_frame: list[list[int]]) -> int:
    """统计 ID 切换次数。

    这里用的近似定义：**每帧新增的、之前从未出现过的 ID 数量之和**。

    一个健康的跟踪器，新增 ID 数应接近"场景中真实目标数"；
    若跟踪频繁失败重建，这个数字会远超真实目标数。

    注意这不是标准的 MOTA/IDF1 —— 那些需要逐帧的 GT 对应关系，
    而我们的合成序列没有这种标注。用"新增 ID 数"作为代理指标，
    足以反映"ID 是否稳定"这一核心性质。
    """
    seen: set[int] = set()
    switches = 0
    for ids in track_ids_per_frame:
        for tid in ids:
            if tid not in seen:
                seen.add(tid)
                switches += 1
    return switches


def id_stability_ratio(
    track_ids_per_frame: list[list[int]],
    *,
    expected_targets: int,
) -> float:
    """ID 稳定度：真实目标数 / 观测到的 ID 数。

    接近 1.0 表示每个真实目标自始至终用的是同一个 ID；
    接近 0 表示 ID 疯狂重建。**越高越好**。
    """
    if expected_targets <= 0:
        return 0.0
    observed = tracking_id_switches(track_ids_per_frame)
    if observed == 0:
        return 0.0
    return float(min(expected_targets / observed, 1.0))
