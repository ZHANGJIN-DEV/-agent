"""Baseline 实现 —— 用于量化本项目各组件的增量贡献。

==============================================================================
为什么 baseline 必须认真实现
==============================================================================

技术报告的说服力不来自「我们的系统能跑起来」，而来自
**每个设计决策各自贡献了多少**。没有 baseline 对比，
"Ours" 就只是一个 demo，无法回答"轨迹优化到底有没有用"这类问题。

这里实现 4 个**递进**的基线，每一层增加一个组件：

    CenterCrop       相机固定看正前方            地板线
    SaliencyGreedy   每帧对准最显著目标          证明「逐帧最优 ≠ 全局最优」
    SaliencySmooth   上一个 + 低通滤波           Pano2Vid 时代的标准做法
    Ours             完整方案（含约束优化）       本项目的贡献

==============================================================================
关键预期（也是这套 baseline 的意义所在）
==============================================================================

**SaliencyGreedy 会呈现一个非常有说服力的反差**：

    居中率    很高  —— 因为它每帧都精确对准了主体
    平滑度    极差  —— 因为逐帧贪心必然抖动

这说明"每帧画面都对"并不等于"成片好看"。观众看的是**运动**，
而贪心策略产生的轨迹在人眼里就是不停顿挫的抖动。

这正是"需要轨迹优化"的核心论据 —— 也是 Pano2Vid（ACCV 2016）
当年提出自动运镜问题时首先要解决的问题。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..geometry.sphere import unwrap_yaw, wrap_yaw
from ..optimize.solver import CameraTrajectory, TrajectorySolver

BaselineName = str  # "center_crop" | "saliency_greedy" | "saliency_smooth" | "ours"


@dataclass
class BaselineConfig:
    """所有 baseline 共用的配置。"""

    fps: float = 15.0
    target_fov_deg: float = 70.0
    max_angular_vel_dps: float = 45.0

    smooth_cutoff_hz: float = 0.8
    """saliency_smooth 的低通截止频率。

    取 0.8Hz 是刻意的：太低会让它跟不上主体（居中率掉下来），
    太高则平滑不了抖动（与 greedy 无异）。选一个让它"表现最好"的值，
    才能构成**公平的**对照 —— 不能让 baseline 因为参数没调好而输。
    """


# ============================================================== 各 baseline


def baseline_center_crop(
    n_frames: int,
    times: NDArray,
    *,
    config: BaselineConfig | None = None,
) -> CameraTrajectory:
    """地板线：相机固定看正前方。

    如果连这个都打不过，说明整个系统没有存在价值。
    """
    cfg = config or BaselineConfig()
    n = n_frames
    return CameraTrajectory(
        times=np.asarray(times, dtype=np.float64),
        yaw_deg=np.zeros(n),
        pitch_deg=np.zeros(n),
        fov_deg=np.full(n, cfg.target_fov_deg),
        yaw_continuous_deg=np.zeros(n),
    )


def baseline_saliency_greedy(
    subj_yaw_deg: ArrayLike,
    subj_pitch_deg: ArrayLike,
    times: NDArray,
    *,
    config: BaselineConfig | None = None,
) -> CameraTrajectory:
    """逐帧贪心：相机精确对准主体，**不做任何时序平滑**。

    这模拟了"每帧选最显著目标"的朴素做法 —— 也是 Pano2Vid 论文里
    作为对照的那类方法。

    预期表现：居中率极高、平滑度极差。
    """
    cfg = config or BaselineConfig()
    yaw = np.asarray(subj_yaw_deg, dtype=np.float64)
    pitch = np.asarray(subj_pitch_deg, dtype=np.float64)
    yaw_c = np.asarray(unwrap_yaw(yaw), dtype=np.float64)

    return CameraTrajectory(
        times=np.asarray(times, dtype=np.float64),
        yaw_deg=wrap_yaw(yaw_c),
        pitch_deg=np.clip(pitch, -90.0, 90.0),
        fov_deg=np.full(len(yaw_c), cfg.target_fov_deg),
        yaw_continuous_deg=yaw_c,
    )


def baseline_saliency_smooth(
    subj_yaw_deg: ArrayLike,
    subj_pitch_deg: ArrayLike,
    times: NDArray,
    *,
    config: BaselineConfig | None = None,
) -> CameraTrajectory:
    """显著性 + 低通滤波 —— Pano2Vid 时代的标准做法。

    比 greedy 多了一个零相位低通（`scipy.signal.filtfilt`），
    用来抑制逐帧抖动。

    **零相位**很重要：普通 IIR 滤波会引入相位延迟，
    让相机系统性地落后于主体 —— 那会让 baseline 显得比实际更差，
    不是公平的比较。
    """
    from scipy.signal import butter, filtfilt

    cfg = config or BaselineConfig()
    t = np.asarray(times, dtype=np.float64)
    yaw_c = np.asarray(unwrap_yaw(np.asarray(subj_yaw_deg, dtype=np.float64)))
    pitch = np.asarray(subj_pitch_deg, dtype=np.float64)

    if len(yaw_c) < 8:
        # 太短，滤波没有意义
        return baseline_saliency_greedy(subj_yaw_deg, subj_pitch_deg, times, config=cfg)

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / cfg.fps
    nyq = 0.5 / max(dt, 1e-9)
    cutoff = min(cfg.smooth_cutoff_hz, nyq * 0.9)
    b, a = butter(2, cutoff / nyq, btype="low")

    yaw_f = filtfilt(b, a, yaw_c)
    pitch_f = filtfilt(b, a, pitch)

    return CameraTrajectory(
        times=t,
        yaw_deg=wrap_yaw(yaw_f),
        pitch_deg=np.clip(pitch_f, -90.0, 90.0),
        fov_deg=np.full(len(yaw_f), cfg.target_fov_deg),
        yaw_continuous_deg=yaw_f.copy(),
    )


def baseline_ours(
    subj_yaw_deg: ArrayLike,
    subj_pitch_deg: ArrayLike,
    times: NDArray,
    *,
    config: BaselineConfig | None = None,
    motion: str = "follow",
    rule_of_thirds: bool = False,
) -> CameraTrajectory:
    """完整方案：约束优化求解轨迹。"""
    cfg = config or BaselineConfig()
    solver = TrajectorySolver()
    if cfg.max_angular_vel_dps != solver.config.max_angular_vel_dps:
        from ..optimize.solver import TrajectoryConfig

        solver.config = TrajectoryConfig(max_angular_vel_dps=cfg.max_angular_vel_dps)

    return solver.solve_shot(
        subj_yaw_deg,
        subj_pitch_deg,
        times=np.asarray(times, dtype=np.float64),
        motion=motion,  # type: ignore[arg-type]
        target_fov_deg=cfg.target_fov_deg,
        rule_of_thirds=rule_of_thirds,
    )


# ============================================================== 统一入口

ALL_BASELINES: tuple[str, ...] = (
    "center_crop",
    "saliency_greedy",
    "saliency_smooth",
    "ours",
)


def run_baseline(
    name: str,
    subj_yaw_deg: ArrayLike,
    subj_pitch_deg: ArrayLike,
    times: NDArray,
    *,
    config: BaselineConfig | None = None,
) -> CameraTrajectory:
    """按名字运行指定的 baseline。"""
    cfg = config or BaselineConfig()
    t = np.asarray(times, dtype=np.float64)
    n = len(t)

    if name == "center_crop":
        return baseline_center_crop(n, t, config=cfg)
    if name == "saliency_greedy":
        return baseline_saliency_greedy(subj_yaw_deg, subj_pitch_deg, t, config=cfg)
    if name == "saliency_smooth":
        return baseline_saliency_smooth(subj_yaw_deg, subj_pitch_deg, t, config=cfg)
    if name == "ours":
        return baseline_ours(subj_yaw_deg, subj_pitch_deg, t, config=cfg)

    raise ValueError(f"未知的 baseline: {name!r}（可选: {', '.join(ALL_BASELINES)}）")
