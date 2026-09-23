"""感知层：从 ERP 全景视频中提取结构化的目标轨迹与显著性。

本层的**统一约束**：所有几何量一律使用球面角度（度），不使用像素。
理由见 `geometry.sphere` —— ERP 是非均匀采样，同样 10 像素的间隔
在赤道与高纬代表的角距离可相差十倍以上。

模块划分:
    types.py            公共数据结构（检测、轨迹状态）
    kalman.py           轻量常量速度 Kalman 滤波器
    track_spherical.py  ★ 球面多目标跟踪（跨 ERP 边界）
    detect.py           YOLO 检测包装（ERP 适配）
    saliency.py         显著性预测（cos φ 加权）
    flow.py             光流（DIS / RAFT）
    registry.py         ★ 子进程模型隔离 + 显存预算调度
"""

from .types import FrameTracks, SphericalDetection, TrackState

__all__ = ["FrameTracks", "SphericalDetection", "TrackState"]
