"""SIR —— Structured Intermediate Representation（结构化中间表示）。

本包实现 **Panorama Scene Graph（PSG）**，是全项目的技术核心。
它把感知层的连续输出转换为一种**同时服务于 LLM 规划与轨迹优化**的
结构化表示。

设计要点见 `schema` 模块的文档字符串。
"""

from .build import (
    BuildConfig,
    build_psg,
    load_numeric_layer,
    write_numeric_layer,
)
from .schema import (
    AudioInfo,
    EventNode,
    PanoramaSceneGraph,
    ShotNode,
    TrackNode,
    TrackStats,
    VideoMeta,
)

__all__ = [
    "AudioInfo",
    "BuildConfig",
    "EventNode",
    "PanoramaSceneGraph",
    "ShotNode",
    "TrackNode",
    "TrackStats",
    "VideoMeta",
    "build_psg",
    "load_numeric_layer",
    "write_numeric_layer",
]
