"""轨迹优化层：把 Edit DSL 的约束编译成连续相机轨迹。

见 `solver` 模块的文档 —— 核心是在 **unwrapped** 角度空间求解，
这是本项目"周期量在数据边界处理一次"原则的第三次应用。
"""

from .solver import CameraTrajectory, TrajectoryConfig, TrajectorySolver

__all__ = ["CameraTrajectory", "TrajectoryConfig", "TrajectorySolver"]
