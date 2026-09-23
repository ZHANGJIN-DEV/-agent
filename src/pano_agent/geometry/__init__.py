"""球面几何层 —— 全项目的地基。

这一层是纯 numpy 实现，**不依赖 torch 或任何深度学习框架**。
这既是有意为之（几何是数学，不是学习问题），也带来工程上的好处：
它能在没有 GPU、没有完整环境的情况下独立开发与测试。

公共坐标约定见 `sphere` 模块的文档字符串。所有模块必须遵守该约定。
"""

from .sphere import (
    GREAT_CIRCLE_EPS,
    SPHERE_AREA_SR,
    fibonacci_sphere,
    great_circle_distance,
    sph_to_erp_pixel,
    erp_pixel_to_sph,
    sph_to_vec,
    solid_angle_weights,
    unwrap_yaw,
    vec_to_sph,
    wrap_pitch,
    wrap_yaw,
)

__all__ = [
    "GREAT_CIRCLE_EPS",
    "SPHERE_AREA_SR",
    "erp_pixel_to_sph",
    "fibonacci_sphere",
    "great_circle_distance",
    "sph_to_erp_pixel",
    "sph_to_vec",
    "solid_angle_weights",
    "unwrap_yaw",
    "vec_to_sph",
    "wrap_pitch",
    "wrap_yaw",
]
