"""球面区域表示与球面 IoU。

==============================================================================
为什么必须重新定义「框」和「IoU」
==============================================================================

在 ERP 图像上画一个矩形框，它对应的**球面区域不是矩形**：
- 在赤道附近，一个 w×h 像素的框近似是球面矩形，尚可接受；
- 在高纬，同样的框会被 cos φ 压缩，实际角宽度急剧变小；
- 跨过 ±180° 经线时，框在图像上被切成互不相连的两段。

因此 360VOT 提出用 **BFoV**（Bounding Field of View，球面包围视场）来
描述全景目标：用四元组 (中心 yaw, 中心 pitch, 角宽, 角高) 表示一个
**以中心为原点的切平面矩形**。旋转版本 rBFoV 再加一个 roll 角。

对应地，IoU 也必须在球面上计算：
    把区域判断下沉到球面，用球面均匀采样点做积分 —— 而不是在 ERP 上数像素。

若直接用像素 IoU，同样的数值在赤道与高纬的物理含义相差十倍；
实验中最直接的后果是：模型在高纬度的表现被系统性高估。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .sphere import (
    erp_pixel_to_sph,
    fibonacci_sphere,
    sph_to_vec,
    vec_to_sph,
    wrap_yaw,
)

# 球面 IoU 的默认采样点数。10000 点在球面上的面积误差约 0.5%，
# 对 IoU 的影响远小于标注本身的不确定性。
DEFAULT_IOU_SAMPLES = 10_000


def _tangent_basis(center_vec: NDArray[np.float64]) -> tuple[NDArray, NDArray, NDArray]:
    """构造以 center_vec 为法向的局部切平面基 (u, t, n)。

    处理极点退化：当光轴接近 ±Y 时，用 X 轴作为参考方向而非 Y 轴，
    否则叉积退化。
    """
    n = center_vec / np.linalg.norm(center_vec)
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.999 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n)
    u /= np.linalg.norm(u)
    t = np.cross(n, u)
    return u, t, n


def angular_offsets(
    yaw_deg: ArrayLike,
    pitch_deg: ArrayLike,
    center_yaw_deg: float,
    center_pitch_deg: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """把球面点表示为相对某个中心的**切平面角度偏移** (α, β)，单位为度。

    这是 gnomonic（球心）投影的角度形式：
        α = atan2(v·u, v·n)   —— 水平偏移
        β = atan2(v·t, v·n)   —— 垂直偏移

    与直接用 (Δyaw, Δpitch) 的差别在高纬和边缘处很大 —— 后者是
    「经纬度差」，不是「视线角差」。
    """
    v = sph_to_vec(yaw_deg, pitch_deg)
    center = sph_to_vec(center_yaw_deg, center_pitch_deg)
    u, t, n = _tangent_basis(center)

    vn = v @ n
    alpha = np.degrees(np.arctan2(v @ u, vn))
    beta = np.degrees(np.arctan2(v @ t, vn))
    return alpha, beta


@dataclass(frozen=True)
class SphericalBox:
    """球面上的一个包围视场（BFoV / rBFoV）。

    Attributes:
        yaw_deg:    中心方位角（度）。
        pitch_deg:  中心俯仰角（度）。
        width_deg:  水平角展度（度）。注意这是**角度**，不是像素。
        height_deg: 垂直角展度（度）。
        roll_deg:   切平面内的旋转角（度）。非零即 rBFoV。
    """

    yaw_deg: float
    pitch_deg: float
    width_deg: float
    height_deg: float
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.width_deg <= 0.0 or self.height_deg <= 0.0:
            raise ValueError(
                f"角展度必须为正，实际为 {self.width_deg}×{self.height_deg}"
            )
        if self.width_deg > 360.0 or self.height_deg > 180.0:
            raise ValueError("角展度超过整个球面")

    # ------------------------------------------------------------ 属性

    @property
    def center_vec(self) -> NDArray[np.float64]:
        return sph_to_vec(self.wrap_yaw(), self.pitch_deg)

    def wrap_yaw(self) -> float:
        """中心 yaw 的归一化值（在 [-180,180) 内）。"""
        return float(wrap_yaw(self.yaw_deg))

    @property
    def solid_angle_sr(self) -> float:
        """该区域近似的立体角（球面度）。

        用「角宽 × 角高（弧度）· cos(中心纬度)」近似 —— 对小区域足够准确，
        面积较大时会低估，仅供排序与筛选使用，不用于精确的 IoU 分母。
        """
        return float(
            np.radians(self.width_deg)
            * np.radians(self.height_deg)
            * np.cos(np.radians(self.pitch_deg))
        )

    # ------------------------------------------------------------ 判定

    def contains(self, yaw_deg: ArrayLike, pitch_deg: ArrayLike) -> NDArray[np.bool_]:
        """判断球面点是否落在该区域内。

        实现在中心切平面上做：把点投影为 (α, β) 偏移，再（对 rBFoV）
        反旋转 roll，最后与半宽半高比较。这样跨 ERP 边界的目标会被
        自然地判定为「一个连通区域」，不需要特殊处理。
        """
        alpha, beta = angular_offsets(yaw_deg, pitch_deg, self.wrap_yaw(), self.pitch_deg)

        if self.roll_deg != 0.0:
            r = np.radians(-self.roll_deg)
            cos_r, sin_r = np.cos(r), np.sin(r)
            alpha, beta = alpha * cos_r - beta * sin_r, alpha * sin_r + beta * cos_r

        return (np.abs(alpha) <= self.width_deg / 2.0) & (
            np.abs(beta) <= self.height_deg / 2.0
        )

    # ------------------------------------------------------------ IoU

    def iou(
        self,
        other: SphericalBox,
        *,
        n_samples: int = DEFAULT_IOU_SAMPLES,
    ) -> float:
        """球面 IoU。用 Fibonacci 球面均匀采样做蒙特卡洛积分。

        Fibonacci 采样在球面上近似等面积，因此各点权重相同，
        不需要再做 cos φ 加权 —— 这正是选择它而非经纬网格的原因。

        Returns:
            [0, 1] 之间的 IoU。
        """
        points = fibonacci_sphere(n_samples)
        yaw, pitch = vec_to_sph(points)

        in_self = self.contains(yaw, pitch)
        in_other = other.contains(yaw, pitch)

        union = np.count_nonzero(in_self | in_other)
        if union == 0:
            # 两个区域都为空 —— 定义 IoU 为 0（而非 NaN）
            return 0.0
        return float(np.count_nonzero(in_self & in_other) / union)

    def intersection_over_smaller(self, other: SphericalBox, *, n_samples: int = DEFAULT_IOU_SAMPLES) -> float:
        """交并比的小面积版本：交 / min(面积)。小目标评测常用。"""
        points = fibonacci_sphere(n_samples)
        yaw, pitch = vec_to_sph(points)
        in_self = np.count_nonzero(self.contains(yaw, pitch))
        in_other = np.count_nonzero(other.contains(yaw, pitch))
        inter = np.count_nonzero(self.contains(yaw, pitch) & other.contains(yaw, pitch))
        denom = min(in_self, in_other)
        return float(inter / denom) if denom > 0 else 0.0

    def center_distance_deg(self, other: SphericalBox) -> float:
        """两个区域中心的大圆角距离（度）。"""
        from .sphere import great_circle_distance

        return float(
            great_circle_distance(
                self.wrap_yaw(), self.pitch_deg, other.wrap_yaw(), other.pitch_deg
            )
        )

    # ------------------------------------------------------------ 构造

    @classmethod
    def from_erp_pixels(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        height: int,
        *,
        roll_deg: float = 0.0,
    ) -> SphericalBox:
        """从 ERP 上的像素框构造 BFoV。

        **支持跨接缝的框**：x 允许超出 [0, width]。此时框表示的是跨越
        ERP 左右边界的连续区域 —— 例如 `x1=2018, x2=2068` 表示一个
        中心在接缝处、宽 50 像素的目标。上游的
        `perception.detect._merge_seam_fragments` 会产出这种坐标。

        中心 yaw 用**圆统计**（circular mean）计算，而不是对端点取平均：
        对一个跨接缝的框，端点平均会得到 0°（正好是接缝的对面），
        而圆均值正确地给出 ±180°。

        注意这仍是一个**有损近似** —— ERP 上的轴对齐矩形在球面上一般不是
        BFoV。小框、非极高纬时误差可接受；用作 GT 时应直接使用数据集
        提供的球面标注。
        """
        # 四个角的球面坐标（不取中心点 —— 跨接缝时"中心"无意义）
        xs = np.array([x1, x1, x2, x2], dtype=np.float64)
        ys = np.array([y1, y2, y1, y2], dtype=np.float64)
        yaw, pitch = erp_pixel_to_sph(xs, ys, width, height)

        # 中心 yaw：圆均值。对 [174.8, 183.6] 给出 ≈179.2 而非 0。
        rad = np.radians(yaw)
        center_yaw = float(
            np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))
        )

        # 中心 pitch：垂直方向无周期性，直接取中点。
        # **必须 +0.5** —— 遵循「整数索引即像素中心」的全局约定，
        # 与 `erp_pixel_to_sph` 保持严格一致。漏掉它会让中心纬度偏移半像素，
        # 虽然肉眼难辨，但在往返测试里会稳定表现为 0.50px 的固定偏差。
        center_pitch = float(
            np.clip(90.0 - ((y1 + y2) / 2.0 + 0.5) / height * 180.0, -90.0, 90.0)
        )

        # 角展度：四角相对中心的切平面偏移的最大绝对值
        alpha, beta = angular_offsets(yaw, pitch, center_yaw, center_pitch)
        half_w = float(np.max(np.abs(alpha)))
        half_h = float(np.max(np.abs(beta)))

        return cls(
            yaw_deg=center_yaw,
            pitch_deg=center_pitch,
            # 理论上不会超过 360°，但极端输入下仍需夹紧以满足不变量
            width_deg=float(np.clip(2.0 * half_w, 1e-6, 360.0)),
            height_deg=float(np.clip(2.0 * half_h, 1e-6, 180.0)),
            roll_deg=roll_deg,
        )

    @classmethod
    def from_centered_points(
        cls,
        center_yaw_deg: float,
        center_pitch_deg: float,
        yaw_deg: ArrayLike,
        pitch_deg: ArrayLike,
    ) -> SphericalBox:
        """从给定的中心与一组边界点构造球面框。

        用于**投影域检测的反投影**：已知立方体面上框的四角方向，
        求覆盖它们的最小切平面矩形。

        与 `from_erp_pixels` 的区别在于中心是**已知的**（由圆统计给出），
        而非从像素坐标反推 —— 后者在跨接缝时会得到错误的中心。
        """
        alpha, beta = angular_offsets(
            yaw_deg, pitch_deg, center_yaw_deg, center_pitch_deg
        )
        half_w = float(np.max(np.abs(alpha)))
        half_h = float(np.max(np.abs(beta)))

        return cls(
            yaw_deg=float(center_yaw_deg),
            pitch_deg=float(np.clip(center_pitch_deg, -90.0, 90.0)),
            width_deg=float(np.clip(2.0 * half_w, 1e-6, 360.0)),
            height_deg=float(np.clip(2.0 * half_h, 1e-6, 180.0)),
        )

    @classmethod
    def from_bfov_values(cls, values: ArrayLike, *, is_rotated: bool = False) -> SphericalBox:
        """从 360VOT 标注格式构造。

        BFoV 为 (x, y, w, h)；rBFoV 为 (x, y, w, h, θ)，其中 θ 单位为**弧度**。
        """
        arr = np.asarray(values, dtype=np.float64).ravel()
        expected = 5 if is_rotated else 4
        if arr.size != expected:
            raise ValueError(f"{'rBFoV' if is_rotated else 'BFoV'} 需要 {expected} 个值，实际 {arr.size}")
        roll = float(np.degrees(arr[4])) if is_rotated else 0.0
        return cls(
            yaw_deg=float(arr[0]),
            pitch_deg=float(arr[1]),
            width_deg=float(arr[2]),
            height_deg=float(arr[3]),
            roll_deg=roll,
        )

    def to_bfov_values(self, *, as_rotated: bool = False) -> NDArray[np.float64]:
        """转回 360VOT 标注格式。rBFoV 的 roll 输出为弧度。"""
        base = [self.yaw_deg, self.pitch_deg, self.width_deg, self.height_deg]
        if as_rotated:
            base.append(float(np.radians(self.roll_deg)))
        return np.array(base, dtype=np.float64)

    def __repr__(self) -> str:
        roll = f", roll={self.roll_deg:.1f}°" if self.roll_deg else ""
        return (
            f"SphericalBox(yaw={self.yaw_deg:.2f}°, pitch={self.pitch_deg:.2f}°, "
            f"{self.width_deg:.2f}°×{self.height_deg:.2f}°{roll})"
        )
