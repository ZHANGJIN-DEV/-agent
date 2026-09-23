"""虚拟相机模型：全景视频中「取景」的数学定义。

==============================================================================
相机模型
==============================================================================

一台虚拟相机由 (yaw, pitch, roll, fov_x) 定义，输出的是一张 W×H 的
透视（NFOV, narrow field of view）图像。

相机坐标系约定（与球面坐标系一致，右手系）：
    +Z_cam  光轴（看向前方）
    +X_cam  右方
    +Y_cam  上方

世界系 ← 相机系 的旋转矩阵：

    R = R_y(yaw) · R_x(-pitch) · R_z(roll)

推导（从单位姿态 v0=(0,0,1) 出发）：
    1. 绕 X 轴抬升 pitch：R_x(-φ)·(0,0,1) = (0, sin φ, cos φ)
    2. 绕 Y 轴偏转 yaw：  R_y(λ)·(0, sin φ, cos φ) = (cos φ·sin λ, sin φ, cos φ·cos λ)
    3. 绕光轴旋转 roll（不改变光轴方向）

第 2 步的结果与 `sphere.sph_to_vec` 的定义完全一致 —— 这是两层之间
唯一的对接点，一旦不一致，所有渲染都会错位。

==============================================================================
为什么不能走捷径
==============================================================================

「把主体放到画面三分点」不是线性映射，而是：
    球面坐标 → 世界单位向量 →（R⁻¹）→ 相机系 →（K）→ 像素坐标
跳过任何一步都会在画面边缘产生肉眼可见的误差，因为透视投影是非线性的。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .sphere import sph_to_vec, vec_to_sph


def rotation_y(angle_deg: float) -> NDArray[np.float64]:
    """绕 Y 轴旋转（右手定则）。"""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotation_x(angle_deg: float) -> NDArray[np.float64]:
    """绕 X 轴旋转（右手定则）。"""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rotation_z(angle_deg: float) -> NDArray[np.float64]:
    """绕 Z 轴旋转（右手定则）。"""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass(frozen=True)
class VirtualCamera:
    """一台全景世界中的虚拟相机。

    Attributes:
        yaw_deg:   光轴方位角，度。
        pitch_deg: 光轴俯仰角，度。
        roll_deg:  绕光轴的旋转，度。正值使画面内容逆时针旋转。
        fov_x_deg: **水平**视场角（度）。垂直视场角由宽高比决定。
        width:     输出图像宽度（像素）。
        height:    输出图像高度（像素）。
    """

    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    fov_x_deg: float = 90.0
    width: int = 1920
    height: int = 1080

    def __post_init__(self) -> None:
        if not (0.0 < self.fov_x_deg < 180.0):
            raise ValueError(f"水平视场角必须在 (0, 180) 内，实际为 {self.fov_x_deg}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"图像尺寸必须为正，实际为 {self.width}×{self.height}")

    # ------------------------------------------------------------ 基础量

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def focal_px(self) -> float:
        """焦距（像素）。由水平视场角定义。"""
        return (self.width / 2.0) / np.tan(np.radians(self.fov_x_deg) / 2.0)

    @property
    def fov_y_deg(self) -> float:
        """垂直视场角（度），由宽高比导出。

        注意：fov_y ≠ fov_x / aspect —— 透视投影下它们不是线性关系。
        这是「景别」映射中容易出错的地方。
        """
        return float(np.degrees(2.0 * np.arctan(np.tan(np.radians(self.fov_x_deg) / 2.0) / self.aspect)))

    # ------------------------------------------------------------ 矩阵

    def rotation_matrix(self) -> NDArray[np.float64]:
        """世界系 ← 相机系 的 3×3 旋转矩阵。"""
        return (
            rotation_y(self.yaw_deg)
            @ rotation_x(-self.pitch_deg)
            @ rotation_z(self.roll_deg)
        )

    def intrinsics(self) -> NDArray[np.float64]:
        """3×3 内参矩阵 K。

        主点取 (W-1)/2 而非 W/2 —— 这是「整数索引 = 像素中心」约定的必然结果。
        对偶数尺寸两者相差 0.5 像素，与 `sphere.sph_to_erp_pixel` 的偏移同源。
        """
        f = self.focal_px
        cx, cy = (self.width - 1) / 2.0, (self.height - 1) / 2.0
        return np.array(
            [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    # ------------------------------------------------------------ 光线生成

    def pixel_rays(self) -> NDArray[np.float64]:
        """返回每个像素对应的**世界系单位方向**，形状 (H, W, 3)。

        这是渲染的入口：对每个像素求一条光线方向，再在 ERP 上按方向采样。
        """
        # 像素中心约定：整数索引即像素中心（与 sphere 模块保持一致）
        xs = np.arange(self.width, dtype=np.float64)
        ys = np.arange(self.height, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys)  # (H, W)

        f = self.focal_px
        cx, cy = (self.width - 1) / 2.0, (self.height - 1) / 2.0

        # 相机系方向：+Z 为光轴，+X 右，+Y 上。
        # 图像 y 向下，故此处取负号。
        x_cam = (grid_x - cx) / f
        y_cam = -(grid_y - cy) / f
        z_cam = np.ones_like(x_cam)

        dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
        dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

        # 转到世界系：v_world = R @ v_cam
        rot = self.rotation_matrix()
        return dirs_cam @ rot.T

    def pixel_angles(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """返回每个像素对应的 (yaw, pitch) 网格，形状均为 (H, W)，单位为度。

        这是 `pixel_rays()` 的角度形式，供需要直接按角度采样而非向量运算的
        场合使用（例如按 ERF 像素索引做双线性插值）。
        """
        return vec_to_sph(self.pixel_rays())

    # ------------------------------------------------------------ 正向投影

    def project(
        self, yaw_deg: ArrayLike, pitch_deg: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """把世界系中的球面点投影到相机像素坐标。

        这是 `pixel_rays()` 的逆过程，构图约束（主体居中、三分法）都依赖它。

        Args:
            yaw_deg, pitch_deg: 球面角（度），可广播。

        Returns:
            (x_px, y_px, visible)：
                x_px, y_px —— 浮点像素坐标（像素中心约定）
                visible    —— 该点是否落在相机前方且视锥之内。
                              **必须检查此标志**：相机背后的点经过透视除法
                              后会得到一个看似合理的坐标，是隐蔽的 bug 来源。
        """
        v_world = sph_to_vec(yaw_deg, pitch_deg)
        rot = self.rotation_matrix()
        # 世界 → 相机：v_cam = Rᵀ @ v_world
        v_cam = v_world @ rot

        z = v_cam[..., 2]
        # 视锥背面判定：z <= 0 的点在相机后方
        front = z > 1e-9
        safe_z = np.where(front, z, 1.0)

        f = self.focal_px
        cx, cy = (self.width - 1) / 2.0, (self.height - 1) / 2.0
        x_px = f * v_cam[..., 0] / safe_z + cx
        y_px = -f * v_cam[..., 1] / safe_z + cy

        # 像素 i 覆盖区间 [i-0.5, i+0.5)，因此有效坐标范围是 [-0.5, size-0.5)
        visible = (
            front
            & (x_px >= -0.5)
            & (x_px < self.width - 0.5)
            & (y_px >= -0.5)
            & (y_px < self.height - 0.5)
        )
        return x_px, y_px, visible

    # ------------------------------------------------------------ 视场判定

    def angular_radius_deg(self) -> float:
        """相机视锥的外接角半径（度）—— 从光轴到最远角点的角距离。

        用于快速剔除：球面点与光轴的角距离若大于此值，必然不在画面内。
        """
        half_diag = np.hypot(self.width / 2.0, self.height / 2.0)
        return float(np.degrees(np.arctan(half_diag / self.focal_px)))

    def __repr__(self) -> str:
        return (
            f"VirtualCamera(yaw={self.yaw_deg:.2f}°, pitch={self.pitch_deg:.2f}°, "
            f"roll={self.roll_deg:.2f}°, fov_x={self.fov_x_deg:.1f}°, "
            f"{self.width}×{self.height})"
        )
