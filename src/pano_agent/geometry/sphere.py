"""球面几何基础：坐标约定、距离度量、球面均匀采样。

==============================================================================
坐标约定（全项目统一，任何模块都不得偏离）
==============================================================================

角度（单位：度）
    yaw   （经度 λ）∈ [-180, 180)   0° 指向 +Z（正前方），+90° 指向 +X（右方）
    pitch （纬度 φ）∈ [-90, 90]     +90° 指向 +Y（天顶），-90° 指向 -Y（天底）
    roll             ∈ [-180, 180)  绕视线轴旋转

单位向量（右手系，Y 轴向上）
    x = cos(φ) · sin(λ)
    y = sin(φ)
    z = cos(φ) · cos(λ)

ERP 图像（等距柱状投影，尺寸 W × H）
    x_px = (λ + 180) / 360 · W      ← 水平环绕：x=0 与 x=W 是同一条经线
    y_px = (90 - φ) / 180 · H       ← 垂直：图像上方对应 +φ

    **像素中心约定**：像素 (i, j) 的中心位于 (i+0.5, j+0.5)。
    本模块所有像素↔角度换算都遵循该约定。差半个像素在球面 IoU 上会体现为
    系统性偏差，且极难察觉 —— 务必保持前后一致。

==============================================================================
为什么这些函数必须存在
==============================================================================

ERP 是球面的**非均匀采样**：x 方向每像素对应的球面弧长正比于 cos φ。
    赤道处 10 像素 ≈ 0.3°，高纬处 10 像素 ≈ 3° —— 相差十倍。
因此任何直接在 ERP 像素上做的距离、面积、IoU 计算都是错的，
必须经本模块换算到球面角度或单位向量上再做。这是整个项目的第一性原理。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# 球面总立体角：4π 球面度
SPHERE_AREA_SR = 4.0 * np.pi

# arccos/arcsin 的数值保护阈值。浮点误差可能让参数略微超出 [-1, 1]，
# 直接传入会产生 NaN —— 这是球面几何实现中最常见的静默 bug 来源。
GREAT_CIRCLE_EPS = 1e-12

# 黄金角，用于 Fibonacci 球面采样
_GOLDEN_ANGLE_RAD = np.pi * (3.0 - np.sqrt(5.0))


# ============================================================== 角度归一化


def wrap_yaw(yaw: ArrayLike) -> NDArray[np.float64] | float:
    """把 yaw 归一化到 [-180, 180)。

    球面上 yaw 是周期量：181° 与 -179° 是同一个方向。
    滤波、求差、插值之前必须先归一化，否则 ±180° 处会出现假跳变。
    """
    arr = np.asarray(yaw, dtype=np.float64)
    wrapped = (arr + 180.0) % 360.0 - 180.0
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def wrap_pitch(pitch: ArrayLike) -> NDArray[np.float64] | float:
    """把 pitch 截断到 [-90, 90]。

    pitch 不是周期量而是有界量：越过天顶在物理上意味着同时翻转 yaw，
    那是另一种参数化问题。本项目中将其截断（相机不会翻到背面）。
    """
    arr = np.asarray(pitch, dtype=np.float64)
    clipped = np.clip(arr, -90.0, 90.0)
    return float(clipped) if clipped.ndim == 0 else clipped


def unwrap_yaw(yaw: ArrayLike) -> NDArray[np.float64]:
    """把 yaw 序列从 S¹「提升」到 ℝ，使相邻差值的绝对值不超过 180°。

    这是本项目最关键的一个技巧：**yaw 生活在圆 S¹ 上，但只能在线性的 ℝ 上做
    滤波、求导与优化。**

    若不 lift 直接对 [-180,180) 的序列做 Kalman 滤波或轨迹优化，相机沿同一方向
    连续转动经过 ±180° 时，轨迹会在数值上从 +179 跳到 -179 —— 滤波器会认为
    发生了 358° 的瞬时转动，轨迹随之出现尖峰与抖动。

    提升后，179° → 181° → 185° 是连续的数值，物理意义同样是连续的旋转。
    这与 DA360 用 circular padding 消除 ERP 接缝是同一思想的两种体现：
    **把周期性打包进表示，而不是在每个使用者处单独处理。**

    Args:
        yaw: 角度序列（度）。标量输入会被视为长度为 1 的序列。

    Returns:
        提升后的 float64 数组，首元素与输入相同。
    """
    arr = np.atleast_1d(np.asarray(yaw, dtype=np.float64))
    if arr.size == 0:
        return arr.copy()
    # np.unwrap 以 period 为周期累积最小跳变；以 360° 为周期正合适
    return np.unwrap(arr, period=360.0)


# ============================================================== 球面 ↔ 向量


def sph_to_vec(yaw: ArrayLike, pitch: ArrayLike) -> NDArray[np.float64]:
    """球面角 → 单位向量。

    Args:
        yaw, pitch: 度。可为标量、数组或可广播的组合。

    Returns:
        形状为 (..., 3) 的数组，末维是 (x, y, z)，且模长为 1。
    """
    lam = np.radians(np.asarray(yaw, dtype=np.float64))
    phi = np.radians(np.asarray(pitch, dtype=np.float64))
    cos_phi = np.cos(phi)
    x = cos_phi * np.sin(lam)
    y = np.sin(phi)
    z = cos_phi * np.cos(lam)
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def vec_to_sph(vec: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """单位向量 → 球面角 (yaw, pitch)，单位为度。

    内部用 arctan2 而非 arcsin/arccos，好处有二：
    1. 全象限正确，不需要额外判断符号；
    2. 在两个极点附近数值稳定（arccos 在 ±1 附近导数发散）。
    """
    v = np.asarray(vec, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"末维必须是 3（xyz），实际为 {v.shape[-1]}")

    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm < GREAT_CIRCLE_EPS):
        raise ValueError("零向量无法转换为球面角")
    x, y, z = x / norm[..., 0], y / norm[..., 0], z / norm[..., 0]

    yaw = np.degrees(np.arctan2(x, z))
    # 输入已归一化，故 clip 只用于吸收浮点误差
    pitch = np.degrees(np.arcsin(np.clip(y, -1.0, 1.0)))
    return wrap_yaw(yaw), pitch


# ============================================================== 距离度量


def great_circle_distance(
    yaw_a: ArrayLike,
    pitch_a: ArrayLike,
    yaw_b: ArrayLike,
    pitch_b: ArrayLike,
) -> NDArray[np.float64] | float:
    """球面上两点的大圆（测地线）距离，单位为**度**。

    这是本项目中「两个目标有多远」的唯一正确度量。像素欧氏距离在 ERP 上
    没有物理意义：同样 100 像素的间隔，在赤道与在高纬代表的实际角距离
    可以相差十倍以上。

    使用 haversine 形式而非直接 arccos(dot)，因为后者在两点接近时
    精度急剧下降（acos 在 1 附近导数发散）。

    Returns:
        角距离（度），范围 [0, 180]。
    """
    lam1, phi1 = np.radians(yaw_a), np.radians(pitch_a)
    lam2, phi2 = np.radians(yaw_b), np.radians(pitch_b)
    lam1, phi1, lam2, phi2 = np.broadcast_arrays(lam1, phi1, lam2, phi2)

    d_phi = phi2 - phi1
    d_lam = lam2 - lam1

    h = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lam / 2.0) ** 2
    h = np.clip(h, 0.0, 1.0)
    dist = np.degrees(2.0 * np.arcsin(np.sqrt(h)))

    return float(dist) if dist.ndim == 0 else dist


# ============================================================== 球面采样


def fibonacci_sphere(n: int, *, return_areas: bool = False):
    """在单位球面上生成 n 个近似均匀分布的点（Fibonacci / 黄金螺旋）。

    用途：球面 IoU、显著性评测的积分节点。相比按经纬度网格采样，
    Fibonacci 采样在高纬不会过度密集，因此**无需再做 cos φ 加权**——
    每个点代表的立体角近似相同。

    Args:
        n: 采样点数。
        return_areas: 为 True 时同时返回每个点代表的立体角（球面度），
            各元素之和严格等于 4π。

    Returns:
        (n, 3) 的单位向量数组；若 return_areas 则返回 (vecs, areas)。
    """
    if n <= 0:
        raise ValueError(f"采样点数必须为正，实际为 {n}")

    idx = np.arange(n, dtype=np.float64)
    # 沿 y 轴等间距分层，保证面积均匀
    y = 1.0 - 2.0 * (idx + 0.5) / n
    radius = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = _GOLDEN_ANGLE_RAD * idx

    vecs = np.stack([radius * np.cos(theta), y, radius * np.sin(theta)], axis=-1)
    # 数值修正：强行归一到单位长度，消除 sqrt 的累积误差
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)

    if not return_areas:
        return vecs
    areas = np.full(n, SPHERE_AREA_SR / n, dtype=np.float64)
    return vecs, areas


def solid_angle_weights(pitch: ArrayLike) -> NDArray[np.float64] | float:
    """给定纬度（度），返回该纬度处单位角度的立体角权重 ∝ cos φ。

    用于在 **ERP 网格**上做积分（显著性评估、球面 IoU 的栅格实现等）。
    ERP 网格在高纬被严重压缩，若不加权，极点附近几行像素会贡献
    与赤道同样多的权重 —— 这是全景评测中最常见的错误。

    返回值为归一化前的相对权重；需要绝对立体角时乘以 (Δλ·Δφ) 的弧度值。
    """
    phi = np.radians(np.asarray(pitch, dtype=np.float64))
    w = np.cos(phi)
    return float(w) if w.ndim == 0 else w


def erp_row_solid_angles(height: int) -> NDArray[np.float64]:
    """返回 ERP 图像每一行所代表的立体角（球面度），总和为 4π。

    行 j 覆盖纬度区间 φ ∈ [90 - (j+1)·180/H, 90 - j·180/H]。
    精确的立体角为 2π/H · (sin φ_upper - sin φ_lower)。

    Args:
        height: ERP 图像高度（像素）。

    Returns:
        长度为 height 的数组，sum(...) == 4π（在浮点精度内）。
    """
    if height <= 0:
        raise ValueError(f"高度必须为正，实际为 {height}")

    edges = np.radians(90.0 - 180.0 * np.arange(height + 1) / height)
    # 每行覆盖完整一圈经度（Δλ = 2π），因此立体角为
    #     ∫∫ cos φ dφ dλ = 2π · (sin φ_上 - sin φ_下)
    # 注意这里**不能**再除以 H —— Δλ 已经是整圈 2π，而 H 只影响 φ 方向的分割。
    return 2.0 * np.pi * np.abs(np.sin(edges[:-1]) - np.sin(edges[1:]))


# ============================================================== ERP 像素换算


def sph_to_erp_pixel(
    yaw: ArrayLike,
    pitch: ArrayLike,
    width: int,
    height: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """球面角 → ERP 浮点像素坐标（遵循像素中心约定）。

    返回的坐标是浮点数，可能落在 [0, W) 之外（yaw 绕行）或被截断（极点附近）。
    调用方若要用作索引，需自行处理环绕与截断 —— 本函数不做隐式假设。
    """
    yaw_arr = np.asarray(yaw, dtype=np.float64)
    pitch_arr = np.asarray(pitch, dtype=np.float64)

    # 减去 0.5 使「整数坐标 = 像素中心」：yaw=-180 落在最左边缘（坐标 -0.5），
    # 像素 0 的中心位于坐标 0。漏掉这个 -0.5 会引入系统性的半像素偏移，
    # 在球面 IoU 与轨迹优化里都极难察觉。
    x = (yaw_arr + 180.0) / 360.0 * width - 0.5
    y = (90.0 - pitch_arr) / 180.0 * height - 0.5
    return x, y


def erp_pixel_to_sph(
    x: ArrayLike,
    y: ArrayLike,
    width: int,
    height: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """ERP 像素坐标（像素中心）→ 球面角 (yaw, pitch)。

    与 sph_to_erp_pixel 严格互逆。
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    yaw = (x_arr + 0.5) / width * 360.0 - 180.0
    pitch = 90.0 - (y_arr + 0.5) / height * 180.0
    return wrap_yaw(yaw), pitch


def erp_pixel_angular_size(width: int, height: int) -> tuple[float, float]:
    """返回 ERP 图像中心（赤道）处单个像素的角尺寸 (Δyaw, Δpitch)，单位为度。

    这是理解「为什么不能用像素度量」的最直接数字：
    赤道处 Δyaw = 360/W，而在纬度 φ 处实际角宽为 (360/W)·cos φ。

    对 2048×1024 的 ERP，赤道处 Δyaw = 0.176°，Δpitch = 0.176°。
    """
    return 360.0 / width, 180.0 / height
