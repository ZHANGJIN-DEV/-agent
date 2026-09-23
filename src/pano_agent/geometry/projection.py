"""ERP ↔ 透视投影：全景取景的渲染核心。

==============================================================================
三条必须遵守的规则
==============================================================================

1. **水平方向环绕，垂直方向截断**
   ERP 的左右边界是同一条经线（x=0 与 x=W 相邻），而上下边界是南北极点
   （不连续）。因此所有采样、卷积、padding 在水平方向都要 wrap，
   在垂直方向只能 clamp。混合二者会同时产生接缝伪影和极点污染。

2. **采样点必须落在像素中心**
   本项目的像素中心约定为 (i+0.5, j+0.5)。在做双线性插值时，
   浮点坐标需先减去 0.5 再取整，否则整幅图像会有半个像素的系统性偏移。

3. **极点附近不插值、只取最近行**
   极点是 ERP 的奇点：整行像素对应球面上的同一个点。若在垂直方向做
   双线性插值，会把极点与相邻纬度的像素混合，产生扇形拖影。
   本实现在极点行附近自动退化为最近邻。

==============================================================================
与 circular padding 的关系
==============================================================================

DA360 用 circular padding 让卷积网络「看见」接缝两侧的连续性。
本模块的 `circular_pad_erp` 是同一思想在经典图像处理层面的体现：
把水平方向卷成环，使任何局部算子都天然跨接缝工作，
而不需要在每个算子内部单独处理边界。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .camera import VirtualCamera
from .sphere import sph_to_erp_pixel

InterpMode = Literal["bilinear", "nearest"]


# ============================================================== 环形 padding


def circular_pad_erp(image: NDArray, pad: int, *, axis: int = 1) -> NDArray:
    """对 ERP 图像沿**经度轴**做环绕 padding，其余轴做边缘 padding。

    这是让局部算子（卷积、平滑、梯度）正确跨接缝工作的标准做法。

    Args:
        image: 形状 (H, W) 或 (H, W, C) 的 ERP 图像。
        pad: 向两侧各扩展的像素数。
        axis: 经度轴。对 (H, W, C) 应为 1。

    Returns:
        扩展后的数组。沿 axis 方向长度为 W + 2*pad，且首尾连续。
    """
    if pad < 0:
        raise ValueError(f"padding 不能为负，实际为 {pad}")
    if pad == 0:
        return image.copy()

    pads = [(0, 0)] * image.ndim
    pads[axis] = (pad, pad)
    return np.pad(image, pads, mode="wrap")


def circular_crop_erp(padded: NDArray, pad: int, *, axis: int = 1) -> NDArray:
    """`circular_pad_erp` 的逆操作。"""
    if pad == 0:
        return padded.copy()
    sl = [slice(None)] * padded.ndim
    sl[axis] = slice(pad, -pad)
    return padded[tuple(sl)]


def horizontal_gradient_erp(image: NDArray) -> NDArray:
    """ERP 图像的水平（经度方向）梯度，**跨接缝连续**。

    直接用 np.gradient 会在 x=0 与 x=W-1 处产生虚假的巨大梯度，
    这正是"拼接伪影"在数值上的一种来源。本函数用环形差分避免该问题。
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim < 2:
        raise ValueError("需要至少二维数组")

    # 用环绕索引计算中心差分
    left = np.roll(img, 1, axis=1)
    right = np.roll(img, -1, axis=1)
    return (right - left) / 2.0


# ============================================================== 球面采样


def sample_erp(
    erp: NDArray,
    yaw_deg: ArrayLike,
    pitch_deg: ArrayLike,
    *,
    mode: InterpMode = "bilinear",
    pole_epsilon_deg: float = 2.0,
) -> NDArray:
    """在 ERP 图像上按**球面角度**采样。

    这是全景渲染的最小构件：给定一组方向，取回对应的颜色/特征。

    Args:
        erp: 形状 (H, W) 或 (H, W, C) 的 ERP 图像。
        yaw_deg, pitch_deg: 采样方向（度），形状需可广播。
        mode: 'bilinear' 或 'nearest'。
        pole_epsilon_deg: 距极点多少度以内退化为最近邻，避免极点拖影。

    Returns:
        与输入角度形状相同的采样结果，末尾追加通道维（若输入有通道）。
    """
    erp = np.asarray(erp)
    if erp.ndim not in (2, 3):
        raise ValueError(f"ERP 必须是 (H,W) 或 (H,W,C)，实际为 {erp.shape}")

    height, width = erp.shape[:2]
    has_channel = erp.ndim == 3

    yaw = np.broadcast_to(np.asarray(yaw_deg, dtype=np.float64), np.broadcast_shapes(
        np.shape(yaw_deg), np.shape(pitch_deg)
    ))
    pitch = np.broadcast_to(np.asarray(pitch_deg, dtype=np.float64), yaw.shape)

    x, y = sph_to_erp_pixel(yaw, pitch, width, height)

    # 水平：环绕。模运算保证 x ∈ [0, W)
    x = np.mod(x, width)
    # 垂直：截断到合法像素索引范围
    y = np.clip(y, 0.0, height - 1.0)

    if mode == "nearest":
        xi = np.mod(np.floor(x).astype(np.int64), width)
        yi = np.clip(np.floor(y).astype(np.int64), 0, height - 1)
        return erp[yi, xi] if not has_channel else erp[yi, xi]

    # ---- 双线性 ----
    # 极点附近退化：整行代表同一球面点，插值只会制造拖影
    near_pole = np.abs(np.abs(pitch) - 90.0) <= pole_epsilon_deg
    if np.any(near_pole):
        y = np.where(near_pole, np.where(pitch > 0, 0.0, height - 1.0), y)

    # 坐标已遵循「整数 = 像素中心」约定，直接 floor 即得左/上邻像素索引。
    # 全局只有 sph_to_erp_pixel 负责那一次 -0.5 偏移，这里不得重复施加。
    x0 = np.floor(x)
    y0 = np.floor(y)
    tx = x - x0
    ty = y - y0

    x0i = np.mod(x0.astype(np.int64), width)
    x1i = np.mod(x0i + 1, width)  # 水平相邻像素环绕
    y0i = np.clip(y0.astype(np.int64), 0, height - 1)
    y1i = np.clip(y0i + 1, 0, height - 1)

    if has_channel:
        tx = tx[..., None]
        ty = ty[..., None]

    v00 = erp[y0i, x0i]
    v01 = erp[y0i, x1i]
    v10 = erp[y1i, x0i]
    v11 = erp[y1i, x1i]

    top = v00 * (1.0 - tx) + v01 * tx
    bottom = v10 * (1.0 - tx) + v11 * tx
    return top * (1.0 - ty) + bottom * ty


# ============================================================== 渲染


def render_nfov(
    erp: NDArray,
    camera: VirtualCamera,
    *,
    mode: InterpMode = "bilinear",
) -> NDArray:
    """从 ERP 渲染出一张透视（NFOV）图像。

    Args:
        erp: 形状 (H, W) 或 (H, W, C) 的 ERP 图像。
        camera: 虚拟相机参数。
        mode: 插值方式。

    Returns:
        形状 (camera.height, camera.width) 或 (..., C) 的透视图像。
    """
    yaw, pitch = camera.pixel_angles()
    return sample_erp(erp, yaw, pitch, mode=mode)


def render_nfov_sequence(
    erp_frames: ArrayLike,
    cameras: list[VirtualCamera],
    *,
    mode: InterpMode = "bilinear",
) -> NDArray:
    """对一帧 ERP 渲染多台相机（或对多帧渲染对应相机）。

    Args:
        erp_frames: 形状 (N, H, W) 或 (N, H, W, C)，或单帧 (H, W[, C])。
        cameras: 长度需与 N 一致，或为 1（此时对每帧用同一相机）。

    Returns:
        形状 (N, out_h, out_w[, C]) 的透视序列。
    """
    frames = np.asarray(erp_frames)
    single = frames.ndim in (2, 3)
    if single:
        frames = frames[None, ...]

    if len(cameras) == 1:
        cameras = cameras * len(frames)
    if len(cameras) != len(frames):
        raise ValueError(
            f"相机数量({len(cameras)})需为 1 或与帧数({len(frames)})一致"
        )

    out = [render_nfov(f, c, mode=mode) for f, c in zip(frames, cameras, strict=True)]
    return np.stack(out, axis=0)


# ============================================================== CubeMap


_CUBE_FACE_UV = {
    # 名称: (u_axis, v_axis, forward) —— 均为世界系基向量
    "front": (1, 1, 2),   # +Z
    "back": (-1, 1, -2),
    "right": (-2, 1, 1),  # +X
    "left": (2, 1, -1),
    "up": (1, -2, 1),     # +Y
    "down": (1, 2, -1),
}


def erp_to_cubemap(erp: NDArray, face_size: int) -> dict[str, NDArray]:
    """ERP → 立方体贴图（6 个面）。

    立方体投影在极点附近畸变远小于 ERP，是几何敏感的视觉任务
    （深度、光流、匹配）的常用中间表示。本项目用它做可视化互验与
    可选的几何预处理。

    Args:
        erp: 形状 (H, W) 或 (H, W, C) 的 ERP 图像。
        face_size: 每个面的边长（像素）。

    Returns:
        {'front'|'back'|'left'|'right'|'up'|'down': (face_size, face_size[, C])}
    """
    erp = np.asarray(erp)
    grid = (np.arange(face_size, dtype=np.float64) + 0.5) / face_size * 2.0 - 1.0

    # 逐面构造方向向量，复用 sample_erp 完成采样
    from .sphere import vec_to_sph  # 局部导入避免循环依赖

    faces: dict[str, NDArray] = {}
    for name, (u_axis, v_axis, forward_axis) in _CUBE_FACE_UV.items():
        u = grid[:, None]
        v = grid[None, :]
        dirs = np.zeros((face_size, face_size, 3), dtype=np.float64)
        dirs[..., forward_axis - 1 if forward_axis > 0 else -forward_axis - 1] = 1.0
        # 按符号放置 u/v 分量
        dirs[..., abs(u_axis) - 1] += np.sign(u_axis) * u
        dirs[..., abs(v_axis) - 1] += np.sign(v_axis) * v
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        yaw, pitch = vec_to_sph(dirs)
        faces[name] = sample_erp(erp, yaw, pitch)

    return faces


# ============================================================== 工具


def erp_seam_difference(erp: ArrayLike) -> float:
    """度量 ERP 接缝处的不连续性：接缝差 / 内部平均相邻差。

    用于验证转码、拼接、滤波等操作是否破坏了环绕连续性。

    **取值解释**：这是一个相对指标，其绝对值依赖信号形状 ——
    在极值点附近接缝差天然偏小（如 cos 在接缝处导数为零，比值可低至 0.1），
    在过零点附近偏大（如 sin，比值约 1.6）。
    因此判据是「**远大于 1 即表示接缝被破坏**」，而不是「接近 1 即正常」。
    破坏与正常的典型值相差 2-3 个数量级，判据非常稳健。

    Returns:
        比值。连续信号通常 < 3；接缝断裂时可达 100 以上。
    """
    img = np.asarray(erp, dtype=np.float64)
    if img.shape[1] < 2:
        raise ValueError("宽度至少为 2")

    seam = np.abs(img[:, 0] - img[:, -1]).mean()
    interior = np.abs(img[:, 1:] - img[:, :-1]).mean()
    if interior < 1e-12:
        return 0.0 if seam < 1e-12 else float("inf")
    return float(seam / interior)
