"""光流估计与运动能量 —— 高光检测的基础信号。

==============================================================================
在全景图上算光流的两个坑
==============================================================================

**坑 1：水平方向不连续**

光流算法默认图像边界之外「没有内容」，因此边界附近的光流会被施加
零梯度或复制边缘之类的约束。但 ERP 的左右边界是**同一条经线**，
那里本应连续。

后果：跨越接缝移动的目标，其光流在接缝处会出现巨大的虚假位移，
进而污染运动能量、显著性、高光检测等全部下游信号。

→ 解法：算光流前对图像做**环形 padding**，算完再裁回。
  这与 DA360 用 circular padding 消除 ERP 接缝是同一手法，
  也与 `track_spherical` 的 yaw 提升同源。

**坑 2：面积权重不均**

ERP 在高纬被严重压缩。若直接对所有像素的运动能量取平均，
极点附近的少数像素与赤道区域的等量像素贡献相同权重 ——
而极点恰恰是畸变最大、光流最不可靠的地方。

→ 解法：聚合时用 **cos φ 加权**，权重正比于该行代表的真实立体角。

==============================================================================
为什么主力用 DIS 而不是 RAFT
==============================================================================

DIS（Dense Inverse Search）是 OpenCV 内置的变分光流：纯 CPU，
512×256 下约 15ms/帧，**零显存占用**。

RAFT 精度更高，但在本项目 2.5GB 的可用显存下属于**独占模型**
（约 1.1GB），且 4K 输入直接爆显存。

因此策略是：**DIS 跑全片，RAFT 只在小样本上做精度对比** ——
这正好构成技术报告里「为什么选 DIS」的实验依据。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from ..geometry.projection import circular_crop_erp, circular_pad_erp
from ..geometry.sphere import erp_row_solid_angles

logger = logging.getLogger(__name__)

PresetName = Literal["ultrafast", "fast", "medium"]

_DIS_PRESETS = {
    "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
}

# 纬度分带（度）—— 用于分带运动统计
_LATITUDE_BANDS: dict[str, tuple[float, float]] = {
    "equatorial": (-30.0, 30.0),
    "mid_latitude": (30.0, 60.0),
    "high_latitude": (60.0, 90.0),
}


@dataclass
class FlowConfig:
    """光流配置。"""

    preset: PresetName = "medium"
    seam_pad_px: int = 64
    """环形 padding 宽度。应大于一帧内目标的最大位移，
    否则跨接缝运动仍会被边界约束污染。0 表示关闭（用于消融实验）。"""

    work_scale: float = 0.5
    """工作分辨率缩放（相对输入）。光流是低频信号，
    降采样几乎不损失信息，却能显著提速。"""


@dataclass
class MotionEnergy:
    """一帧对的运动能量统计。

    所有数值均基于 **cos φ 加权** 的球面平均，而非简单的像素平均 ——
    否则高纬区域的噪声会主导指标。
    """

    total: float
    """加权总运动能量。"""

    mean_weighted: float
    """按立体角加权的平均每像素位移（像素单位，工作分辨率下）。"""

    mean_naive: float
    """未加权的平均位移 —— 保留用于对比，展示加权带来的差异。"""

    max_magnitude: float
    """最大位移幅值（未加权）。"""

    band_energies: dict[str, float] = field(default_factory=dict)
    """按纬度分带的加权运动能量。"""

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "mean_weighted": self.mean_weighted,
            "mean_naive": self.mean_naive,
            "max_magnitude": self.max_magnitude,
            "band_energies": dict(self.band_energies),
        }


class DisFlowEstimator:
    """基于 OpenCV DIS 的光流估计器（CPU，零显存）。"""

    def __init__(self, config: FlowConfig | None = None) -> None:
        self.config = config or FlowConfig()
        self._dis = cv2.DISOpticalFlow_create(_DIS_PRESETS[self.config.preset])
        # 迭代次数：DIS 的默认值偏小，中等预设下适当提高可改善大位移
        self._dis.setFinestScale(1)
        self._dis.setPatchSize(8)
        self._dis.setPatchStride(4)

    # ------------------------------------------------------------ 光流

    def compute(
        self,
        prev: NDArray,
        curr: NDArray,
        *,
        use_seam_padding: bool = True,
    ) -> NDArray[np.float32]:
        """计算从 prev 到 curr 的光流。

        Args:
            prev, curr: 形状 (H, W) 或 (H, W, C) 的 ERP 图像。
            use_seam_padding: 是否启用环形 padding。
                设为 False 可用于消融实验，量化"跨接缝污染"的影响。

        Returns:
            形状 (H, W, 2) 的 float32 光流，单位是**工作分辨率**下的像素。
        """
        h, w = prev.shape[:2]
        if curr.shape[:2] != (h, w):
            raise ValueError(f"两帧尺寸不一致: {prev.shape[:2]} vs {curr.shape[:2]}")

        prev_gray = _to_gray(prev)
        curr_gray = _to_gray(curr)

        scale = self.config.work_scale
        if scale != 1.0:
            new_size = (max(int(round(w * scale)), 8), max(int(round(h * scale)), 8))
            prev_gray = cv2.resize(prev_gray, new_size, interpolation=cv2.INTER_AREA)
            curr_gray = cv2.resize(curr_gray, new_size, interpolation=cv2.INTER_AREA)

        pad = self.config.seam_pad_px if use_seam_padding else 0
        if pad > 0:
            pad_scaled = max(int(round(pad * scale)), 1)
            prev_gray = circular_pad_erp(prev_gray, pad_scaled)
            curr_gray = circular_pad_erp(curr_gray, pad_scaled)

        flow = self._dis.calc(prev_gray, curr_gray, None)

        if pad > 0:
            flow = circular_crop_erp(flow, max(int(round(pad * scale)), 1))

        return flow.astype(np.float32)

    # ------------------------------------------------------------ 能量

    def motion_energy(
        self,
        prev: NDArray,
        curr: NDArray,
        *,
        use_seam_padding: bool = True,
    ) -> MotionEnergy:
        """计算两帧之间的球面加权运动能量。"""
        flow = self.compute(prev, curr, use_seam_padding=use_seam_padding)
        return spherical_motion_energy(flow)


def spherical_motion_energy(flow: NDArray) -> MotionEnergy:
    """把光流场聚合为**球面加权**的运动能量。

    权重取自 `erp_row_solid_angles` —— 每一行代表的真实立体角。
    这样高纬像素不会再因为"数量多"而获得不成比例的权重。

    Args:
        flow: 形状 (H, W, 2) 的光流场。

    Returns:
        MotionEnergy 统计。
    """
    flow = np.asarray(flow, dtype=np.float64)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"光流场形状必须是 (H,W,2)，实际 {flow.shape}")

    height, width = flow.shape[:2]

    magnitude = np.hypot(flow[..., 0], flow[..., 1])
    total_pixels = float(height * width)

    # 每行的立体角权重（总和为 4π）
    row_weights = erp_row_solid_angles(height)
    weight_map = np.broadcast_to(row_weights[:, None], (height, width))
    weight_sum = float(weight_map.sum())

    mean_weighted = float((magnitude * weight_map).sum() / weight_sum)
    mean_naive = float(magnitude.mean())

    # 分带统计：把纬度范围映射到行范围
    bands: dict[str, float] = {}
    for name, (lat_lo, lat_hi) in _LATITUDE_BANDS.items():
        r0, r1 = _latitude_to_rows(lat_lo, lat_hi, height)
        if r1 <= r0:
            bands[name] = 0.0
            continue
        band_mag = magnitude[r0:r1]
        band_w = weight_map[r0:r1]
        ws = float(band_w.sum())
        bands[name] = float((band_mag * band_w).sum() / ws) if ws > 0 else 0.0

    return MotionEnergy(
        total=float((magnitude * weight_map).sum()),
        mean_weighted=mean_weighted,
        mean_naive=mean_naive,
        max_magnitude=float(magnitude.max()),
        band_energies=bands,
    )


def _latitude_to_rows(lat_lo: float, lat_hi: float, height: int) -> tuple[int, int]:
    """把纬度区间（度）映射为 ERP 行区间 [r0, r1)。

    ERP 中行 j 覆盖纬度 [90 - (j+1)·180/H, 90 - j·180/H]，
    因此纬度 lat 对应行 (90 - lat)/180 · H。
    """
    lat_lo, lat_hi = min(lat_lo, lat_hi), max(lat_lo, lat_hi)

    # 行号随纬度升高而减小
    row_of_lat = lambda lat: (90.0 - lat) / 180.0 * height  # noqa: E731
    r0 = int(np.floor(row_of_lat(lat_hi)))
    r1 = int(np.ceil(row_of_lat(lat_lo)))

    return max(r0, 0), min(r1, height)


def _to_gray(image: NDArray) -> NDArray[np.uint8]:
    """转灰度 uint8。"""
    img = np.asarray(image)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = img[..., 0]

    if img.dtype != np.uint8:
        # 假设输入在 [0,1] 或任意范围，归一化到 uint8
        arr = img.astype(np.float64)
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        if hi - lo > 1e-9:
            arr = (arr - lo) / (hi - lo) * 255.0
        else:
            arr = np.zeros_like(arr)
        img = arr.astype(np.uint8)

    return np.ascontiguousarray(img)
