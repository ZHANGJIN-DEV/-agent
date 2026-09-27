"""从实例分割图提取球面检测 —— 替代 YOLO 的感知源。

==============================================================================
为什么放弃 YOLO
==============================================================================

本项目最初计划用 YOLO 做检测，**实测证明方向错了**。

在 6 张真实全景图上跑（conf=0.05，已经是极低阈值）共 60 个检测：

    car 27 ✓    traffic light 8 ✓    truck 3 ✓    bus 1 ✓
    umbrella 8 ✗    airplane 6 ✗    clock 4 ✗    boat 3 ✗

    （城市街道上出现雨伞、飞机、船 —— 全是误检）

真实类别只有 4 类。根因是 **COCO 80 类与城市街景的交集极小**：
COCO 里根本没有 building / road / sign / window 这些街景主体。

当初我把"检测不到"归因于 ERP 畸变（"透视→全景鸿沟"），
**那是过度归因** —— 主因是类别不匹配。畸变确实存在，但不是主要矛盾。

==============================================================================
数据集自带的实例标注远好于 YOLO
==============================================================================

    YOLO                       实例分割（本模块）
    ──────────────────────     ──────────────────────────────
    COCO 80 类，与街景不匹配     25 类街景类别
    模型预测，有噪声             逐像素完美标注
    需 GPU 推理                  纯 CPU，零显存
    受 ERP 畸变影响              本身就是全景标注，无畸变问题

**这个数据集自带比 YOLO 更好的东西，绕过去用 COCO 是决策失误。**

==============================================================================
数据格式（实测确认）
==============================================================================

    A 通道  = 类别 ID（值域 [3, 28]，与语义分割图**完全一致**，已验证）
    R/G/B   = (类别, 实例) 的联合彩色编码 —— 每个独立个体一种独特颜色

    实测：类别 5 (concreteblock) 有 22 种颜色 = 22 个实例
          类别 3 (sky) 只有 1 种颜色 = 天空整体算一个

**没有独立的实例 ID 通道**，实例身份由 RGB 三元组承载。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..geometry.region import SphericalBox
from .types import SphericalDetection

logger = logging.getLogger(__name__)

# 背景类别 —— 面积巨大且无剪辑价值，不作为"目标"
BACKGROUND_CLASSES: frozenset[int] = frozenset({3, 4, 25, 26, 27})
"""sky(3) / land(4) / road(25) / sidewalk(26) / grounddirt(27)"""

CLASS_NAMES: dict[int, str] = {
    3: "sky", 4: "land", 5: "concreteblock", 6: "streetprops",
    7: "plasticcone", 8: "metalfence", 9: "pillar", 10: "lamp",
    11: "trashcan", 12: "postbox", 13: "umbrella", 14: "table",
    15: "chair", 16: "greenpot", 17: "roadcolumn", 18: "adplane",
    19: "buildingawning", 20: "scaffolding", 22: "plant",
    23: "ventilationtube", 24: "building", 25: "road",
    26: "sidewalk", 27: "grounddirt", 28: "hotdogpot",
}


@dataclass
class InstanceExtractorConfig:
    """实例提取配置。"""

    min_area_px: int = 300
    """最小实例面积。低于此值视为噪点（全景图里几个像素的区域无意义）。"""

    max_aspect: float = 25.0
    """最大长宽比。过于细长的区域通常是高纬畸变条带，不是真实目标。"""

    max_boxes_per_class: int = 8
    """每个类别最多保留多少个实例（按面积降序）。防止一个类别刷满列表。"""

    include_background: bool = False
    """是否包含背景类。默认 False —— 天空/道路不是剪辑关心的"主体"。"""

    min_confidence_area: float = 1.0
    """用于把实例面积映射为 confidence 的尺度（像素面积对数尺度）。"""


class InstanceExtractor:
    """从 omnidirectional 实例分割图提取球面检测。

    纯 CPU 实现（仅用 cv2 的连通域分析），**零显存占用** ——
    这也意味着它可以与 GPU 上的其他模型并行，不争抢那 3GB 的预算。
    """

    def __init__(self, config: InstanceExtractorConfig | None = None) -> None:
        self.config = config or InstanceExtractorConfig()

    def extract(self, instance_rgba: NDArray) -> list[SphericalDetection]:
        """从实例分割图提取球面检测。

        Args:
            instance_rgba: (H, W, 4) 的 uint8 数组。
                A 通道为类别 ID，RGB 为实例彩色编码。

        Returns:
            球面表示的检测列表，按置信度（≈实例大小）降序。
        """
        arr = np.asarray(instance_rgba)
        if arr.ndim != 3 or arr.shape[2] < 4:
            raise ValueError(f"需要 (H,W,4) 的 RGBA 实例图，实际 {arr.shape}")

        height, width = arr.shape[:2]
        label_map, label_info = self._build_label_map(arr, width, height)
        if not label_info:
            return []

        # 环形 padding：让跨 ERP 接缝的实例连成一体
        pad = max(width // 8, 32)
        padded = np.pad(label_map, ((0, 0), (pad, pad)), mode="wrap")

        # 一次性求出所有标签的包围盒与面积 —— 这是性能关键。
        # 逐实例做全图扫描（`rgb == color` + connectedComponents）在
        # 1024×2048 上要 17 秒；本实现是单次 O(N) 扫描，约快两个数量级。
        #
        # ⚠ 这里**刻意不用 scipy.ndimage.find_objects**：它返回的列表长度
        # 与 `np.bincount` 的结果并不总是一致（实测标签集为 {0,1,2,3} 时
        # 它返回 3 个元素而非 4 个），两者的下标含义对不上会导致
        # 每个实例拿到邻居的包围盒。用显式的 min/max 归约语义确定，
        # 不依赖任何库的索引约定。
        n_labels = int(padded.max()) + 1
        ys, xs = np.nonzero(padded)
        labs = padded[ys, xs]

        counts = np.bincount(labs, minlength=n_labels)
        min_y = np.full(n_labels, padded.shape[0], dtype=np.int64)
        min_x = np.full(n_labels, padded.shape[1], dtype=np.int64)
        max_y = np.zeros(n_labels, dtype=np.int64)
        max_x = np.zeros(n_labels, dtype=np.int64)
        np.minimum.at(min_y, labs, ys)
        np.maximum.at(max_y, labs, ys)
        np.minimum.at(min_x, labs, xs)
        np.maximum.at(max_x, labs, xs)

        detections: list[SphericalDetection] = []
        per_class_count: dict[str, int] = {}

        for label_id in range(1, n_labels):
            cid = label_info.get(label_id)
            if cid is None:
                continue  # 背景或非目标类别

            area = int(counts[label_id]) if label_id < len(counts) else 0
            if area < self.config.min_area_px:
                continue

            name = CLASS_NAMES[cid]
            if per_class_count.get(name, 0) >= self.config.max_boxes_per_class:
                continue

            x1 = float(min_x[label_id] - pad)
            y1 = float(min_y[label_id])
            x2 = float(max_x[label_id] + 1 - pad)
            y2 = float(max_y[label_id] + 1)

            pw, ph = x2 - x1, y2 - y1
            if pw <= 0 or ph <= 0:
                continue
            if max(pw, ph) / max(min(pw, ph), 1.0) > self.config.max_aspect:
                continue
            # 完全落在 padding 副本里的重复分量
            if x1 >= width or x2 <= 0:
                continue

            box = SphericalBox.from_erp_pixels(x1, y1, x2, y2, width, height)

            # 面积的对数尺度 → 置信度。这是**启发式**（大目标更值得关注），
            # 不是模型置信度 —— 与 highlight_score 同样是弱信号。
            conf = float(
                np.clip(
                    np.log1p(area) / np.log1p(self.config.min_confidence_area * 10000),
                    0.05, 1.0,
                )
            )

            detections.append(
                SphericalDetection(
                    yaw_deg=box.yaw_deg,
                    pitch_deg=box.pitch_deg,
                    width_deg=box.width_deg,
                    height_deg=box.height_deg,
                    confidence=conf,
                    category=name,
                )
            )
            per_class_count[name] = per_class_count.get(name, 0) + 1

        detections.sort(key=lambda d: -d.confidence)
        return detections

    # ------------------------------------------------------------ 内部

    def _build_label_map(
        self, arr: NDArray, width: int, height: int
    ) -> tuple[NDArray[np.int32], dict[int, int]]:
        """把 (类别, RGB) 组合编码为紧凑的整数标签图。

        Returns:
            (label_map, {label_id: class_id})

        编码的必要性：直接用 RGB 三元组判断"是否同一实例"需要三通道比较，
        而把它压成单个整数后可以用 `np.unique(..., return_inverse=True)`
        一次性完成，后续所有操作都是单通道的。
        """
        rgb = arr[..., :3].astype(np.int32)
        classes = arr[..., 3].astype(np.int32)

        # 类别 << 24 | R << 16 | G << 8 | B
        combined = (classes << 24) | (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]

        # 一次性得到紧凑标签（0 保留给"未使用"）
        _, inverse = np.unique(combined, return_inverse=True)
        label_map = (inverse.reshape(height, width) + 1).astype(np.int32)

        # 每个标签对应的类别；同时过滤掉类别不在白名单/属于背景的标签
        flat_combined = combined.ravel()
        n_labels = int(label_map.max()) + 1
        first_pixel = np.full(n_labels, -1, dtype=np.int64)
        uniq_labels, first_idx = np.unique(label_map.ravel(), return_index=True)
        first_pixel[uniq_labels] = first_idx

        # 筛出属于目标的标签（排除背景类与未知类）
        target_labels: list[int] = []
        target_classes: dict[int, int] = {}
        for lid in uniq_labels:
            if lid == 0:
                continue
            cid = int((flat_combined[first_pixel[lid]] >> 24) & 0xFF)
            if cid not in CLASS_NAMES:
                continue
            if not self.config.include_background and cid in BACKGROUND_CLASSES:
                continue
            target_labels.append(int(lid))
            target_classes[int(lid)] = cid

        # ★ 把目标标签**重新编号为连续的 1..N**。
        #
        # 这一步是必需的，而不是可有可无的美化：
        # `scipy.ndimage.find_objects` 返回的列表**按标签值索引**，
        # 若标签跳号（例如筛掉背景后剩下 {0, 2, 3, 4}，1 被空出），
        # 用「循环下标」当「标签值」就会取到隔壁实例的包围盒。
        # 真实数据上可能因标签恰好连续而侥幸不暴露，合成数据会立刻崩。
        #
        # 重编号后不变量成立：label_map 的取值是 0..N 连续整数，
        # 且 slices[i] 恰好对应标签 i。
        remap = np.zeros(n_labels, dtype=np.int32)
        for new_id, old_id in enumerate(target_labels, start=1):
            remap[old_id] = new_id

        label_info = {
            new_id: target_classes[old_id]
            for new_id, old_id in enumerate(target_labels, start=1)
        }
        label_map = remap[label_map]

        return label_map, label_info


def load_instance_map(path) -> NDArray:
    """读取实例分割图（RGBA）。"""
    from pathlib import Path

    from PIL import Image

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"实例分割图不存在: {p}")

    arr = np.asarray(Image.open(p))
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"期望 RGBA，实际 {arr.shape}")
    return arr
