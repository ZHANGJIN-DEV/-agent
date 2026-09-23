"""ERP 全景图上的目标检测。

==============================================================================
在全景图上跑检测器的三个特有困难
==============================================================================

1. **跨接缝的目标被切成两段**

   位于 yaw ≈ ±180° 的目标，在 ERP 上被切成左右两个互不相连的碎片。
   普通检测器会输出两个残缺的框（或什么都检不到），
   而下游的跟踪器会看到两个目标凭空出现又消失。

   → 解法：推理前对图像做**水平环形 padding**，推理后再把坐标折回。
     这与 DA360 用 circular padding 消除接缝是同一手法。

2. **两极严重畸变**

   ERP 在极点附近的面积畸变因子 1/cos φ 发散：一个正方形目标在高纬
   会被拉伸成极宽的条带，检测器的先验框完全失效。

   → 解法：可选极点分块（把极区重投影到立方体面后再推理）。
     本模块保留该扩展点，但默认不启用 —— 它是 W4 止损点的优先裁剪项。

3. **先验框宽高比失配**

   检测器通常在透视图像上训练，而 ERP 目标的宽高比随纬度剧烈变化。

   → 本模块选择「在 ERP 上直接推理 + 后处理修正」，
     而不是「在透视图上推理再拼回」—— 后者会把问题从一个接缝
     变成多个接缝，且引入重投影误差。

==============================================================================
输出约定
==============================================================================

检测结果一律转换为 `SphericalDetection`（球面角度），
**绝不向下游传递像素坐标**。这样跟踪、显著性、轨迹优化等模块
都不需要关心 ERP 的分辨率与投影细节。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..geometry.projection import circular_pad_erp
from ..geometry.region import SphericalBox
from .types import SphericalDetection

if TYPE_CHECKING:  # pragma: no cover
    from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """检测器配置。默认值针对 RTX 3050 (2.5GB 可用显存) 调优。"""

    weights: str = "yolo11n.pt"
    device: str = "cuda"
    imgsz: tuple[int, int] = (640, 1280)
    """推理尺寸 (H, W)。ERP 是 2:1，故宽是高的两倍。
    在 2.5GB 显存下 1280×640 是安全上限。"""

    conf: float = 0.25
    iou: float = 0.5
    max_det: int = 300
    half: bool = True
    """FP16 推理。3050 支持，约省一半显存。"""

    seam_pad_px: int = 160
    """接缝处理的环形 padding 宽度（原图像素）。
    应大于最大目标的半宽，否则跨接缝目标仍会被切断。
    0 表示关闭该处理（用于消融实验）。"""


class ErpDetector:
    """在 ERP 全景图上运行的检测器。

    依赖 ultralytics，但采用**延迟导入** ——
    这样在未安装 torch 的环境里仍可导入本模块、运行其纯几何部分
    （如坐标转换与跨缝合并逻辑）的单元测试。
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._model: YOLO | None = None

    # ------------------------------------------------------------ 模型

    @property
    def model(self) -> YOLO:
        """延迟加载模型（首次调用时才占显存）。"""
        if self._model is None:
            from ultralytics import YOLO  # 延迟导入，避免无 torch 环境下导入失败

            logger.info("加载检测模型 %s", self.config.weights)
            self._model = YOLO(self.config.weights)
        return self._model

    def release(self) -> None:
        """释放模型引用。

        ⚠ 注意：这**不会**真正归还显存给驱动 —— PyTorch 的缓存分配器
        不保证释放。唯一可靠的方式是让进程退出，这也是 `registry.py`
        用子进程隔离模型的原因。
        """
        self._model = None

    # ------------------------------------------------------------ 推理

    def detect(self, erp_image: NDArray) -> list[SphericalDetection]:
        """对一帧 ERP 图像做检测。

        Args:
            erp_image: 形状 (H, W) 或 (H, W, C) 的 ERP 图像。

        Returns:
            球面表示的检测结果。
        """
        img = np.asarray(erp_image)
        if img.ndim not in (2, 3):
            raise ValueError(f"ERP 图像必须是 (H,W) 或 (H,W,C)，实际 {img.shape}")

        height, width = img.shape[:2]
        pad = self.config.seam_pad_px

        # ---- 1. 环形 padding，让跨接缝目标变完整 ----
        if pad > 0:
            padded = circular_pad_erp(img, pad)
            x_offset = pad
        else:
            padded = img
            x_offset = 0

        # ---- 2. 推理 ----
        results = self.model.predict(
            padded,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            max_det=self.config.max_det,
            device=self.config.device,
            half=self.config.half and self.config.device == "cuda",
            verbose=False,
        )

        # ---- 3. 解析框并折回原图坐标 ----
        boxes = self._extract_boxes(results[0])
        if boxes.size == 0:
            return []

        # 缩放回 padded 图坐标（YOLO 内部会 letterbox 到 imgsz）
        scale_x, scale_y = self._letterbox_scale(padded.shape[:2], self.config.imgsz)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # 减去 padding 偏移
        boxes[:, [0, 2]] -= x_offset

        # ---- 4. 跨接缝的碎片合并 ----
        boxes = _merge_seam_fragments(boxes, width, height)

        # ---- 5. 像素框 → 球面框 ----
        detections: list[SphericalDetection] = []
        for x1, y1, x2, y2, conf, cls in boxes:
            box = SphericalBox.from_erp_pixels(
                float(x1), float(y1), float(x2), float(y2), width, height
            )
            detections.append(
                SphericalDetection(
                    yaw_deg=box.yaw_deg,
                    pitch_deg=box.pitch_deg,
                    width_deg=box.width_deg,
                    height_deg=box.height_deg,
                    confidence=float(conf),
                    category=str(int(cls)),
                )
            )

        return detections

    # ------------------------------------------------------------ 内部

    @staticmethod
    def _extract_boxes(result) -> NDArray[np.float64]:
        """把 ultralytics 的 Results 转成 (N, 6) 数组 [x1,y1,x2,y2,conf,cls]。"""
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 6), dtype=np.float64)

        xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float64)
        conf = result.boxes.conf.cpu().numpy().astype(np.float64).reshape(-1, 1)
        cls = result.boxes.cls.cpu().numpy().astype(np.float64).reshape(-1, 1)
        return np.concatenate([xyxy, conf, cls], axis=1)

    @staticmethod
    def _letterbox_scale(
        padded_hw: tuple[int, int], imgsz: tuple[int, int]
    ) -> tuple[float, float]:
        """计算 letterbox 的缩放系数。

        ultralytics 默认保持宽高比并补边，因此 x 与 y 的缩放系数相同
        （取较小者），补边造成的偏移在后续会被 xyxy 的原始尺度还原。
        这里返回「原图尺寸 / 推理尺寸」的倒数关系。
        """
        ph, pw = padded_hw
        ih, iw = imgsz
        gain = min(ih / ph, iw / pw)
        # boxes 已是推理尺度，乘以 1/gain 还原到原图尺度
        inv = 1.0 / gain if gain > 0 else 1.0
        return inv, inv


def _merge_seam_fragments(
    boxes: NDArray[np.float64],
    width: int,
    height: int,
    *,
    tol_px: float = 40.0,
) -> NDArray[np.float64]:
    """合并被 ERP 接缝切开的同一目标的碎片。

    当检测在 padded 图上完成时，同一目标在接缝处可能产生两个框：
    一个贴左边（x1 ≈ 0）、一个贴右边（x2 ≈ width）。
    它们在 padded 图上其实是同一段，折回原坐标后被分到了两端。

    ------------------------------------------------------------------
    ⚠ 合并结果的坐标表示（容易搞错的地方）
    ------------------------------------------------------------------

    不能简单地写成 `[左碎片.x1, ..., 右碎片.x2] = [0, ..., 2047]` ——
    那样在像素上"跨越整个宽度"，实际目标却只是接缝附近的一小块，
    转换到球面时角展度会算成 360°。

    正确做法是在**接缝连续坐标系**下合并：以接缝为原点把两段接起来。
    设接缝位于 x = width ≡ 0，则

        右碎片 [2018, 2047]  位于接缝**左侧**
        左碎片 [0, 20]       位于接缝**右侧**，平移一个周期后是 [2048, 2068]

    合并结果便是 `[2018, 2068]` —— 一个跨越接缝、宽 50 像素的连续框，
    中心落在 2043（即 yaw ≈ 179°），角展度 8.8°。这才是正确的语义。

    因此输出框的 x 坐标**允许超出 [0, width]**，
    调用方 `SphericalBox.from_erp_pixels` 需支持这种连续坐标。

    Returns:
        合并后的框数组，列含义同输入；跨接缝框的 x 可能超出 [0, width]。
    """
    if boxes.size == 0:
        return boxes

    left_mask = boxes[:, 0] < tol_px
    right_mask = boxes[:, 2] > width - tol_px
    if not left_mask.any() or not right_mask.any():
        return boxes

    left_idx = np.where(left_mask)[0]
    right_idx = np.where(right_mask)[0]

    consumed: set[int] = set()
    merged: list[NDArray[np.float64]] = []

    for li in left_idx:
        lb = boxes[li]
        best: tuple[float, int] | None = None

        for ri in right_idx:
            if ri in consumed or ri == li:
                continue
            rb = boxes[ri]

            # 垂直方向必须有重叠
            v_overlap = min(lb[3], rb[3]) - max(lb[1], rb[1])
            if v_overlap <= 0:
                continue
            # 垂直重叠比例（相对较矮的那个）
            min_h = min(lb[3] - lb[1], rb[3] - rb[1])
            if min_h <= 0 or v_overlap / min_h < 0.5:
                continue

            # 把左边的框右移一个周期后的水平间距
            gap = abs((lb[0] + width) - rb[2])
            if gap > tol_px:
                continue
            if best is None or gap < best[0]:
                best = (gap, ri)

        if best is None:
            continue

        ri = best[1]
        rb = boxes[ri]
        consumed.add(li)
        consumed.add(ri)

        # 在接缝连续坐标系下合并：右碎片在接缝左侧保持不变，
        # 左碎片整体右移一个周期接到它后面。
        merged.append(
            np.array(
                [
                    rb[0],                 # 右碎片的左边界（接缝左侧）
                    min(lb[1], rb[1]),     # 垂直范围取并集
                    lb[2] + width,         # 左碎片的右边界 + 一个周期
                    max(lb[3], rb[3]),
                    max(lb[4], rb[4]),     # 置信度取较高者
                    rb[5] if rb[4] >= lb[4] else lb[5],
                ],
                dtype=np.float64,
            )
        )

    keep = [boxes[i] for i in range(len(boxes)) if i not in consumed]
    if merged:
        keep.extend(merged)
    return np.array(keep, dtype=np.float64) if keep else np.empty((0, 6))
