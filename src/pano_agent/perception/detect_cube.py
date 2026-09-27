"""立方体贴图检测 —— 解决「透视域检测器在 ERP 上失效」的问题。

==============================================================================
问题：为什么 YOLO 在 ERP 上几乎检测不到东西
==============================================================================

实测（yolo11n 在 2048×1024 真实全景图上）：

    conf=0.15（默认）→ 0 个检测
    conf=0.05        → 0 个
    conf=0.01        → 17 个

置信度崩塌式下降，根源是三个因素叠加 —— 它们都源于同一件事：
**ERP 的内容分布与 COCO 的训练分布严重不符**。

1. **两极畸变**：极点附近的面积畸变因子 1/cos φ 发散，
   一个正方形目标在高纬被拉伸成极宽的条带。检测器的先验框完全失效。

2. **宽扁画幅**：2:1 的画幅 letterbox 到 1280×640 后，
   原本就小的目标进一步缩小到十几个像素。

3. **采样非均匀**：同样的目标，在赤道与在高纬占据的像素数相差十倍，
   检测器无法用统一的尺度先验应对。

==============================================================================
解法：换投影，而不是调阈值
==============================================================================

**降低 conf 阈值不是解决方案** —— 它只是把"漏检"换成了"虚警"。

正确做法是：**把图像投影到检测器熟悉的域**。

立方体贴图恰好满足这个需求：

    6 个面，每个是 90°×90° 的**标准透视投影** → 匹配 COCO 的成像模型
    正方形画幅（N×N）                        → 无 letterbox 损失
    极点区域被展成正常视角                    → 畸变消失

这与 DA360 用 circular padding + 立方体投影处理 ERP 是同一思路，
也是 Insta360 研究院综述里「Projection-Driven（投影驱动）」方法族的
直接应用 —— 该族方法适合几何敏感的任务，检测正属于此类。

==============================================================================
流程
==============================================================================

    ERP (2:1)
      ↓ erp_to_cubemap      6 个 N×N 透视面
      ↓ YOLO 逐面推理        每面匹配预训练分布
      ↓ 面内框 → 球面框      用该面的投影参数反算
      ↓ 跨面去重            同一目标在相邻面边界会被检出两次
    球面检测列表

去重的关键：**在球面上算 IoU**，而不是在像素上。相邻面的边界处，
同一个目标的两半在不同面里有不同的像素形状，像素 IoU 会失效。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..geometry.projection import erp_to_cubemap
from ..geometry.region import SphericalBox
from .types import SphericalDetection

if TYPE_CHECKING:  # pragma: no cover
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

# 立方体六个面在世界系中的朝向：
#   面名 → (法向轴索引, 符号)，以及该面的 u/v 轴
# 与 geometry.projection._CUBE_FACE_UV 必须保持一致
_FACE_ORIENTATION: dict[str, tuple[NDArray, NDArray, NDArray]] = {
    # name: (forward, right(+u), up(+v))
    "front": (np.array([0, 0, 1.0]), np.array([1, 0, 0.0]), np.array([0, 1, 0.0])),
    "back": (np.array([0, 0, -1.0]), np.array([-1, 0, 0.0]), np.array([0, 1, 0.0])),
    "right": (np.array([1, 0, 0.0]), np.array([0, 0, -1.0]), np.array([0, 1, 0.0])),
    "left": (np.array([-1, 0, 0.0]), np.array([0, 0, 1.0]), np.array([0, 1, 0.0])),
    "up": (np.array([0, 1, 0.0]), np.array([1, 0, 0.0]), np.array([0, 0, -1.0])),
    "down": (np.array([0, -1, 0.0]), np.array([1, 0, 0.0]), np.array([0, 0, 1.0])),
}


@dataclass
class CubeDetectorConfig:
    """立方体检测配置。"""

    weights: str = "yolo11n.pt"
    device: str = "cuda"
    face_size: int = 1024
    """每个面的边长。1024 时 6 面共 6MP，与单张 2K ERP 相当。
    在 2.5-3GB 显存下可逐面推理（不必同时驻留）。"""

    conf: float = 0.15
    """立方体域可以直接用 COCO 的标准阈值 —— 这正是换投影的价值所在。"""

    iou: float = 0.5
    half: bool = True
    max_det: int = 300
    merge_iou: float = 0.5
    """跨面去重的**球面** IoU 阈值。"""


class CubeDetector:
    """基于立方体贴图的 ERP 检测器。

    与 `ErpDetector` 的分工：
        ErpDetector   直接在 ERP 上推理。快，但置信度低（见模块文档）。
        CubeDetector  投影到 6 个透视面推理。慢 6 倍，但置信度正常。

    两者都输出统一的 `SphericalDetection`，下游无需感知差异。
    """

    def __init__(self, config: CubeDetectorConfig | None = None) -> None:
        self.config = config or CubeDetectorConfig()
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            from ultralytics import YOLO

            logger.info("加载检测模型 %s", self.config.weights)
            self._model = YOLO(self.config.weights)
        return self._model

    def release(self) -> None:
        self._model = None

    # ------------------------------------------------------------ 检测

    def detect(self, erp_image: NDArray) -> list[SphericalDetection]:
        """对一帧 ERP 做立方体域检测。"""
        img = np.asarray(erp_image)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]

        height, width = img.shape[:2]
        face_size = self.config.face_size

        # ---- 1. ERP → 6 个透视面 ----
        faces = erp_to_cubemap(img, face_size)

        # ---- 2. 逐面推理并反投影到球面 ----
        all_dets: list[SphericalDetection] = []
        for face_name, face_img in faces.items():
            boxes = self._detect_face(face_img)
            for x1, y1, x2, y2, conf, cls in boxes:
                det = self._face_box_to_spherical(
                    face_name, x1, y1, x2, y2, face_size, conf, cls
                )
                if det is not None:
                    all_dets.append(det)

        # ---- 3. 跨面去重（在球面上做）----
        return self._merge_across_faces(all_dets)

    # ------------------------------------------------------------ 内部

    def _detect_face(
        self, face_img: NDArray
    ) -> list[tuple[float, float, float, float, float, int]]:
        """在单个立方体面上推理。

        面是标准透视投影 + 正方形画幅，因此可以直接用 COCO 的默认阈值。
        """
        results = self.model.predict(
            face_img,
            imgsz=self.config.face_size,
            conf=self.config.conf,
            iou=self.config.iou,
            max_det=self.config.max_det,
            device=self.config.device,
            half=self.config.half and self.config.device == "cuda",
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        return [
            (float(a), float(b), float(c), float(d), float(e), int(f))
            for (a, b, c, d), e, f in zip(xyxy, conf, cls, strict=True)
        ]

    def _face_box_to_spherical(
        self,
        face_name: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        face_size: int,
        conf: float,
        cls: int,
    ) -> SphericalDetection | None:
        """把立方体面上的像素框转换为球面检测。

        立方体面的投影是**标准 gnomonic**：面内某点 (u, v)（归一化到 [-1,1]）
        对应世界系方向

            d = forward + u·right + v·(-up)

        注意 v 轴取负：图像 y 向下，而 `up` 是世界系的上方。

        取框的四角方向，用**圆统计**求中心 yaw（避免跨越 ±180° 时
        端点平均给出错误中心 —— 与 `region.from_erp_pixels` 同样的坑）。
        """
        orientation = _FACE_ORIENTATION.get(face_name)
        if orientation is None:
            return None
        forward, right, up = orientation

        # 像素 → 归一化切平面坐标
        corners = np.array(
            [
                [x1, y1],
                [x1, y2],
                [x2, y1],
                [x2, y2],
            ],
            dtype=np.float64,
        )
        norm = (corners / face_size) * 2.0 - 1.0  # → [-1, 1]

        dirs = []
        for u, v in norm:
            d = forward + u * right - v * up
            n = np.linalg.norm(d)
            if n < 1e-9:
                return None
            dirs.append(d / n)

        dirs = np.asarray(dirs)
        from ..geometry.sphere import vec_to_sph

        yaw, pitch = vec_to_sph(dirs)

        # 中心用**球面质心**（向量平均），不能对角度分别取算术平均：
        #
        #   - 极点附近 pitch 强烈非线性。实测：`up` 面的四角 pitch 均为
        #     83.70°（彼此对称），算术平均仍是 83.70°，**而真实中心是 90°**。
        #     这是 pitch = arcsin(y) 在 y→1 处的导数发散所致。
        #
        #   - 跨越 ±180° 时 yaw 的算术平均会落到反面（0°）而非 ±180°。
        #
        # 在 3D 单位向量空间取质心则对两种情况都自然正确 ——
        # 因为球面在该空间中是光滑且无边界的。
        mean_vec = dirs.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm < 1e-9:
            return None  # 四角方向恰好对跖，退化为无定义
        center_yaw, center_pitch = vec_to_sph(mean_vec / norm)
        center_yaw = float(center_yaw)
        center_pitch = float(np.clip(center_pitch, -90.0, 90.0))

        box = SphericalBox.from_centered_points(
            center_yaw, center_pitch, yaw, pitch
        )
        return SphericalDetection(
            yaw_deg=box.yaw_deg,
            pitch_deg=box.pitch_deg,
            width_deg=box.width_deg,
            height_deg=box.height_deg,
            confidence=conf,
            category=str(cls),
        )

    def _merge_across_faces(
        self, detections: list[SphericalDetection]
    ) -> list[SphericalDetection]:
        """合并相邻面边界处被重复检出的同一目标。

        去重判据是**球面 IoU**：同一目标在相邻两个面里会呈现不同的像素形状
        （各自被裁掉一半），像素 IoU 在此完全失效，而球面 IoU 天然正确。
        """
        if len(detections) <= 1:
            return detections

        # 按置信度降序，优先保留高置信的
        ordered = sorted(detections, key=lambda d: -d.confidence)
        kept: list[SphericalDetection] = []

        for det in ordered:
            box = det.to_box()
            duplicate = False
            for existing in kept:
                if existing.category != det.category:
                    continue
                iou = box.iou(existing.to_box(), n_samples=2000)
                if iou > self.config.merge_iou:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)

        return kept
