"""显著性预测 —— 高光检测的基础信号。

==============================================================================
为什么损失函数必须用 cos φ 加权
==============================================================================

显著性损失通常是逐像素的（BCE / MSE）。在 ERP 上直接这么做会造成
**双重错误**：

1. **梯度权重错配**
   ERP 高纬区域被严重压缩：靠近极点的少数几行代表了一大片球面区域，
   但它们在损失中的权重与赤道的任意一行完全相同。
   结果是模型在极区学得不足，而极区恰恰是畸变最严重的地方。

2. **评测偏差**
   AUC / CC 等指标若逐像素计算，极点附近会不成比例地主导分数。
   一个"赤道完美、极区全错"的模型仍可能得到漂亮的结果。

正确做法是让每个像素的损失权重正比于它代表的立体角，即 ∝ cos φ。
`latitude_weight_map` 用 `sphere.erp_row_solid_angles` 给出精确权重
（各行之和严格等于 4π）。

这与 DA360 用 circular padding + learnable shift 处理 ERP 接缝、
`track_spherical` 用 yaw 提升处理周期边界是同一类思路：
**尊重 ERP 的几何，而不是把它当成普通图像。**

==============================================================================
标注来源与诚实声明
==============================================================================

本项目**没有**真实的人眼注视 GT（Wild360 等数据集的许可条款不明确，
不宜作为公开成果的基础）。因此采用**弱监督**：

    语义 mask 中的"前景类"（建筑、街道设施、行人、车辆）→ 正样本
    天空、道路、地面                              → 负样本

这不是理想的显著性标注 —— 人看的地方未必是语义前景。
因此技术报告中必须**明确声明这是弱监督**，并把它列为局限，
而不是当作贡献点。诚实地说明局限比包装成亮点更有说服力。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..geometry.sphere import erp_row_solid_angles

logger = logging.getLogger(__name__)

try:  # PyTorch 是可选依赖 —— 让 numpy 部分（权重图）能独立测试
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - 取决于环境
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "显著性模型需要 PyTorch。请先完成环境安装（见 README 的环境搭建章节），"
            "注意 torch 必须从 CUDA 源单独安装。"
        )


# ============================================================== 纬度权重


def latitude_weight_map(height: int, *, normalize: bool = True) -> NDArray[np.float32]:
    """ERP 的纬度权重图，形状 (H, 1)，权重 ∝ 该行代表的立体角。

    Args:
        height: ERP 图像高度。
        normalize: 是否把权重归一化到「平均值为 1」。
            归一化后数值更直观（1.0 表示该行代表平均大小的球面面积），
            且改变输入高度不会改变总损失量级。

    Returns:
        (H, 1) 的 float32 权重。
    """
    if height <= 0:
        raise ValueError(f"高度必须为正，实际为 {height}")

    weights = erp_row_solid_angles(height).astype(np.float32)

    if normalize:
        mean = float(weights.mean())
        if mean > 0:
            weights = weights / mean

    return weights[:, None]


def latitude_weight_map_2d(height: int, width: int) -> NDArray[np.float32]:
    """(H, W) 的二维权重图，便于直接与图像相乘或送入 torch。"""
    if width <= 0:
        raise ValueError(f"宽度必须为正，实际为 {width}")
    return np.broadcast_to(latitude_weight_map(height), (height, width)).copy()


def weighted_mean(
    values: NDArray, height: int, width: int
) -> float:
    """按球面立体角加权求平均。

    这是本项目中「任何在全景图上的标量聚合」的正确做法。
    直接 `values.mean()` 会高估高纬区域的影响。
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (height, width):
        raise ValueError(f"期望形状 ({height},{width})，实际 {arr.shape}")

    weights = latitude_weight_map_2d(height, width).astype(np.float64)
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    return float((arr * weights).sum() / total)


# ============================================================== 网络

if _HAS_TORCH:

    class _DoubleConv(nn.Module):
        """两个 3×3 卷积 + BN + ReLU。"""

        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class SaliencyUNet(nn.Module):
        """轻量 UNet，用于 ERP 显著性预测。

        规模针对 2.5GB 可用显存设计：base=32 时约 2M 参数，
        FP16 推理峰值约 350MB（512×256 输入）。

        显著性本质上是一个低频、全局的任务（"画面中哪里值得看"），
        因此不需要大 backbone —— 与检测任务不同。

        Args:
            base: 基础通道数。32 对应约 2M 参数。
            in_channels: 输入通道（RGB 为 3）。
        """

        def __init__(self, base: int = 32, in_channels: int = 3) -> None:
            super().__init__()
            _require_torch()

            self.base = base
            self.enc1 = _DoubleConv(in_channels, base)
            self.enc2 = _DoubleConv(base, base * 2)
            self.enc3 = _DoubleConv(base * 2, base * 4)
            self.bottleneck = _DoubleConv(base * 4, base * 8)

            self.pool = nn.MaxPool2d(2)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)

            self.dec3 = _DoubleConv(base * 8, base * 4)
            self.dec2 = _DoubleConv(base * 4, base * 2)
            self.dec1 = _DoubleConv(base * 2, base)

            self.head = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            # 编码
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))

            # 解码（跳跃连接）
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

            return self.head(d1)  # logits，未经 sigmoid

        def num_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------------------------------------------------------- 损失

    def weighted_bce_loss(
        logits,
        targets,
        *,
        pos_weight: float = 1.0,
        weight_map=None,
    ):
        """按球面立体角加权的二元交叉熵。

        Args:
            logits: 模型输出，形状 (B, 1, H, W)，**未经 sigmoid**。
            targets: 目标显著图，形状 (B, 1, H, W)，取值 [0, 1]。
            pos_weight: 正样本权重。显著性数据通常正样本稀少。
            weight_map: 形状 (1, 1, H, W) 或 (H, W) 的纬度权重。
                None 时自动按输入尺寸构造（**推荐**，避免手工传错）。

        Returns:
            标量损失。
        """
        _require_torch()

        if weight_map is None:
            h, w = logits.shape[-2:]
            weight_map = torch.from_numpy(latitude_weight_map_2d(h, w)).to(
                logits.device, logits.dtype
            )[None, None]

        elif weight_map.dim() == 2:
            weight_map = weight_map[None, None].to(logits.device, logits.dtype)

        per_pixel = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=torch.tensor(pos_weight)
            if pos_weight != 1.0 else None,
        )
        return (per_pixel * weight_map).sum() / weight_map.expand_as(per_pixel).sum()

    def weighted_iou(
        logits, targets, *, threshold: float = 0.5
    ):
        """球面加权的 IoU —— 评测用的正确指标。"""
        _require_torch()

        h, w = logits.shape[-2:]
        weight_map = torch.from_numpy(latitude_weight_map_2d(h, w)).to(
            logits.device, logits.dtype
        )[None, None]

        pred = (torch.sigmoid(logits) > threshold).to(logits.dtype)
        target = (targets > threshold).to(logits.dtype)

        inter = (pred * target * weight_map).sum()
        union = ((pred + target).clamp(max=1.0) * weight_map).sum()
        return (inter / union.clamp(min=1e-8)).item()

else:  # pragma: no cover - 无 torch 环境

    class SaliencyUNet:  # type: ignore[no-redef]
        """占位实现 —— 未安装 PyTorch 时给出明确的错误提示。"""

        def __init__(self, *args, **kwargs) -> None:
            _require_torch()

    def weighted_bce_loss(*args, **kwargs):  # type: ignore[misc]
        _require_torch()

    def weighted_iou(*args, **kwargs):  # type: ignore[misc]
        _require_torch()


# ============================================================== 推理包装


@dataclass
class SaliencyConfig:
    """显著性推理配置。"""

    weights: Path | str | None = None
    base: int = 32
    input_hw: tuple[int, int] = (256, 512)
    """推理分辨率 (H, W)。显著性是低频信号，不需要全分辨率。"""

    device: str = "cuda"
    half: bool = True


class SaliencyPredictor:
    """显著性推理包装。"""

    def __init__(self, config: SaliencyConfig | None = None) -> None:
        self.config = config or SaliencyConfig()
        self._model = None

    def load(self) -> None:
        """加载权重并移到目标设备。"""
        _require_torch()

        model = SaliencyUNet(base=self.config.base)
        weights = self.config.weights
        if weights is not None:
            path = Path(weights)
            if not path.is_file():
                raise FileNotFoundError(f"显著性权重不存在: {path}")
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)

        model.eval()
        if self.config.device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
            if self.config.half:
                model = model.half()

        self._model = model
        logger.info("显著性模型就绪（%d 参数）", model.num_parameters())

    def predict(self, erp_image: NDArray) -> NDArray[np.float32]:
        """预测显著图，输出与输入同尺寸，值域 [0, 1]。"""
        _require_torch()
        if self._model is None:
            self.load()

        import cv2

        img = np.asarray(erp_image)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]

        h0, w0 = img.shape[:2]
        hh, ww = self.config.input_hw

        resized = cv2.resize(img, (ww, hh), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(resized.astype(np.float32) / 255.0)
        if x.ndim == 2:
            x = x[None]
        x = x.permute(2, 0, 1)[None]  # (1, C, H, W)

        device = next(self._model.parameters()).device
        x = x.to(device)
        if self.config.half and device.type == "cuda":
            x = x.half()

        with torch.inference_mode():
            logits = self._model(x)

        prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        return cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )

    def predict_weighted_mean(self, erp_image: NDArray) -> float:
        """预测并返回**球面加权**的平均显著度。

        这是「这一帧有多值得看」的标量描述，用于高光检测。
        用朴素平均会让极区主导，因此必须加权。
        """
        sal = self.predict(erp_image)
        return weighted_mean(sal, *sal.shape)
