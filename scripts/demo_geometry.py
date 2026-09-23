"""W1 验收演示：从 ERP 渲染多视角 NFOV，目视验证几何层正确性。

为什么用语义 mask 而不是真实 RGB:
    语义 mask 是**逐像素确定**的 —— 每个类别边界都是已知的几何位置。
    用它渲染 NFOV 时，任何投影错误（符号、半像素偏移、接缝错位）都会
    表现为类别边界与预期不符，比看真实照片容易发现得多。
    真实 RGB 要到 nyc_Raw.zip（11.3GB）下完才有。

用法:
    python scripts/demo_geometry.py
    python scripts/demo_geometry.py --output assets/w1_geometry_demo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from PIL import Image, ImageDraw  # noqa: E402

from pano_agent.geometry.camera import VirtualCamera  # noqa: E402
from pano_agent.geometry.projection import (  # noqa: E402
    erp_seam_difference,
    render_nfov,
)

# 语义类别 → 颜色（对应 semantic_lists_nyc.txt，索引即 alpha 通道取值）
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    3: (135, 206, 235),   # sky         天蓝
    4: (139, 119, 101),   # land        土褐
    5: (169, 169, 169),   # concreteblock 灰
    6: (255, 165, 0),     # streetprops 橙
    7: (255, 99, 71),     # plasticcone 番茄红
    8: (105, 105, 105),   # metalfence  深灰
    9: (192, 192, 192),   # pillar      银
    10: (255, 255, 0),    # lamp        黄
    11: (34, 139, 34),    # trashcan    森林绿
    12: (220, 20, 60),    # postbox     猩红
    13: (255, 192, 203),  # umbrella    粉
    14: (160, 82, 45),    # table       赭
    15: (210, 180, 140),  # chair       棕褐
    16: (50, 205, 50),    # greenpot    亮绿
    17: (255, 215, 0),    # roadcolumn  金
    18: (70, 130, 180),   # adplane     钢蓝
    19: (188, 143, 143),  # buildingawning 玫瑰褐
    20: (128, 128, 0),    # scaffolding 橄榄
    22: (0, 100, 0),      # plant       深绿
    23: (176, 196, 222),  # ventilationtube 浅钢蓝
    24: (128, 128, 128),  # building    灰
    25: (64, 64, 64),     # road        深灰沥青
    26: (211, 211, 211),  # sidewalk    浅灰
    27: (101, 67, 33),    # grounddirt  暗褐
    28: (255, 140, 0),    # hotdogpot   暗橙
}

BACKGROUND_COLOR = (20, 20, 20)


def load_erp_mask(path: Path) -> np.ndarray:
    """读取全景语义 mask（类别 ID 存在 alpha 通道）。"""
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., 3].astype(np.uint8)
    raise ValueError(f"期望 RGBA 图，实际 shape={arr.shape}, mode={img.mode}")


def colorize(class_ids: np.ndarray) -> np.ndarray:
    """类别 ID → RGB 彩色图（未登记类别用背景色）。"""
    h, w = class_ids.shape
    rgb = np.full((h, w, 3), BACKGROUND_COLOR, dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        rgb[class_ids == cid] = color
    return rgb


def make_label_panel(width: int, height: int, text: str) -> Image.Image:
    """生成一个带文字标注的面板，用于拼图。"""
    panel = Image.new("RGB", (width, height), (10, 10, 10))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), text, fill=(230, 230, 230))
    return panel


def build_grid(
    erp_rgb: np.ndarray,
    shots: list[tuple[VirtualCamera, str]],
    *,
    erp_panel_width: int = 1024,
    cell_height: int = 216,
) -> Image.Image:
    """把 ERP 总览 + 多个 NFOV 视角拼成一张对比图。"""
    # 左侧：ERP 缩略图（2:1）
    erp_full = Image.fromarray(erp_rgb)
    erp_thumb = erp_full.resize((erp_panel_width, erp_panel_width // 2), Image.Resampling.LANCZOS)

    # 右侧：每个视角一格
    cell_w = 384
    n_shots = len(shots)
    grid_h = max(erp_thumb.height + 40, n_shots * (cell_height + 30))
    grid_w = erp_panel_width + cell_w + 40

    canvas = Image.new("RGB", (grid_w, grid_h), (16, 16, 16))

    # 左上：ERP 全景（带视角标记）
    erp_with_marks = erp_thumb.copy()
    draw = ImageDraw.Draw(erp_with_marks)
    draw.text((8, 8), "ERP (2048x1024, 2:1)", fill=(255, 255, 255))
    for cam, _ in shots:
        # yaw → x, pitch → y（用与 geometry 一致的映射）
        px = (cam.yaw_deg + 180.0) / 360.0 * erp_thumb.width
        py = (90.0 - cam.pitch_deg) / 180.0 * erp_thumb.height
        r = 5
        draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 60, 60), width=2)
    canvas.paste(erp_with_marks, (10, 10))

    # 右侧：各视角
    for i, (cam, label) in enumerate(shots):
        nfov = render_nfov(erp_rgb, cam)
        panel = Image.fromarray(nfov.astype(np.uint8)).resize(
            (cell_w, cell_height), Image.Resampling.LANCZOS
        )
        y = 10 + i * (cell_height + 30)
        canvas.paste(panel, (erp_panel_width + 20, y))
        d = ImageDraw.Draw(canvas)
        d.text((erp_panel_width + 20, y + cell_height + 2), label, fill=(200, 200, 200))

    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="几何层渲染演示")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("C:/pano_data/datasets/nyc_seg/nyc_seg_panorama/panorama_0.png"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "w1_geometry_demo.png",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"[ERROR] 输入文件不存在: {args.input}")
        print("       请先下载 nyc_seg_panorama.zip（见 docs/datasets.md）")
        return 1

    print(f"读取 ERP: {args.input.name}")
    mask = load_erp_mask(args.input)
    h, w = mask.shape
    print(f"  尺寸 {w}x{h}  比例 {w/h:.4f}  {'[OK] ERP' if abs(w/h - 2.0) < 0.01 else '[WARN] 非 2:1'}")

    classes = np.unique(mask)
    print(f"  出现类别 {len(classes)} 种: {classes.tolist()}")

    erp_rgb = colorize(mask)

    # 接缝连续性检查
    seam = erp_seam_difference(mask.astype(np.float64))
    print(f"  接缝不连续度 {seam:.3f}  (≈1 表示接缝与内部无差别)")

    # 构造一组测试视角
    shots = [
        (VirtualCamera(0, 0, 0, 90, 384, 216), "yaw=0 pitch=0 fov=90  (正前方)"),
        (VirtualCamera(90, 0, 0, 90, 384, 216), "yaw=90 pitch=0 fov=90  (右方)"),
        (VirtualCamera(180, 0, 0, 90, 384, 216), "yaw=180 pitch=0 fov=90 (正后方)"),
        (VirtualCamera(-90, 0, 0, 90, 384, 216), "yaw=-90 pitch=0 fov=90 (左方)"),
        (VirtualCamera(0, 60, 0, 90, 384, 216), "yaw=0 pitch=60 fov=90  (仰视，高纬)"),
        (VirtualCamera(0, -60, 0, 90, 384, 216), "yaw=0 pitch=-60 fov=90 (俯视)"),
        (VirtualCamera(0, 85, 0, 90, 384, 216), "yaw=0 pitch=85 fov=90  (近极点)"),
        (VirtualCamera(179, 0, 0, 60, 384, 216), "yaw=179 fov=60 (跨接缝处)"),
    ]

    print(f"渲染 {len(shots)} 个视角...")
    grid = build_grid(erp_rgb, shots)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"\n[OK] 已保存: {args.output}")
    print(f"     尺寸 {grid.width}x{grid.height}")
    print("\n检查要点:")
    print("  1. 四个水平视角（0/90/180/-90）应无缝拼接成完整的 360° 环绕")
    print("  2. pitch=85 的近极点视角应无扇形拖影")
    print("  3. yaw=179（跨接缝）应无垂直错位")
    return 0


if __name__ == "__main__":
    sys.exit(main())
