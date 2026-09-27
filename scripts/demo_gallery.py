"""跑多个素材，生成效果对比图。

用途：一次性看到系统在**不同场景**下的表现 —— 有助于判断
它是否只在某个特定场景下"碰巧能跑"。

对每个素材：
    1. 走完整 pipeline（实例提取 → 跟踪 → PSG → 规划 → 优化 → 渲染）
    2. 从成片里抽 3 帧
    3. 纵向拼成一行
最后把所有行拼成一张总图。

用法：
    python scripts/demo_gallery.py                          # 默认 3 个素材
    python scripts/demo_gallery.py --count 5 --frames 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DATA_ROOT = Path("C:/pano_data")
INSTANCE_DIR = DATA_ROOT / "datasets" / "nyc_instance" / "nyc_instance_panorama"
RAW_DIR = DATA_ROOT / "datasets" / "nyc_raw" / "nyc_Raw"


def pick_examples(count: int) -> list[Path]:
    """均匀挑选若干素材。

    刻意**均匀取样而非取前 N 个** —— 前几个文件可能来自同一场景批次，
    横向对比的价值会大打折扣。
    """
    files = sorted(INSTANCE_DIR.glob("*.png"))
    if not files:
        return []
    # 优先挑有真实 RGB 的（渲染素材质量更高）
    with_raw = [f for f in files if (RAW_DIR / f.name).is_file()]
    pool = with_raw if len(with_raw) >= count else files
    step = max(len(pool) // count, 1)
    return pool[::step][:count]


def extract_strip(video: Path, out_png: Path, *, n_frames: int = 3, width: int = 480) -> bool:
    """从成片里抽若干帧拼成一条横向图。"""
    import subprocess

    if not video.is_file():
        return False
    height = int(width * 9 / 16)
    # 取视频中段附近，避开开场与结尾
    picks = [int(0.15 * 100), int(0.45 * 100), int(0.75 * 100)]
    # 用 ffmpeg 的 select 表达式按比例抽帧
    expr = "+".join(f"eq(n\\,{p})" for p in [3, 12, 21])
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(video),
        "-vf", f"select='{expr}',scale={width}:{height},tile={n_frames}x1",
        "-frames:v", "1", str(out_png),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        return r.returncode == 0 and out_png.is_file()
    except Exception:  # noqa: BLE001
        return False


def label_row(img: np.ndarray, text: str, *, bar_h: int = 30) -> np.ndarray:
    """在图像顶部加一条标签栏。"""
    from PIL import Image, ImageDraw

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    h, w = img.shape[:2]
    bar = np.full((bar_h, w, 3), 22, dtype=np.uint8)
    out = np.concatenate([bar, img], axis=0)

    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    draw.text((10, 8), text, fill=(230, 230, 230))
    return np.asarray(pil)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多素材效果对比")
    parser.add_argument("--count", type=int, default=3, help="素材数量")
    parser.add_argument("--frames", type=int, default=30, help="每个成片的帧数")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "gallery.png",
    )
    parser.add_argument("--keep-videos", action="store_true", help="保留下每个成片")
    args = parser.parse_args(argv)

    from pano_agent.service.pipeline import PipelineConfig, run_pipeline

    if not INSTANCE_DIR.is_dir():
        print(f"[ERROR] 数据目录不存在: {INSTANCE_DIR}")
        return 1

    examples = pick_examples(args.count)
    if not examples:
        print("[ERROR] 没有可用素材")
        return 1

    print("=" * 72)
    print(f"  多素材对比（{len(examples)} 个）")
    print("=" * 72)

    work_dir = args.output.parent / "gallery_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    rows: list[np.ndarray] = []

    for i, inst_path in enumerate(examples):
        raw_path = RAW_DIR / inst_path.name
        video_path = work_dir / f"{inst_path.stem}.mp4"
        strip_path = work_dir / f"{inst_path.stem}_strip.png"

        print(f"\n[{i + 1}/{len(examples)}] {inst_path.name}")
        t0 = time.perf_counter()

        try:
            result = run_pipeline(
                inst_path,
                raw_path if raw_path.is_file() else None,
                video_path,
                config=PipelineConfig(
                    num_frames=args.frames, fps=args.fps,
                    width=args.width, height=args.height,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"      失败: {type(exc).__name__}: {exc}")
            continue

        dt = time.perf_counter() - t0
        m = result.metrics
        print(
            f"      轨迹 {result.num_tracks} 条 | 跨接缝 {result.seam_crossings} 次 | "
            f"{result.render_frames} 帧 | {dt:.1f}s"
        )
        print(
            f"      平滑度 {m.get('smoothness', 0):.1f} °/s² | "
            f"峰值角速度 {m.get('peak_angular_velocity_dps', 0):.0f}°/s | "
            f"合规 {m.get('angular_velocity_compliance', 0):.0%}"
        )

        if extract_strip(video_path, strip_path):
            from PIL import Image

            arr = np.asarray(Image.open(strip_path))
            label = (
                f"{inst_path.name}   "
                f"轨迹 {result.num_tracks}  跨接缝 {result.seam_crossings}  "
                f"峰值 {m.get('peak_angular_velocity_dps', 0):.0f}°/s  "
                f"({dt:.0f}s)"
            )
            rows.append(label_row(arr, label))
        else:
            print("      抽帧失败")

    if not rows:
        print("\n[ERROR] 没有生成任何对比行")
        return 1

    # 纵向拼接（宽度对齐到最小者）
    min_w = min(r.shape[1] for r in rows)
    rows = [r[:, :min_w] for r in rows]
    from PIL import Image

    gap = np.full((8, min_w, 3), 13, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for i, r in enumerate(rows):
        if i > 0:
            parts.append(gap)
        parts.append(r)

    canvas = np.concatenate(parts, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(args.output)

    print("\n" + "=" * 72)
    print(f"  ✓ 对比图已生成: {args.output}")
    print(f"    尺寸 {canvas.shape[1]}×{canvas.shape[0]}，{len(rows)} 行")
    print("=" * 72)

    if not args.keep_videos:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
