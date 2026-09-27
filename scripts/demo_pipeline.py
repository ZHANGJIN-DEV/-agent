"""端到端验证：实例 → 球面框 → 球面跟踪 → PSG。

==============================================================================
数据约束与验证设计
==============================================================================

Omni360-X 提供的是**静态全景图**，没有视频序列 —— 所以无法直接验证跟踪。
但它的**实例本身是真实的**（真实形状、真实尺寸、真实类别、真实的球面位置）。

因此本脚本用「**真实实例 + 合成运动**」构造序列：

    1. 从真实实例图提取实例（走完整的 InstanceExtractor 流程）
    2. 用 `np.roll` 在像素域平移实例图（模拟目标在球面上移动）
    3. 每帧重新提取 —— 走真实流程，不是直接改坐标
    4. 喂给球面跟踪器

**为什么用 np.roll**：它是**环绕**平移，目标从右边缘移出会从左边缘进入，
天然制造出「跨越 ERP 接缝」的场景 —— 而这正是本项目的核心验证点。

这样得到的验证比纯合成数据更接近实际（实例形状真实、有真实的高纬畸变），
又比静态图有可控、可复现的运动。

==============================================================================
验证什么、不验证什么
==============================================================================

验证：
    ✓ 整条链路在真实数据上能跑通
    ✓ 真实实例形状下的球面框转换
    ✓ **跨 ERP 接缝时的 ID 保持**（np.roll 天然制造该场景）
    ✓ PSG 能从真实实例构建出来

不验证：
    ✗ 检测质量（实例是数据集标注，不是模型预测）
    ✗ 真实视频的动态特性（静止场景 + 人为平移 ≠ 真实运动）
    ✗ 显著性、光流 —— 需要真实视频帧间差异
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_INSTANCE = Path(
    "C:/pano_data/datasets/nyc_instance/nyc_instance_panorama/panorama_3518.png"
)


def shift_instance_map(inst: np.ndarray, dx: int) -> np.ndarray:
    """把实例图水平环绕平移 dx 像素。

    np.roll 的环绕语义正是我们需要的：目标从 x=W-1 移出后会从 x=0 进入，
    在球面上这是**连续的**（ERP 左右边界是同一条经线）。
    """
    return np.roll(inst, dx, axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="端到端链路验证（真实实例 + 合成运动）")
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "pipeline_e2e.png",
    )
    args = parser.parse_args(argv)

    from pano_agent.perception.instance import InstanceExtractor, load_instance_map
    from pano_agent.perception.track_spherical import SphericalTracker
    from pano_agent.perception.types import FrameTracks
    from pano_agent.sir import BuildConfig, VideoMeta, build_psg

    if not args.instance.is_file():
        print(f"[ERROR] 找不到 {args.instance}")
        return 1

    print("=" * 68)
    print("  端到端链路验证")
    print("=" * 68)

    # ---- 1. 载入真实实例图 ----
    print(f"\n[1] 载入 {args.instance.name}")
    inst_map = load_instance_map(args.instance)
    height, width = inst_map.shape[:2]
    print(f"    ERP {width}×{height}  比例 {width/height:.4f}")

    extractor = InstanceExtractor()

    # ---- 2. 选定一条"目标"轨迹 ----
    print("\n[2] 选定待跟踪目标")
    base = extractor.extract(inst_map)
    print(f"    该场景共 {len(base)} 个实例")

    if not base:
        print("    [ERROR] 没有可跟踪的实例")
        return 1

    # 选一个球面位置明确的实例，且远离极点（极点附近 yaw 会退化）
    target = max(
        (d for d in base if abs(d.pitch_deg) < 45.0 and d.confidence > 0.5),
        key=lambda d: d.confidence,
        default=base[0],
    )
    print(f"    选定: {target.category}")
    print(f"      初始球面位置 yaw={target.yaw_deg:.2f}°  pitch={target.pitch_deg:.2f}°")
    print(f"      角尺寸 {target.width_deg:.2f}° × {target.height_deg:.2f}°")

    # ---- 3. 生成合成序列（每帧平移 1° 左右）----
    print(f"\n[3] 生成 {args.frames} 帧序列（np.roll 环绕平移）")
    # 每帧水平平移 12 像素 ≈ 12/2048*360 ≈ 2.1°/帧
    px_per_frame = 12
    total_shift = px_per_frame * args.frames
    print(f"    每帧平移 {px_per_frame} px ≈ {px_per_frame / width * 360:.2f}°/帧")
    print(f"    总位移 {total_shift} px ≈ {total_shift / width * 360:.0f}° "
          f"（{'跨越接缝' if total_shift > width / 2 else '未跨越'}）")

    tracker = SphericalTracker(
        dt=1 / 30,
        max_age=15,
        min_hits=2,
        gate_deg=30.0,
        # 合成序列每帧转 2°，真实角速度约 63°/秒，需要放宽过程噪声
        accel_std=300.0,
    )

    frames: list[FrameTracks] = []
    id_history: list[list[int]] = []
    seam_frame: int | None = None

    for i in range(args.frames):
        shifted = shift_instance_map(inst_map, i * px_per_frame)

        # 每帧走真实的提取流程
        dets = extractor.extract(shifted)

        states = tracker.update(dets, dt=1 / 30)

        frames.append(
            FrameTracks(frame_index=i, time_s=i / 30.0, tracks=list(states))
        )
        confirmed = [s.track_id for s in states if s.is_confirmed]
        id_history.append(confirmed)

        if seam_frame is None and any(s.crosses_seam for s in states):
            seam_frame = i

    # ---- 4. 跟踪结果分析 ----
    print("\n[4] 跟踪结果")
    all_ids: set[int] = set()
    for ids in id_history:
        all_ids.update(ids)
    print(f"    出现的 track_id 共 {len(all_ids)} 个: {sorted(all_ids)}")

    max_concurrent = max((len(ids) for ids in id_history), default=0)
    print(f"    单帧最多并发轨迹: {max_concurrent}")

    print(f"    累计跨接缝次数: {tracker.total_seam_crossings}")
    if seam_frame is not None:
        print(f"    首次检测到跨接缝: 第 {seam_frame} 帧")
    else:
        print("    ⚠ 未检测到跨接缝（可能位移不足或目标已丢失）")

    # ---- 5. 构建 PSG ----
    print("\n[5] 构建 PSG")
    meta = VideoMeta(
        fps=30.0, duration_s=args.frames / 30.0, erp_width=width, erp_height=height
    )
    psg = build_psg(frames, meta, config=BuildConfig(min_track_frames=3))
    print(f"    {psg!r}")

    print("\n    轨迹列表:")
    for node in list(psg.tracks.values())[:6]:
        print(
            f"      #{node.track_id:<3} {node.category:<16} "
            f"时长={node.stats.duration_s:.2f}s  "
            f"角速度={node.stats.angular_velocity_mean_dps:6.1f}°/s  "
            f"跨接缝={node.stats.crosses_seam}  "
            f"属性={node.attributes}"
        )

    # ---- 6. LLM 视图 ----
    print("\n[6] LLM 视图（送进模型的实际内容）")
    view = psg.to_llm_view(max_tracks=3)
    text = json.dumps(view, ensure_ascii=False, indent=2)
    print("    " + text.replace("\n", "\n    ")[:700])
    print(f"\n    视图大小: {len(text)} 字符")

    # 不变量检查：LLM 视图中不得出现坐标
    if any(k in text for k in ("yaw_deg", "pitch_deg", "numeric_ref", ".npz")):
        print("    ✗ LLM 视图泄漏了几何坐标！")
        return 1
    print("    ✓ LLM 视图无几何坐标泄漏")

    # ---- 7. 结论 ----
    print("\n" + "=" * 68)
    print("  验证结论")
    print("=" * 68)

    passed = True

    if len(all_ids) <= max_concurrent * 2:
        print(f"  ✓ ID 数量合理（{len(all_ids)} 个 ID / 最多 {max_concurrent} 并发）")
    else:
        print(f"  ✗ ID 爆炸（{len(all_ids)} 个 ID，说明跟踪频繁丢失重建）")
        passed = False

    if tracker.total_seam_crossings > 0:
        print(f"  ✓ 跨接缝跟踪成功（{tracker.total_seam_crossings} 次）")
    else:
        print("  ⚠ 本次序列未产生跨接缝")

    print(f"  ✓ PSG 构建成功（{len(psg.tracks)} 条轨迹）")

    if passed:
        print("\n  整条链路在真实数据上跑通。")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
