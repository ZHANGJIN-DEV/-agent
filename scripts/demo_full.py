"""完整 Demo：从真实全景图生成一段自动运镜视频。

==============================================================================
这是整个项目的端到端演示
==============================================================================

    真实实例图
        ↓ InstanceExtractor
    球面检测（真实类别、真实形状）
        ↓ np.roll 合成运动
    帧序列
        ↓ SphericalTracker
    逐帧轨迹（跨接缝 ID 保持）
        ↓ build_psg
    Panorama Scene Graph（符号层 + 数值层）
        ↓ EditPlanner（LLM 或降级）
    Edit DSL（只有 track_id，没有坐标）
        ↓ TrajectorySolver（L-BFGS-B on unwrapped yaw）
    相机轨迹
        ↓ NfovRenderer
    成片 mp4

==============================================================================
关于 LLM 规划的说明
==============================================================================

默认使用 `--llm auto`：有 API key 就用真实 LLM，否则自动降级到
纯优化器方案。两种路径产出的都是合法的 EditDSL，下游无差别。

这正好构成一个可复现的消融对比：
    python scripts/demo_full.py --llm off    # 纯优化器
    python scripts/demo_full.py --llm on     # 带 LLM 规划
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_INSTANCE = Path(
    "C:/pano_data/datasets/nyc_instance/nyc_instance_panorama/panorama_3518.png"
)
DEFAULT_RAW = Path("C:/pano_data/datasets/nyc_raw/nyc_Raw/panorama_3518.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="完整端到端 Demo")
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW,
                        help="真实 RGB 全景图（用作渲染素材）")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "assets" / "demo_full.mp4")
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--fps", type=float, default=15.0,
                        help="输出帧率。合成序列用低帧率即可，文件更小。")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--instruction", default="剪一个突出画面中最显眼主体的短片")
    parser.add_argument("--llm", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--shift-px", type=int, default=14,
                        help="每帧水平平移像素数（模拟目标运动）")
    args = parser.parse_args(argv)

    from pano_agent.agent.planner import EditPlanner, PlannerConfig
    from pano_agent.optimize.solver import TrajectorySolver
    from pano_agent.perception.instance import InstanceExtractor, load_instance_map
    from pano_agent.perception.track_spherical import SphericalTracker
    from pano_agent.perception.types import FrameTracks
    from pano_agent.render.nfov import NfovRenderer, RenderConfig
    from pano_agent.sir import BuildConfig, VideoMeta, build_psg

    print("=" * 70)
    print("  PanoDirector 完整流程演示")
    print("=" * 70)

    if not args.instance.is_file():
        print(f"[ERROR] 找不到实例图: {args.instance}")
        return 1

    # ================================================== 1. 实例提取
    print(f"\n[1/6] 实例提取 —— {args.instance.name}")
    inst_map = load_instance_map(args.instance)
    erp_h, erp_w = inst_map.shape[:2]
    print(f"      ERP {erp_w}×{erp_h}  比例 {erp_w/erp_h:.4f}")

    extractor = InstanceExtractor()
    base_dets = extractor.extract(inst_map)
    print(f"      提取到 {len(base_dets)} 个实例")

    if not base_dets:
        print("[ERROR] 没有可用实例")
        return 1

    # 报告类别分布
    from collections import Counter

    dist = Counter(d.category for d in base_dets)
    print(f"      类别分布: {dict(dist.most_common(6))}")

    # 选一个主体用于展示（远离极点、置信度高）
    candidates = [d for d in base_dets if abs(d.pitch_deg) < 45.0]
    subject = max(candidates or base_dets, key=lambda d: d.confidence)
    print(f"      → 演示主体: {subject.category}  "
          f"(yaw={subject.yaw_deg:.1f}° pitch={subject.pitch_deg:.1f}°)")

    # ================================================== 2. 合成运动 + 跟踪
    print(f"\n[2/6] 合成运动并跟踪（{args.frames} 帧，每帧平移 {args.shift_px}px）")
    deg_per_frame = args.shift_px / erp_w * 360.0
    print(f"      角速度约 {deg_per_frame:.2f}°/帧 = "
          f"{deg_per_frame * args.fps:.1f}°/秒")

    tracker = SphericalTracker(
        dt=1 / args.fps, max_age=10, min_hits=2, gate_deg=35.0, accel_std=300.0
    )

    frames_tracks: list[FrameTracks] = []
    for i in range(args.frames):
        shifted = np.roll(inst_map, i * args.shift_px, axis=1)
        dets = extractor.extract(shifted)
        states = tracker.update(dets, dt=1 / args.fps)
        frames_tracks.append(
            FrameTracks(frame_index=i, time_s=i / args.fps, tracks=list(states))
        )

    ids_seen = {s.track_id for ft in frames_tracks for s in ft.tracks}
    print(f"      出现的 track_id: {len(ids_seen)} 个")
    print(f"      跨接缝次数: {tracker.total_seam_crossings}")

    # ================================================== 3. PSG
    print("\n[3/6] 构建 Panorama Scene Graph")
    meta = VideoMeta(
        fps=args.fps, duration_s=args.frames / args.fps,
        erp_width=erp_w, erp_height=erp_h,
    )
    psg = build_psg(frames_tracks, meta, config=BuildConfig(min_track_frames=3))
    print(f"      {psg!r}")

    top = sorted(psg.tracks.values(), key=lambda n: -n.stats.highlight_score)[:3]
    for node in top:
        print(f"        #{node.track_id:<3} {node.category:<16} "
              f"高光={node.stats.highlight_score:.3f} "
              f"角速度={node.stats.angular_velocity_mean_dps:5.1f}°/s")

    # ================================================== 4. 规划
    print("\n[4/6] LLM 规划")
    use_llm = args.llm == "on"
    if args.llm == "auto":
        try:
            from pano_agent.agent.providers import create_provider
            from pano_agent.config import get_settings

            settings = get_settings()
            provider = create_provider("llm", settings=settings)
            use_llm = provider.name != "mock"
            print(f"      provider = {provider.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"      LLM 不可用（{type(exc).__name__}），将使用降级方案")
            provider = None
            use_llm = False
    elif use_llm:
        from pano_agent.agent.providers import create_provider

        provider = create_provider("llm")
    else:
        provider = None

    if use_llm and provider is not None:
        planner = EditPlanner(provider, PlannerConfig(max_retries=3))
        plan = planner.plan(psg, args.instruction, target_duration_s=args.frames / args.fps)
        print(f"      尝试 {plan.attempts} 次，降级={plan.used_fallback}")
        if plan.reasoning:
            print(f"      模型推理: {plan.reasoning[:160]}...")
    else:
        # 强制降级路径
        planner = EditPlanner(None, PlannerConfig()) if False else None
        from pano_agent.agent.planner import EditPlanner as _EP

        class _NullProvider:
            name = "disabled"
            supports_vision = False

            def chat(self, *a, **k):
                raise RuntimeError("LLM 已禁用")

        planner = _EP(_NullProvider(), PlannerConfig(max_retries=1))
        plan = planner.plan(psg, args.instruction, target_duration_s=args.frames / args.fps)
        print("      使用纯优化器降级方案（--llm off）")

    print(f"      {plan.dsl}")
    for i, shot in enumerate(plan.dsl.shots):
        print(f"        镜头 {i}: [{shot.t_start:5.2f}s, {shot.t_end:5.2f}s] "
              f"subject={shot.subject} {shot.shot_type}/{shot.camera_motion} "
              f"fov={shot.fov_deg:.0f}°")

    # ================================================== 5. 轨迹求解
    print("\n[5/6] 求解相机轨迹")
    solver = TrajectorySolver()

    # 取出主体在每帧的球面位置（从 PSG 的数值层/轨迹状态）
    total_frames = args.frames
    all_cams = []

    for shot in plan.dsl.shots:
        f0 = int(round(shot.t_start * args.fps))
        f1 = int(round(shot.t_end * args.fps))
        f0, f1 = max(f0, 0), min(f1, total_frames)
        if f1 <= f0:
            continue

        n = f1 - f0
        if shot.subject is not None:
            # 从跟踪结果里取该主体的逐帧位置
            ys, ps = [], []
            for fi in range(f0, f1):
                match = next(
                    (s for s in frames_tracks[fi].tracks if s.track_id == shot.subject),
                    None,
                )
                if match is not None:
                    ys.append(match.yaw_deg)
                    ps.append(match.pitch_deg)

            if len(ys) < 2:
                # 该主体在此区间可见帧太少，退化为静态机位
                ys = [float(np.mean([s.yaw_deg for s in frames_tracks[fi].tracks] or [0.0]))
                      for fi in range(f0, f1)]
                ps = [0.0] * n
            subj_yaw = np.asarray(ys[:n], dtype=np.float64)
            subj_pitch = np.asarray(ps[:n], dtype=np.float64)
            # 长度对齐
            if len(subj_yaw) < n:
                subj_yaw = np.pad(subj_yaw, (0, n - len(subj_yaw)), mode="edge")
                subj_pitch = np.pad(subj_pitch, (0, n - len(subj_pitch)), mode="edge")
        else:
            # 无主体镜头：缓慢横摇
            subj_yaw = np.linspace(-40.0, 40.0, n)
            subj_pitch = np.zeros(n)

        traj = solver.solve_shot(
            subj_yaw, subj_pitch,
            fps=args.fps,
            motion=shot.camera_motion,
            target_fov_deg=shot.fov_deg,
            rule_of_thirds=plan.dsl.constraints.rule_of_thirds,
        )
        print(f"      镜头[f{f0}-{f1}] {shot.camera_motion:<8} "
              f"平滑度={traj.smoothness():7.3f}  "
              f"峰值角速度={traj.angular_speed_dps().max():5.1f}°/s")
        all_cams.extend(traj.to_cameras(args.width, args.height))

    if not all_cams:
        print("[ERROR] 没有生成任何相机")
        return 1

    print(f"      共 {len(all_cams)} 帧相机")

    # ================================================== 6. 渲染
    print("\n[6/6] 渲染成片")
    if args.raw.is_file():
        from PIL import Image

        source = np.asarray(Image.open(args.raw))[..., :3]
        print(f"      素材: {args.raw.name}  {source.shape}")
    else:
        # 退化为彩色化的实例图
        from pano_agent.perception.instance import CLASS_NAMES
        import cv2

        print(f"      素材: 真实 RGB 不存在，改用彩色化实例图")
        source = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
        for cid in np.unique(inst_map[..., 3]):
            color = tuple(int(c) for c in np.random.default_rng(int(cid)).integers(60, 240, 3))
            source[inst_map[..., 3] == cid] = color

    # 为每一帧构造一个"视频"：真实全景图 + 逐帧环绕平移
    # （模拟相机在静止场景中移动，或者说目标在动 —— 二者在渲染上等价）
    from pano_agent.optimize.solver import CameraTrajectory

    n_render = len(all_cams)
    erp_frames = np.stack(
        [np.roll(source, i * args.shift_px, axis=1) for i in range(n_render)]
    )

    traj_all = CameraTrajectory(
        times=np.arange(n_render) / args.fps,
        yaw_deg=np.array([c.yaw_deg for c in all_cams]),
        pitch_deg=np.array([c.pitch_deg for c in all_cams]),
        fov_deg=np.array([c.fov_x_deg for c in all_cams]),
        yaw_continuous_deg=np.unwrap(np.array([c.yaw_deg for c in all_cams]), period=360.0),
    )

    renderer = NfovRenderer(
        RenderConfig(
            width=args.width, height=args.height,
            fps=args.fps, crf=20, preset="medium",
        )
    )
    out = renderer.render_to_file(erp_frames, traj_all, args.output)

    size_mb = out.stat().st_size / 1048576
    print(f"\n{'=' * 70}")
    print(f"  ✓ 成片已生成: {out}")
    print(f"    {n_render} 帧 / {args.width}×{args.height} / {args.fps:.0f}fps / {size_mb:.2f} MB")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
