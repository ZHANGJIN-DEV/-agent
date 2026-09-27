"""把完整流程封装成可调用的函数 —— 供 API / CLI / 测试三方复用。

==============================================================================
为什么需要这一层
==============================================================================

`scripts/demo_full.py` 是脚本形态：参数来自 argparse、结果靠 print、
无法报告中间进度。这些特性让它**无法被服务调用**。

本模块把同一套流程重构成纯函数：

    输入参数 → PipelineResult（含产物路径、各阶段摘要、指标）

脚本与服务都调用它，**保证两条路径行为一致**。否则极易出现
"脚本能跑但 API 跑不通"这类问题 —— 而那类问题往往到演示当天
才被发现。

==============================================================================
进度回调
==============================================================================

`on_progress` 是一个可选回调 `(stage: str, pct: float, detail: str) -> None`。

Web Demo 用它推送进度；CLI 用它打印；测试传 None。
**不引入任何异步/消息队列机制** —— 回调是同步的，调用方自行决定
怎么用（写日志、更新内存状态、推 SSE）。

==============================================================================
为什么不上 Celery
==============================================================================

本任务是**单机串行**的：一块 GPU、一次处理一个视频。
引入 Celery 需要额外的 Redis 服务、worker 进程、任务序列化，
而换来的"分布式能力"在此场景下毫无用武之地。

FastAPI 的 `BackgroundTasks` 足以承担"提交后异步跑、期间可查状态"。
**拒绝为不需要的复杂度付费** —— 这也是本项目一贯的取舍。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

ProgressCallback = Callable[[str, float, str], None]


@dataclass
class PipelineConfig:
    """流程配置。"""

    fps: float = 15.0
    width: int = 960
    height: int = 540
    num_frames: int = 45
    """合成序列的帧数（数据集是静态图，用平移模拟运动）。"""

    shift_px: int = 14
    min_track_frames: int = 3
    target_fov_deg: float = 70.0
    use_llm: bool = False
    instruction: str = "剪一个突出画面中最显眼主体的短片"


@dataclass
class StageReport:
    """单个阶段的执行记录。"""

    name: str
    duration_s: float
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "stage": self.name,
            "duration_s": round(self.duration_s, 3),
            "detail": self.detail,
        }


@dataclass
class PipelineResult:
    """流程产出。"""

    output_path: Path | None
    stages: list[StageReport] = field(default_factory=list)
    psg_summary: dict = field(default_factory=dict)
    dsl: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    num_tracks: int = 0
    seam_crossings: int = 0
    render_frames: int = 0

    @property
    def total_duration_s(self) -> float:
        return sum(s.duration_s for s in self.stages)

    def to_dict(self) -> dict:
        return {
            "output_path": str(self.output_path) if self.output_path else None,
            "stages": [s.to_dict() for s in self.stages],
            "total_duration_s": round(self.total_duration_s, 3),
            "psg_summary": self.psg_summary,
            "dsl": self.dsl,
            "metrics": self.metrics,
            "num_tracks": self.num_tracks,
            "seam_crossings": self.seam_crossings,
            "render_frames": self.render_frames,
        }


def run_pipeline(
    instance_path: Path | str,
    raw_path: Path | str | None,
    output_path: Path | str,
    *,
    config: PipelineConfig | None = None,
    on_progress: ProgressCallback | None = None,
) -> PipelineResult:
    """执行完整流程。

    Args:
        instance_path: 实例分割图（提供真实目标）。
        raw_path: 真实 RGB 全景图（渲染素材）。None 则回退到彩色化实例图。
        output_path: 输出 mp4 路径。
        config: 配置。
        on_progress: 进度回调 (stage, pct, detail)。

    Returns:
        PipelineResult。
    """
    cfg = config or PipelineConfig()
    instance_path = Path(instance_path)
    output_path = Path(output_path)

    def report(stage: str, pct: float, detail: str = "") -> None:
        if on_progress is not None:
            on_progress(stage, pct, detail)

    result = PipelineResult(output_path=None)
    timers: dict[str, float] = {}

    def begin(name: str) -> float:
        report(name, 0.0)
        return time.perf_counter()

    def end(name: str, t0: float, detail: str = "") -> None:
        dt = time.perf_counter() - t0
        timers[name] = dt
        result.stages.append(StageReport(name=name, duration_s=dt, detail=detail))
        report(name, 1.0, detail)

    # ============================================ 1. 实例提取
    from ..perception.instance import InstanceExtractor, load_instance_map

    t0 = begin("实例提取")
    inst_map = load_instance_map(instance_path)
    erp_h, erp_w = inst_map.shape[:2]
    extractor = InstanceExtractor()
    base_dets = extractor.extract(inst_map)
    if not base_dets:
        raise ValueError("实例图中没有可用的前景实例")
    end("实例提取", t0, f"{len(base_dets)} 个实例, ERP {erp_w}×{erp_h}")

    # ============================================ 2. 合成运动 + 跟踪
    from ..perception.track_spherical import SphericalTracker
    from ..perception.types import FrameTracks

    t0 = begin("球面跟踪")
    tracker = SphericalTracker(
        dt=1 / cfg.fps, max_age=10, min_hits=2, gate_deg=35.0, accel_std=300.0
    )
    frames_tracks: list[FrameTracks] = []
    for i in range(cfg.num_frames):
        shifted = np.roll(inst_map, i * cfg.shift_px, axis=1)
        dets = extractor.extract(shifted)
        states = tracker.update(dets, dt=1 / cfg.fps)
        frames_tracks.append(
            FrameTracks(frame_index=i, time_s=i / cfg.fps, tracks=list(states))
        )
        report("球面跟踪", (i + 1) / cfg.num_frames, f"第 {i + 1}/{cfg.num_frames} 帧")

    ids_seen = {s.track_id for ft in frames_tracks for s in ft.tracks}
    result.seam_crossings = tracker.total_seam_crossings
    end(
        "球面跟踪", t0,
        f"{len(ids_seen)} 个 ID, 跨接缝 {tracker.total_seam_crossings} 次",
    )

    # ============================================ 3. PSG
    from ..sir import BuildConfig, VideoMeta, build_psg

    t0 = begin("构建场景图")
    meta = VideoMeta(
        fps=cfg.fps,
        duration_s=cfg.num_frames / cfg.fps,
        erp_width=erp_w,
        erp_height=erp_h,
    )
    psg = build_psg(
        frames_tracks, meta, config=BuildConfig(min_track_frames=cfg.min_track_frames)
    )
    result.num_tracks = len(psg.tracks)
    result.psg_summary = {
        "num_tracks": len(psg.tracks),
        "num_events": len(psg.events),
        "duration_s": round(meta.duration_s, 2),
    }
    end("构建场景图", t0, f"{len(psg.tracks)} 条轨迹, {len(psg.events)} 个事件")

    # ============================================ 4. 规划
    from ..agent.planner import EditPlanner, PlannerConfig

    t0 = begin("剪辑规划")
    if cfg.use_llm:
        from ..agent.providers import create_provider

        provider = create_provider("llm")
    else:
        class _DisabledProvider:
            name = "disabled"
            supports_vision = False

            def chat(self, *a, **k):
                raise RuntimeError("LLM 未启用")

        provider = _DisabledProvider()

    planner = EditPlanner(provider, PlannerConfig(max_retries=3 if cfg.use_llm else 1))
    plan = planner.plan(psg, cfg.instruction, target_duration_s=meta.duration_s)
    result.dsl = plan.dsl.to_prompt_dict()
    end(
        "剪辑规划", t0,
        f"{'降级' if plan.used_fallback else f'LLM {plan.attempts} 次'} → "
        f"{len(plan.dsl.shots)} 个镜头",
    )

    # ============================================ 5. 轨迹求解
    from ..optimize.solver import CameraTrajectory, TrajectorySolver

    t0 = begin("轨迹优化")
    solver = TrajectorySolver()
    all_cams = []

    for shot in plan.dsl.shots:
        f0 = max(int(round(shot.t_start * cfg.fps)), 0)
        f1 = min(int(round(shot.t_end * cfg.fps)), cfg.num_frames)
        if f1 <= f0:
            continue
        n = f1 - f0

        if shot.subject is not None:
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
                ys = [0.0] * n
                ps = [0.0] * n
            subj_yaw = np.asarray(ys[:n], dtype=np.float64)
            subj_pitch = np.asarray(ps[:n], dtype=np.float64)
            if len(subj_yaw) < n:
                subj_yaw = np.pad(subj_yaw, (0, n - len(subj_yaw)), mode="edge")
                subj_pitch = np.pad(subj_pitch, (0, n - len(subj_pitch)), mode="edge")
        else:
            subj_yaw = np.linspace(-40.0, 40.0, n)
            subj_pitch = np.zeros(n)

        traj = solver.solve_shot(
            subj_yaw, subj_pitch, fps=cfg.fps,
            motion=shot.camera_motion,
            target_fov_deg=shot.fov_deg or cfg.target_fov_deg,
            rule_of_thirds=plan.dsl.constraints.rule_of_thirds,
        )
        all_cams.extend(traj.to_cameras(cfg.width, cfg.height))

    if not all_cams:
        raise ValueError("没有生成任何相机轨迹")

    result.render_frames = len(all_cams)
    end("轨迹优化", t0, f"{len(all_cams)} 帧相机")

    # ============================================ 6. 渲染
    from ..render.nfov import NfovRenderer, RenderConfig

    t0 = begin("渲染成片")
    if raw_path is not None and Path(raw_path).is_file():
        from PIL import Image

        source = np.asarray(Image.open(raw_path))[..., :3]
    else:
        source = _colorize_instances(inst_map)

    n_render = len(all_cams)
    erp_frames = np.stack(
        [np.roll(source, i * cfg.shift_px, axis=1) for i in range(n_render)]
    )

    traj_all = CameraTrajectory(
        times=np.arange(n_render) / cfg.fps,
        yaw_deg=np.array([c.yaw_deg for c in all_cams]),
        pitch_deg=np.array([c.pitch_deg for c in all_cams]),
        fov_deg=np.array([c.fov_x_deg for c in all_cams]),
        yaw_continuous_deg=np.unwrap(
            np.array([c.yaw_deg for c in all_cams]), period=360.0
        ),
    )

    renderer = NfovRenderer(
        RenderConfig(width=cfg.width, height=cfg.height, fps=cfg.fps)
    )
    out = renderer.render_to_file(erp_frames, traj_all, output_path)
    result.output_path = out
    end("渲染成片", t0, f"{out.name} ({out.stat().st_size / 1048576:.2f} MB)")

    # ============================================ 指标
    from ..eval import evaluate_trajectory

    # 收集镜头边界 —— 成片由多个独立求解的镜头拼接而成，
    # 镜头间的跳跃是剪辑的正常行为，必须从平滑度/角速度指标里排除。
    # （不排除的话会得到"峰值 2195°/s、上限 45°/s"这类荒谬结果）
    cut_frames: list[int] = []
    cursor = 0
    for shot in plan.dsl.shots:
        f0 = max(int(round(shot.t_start * cfg.fps)), 0)
        f1 = min(int(round(shot.t_end * cfg.fps)), cfg.num_frames)
        if f1 <= f0:
            continue
        if cursor > 0:
            cut_frames.append(cursor)
        cursor += f1 - f0

    try:
        m = evaluate_trajectory(
            traj_all.yaw_deg, traj_all.pitch_deg, traj_all.fov_deg, traj_all.times,
            traj_all.yaw_deg, traj_all.pitch_deg,  # 自比对：仅用于报告轨迹自身性质
            yaw_continuous_deg=traj_all.yaw_continuous_deg,
            cut_frames=cut_frames,
        )
        result.metrics = {
            "smoothness": round(m.smoothness, 3),
            "peak_angular_velocity_dps": round(m.peak_angular_velocity_dps, 2),
            "angular_velocity_compliance": round(m.angular_velocity_compliance, 4),
            "num_cuts": len(cut_frames),
        }
    except Exception:  # noqa: BLE001 - 指标失败不应让整个任务失败
        result.metrics = {}

    return result


def _colorize_instances(inst_map: np.ndarray) -> np.ndarray:
    """把实例图彩色化，作为渲染素材的降级方案。"""
    h, w = inst_map.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in np.unique(inst_map[..., 3]):
        rng = np.random.default_rng(int(cid))
        color = tuple(int(c) for c in rng.integers(60, 240, 3))
        out[inst_map[..., 3] == cid] = color
    return out
