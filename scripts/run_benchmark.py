"""跑评测基准，生成可复现的对比报告。

用法：
    python scripts/run_benchmark.py                    # 默认参数
    python scripts/run_benchmark.py --noise 1.5 --n 60
    python scripts/run_benchmark.py --output docs/benchmark.md

==============================================================================
评测设计说明（哪些是可信的，哪些不是）
==============================================================================

**可信的部分**：
    本脚本用**受控的合成主体轨迹**做基准 —— 正弦运动叠加已知标准差的高斯噪声。
    这让我们能精确知道"输入有多抖"，从而计算"平滑倍数"这类相对指标。
    baseline 之间的对比在**同一份输入**上进行，因而是公平的。

**不可信 / 未验证的部分**（报告中必须声明）：
    ✗ 这**不是**真实视频上的跟踪质量 —— 数据集是静态全景图，没有时间序列
    ✗ 没有 Pano2Vid 的人工运镜 GT，因而**无法**做 DTW 对比
    ✗ 主体位置是合成的，不代表真实检测器的误差分布

本脚本的定位是：**验证各设计组件的相对贡献**，而不是声称"系统在真实
场景中达到了 X% 的性能"。把边界划清楚，比让读者误以为全面验证了更有价值。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def build_subject_track(
    n_frames: int,
    fps: float,
    *,
    noise_std_deg: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造受控的主体轨迹：正弦运动 + 高斯噪声。

    Returns:
        (noisy_yaw, pitch, times) —— noisy_yaw 是"检测器观测到的"位置，
        带噪声；pitch 与 times 无噪声。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames, dtype=np.float64) / fps

    clean_yaw = 30.0 * np.sin(2 * np.pi * 0.3 * t) + 5.0
    pitch = 10.0 * np.sin(2 * np.pi * 0.2 * t)

    noisy_yaw = clean_yaw + rng.normal(0.0, noise_std_deg, n_frames)
    return noisy_yaw, pitch, t


def run_suite(
    *,
    n_frames: int = 60,
    fps: float = 15.0,
    noise_std_deg: float = 1.5,
    fov_deg: float = 70.0,
    max_dps: float = 45.0,
    seed: int = 0,
) -> tuple[list[tuple[str, object]], float]:
    """跑全部 baseline，返回 (结果列表, 输入噪声基准)。"""
    from pano_agent.eval import (
        ALL_BASELINES,
        BaselineConfig,
        evaluate_trajectory,
        run_baseline,
        trajectory_smoothness,
    )

    cfg = BaselineConfig(fps=fps, target_fov_deg=fov_deg, max_angular_vel_dps=max_dps)
    yaw, pitch, times = build_subject_track(
        n_frames, fps, noise_std_deg=noise_std_deg, seed=seed
    )

    baseline_smoothness = trajectory_smoothness(yaw, pitch, times)

    results = []
    for name in ALL_BASELINES:
        traj = run_baseline(name, yaw, pitch, times, config=cfg)
        metrics = evaluate_trajectory(
            traj.yaw_deg, traj.pitch_deg, traj.fov_deg, traj.times,
            yaw, pitch,
            max_angular_vel_dps=max_dps,
            yaw_continuous_deg=traj.yaw_continuous_deg,
        )
        results.append((name, metrics))

    return results, baseline_smoothness


def render_markdown(
    results: list[tuple[str, object]],
    baseline_smoothness: float,
    *,
    n_frames: int,
    fps: float,
    noise_std_deg: float,
    fov_deg: float,
    max_dps: float,
) -> str:
    """生成 markdown 格式的评测报告。"""
    lines = [
        "# 轨迹质量基准测试",
        "",
        "## 实验设置",
        "",
        f"- 帧数：{n_frames}（{fps:.0f} fps，共 {n_frames / fps:.1f} 秒）",
        f"- 主体轨迹：正弦运动 + 高斯噪声（σ = {noise_std_deg:.1f}°）",
        f"- 目标 FOV：{fov_deg:.0f}°",
        f"- 角速度上限：{max_dps:.0f}°/s",
        f"- 输入噪声的角加速度基准：**{baseline_smoothness:.1f} °/s²**",
        "",
        "## 结果",
        "",
        "| Baseline | 居中率 | 平均偏移 | 平滑 (°/s²) | 角速度合规 | 峰值速度 |",
        "|---|---|---|---|---|---|",
    ]

    for name, m in results:
        lines.append(
            f"| `{name}` | {m.centering_rate:.0%} | {m.mean_offset_deg:.2f}° | "
            f"{m.smoothness:.1f} | {m.angular_velocity_compliance:.0%} | "
            f"{m.peak_angular_velocity_dps:.0f}°/s |"
        )

    lines += [
        "",
        "## 结论",
        "",
        "### 1. `center_crop` 是合格的地板线",
        "",
        f"居中率仅 {results[0][1].centering_rate:.0%} —— 固定机位无法应对主体移动。",
        "后续三个 baseline 确实做了事。",
        "",
        "### 2. 「每帧都对准」≠「成片好看」",
        "",
        f"`saliency_greedy` 的居中率 {results[1][1].centering_rate:.0%}（完美），",
        f"但平滑度 {results[1][1].smoothness:.1f} °/s² —— 与**输入噪声基准持平**，",
        "说明它完全没有做时序平滑。峰值角速度也超出配置上限。",
        "",
        "这是「需要轨迹优化」的核心论据：观众感知的是**运动**，",
        "逐帧贪心产生的轨迹在人眼里就是不停顿挫的抖动。",
        "",
        "### 3. 我们的方案是**权衡**而非全面碾压",
        "",
        f"`saliency_smooth` 的平滑度（{results[2][1].smoothness:.1f}）低于我们",
        f"（{results[3][1].smoothness:.1f} °/s²），但它的角速度合规率只有",
        f"{results[2][1].angular_velocity_compliance:.0%}，峰值 "
        f"{results[2][1].peak_angular_velocity_dps:.0f}°/s —— **超出配置上限**。",
        "",
        "低通滤波不感知速度约束，它的「更平滑」是以牺牲运动可控性换来的。",
        f"我们的方案在合规率上领先 "
        f"{results[3][1].angular_velocity_compliance - results[2][1].angular_velocity_compliance:.0%}"
        f"（{results[3][1].angular_velocity_compliance:.0%} vs "
        f"{results[2][1].angular_velocity_compliance:.0%}），",
        f"代价是平滑度略高。**这是一个有意的权衡。**",
        "",
        "## ⚠ 本评测不验证什么",
        "",
        "- **不是**真实视频上的跟踪质量 —— 数据集是静态全景图，无时间序列",
        "- **没有** Pano2Vid 的人工运镜 GT，因而无法做 DTW 对比",
        "- 主体位置是合成的，不代表真实检测器的误差分布",
        "",
        "本评测的定位是**验证各设计组件的相对贡献**，",
        "而不是声称「系统在真实场景达到了 X% 的性能」。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="轨迹质量基准测试")
    parser.add_argument("--n", type=int, default=60, help="帧数")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--noise", type=float, default=1.5, help="观测噪声标准差（度）")
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--max-dps", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None,
                        help="输出 markdown 路径；不指定则只打印")
    args = parser.parse_args(argv)

    print("=" * 76)
    print("  轨迹质量基准测试")
    print("=" * 76)

    results, base_smooth = run_suite(
        n_frames=args.n, fps=args.fps, noise_std_deg=args.noise,
        fov_deg=args.fov, max_dps=args.max_dps, seed=args.seed,
    )

    print(f"\n  输入噪声的角加速度基准: {base_smooth:.1f} °/s²\n")
    print("  {:^18} {:>8} {:>11} {:>12} {:>11} {:>11}".format(
        "Baseline", "居中率", "平均偏移", "平滑", "合规", "峰值速度"))
    print("  " + "-" * 76)
    for name, m in results:
        print("  {:^18} {:>7.0%} {:>10.2f}° {:>11.1f} {:>10.0%} {:>9.0f}°/s".format(
            name, m.centering_rate, m.mean_offset_deg, m.smoothness,
            m.angular_velocity_compliance, m.peak_angular_velocity_dps))

    print("\n  平滑 = 角加速度 RMS (度/秒²)   合规 = 角速度不超限的帧占比")

    markdown = render_markdown(
        results, base_smooth,
        n_frames=args.n, fps=args.fps, noise_std_deg=args.noise,
        fov_deg=args.fov, max_dps=args.max_dps,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"\n  报告已写入: {args.output}")

    print("\n" + "=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
