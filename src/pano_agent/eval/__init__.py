"""评测层：轨迹质量指标 + baseline 对比 + 消融实验。

设计原则见 `metrics_traj` 与 `baselines` 的模块文档。
一句话概括：**所有距离都在球面上算，所有对比都要公平。**
"""

from .baselines import (
    ALL_BASELINES,
    BaselineConfig,
    baseline_center_crop,
    baseline_ours,
    baseline_saliency_greedy,
    baseline_saliency_smooth,
    run_baseline,
)
from .metrics_traj import (
    TrajectoryMetrics,
    angular_velocity_compliance,
    evaluate_trajectory,
    id_stability_ratio,
    subject_centering_rate,
    subject_visibility_rate,
    tracking_id_switches,
    trajectory_smoothness,
)

__all__ = [
    "ALL_BASELINES",
    "BaselineConfig",
    "TrajectoryMetrics",
    "angular_velocity_compliance",
    "baseline_center_crop",
    "baseline_ours",
    "baseline_saliency_greedy",
    "baseline_saliency_smooth",
    "evaluate_trajectory",
    "id_stability_ratio",
    "run_baseline",
    "subject_centering_rate",
    "subject_visibility_rate",
    "tracking_id_switches",
    "trajectory_smoothness",
]
