"""轨迹优化 —— 把 Edit DSL 编译成连续相机轨迹。

==============================================================================
问题形式化
==============================================================================

已知：
    主体逐帧球面位置   (yaw_t, pitch_t)    来自 PSG 的数值层
    镜头语义约束       拍谁 / 景别 / 运镜    来自 Edit DSL

求解：
    相机逐帧参数       (yaw_t, pitch_t, fov_t)

使目标函数最小：

    E = w_center · L_center      主体居中（大圆距离）
      + w_smooth · L_smooth      轨迹平滑（二阶差分）
      + w_fov    · L_fov         FOV 平滑 + 景别匹配
      + w_thirds · L_thirds      三分法（可选）

==============================================================================
为什么必须在 unwrapped 空间求解
==============================================================================

与 `track_spherical` **完全同源**的问题：

    yaw 生活在圆 S¹ 上，而优化器只理解线性的 ℝ。

若直接在 [-180, 180) 上优化，相机从 179° 转到 181° 时，
参数上表现为 179 → -179 的跳变。梯度会认为需要跨越 358°，
于是要么收敛到错误解，要么在接缝处产生剧烈抖动。

解法同样是**把周期性挪到数据进入系统的边界**：

    1. 进入时：把主体 yaw 序列 unwrap 到 ℝ（可能超出 ±180°）
    2. 求解全程在 ℝ 上进行（连续性天然成立）
    3. 输出时：wrap 回 [-180, 180)

这是本项目的第三次应用同一原则（前两次：跟踪器的 yaw 提升、
检测框的接缝连续坐标）。

==============================================================================
目标函数的各项
==============================================================================

**L_center**（主体居中）
    主体球面位置与相机光轴的**大圆距离**。用角度而非像素 ——
    像素距离在高纬会失效，导致构图约束在高纬区域失灵。

**L_smooth**（轨迹平滑）
    相机角位置序列的二阶差分（即角加速度）。
    惩罚它可以让运镜平稳，避免生硬的甩镜。

**L_fov**
    两项：FOV 变化率（避免突兀变焦）+ 与目标景别的偏差。

**L_thirds**（三分法）
    切平面意义下把主体放到三分点。需要走完整的虚拟相机投影链
    （`geometry.camera`），不是简单的线性映射。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from ..geometry.sphere import great_circle_distance, unwrap_yaw, wrap_pitch, wrap_yaw

logger = logging.getLogger(__name__)

MotionName = Literal["static", "follow", "lead", "orbit", "push_in", "pull_out", "pan"]


@dataclass
class TrajectoryConfig:
    """轨迹优化配置。各项权重决定"什么样的运镜算好"。"""

    w_center: float = 1.0
    """主体居中项的权重。这是最重要的构图约束。"""

    w_smooth: float = 0.5
    """平滑项的权重。越大越平稳，但跟随快速主体时会滞后。"""

    w_fov_smooth: float = 0.3
    w_fov_target: float = 0.2
    w_thirds: float = 0.0
    """三分法权重。默认 0 —— 它是个审美偏好，不是硬性正确性。"""

    max_angular_vel_dps: float = 45.0
    max_fov_rate_dps: float = 20.0

    # 运镜的语义实现参数
    lead_offset_deg: float = 15.0
    """lead 运镜：相机领先主体这么多度。"""

    static_fov_deg: float = 85.0
    """static 运镜使用的固定 FOV。"""

    max_iter: int = 200


@dataclass
class CameraTrajectory:
    """求解出的相机轨迹。

    `yaw_deg` 是对外的归一化值（∈ [-180, 180)），
    `yaw_continuous_deg` 是求解空间的连续值（可能超出 ±180°）。

    两者**必须同时保存**：前者用于渲染（球面坐标本身就是周期的），
    后者用于计算平滑度 —— 若对归一化值求差分，跨接缝处会得到
    虚假的 358° 巨值，让"最平滑"的轨迹看起来最不平滑。
    """

    times: NDArray[np.float64]
    yaw_deg: NDArray[np.float64]
    pitch_deg: NDArray[np.float64]
    fov_deg: NDArray[np.float64]
    yaw_continuous_deg: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.yaw_continuous_deg is None:
            # 未提供时退化为 unwrap 后的归一化值（对单镜头通常等价）
            self.yaw_continuous_deg = unwrap_yaw(self.yaw_deg)

    def __len__(self) -> int:
        return len(self.times)

    def at(self, index: int, width: int = 1920, height: int = 1080):
        """取第 index 帧的相机对象。"""
        from ..geometry.camera import VirtualCamera

        i = int(np.clip(index, 0, len(self) - 1))
        return VirtualCamera(
            yaw_deg=float(self.yaw_deg[i]),
            pitch_deg=float(self.pitch_deg[i]),
            roll_deg=0.0,
            fov_x_deg=float(self.fov_deg[i]),
            width=width,
            height=height,
        )

    def to_cameras(self, width: int = 1920, height: int = 1080) -> list:
        return [self.at(i, width, height) for i in range(len(self))]

    # ------------------------------------------------------ 质量指标

    def angular_speed_dps(self) -> NDArray[np.float64]:
        """逐帧相机角速度（度/秒），**跨接缝感知**。"""
        if len(self) < 2:
            return np.zeros(1)
        d = np.abs(
            great_circle_distance(
                self.yaw_deg[:-1], self.pitch_deg[:-1],
                self.yaw_deg[1:], self.pitch_deg[1:],
            )
        )
        dt = np.diff(self.times)
        dt = np.where(dt > 1e-9, dt, 1e-9)
        return np.concatenate([[0.0], d / dt])

    def smoothness(self) -> float:
        """角加速度的均方，越小越平滑。

        **必须在连续 yaw 上计算** —— 直接对归一化的 yaw_deg 求差分，
        跨接缝处会得到虚假的 358° 巨值，让最平滑的轨迹看起来最糟。
        """
        if len(self) < 3:
            return 0.0
        y = np.asarray(self.yaw_continuous_deg, dtype=np.float64)
        p = np.asarray(self.pitch_deg, dtype=np.float64)
        return float(np.mean(np.diff(y, n=2) ** 2 + np.diff(p, n=2) ** 2))

    def __repr__(self) -> str:
        return (
            f"<CameraTrajectory {len(self)} 帧 / {self.times[-1]:.2f}s "
            f"峰值角速度 {self.angular_speed_dps().max():.1f}°/s>"
        )


class TrajectorySolver:
    """把镜头规格与主体轨迹求解成相机轨迹。"""

    def __init__(self, config: TrajectoryConfig | None = None) -> None:
        self.config = config or TrajectoryConfig()

    # ------------------------------------------------------------ 主入口

    def solve_shot(
        self,
        subject_yaw_deg: ArrayLike,
        subject_pitch_deg: ArrayLike,
        *,
        times: ArrayLike | None = None,
        fps: float = 30.0,
        motion: MotionName = "follow",
        target_fov_deg: float = 70.0,
        rule_of_thirds: bool = False,
        yaw_sign: float = 1.0,
    ) -> CameraTrajectory:
        """求解一个镜头的相机轨迹。

        Args:
            subject_yaw_deg: 主体逐帧 yaw（**会被 unwrap**）。
            subject_pitch_deg: 主体逐帧 pitch。
            times: 时间戳；None 则按 fps 生成。
            motion: 运镜方式，影响初始猜测与约束。
            target_fov_deg: 目标景别对应的 FOV。
            rule_of_thirds: 是否启用三分法约束。
            yaw_sign: 三分法时主体偏向画面哪一侧（+1 右 / -1 左）。

        Returns:
            CameraTrajectory。
        """
        subj_yaw = unwrap_yaw(np.asarray(subject_yaw_deg, dtype=np.float64))
        subj_pitch = np.asarray(subject_pitch_deg, dtype=np.float64)
        n = len(subj_yaw)

        if n == 0:
            raise ValueError("主体轨迹为空")
        if len(subj_pitch) != n:
            raise ValueError(
                f"yaw 与 pitch 长度不一致: {n} vs {len(subj_pitch)}"
            )

        if times is None:
            times = np.arange(n, dtype=np.float64) / fps
        times = np.asarray(times, dtype=np.float64)

        # ---- 初始猜测：直接跟随主体 ----
        init_yaw = self._initial_yaw(subj_yaw, subj_pitch, motion)
        init_pitch = subj_pitch.copy() * self._pitch_tracking_gain(motion)

        x0 = np.concatenate(
            [
                init_yaw,
                init_pitch,
                np.full(n, float(target_fov_deg)),
            ]
        )

        # ---- 边界（L-BFGS-B）----
        bounds = (
            [(None, None)] * n                      # yaw 无界（已在 ℝ 上）
            + [(-90.0, 90.0)] * n                   # pitch 有界
            + [(15.0, 130.0)] * n                   # fov 有界
        )

        result = minimize(
            self._objective,
            x0,
            args=(subj_yaw, subj_pitch, times, target_fov_deg, rule_of_thirds, yaw_sign),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.config.max_iter, "ftol": 1e-8},
        )

        if not result.success:
            logger.warning("轨迹优化未完全收敛: %s（继续使用当前解）", result.message)

        sol = result.x
        return CameraTrajectory(
            times=times,
            # 对外：归一化到 [-180, 180)（球面坐标本身就是周期的）
            yaw_deg=wrap_yaw(sol[:n]),
            pitch_deg=wrap_pitch(sol[n : 2 * n]),
            fov_deg=np.clip(sol[2 * n :], 15.0, 130.0),
            # 对内：保留求解空间的连续值，供 smoothness() 使用
            yaw_continuous_deg=sol[:n].copy(),
        )

    # ------------------------------------------------------------ 内部

    def _initial_yaw(
        self, subj_yaw: NDArray, subj_pitch: NDArray, motion: MotionName
    ) -> NDArray:
        """不同运镜的初始猜测。

        好的初值对非凸优化很关键 —— 否则可能收敛到一个
        "相机不动、主体在画面里乱跑"的局部极小。
        """
        cfg = self.config
        if motion == "lead":
            # 领先：相机跑到主体前方（沿主体运动方向）
            velocity = np.gradient(subj_yaw)
            direction = np.sign(velocity)
            return subj_yaw + direction * cfg.lead_offset_deg
        if motion == "static":
            # 静止：取主体的时间平均位置
            return np.full_like(subj_yaw, float(np.mean(subj_yaw)))
        if motion == "pan":
            # 横摇：线性扫过主体范围
            return np.linspace(subj_yaw[0], subj_yaw[-1], len(subj_yaw))
        # follow / orbit / push_in / pull_out 都以跟随为初值
        return subj_yaw.copy()

    @staticmethod
    def _pitch_tracking_gain(motion: MotionName) -> float:
        """pitch 的跟随增益。

        俯仰方向不做满跟随 —— 完全跟随会让画面随主体的每一次
        上下抖动而晃动，观感很差。0.6 左右是常见折中。
        """
        return 1.0 if motion in ("follow", "lead") else 0.6

    def _objective(
        self,
        x: NDArray,
        subj_yaw: NDArray,
        subj_pitch: NDArray,
        times: NDArray,
        target_fov: float,
        rule_of_thirds: bool,
        yaw_sign: float,
    ) -> float:
        """目标函数。"""
        n = len(subj_yaw)
        cam_yaw = x[:n]
        cam_pitch = x[n : 2 * n]
        cam_fov = x[2 * n :]

        cfg = self.config

        # ---- L_center：主体到光轴的大圆距离 ----
        d = great_circle_distance(cam_yaw, cam_pitch, subj_yaw, subj_pitch)
        l_center = float(np.mean(d**2))

        # ---- L_smooth：二阶差分（角加速度）----
        if n >= 3:
            acc_yaw = np.diff(cam_yaw, n=2)
            acc_pitch = np.diff(cam_pitch, n=2)
            l_smooth = float(np.mean(acc_yaw**2 + acc_pitch**2))
        else:
            l_smooth = 0.0

        # ---- L_fov ----
        if n >= 2:
            fov_rate = np.diff(cam_fov) / np.maximum(np.diff(times), 1e-9)
            l_fov_smooth = float(np.mean(fov_rate**2))
        else:
            l_fov_smooth = 0.0
        l_fov_target = float(np.mean((cam_fov - target_fov) ** 2))

        # ---- L_thirds（可选）----
        l_thirds = 0.0
        if rule_of_thirds and n >= 1:
            # 三分法：把主体放在画面水平三分点。
            # 在切平面近似下，主体与光轴的 yaw 差应约为 ±fov/3
            desired = cam_fov / 3.0 * yaw_sign
            delta = wrap_yaw(subj_yaw - cam_yaw)
            l_thirds = float(np.mean((delta - desired) ** 2))

        return (
            cfg.w_center * l_center
            + cfg.w_smooth * l_smooth
            + cfg.w_fov_smooth * l_fov_smooth
            + cfg.w_fov_target * l_fov_target
            + cfg.w_thirds * l_thirds
        )
