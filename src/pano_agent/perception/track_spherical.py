"""球面多目标跟踪 —— 解决 360° 全景特有的「跨接缝丢 ID」问题。

==============================================================================
问题的本质
==============================================================================

在普通视频里，目标从画面右边出去就是真的出去了。全景视频不同：
**ERP 的左右边界是同一条经线**，一个目标从 x=W-1 走出去，下一帧会从
x=0 走进来 —— 它从未离开过视野，只是在图像坐标上"绕了一圈"。

朴素的跟踪器（SORT / ByteTrack 直接在像素坐标上工作）在这里必然失败：

    帧 t   : yaw = +179°    (图像最右侧)
    帧 t+1 : yaw = -179°    (图像最左侧)
    → 像素距离 = W - 2 像素，滤波器认为发生了巨大位移
    → 关联失败，原轨迹丢失，目标被分配一个新 ID

这正是 360VOT 所定义的 7 个 360° 特有挑战之一（CROSS_BOUNDARY），
也是最容易在真实全景视频里肉眼观察到的跟踪缺陷。

==============================================================================
解法：把周期量打包进表示，而不是在每个使用者处单独处理
==============================================================================

核心操作只有一行 —— 把观测 yaw「提升」到与预测值同一分支：

    k = round((predicted_unwrapped - observed_wrapped) / 360)
    lifted = observed_wrapped + k * 360

于是 +179° → -179° 的跳变在滤波器眼中变成 +179° → +181°，
速度估计是正常的 2°/帧，轨迹平滑连续，ID 得以保持。

这与 DA360 用 circular padding 消除 ERP 接缝、`sphere.unwrap_yaw` 用于
轨迹优化是**同一思想的三种体现**：
    周期性的表示（S¹ → ℝ 的提升）应当发生在**数据进入系统的边界**，
    之后的每一层都可以当它是普通线性量。

==============================================================================
与 SORT / ByteTrack 的差异
==============================================================================

    SORT / ByteTrack          本实现
    ───────────────────       ──────────────────────────────
    状态在像素坐标            状态在球面角度（yaw/pitch）
    关联用像素 IoU            关联用 great-circle 距离
    yaw 无特殊处理            yaw 在 unwrapped 空间滤波
    线性匀速运动模型          相同（在 unwrapped 角度空间近似线性）

注意最后一行：我们**没有**引入更复杂的运动模型（如恒速转弯曲率），
因为在 unwrapped 角度空间里，"匀速转动"就是线性运动，
这正是提升操作带来的简化。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

from ..geometry.sphere import great_circle_distance, wrap_pitch, wrap_yaw
from .kalman import ConstantVelocityKalman
from .types import SphericalDetection, TrackState

# 匈牙利算法中代表「禁止匹配」的大代价。
# 用一个大但有限的值而非 inf —— scipy 的 linear_sum_assignment 在含 inf 的
# 代价矩阵上行为不确定，而大有限值配合事后的门限过滤是等价的且更稳健。
_FORBIDDEN = 1e6


def lift_yaw(observed_yaw_deg: float, reference_yaw_deg: float) -> float:
    """把观测 yaw 提升到与参考值最接近的分支。

    这是整个模块的核心操作，只有三行，但决定了跨接缝跟踪的成败。

    Args:
        observed_yaw_deg: 观测角，通常来自检测器，已在 [-180, 180) 内。
        reference_yaw_deg: 参考角（预测值），可以是任意实数（已 unwrapped）。

    Returns:
        与 observed 相差 360° 整数倍、且与 reference 最接近的等价角度。

    Examples:
        >>> lift_yaw(-179.0, 181.0)
        181.0
        >>> lift_yaw(179.0, -181.0)
        -181.0
    """
    # 先折叠到同一周期，再选择最接近的代表元
    delta = wrap_yaw(observed_yaw_deg - reference_yaw_deg)
    return reference_yaw_deg + delta


class _Track:
    """内部轨迹对象 —— 持有 Kalman 状态与生命周期计数。

    与对外的 `TrackState` 分开，是因为前者需要可变状态与滤波器实例，
    后者应是对外只读的快照。
    """

    __slots__ = (
        "track_id", "kf", "width_deg", "height_deg", "category",
        "age", "hits", "time_since_update", "crosses_seam",
        "confidence", "embedding", "_max_unwrapped_yaw",
    )

    def __init__(
        self,
        track_id: int,
        detection: SphericalDetection,
        tracker: SphericalTracker,
    ) -> None:
        self.track_id = track_id
        self.width_deg = detection.width_deg
        self.height_deg = detection.height_deg
        self.category = detection.category
        self.confidence = detection.confidence
        self.embedding = detection.embedding

        # 状态向量 [yaw, pitch, v_yaw, v_pitch]
        # **yaw 从此刻起就是 unwrapped 的**，后续所有运算都在 ℝ 上进行
        self.kf = ConstantVelocityKalman(
            ndim=2,
            dt=tracker.dt,
            accel_std=tracker.accel_std,
            meas_std=tracker.meas_std,
        )
        self.kf.reset(
            position=[detection.yaw_deg, detection.pitch_deg],
            velocity=[0.0, 0.0],
        )

        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.crosses_seam = False
        self._max_unwrapped_yaw = abs(detection.yaw_deg)

    # ------------------------------------------------------------ 属性

    @property
    def unwrapped_yaw(self) -> float:
        """提升空间中的 yaw（可能超出 ±180，这是正常的）。"""
        return float(self.kf.position[0])

    @property
    def wrapped_yaw(self) -> float:
        """归一化到 [-180, 180) 的 yaw —— 对外的值。"""
        return float(wrap_yaw(self.unwrapped_yaw))

    @property
    def pitch(self) -> float:
        return float(np.clip(self.kf.position[1], -90.0, 90.0))

    # ------------------------------------------------------------ 生命周期

    def predict(self, dt: float | None = None) -> None:
        """外推一步。dt 变化时重建矩阵（视频帧率可能不恒定）。"""
        if dt is not None and abs(dt - self.kf.dt) > 1e-9:
            self.kf.dt = float(dt)
            self.kf._build_static_matrices()
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: SphericalDetection) -> None:
        """用匹配到的检测校正状态。"""
        # ★ 关键：先把观测提升到滤波器所在的分支，再送进 Kalman
        lifted_yaw = lift_yaw(detection.yaw_deg, self.unwrapped_yaw)
        pitch = float(np.clip(detection.pitch_deg, -90.0, 90.0))

        self.kf.update([lifted_yaw, pitch])

        # 尺寸与外观用滑动平均更新，避免单帧检测抖动直接传导
        alpha = 0.3
        self.width_deg = (1 - alpha) * self.width_deg + alpha * detection.width_deg
        self.height_deg = (1 - alpha) * self.height_deg + alpha * detection.height_deg
        self.confidence = (1 - alpha) * self.confidence + alpha * detection.confidence
        if detection.embedding is not None:
            self.embedding = detection.embedding

        self.hits += 1
        self.time_since_update = 0

        # 跨缝判定：unwrapped yaw 跑出 [-180, 180) 即说明发生过绕行
        self._max_unwrapped_yaw = max(self._max_unwrapped_yaw, abs(self.unwrapped_yaw))
        if self._max_unwrapped_yaw > 180.0:
            self.crosses_seam = True

    def to_state(self, *, is_confirmed: bool) -> TrackState:
        vel = self.kf.velocity
        return TrackState(
            track_id=self.track_id,
            yaw_deg=self.wrapped_yaw,
            pitch_deg=self.pitch,
            width_deg=float(self.width_deg),
            height_deg=float(self.height_deg),
            v_yaw_dps=float(vel[0]),
            v_pitch_dps=float(vel[1]),
            confidence=float(self.confidence),
            category=self.category,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            is_confirmed=is_confirmed,
            crosses_seam=self.crosses_seam,
        )


class SphericalTracker:
    """球面多目标跟踪器。

    Args:
        dt: 默认帧间隔（秒）。
        max_age: 连续多少帧未匹配后删除轨迹。
        min_hits: 确认一条轨迹所需的最少累计匹配次数
            （用于抑制单帧虚警产生的"幽灵轨迹"）。
        gate_deg: 关联门限 —— 预测位置与检测的大圆距离超过此值则禁止匹配。
            太小会导致快速运动的目标被反复重建 ID，太大则容易误关联。
        accel_std: 过程噪声（加速度标准差，度/秒²）。
        meas_std: 观测噪声（度）。
        appearance_weight: 外观代价的权重（0 表示纯几何关联）。
            需要检测结果携带 embedding 才生效。
    """

    def __init__(
        self,
        *,
        dt: float = 1.0 / 30.0,
        max_age: int = 30,
        min_hits: int = 3,
        gate_deg: float = 30.0,
        accel_std: float = 200.0,
        meas_std: float = 1.5,
        appearance_weight: float = 0.0,
    ) -> None:
        if max_age < 0:
            raise ValueError(f"max_age 不能为负，实际为 {max_age}")
        if min_hits < 1:
            raise ValueError(f"min_hits 至少为 1，实际为 {min_hits}")
        if gate_deg <= 0:
            raise ValueError(f"gate_deg 必须为正，实际为 {gate_deg}")

        self.dt = float(dt)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.gate_deg = float(gate_deg)
        self.accel_std = float(accel_std)
        self.meas_std = float(meas_std)
        self.appearance_weight = float(appearance_weight)

        self.tracks: list[_Track] = []
        self._next_id = 1
        self.frame_index = 0
        self.total_seam_crossings = 0
        """累计发生过的跨接缝次数 —— 用于评测时定位"本该失败但没有失败"的样本。"""

    # ------------------------------------------------------------ 主循环

    def update(
        self,
        detections: list[SphericalDetection],
        *,
        dt: float | None = None,
        time_s: float | None = None,
    ) -> list[TrackState]:
        """处理一帧检测结果，返回当前所有轨迹。

        Args:
            detections: 本帧的检测（球面表示）。
            dt: 距上一帧的间隔；None 则用构造时的默认值。
            time_s: 本帧时间戳，仅用于日志。

        Returns:
            轨迹状态列表（含未确认的，调用方可用 `is_confirmed` 过滤）。
        """
        step = dt if dt is not None else self.dt
        self.frame_index += 1

        # ---- 1. 预测 ----
        for track in self.tracks:
            track.predict(step)

        # ---- 2. 关联 ----
        matches, unmatched_tracks, unmatched_dets = self._associate(detections)

        # ---- 3. 用匹配到的检测更新 ----
        for ti, di in matches:
            was_seam = self.tracks[ti].crosses_seam
            self.tracks[ti].update(detections[di])
            if self.tracks[ti].crosses_seam and not was_seam:
                self.total_seam_crossings += 1

        # ---- 4. 未匹配的检测 → 新轨迹 ----
        for di in unmatched_dets:
            self.tracks.append(_Track(self._next_id, detections[di], self))
            self._next_id += 1

        # ---- 5. 老化与清理 ----
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # ---- 6. 输出 ----
        return [
            t.to_state(is_confirmed=t.hits >= self.min_hits and t.time_since_update == 0)
            for t in self.tracks
        ]

    # ------------------------------------------------------------ 关联

    def _build_cost_matrix(
        self, detections: list[SphericalDetection]
    ) -> NDArray[np.float64]:
        """构造关联代价矩阵。

        代价 = 预测位置与检测的**大圆距离**（度）。

        绝不能用像素距离：同样的像素间隔在赤道与高纬的物理意义相差十倍，
        用像素距离会让门限在高纬过松、赤道过紧，行为无法解释。
        """
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        cost = np.full((n_tracks, n_dets), _FORBIDDEN, dtype=np.float64)

        for i, track in enumerate(self.tracks):
            pred_yaw = track.unwrapped_yaw
            pred_pitch = track.pitch

            for j, det in enumerate(detections):
                # 检测的 yaw 先提升到该轨迹所在的分支，再算距离
                det_yaw_lifted = lift_yaw(det.yaw_deg, pred_yaw)

                d = great_circle_distance(
                    pred_yaw, pred_pitch, det_yaw_lifted, det.pitch_deg
                )
                if d > self.gate_deg:
                    continue  # 超过门限，保持 _FORBIDDEN

                w = 1.0
                if (
                    self.appearance_weight > 0.0
                    and track.embedding is not None
                    and det.embedding is not None
                ):
                    # 余弦距离，映射到 [0, 2]
                    cos = float(
                        np.dot(track.embedding, det.embedding)
                        / (np.linalg.norm(track.embedding) * np.linalg.norm(det.embedding) + 1e-9)
                    )
                    w += self.appearance_weight * (1.0 - cos)

                cost[i, j] = float(d) * w

        return cost

    def _associate(
        self, detections: list[SphericalDetection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """匈牙利算法求最优匹配，并过滤掉超过门限的配对。

        Returns:
            (matches, unmatched_track_indices, unmatched_detection_indices)
        """
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))

        cost = self._build_cost_matrix(detections)
        rows, cols = linear_sum_assignment(cost)

        matches: list[tuple[int, int]] = []
        for r, c in zip(rows, cols, strict=True):
            # 匈牙利求的是全局最小代价，可能仍包含被标记为禁止的配对 —— 必须显式过滤
            if cost[r, c] < _FORBIDDEN:
                matches.append((int(r), int(c)))

        matched_tracks = {r for r, _ in matches}
        matched_dets = {c for _, c in matches}
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]

        return matches, unmatched_tracks, unmatched_dets

    # ------------------------------------------------------------ 工具

    def reset(self) -> None:
        """清空所有状态，用于处理新视频。"""
        self.tracks.clear()
        self._next_id = 1
        self.frame_index = 0
        self.total_seam_crossings = 0

    @property
    def num_confirmed(self) -> int:
        return sum(1 for t in self.tracks if t.hits >= self.min_hits)

    def __repr__(self) -> str:
        return (
            f"<SphericalTracker tracks={len(self.tracks)} "
            f"confirmed={self.num_confirmed} frame={self.frame_index} "
            f"seam_crossings={self.total_seam_crossings}>"
        )
