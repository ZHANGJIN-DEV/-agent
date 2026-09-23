"""轻量常量速度 Kalman 滤波器。

==============================================================================
为什么不用 filterpy
==============================================================================

本项目的状态维度很低（每维一个位置 + 一个速度），自实现仅数十行，
换来的是**零额外依赖**与**完全透明的行为** —— 面试时可以逐行讲清每个
矩阵的物理含义，而通用库反而把这一层掩盖了。

==============================================================================
模型（各维独立同构）
==============================================================================

    状态  x = [位置, 速度]ᵀ
    转移  F = [[1, dt], [0, 1]]      —— 常量速度假设
    观测  H = [1, 0]                 —— 只观测位置

过程噪声采用**连续白噪声加速度模型（CWNA）**，它对「目标机动」的刻画
比简单的对角 Q 更符合物理直觉：

    Q = σ_a² · [[dt⁴/4, dt³/2],
                [dt³/2, dt²  ]]

其中 σ_a 是加速度标准差。对球面跟踪，其单位是「度/秒²」。

==============================================================================
⚠ 与球面跟踪配合时的关键约束
==============================================================================

传入本滤波器的 **yaw 必须已经是连续角度（unwrapped）**。

若把 [-180, 180) 的 yaw 直接送入，跨越 ±180° 时滤波器会观测到 358° 的
瞬时转动，由此产生巨大的速度估计，并需要很长时间才能"过冲-回摆"回来。

本模块**刻意不做任何角度归一化** —— 那是 `track_spherical` 的职责。
把周期量的处理集中在一处，比在每个使用者处分别处理更不容易出错。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ConstantVelocityKalman:
    """N 维常量速度 Kalman 滤波器。

    Args:
        ndim: 状态的位置维度（球面跟踪中通常为 2：yaw + pitch）。
        dt: 帧间隔（秒）。
        accel_std: 加速度标准差，单位与位置一致（球面跟踪中为 度/秒²）。
            越大表示越相信目标会机动，滤波结果越贴近观测、越不平滑。
        meas_std: 观测噪声标准差（度）。
        initial_pos_std: 初始位置不确定度。
        initial_vel_std: 初始速度不确定度。
    """

    def __init__(
        self,
        ndim: int = 2,
        dt: float = 1.0 / 30.0,
        accel_std: float = 200.0,
        meas_std: float = 1.5,
        initial_pos_std: float = 10.0,
        initial_vel_std: float = 100.0,
    ) -> None:
        if ndim < 1:
            raise ValueError(f"ndim 必须为正，实际为 {ndim}")
        if dt <= 0:
            raise ValueError(f"dt 必须为正，实际为 {dt}")

        self.ndim = int(ndim)
        self.dt = float(dt)
        self.accel_std = float(accel_std)
        self.meas_std = float(meas_std)
        self.initial_pos_std = float(initial_pos_std)
        self.initial_vel_std = float(initial_vel_std)

        n = 2 * self.ndim
        self.x = np.zeros(n, dtype=np.float64)
        self.P = np.eye(n, dtype=np.float64)

        self._build_static_matrices()
        self.reset()

    # ------------------------------------------------------------ 矩阵

    def _build_static_matrices(self) -> None:
        nd, dt = self.ndim, self.dt
        eye = np.eye(nd)
        zero = np.zeros((nd, nd))

        self.F = np.block([[eye, dt * eye], [zero, eye]])
        self.H = np.block([eye, zero])

        # 连续白噪声加速度模型：对每一维构造 2×2 块，再按维展开
        q = self.accel_std**2
        block = np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=np.float64
        ) * q
        # np.kron 会把 2×2 块与 nd×nd 单位阵做张量积，正好得到
        # 各维独立、块间不耦合的 Q
        self.Q = np.kron(block, eye)

        self.R = np.eye(nd, dtype=np.float64) * self.meas_std**2
        self._I = np.eye(2 * nd, dtype=np.float64)

    # ------------------------------------------------------------ 状态

    def reset(
        self,
        position: ArrayLike | None = None,
        velocity: ArrayLike | None = None,
    ) -> None:
        """重置状态。不传参数则回到零位置、零速度。"""
        nd = self.ndim
        self.x = np.zeros(2 * nd, dtype=np.float64)

        if position is not None:
            pos = np.asarray(position, dtype=np.float64).ravel()
            if pos.size != nd:
                raise ValueError(f"position 需要 {nd} 维，实际 {pos.size}")
            self.x[:nd] = pos

        if velocity is not None:
            vel = np.asarray(velocity, dtype=np.float64).ravel()
            if vel.size != nd:
                raise ValueError(f"velocity 需要 {nd} 维，实际 {vel.size}")
            self.x[nd:] = vel

        self.P = np.diag(
            np.concatenate(
                [
                    np.full(nd, self.initial_pos_std**2),
                    np.full(nd, self.initial_vel_std**2),
                ]
            )
        )

    @property
    def position(self) -> NDArray[np.float64]:
        return self.x[: self.ndim].copy()

    @property
    def velocity(self) -> NDArray[np.float64]:
        return self.x[self.ndim :].copy()

    @property
    def position_std(self) -> NDArray[np.float64]:
        """位置分量的标准差（取协方差对角块的开方）。"""
        return np.sqrt(np.maximum(np.diag(self.P)[: self.ndim], 0.0))

    # ------------------------------------------------------------ 滤波

    def predict(self) -> NDArray[np.float64]:
        """按常量速度模型外推一步。"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.position

    def update(self, z: ArrayLike) -> NDArray[np.float64]:
        """用观测 z 校正状态。

        协方差更新采用 **Joseph 形式**：

            P = (I - KH) P (I - KH)ᵀ + K R Kᵀ

        而非教科书上的 `(I - KH) P`。二者在数学上等价，但后者在数值上
        不保证对称正定，长期迭代后可能出现负方差导致滤波发散 ——
        这对需要连续处理上万帧的视频跟踪来说是必须避免的。
        """
        z = np.asarray(z, dtype=np.float64).ravel()
        if z.size != self.ndim:
            raise ValueError(f"观测需要 {self.ndim} 维，实际 {z.size}")

        y = z - self.H @ self.x                       # 新息
        S = self.H @ self.P @ self.H.T + self.R       # 新息协方差
        K = self.P @ self.H.T @ np.linalg.inv(S)      # 增益

        self.x = self.x + K @ y

        I_KH = self._I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.position

    # ------------------------------------------------------------ 关联辅助

    def mahalanobis_d2(self, z: ArrayLike) -> float:
        """观测相对预测分布的平方马氏距离。

        用于**尺度感知**的关联门限：位置不确定度大时允许更远的匹配，
        不确定度小时收紧。这比固定阈值更合理，尤其在目标长时间
        未被观测、协方差已经膨胀的情况下。
        """
        z = np.asarray(z, dtype=np.float64).ravel()
        if z.size != self.ndim:
            raise ValueError(f"观测需要 {self.ndim} 维，实际 {z.size}")

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(y @ np.linalg.solve(S, y))

    def copy(self) -> ConstantVelocityKalman:
        """深拷贝 —— 用于"试探性更新"（如关联前先克隆做假设检验）。"""
        other = ConstantVelocityKalman.__new__(type(self))
        other.__dict__.update(
            {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        )
        return other

    def __repr__(self) -> str:
        pos = np.array2string(self.position, precision=2, suppress_small=True)
        vel = np.array2string(self.velocity, precision=2, suppress_small=True)
        return f"<CVKalman ndim={self.ndim} dt={self.dt:.4f}s pos={pos} vel={vel}>"
