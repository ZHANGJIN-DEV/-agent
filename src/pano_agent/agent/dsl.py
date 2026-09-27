"""Edit DSL —— LLM 输出与几何求解之间的类型契约。

==============================================================================
整个架构的立论基础
==============================================================================

**本文件中不存在任何表示相机位置或目标坐标的 float 字段。**
这不是疏漏，是刻意设计。

LLM 能说的只有：
    - 拍**谁**（`subject: track_id`）
    - 什么**景别**（`shot_type: "close_up"`）
    - 什么**运镜**（`camera_motion: "follow"`）
    - 什么**节奏**（`t_start` / `t_end`）
    - 什么**风格**（`style: "energetic"`）

LLM **不能**说"相机在 yaw=37.5°、pitch=12.3°"。那是优化器的职责。

------------------------------------------------------------------------------
为什么这个约束是必要的
------------------------------------------------------------------------------

**LLM 输出的是「约束」，不是「解」。**

    约束（constraint）  可校验、可拒绝、可兜底、可回退
    解（solution）      不可校验 —— 错了也不知道，只能靠肉眼看出来

具体到本任务：

1. LLM **没有度量几何的概念**。30fps × 15s = 450 个连续角度值，
   它无法判断哪个值对应"主体在画面中央"，也无法保证帧间连续。
2. **token 成本**：450 个浮点数既昂贵又必然前后矛盾。
3. **无法验证**：拿到一串角度，我们没有任何自动手段判断它对不对。
   而拿到 `{subject: 3, shot_type: "medium"}`，我们可以立即检查
   "track_id=3 存在吗"、"这个景别对应的 FOV 在合法区间吗"。

于是职责天然分离：

    LLM 层        语义决策（拍谁、什么景别、什么节奏）
    优化层        几何求解（满足约束的连续轨迹）

这个分离由 **pydantic 的类型系统强制执行** —— 不是靠 Prompt 里写
"请不要输出坐标"，而是让 LLM **根本没有字段可以填**。

------------------------------------------------------------------------------
为什么保留 fov_deg
------------------------------------------------------------------------------

`fov_deg` 是 float，看似违反上述原则，实则不然：

    它描述的是**景别的具体取值**（"用 65° 视场拍这个中景"），
    不是**相机的位置**。前者是约束参数，后者是解。

景别本来就有模糊性（"中景"是 50° 还是 70°？），让 LLM 给出具体
FOV 是合理的 —— 它需要语义知识（"特写要更窄"），但不需要几何知识。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ShotType = Literal["extreme_wide", "wide", "medium", "close_up", "extreme_close_up"]
"""景别。对应的 FOV 区间见 `SHOT_TYPE_FOV_RANGE`。"""

CameraMotion = Literal[
    "static",     # 静止
    "follow",     # 跟随主体（保持其在画面同一位置）
    "lead",       # 领先于主体（主体偏后）
    "orbit",      # 环绕
    "push_in",    # 推近
    "pull_out",   # 拉远
    "pan",        # 横摇
]

EditStyle = Literal["energetic", "calm", "cinematic", "documentary"]

# 各景别的 FOV 参考区间（度）。用于校验 LLM 给出的 fov_deg 是否自洽。
SHOT_TYPE_FOV_RANGE: dict[str, tuple[float, float]] = {
    "extreme_wide": (100.0, 120.0),
    "wide": (70.0, 100.0),
    "medium": (45.0, 75.0),
    "close_up": (25.0, 50.0),
    "extreme_close_up": (15.0, 30.0),
}


class ShotSpec(BaseModel):
    """一个镜头的规格。

    这是 LLM 能表达的**最小决策单元**。注意它描述的全是"意图"，
    没有一个是"坐标"。
    """

    model_config = ConfigDict(extra="forbid")
    """extra="forbid"：LLM 若捏造字段（比如自作主张加一个 `yaw_deg`），
    校验会直接失败而不是默默忽略。这是「类型隔离」的一部分。"""

    t_start: float = Field(ge=0, description="起始时间（秒）")
    t_end: float = Field(gt=0, description="结束时间（秒）")

    subject: int | None = Field(
        default=None,
        description="主体 track_id。None 表示无特定主体（如纯风光镜头）。"
        "**注意：这是 ID，不是坐标** —— LLM 永远不能直接指定角度。",
    )

    shot_type: ShotType = "medium"
    camera_motion: CameraMotion = "static"
    fov_deg: float = Field(
        default=60.0,
        ge=10.0,
        le=140.0,
        description="水平视场角。这是**景别的具体取值**，不是相机位置。",
    )

    @model_validator(mode="after")
    def _check_time_range(self) -> ShotSpec:
        if self.t_start >= self.t_end:
            raise ValueError(
                f"镜头起止时间非法: [{self.t_start}, {self.t_end}]（起点必须早于终点）"
            )
        return self

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


class GlobalConstraints(BaseModel):
    """全局构图与运动约束 —— 交给优化器的软/硬约束声明。"""

    model_config = ConfigDict(extra="forbid")

    rule_of_thirds: bool = Field(
        default=False, description="是否启用三分法构图（球面切平面意义下）"
    )
    keep_subject_in_frame: bool = Field(default=True, description="主体必须始终在画面内")
    max_angular_vel_dps: float = Field(
        default=45.0,
        gt=0,
        le=360,
        description="相机角速度上限（度/秒）。限制过快的甩镜。",
    )
    max_fov_rate_dps: float = Field(
        default=20.0, gt=0, description="FOV 变化率上限（度/秒）。限制突兀的变焦。"
    )
    smoothness_weight: float = Field(
        default=1.0, ge=0, description="轨迹平滑项的权重（越大越平稳）"
    )


class EditDSL(BaseModel):
    """LLM 输出的完整剪辑方案。

    这是 `agent` 层与 `optimize` 层之间的接口。下游的 Compiler 会把它
    翻译成约束集，交给求解器算出实际的相机轨迹。
    """

    model_config = ConfigDict(extra="forbid")

    shots: list[ShotSpec] = Field(min_length=1, description="镜头序列，按时间排序")
    style: EditStyle = "cinematic"
    constraints: GlobalConstraints = Field(default_factory=GlobalConstraints)
    audio_sync: bool = Field(default=False, description="是否把剪辑点吸附到音频节拍")

    @field_validator("shots")
    @classmethod
    def _check_shots(cls, v: list[ShotSpec]) -> list[ShotSpec]:
        if not v:
            raise ValueError("至少要有一个镜头")
        # 按时间排序，并检查是否重叠
        ordered = sorted(v, key=lambda s: s.t_start)
        for a, b in zip(ordered, ordered[1:], strict=False):
            if b.t_start < a.t_end - 1e-6:
                raise ValueError(
                    f"镜头时间重叠: [{a.t_start}, {a.t_end}] 与 [{b.t_start}, {b.t_end}]"
                )
        return ordered

    # ------------------------------------------------------------ 校验

    def validate_against_tracks(self, valid_track_ids: set[int]) -> list[str]:
        """检查引用的 track_id 是否都真实存在。

        LLM 可能幻觉出不存在的 id。**这是防御的第一道关卡**：
        返回的错误列表可被用作重试提示，或据此决定降级到纯优化器方案。

        Returns:
            问题描述列表；为空表示通过。
        """
        problems: list[str] = []
        for i, shot in enumerate(self.shots):
            if shot.subject is not None and shot.subject not in valid_track_ids:
                problems.append(
                    f"镜头 {i}（[{shot.t_start:.1f}s, {shot.t_end:.1f}s]）"
                    f"引用了不存在的 track_id={shot.subject}；"
                    f"可用的是 {sorted(valid_track_ids)}"
                )
        return problems

    def validate_fov_consistency(self) -> list[str]:
        """检查 fov_deg 与 shot_type 是否自洽。

        这是**软检查**（作为提示返回给 LLM 参考），不阻断流程 ——
        景别与 FOV 的对应本就有主观空间。
        """
        warnings: list[str] = []
        for i, shot in enumerate(self.shots):
            lo, hi = SHOT_TYPE_FOV_RANGE.get(shot.shot_type, (0.0, 180.0))
            if not (lo <= shot.fov_deg <= hi):
                warnings.append(
                    f"镜头 {i}: shot_type={shot.shot_type} 的常见 FOV 区间是 "
                    f"[{lo:.0f}, {hi:.0f}]°，但 fov_deg={shot.fov_deg:.0f}°"
                )
        return warnings

    @property
    def total_duration_s(self) -> float:
        return max((s.t_end for s in self.shots), default=0.0)

    @property
    def referenced_tracks(self) -> set[int]:
        return {s.subject for s in self.shots if s.subject is not None}

    # ------------------------------------------------------------ 给 LLM

    def to_prompt_dict(self) -> dict[str, Any]:
        """转成可读的字典（用于日志、或回灌给 LLM 做自省）。"""
        return {
            "style": self.style,
            "audio_sync": self.audio_sync,
            "shots": [
                {
                    "t_start": round(s.t_start, 2),
                    "t_end": round(s.t_end, 2),
                    "subject": s.subject,
                    "shot_type": s.shot_type,
                    "camera_motion": s.camera_motion,
                    "fov_deg": round(s.fov_deg, 1),
                }
                for s in self.shots
            ],
            "constraints": self.constraints.model_dump(),
        }

    def __repr__(self) -> str:
        subjects = sorted(self.referenced_tracks)
        return (
            f"<EditDSL {len(self.shots)} 镜头 / {self.total_duration_s:.1f}s "
            f"style={self.style} subjects={subjects}>"
        )


def get_edit_dsl_json_schema() -> dict[str, Any]:
    """导出 JSON Schema，供 LLM 的 structured output / function calling 使用。

    有了它，模型端的约束与代码端的 pydantic 校验就来自**同一个定义**，
    不会出现"提示词说的和代码查的不一致"这类经典问题。
    """
    return EditDSL.model_json_schema()


# 显式声明：本模块**没有**、且**不应有**任何表示坐标的字段。
# 这个常量用于测试断言，防止日后有人"顺手"加一个 yaw_deg 进来。
FORBIDDEN_COORDINATE_FIELDS: frozenset[str] = frozenset(
    {
        "yaw_deg", "pitch_deg", "roll_deg",
        "yaw", "pitch", "roll",
        "center_yaw_deg", "center_pitch_deg",
        "x", "y", "position", "camera_position",
        "trajectory", "waypoints", "keyframes",
    }
)
