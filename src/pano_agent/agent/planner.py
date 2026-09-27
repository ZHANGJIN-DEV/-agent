"""LLM 规划器 —— 把自然语言指令翻译成 Edit DSL。

==============================================================================
为什么不用 LangGraph / AutoGen
==============================================================================

本任务是**单链路**的：读 PSG → 思考 → 输出 DSL。
没有多智能体协商、没有循环图、没有人类介入节点。

引入图编排框架会带来额外抽象层、不可控的 token 开销（框架往往塞入
大量系统提示）、以及与 pydantic 校验的集成成本。

而本任务真正需要的只有三件事：

    1. 把 PSG 的符号层视图塞进 prompt
    2. 调用 LLM 并要求 JSON 输出
    3. 用 pydantic 校验，失败则带错误信息重试

**三十行 while 循环就够了。**

这不是"重复造轮子"，而是"拒绝为不需要的复杂度付费"。
日后若任务真的变成多轮协商（比如需要反问用户澄清意图），
再带着真实约束去评估迁移也不迟。

==============================================================================
重试与降级：让 pipeline 永不中断
==============================================================================

LLM 会犯两类错误：

    1. **格式错误**   输出的 JSON 不符合 schema
    2. **幻觉引用**   引用了不存在的 track_id

处理方式是把 **pydantic 的报错原文**与**可用 track_id 列表**回灌给它，
最多重试 N 次。

若仍然失败 —— **降级到纯优化器方案**（不依赖 LLM，直接用显著性
挑主体 + 固定的景别/运镜规则）。这是工程鲁棒性的要求：
演示时不能因为模型抽风就整个挂掉。

这个降级路径本身也值得写进报告：它同时是一个**消融实验的对照组**
（"有 LLM 规划 vs 纯优化器"），正好是计划里要求的三个关键消融之一。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from .dsl import EditDSL, GlobalConstraints, ShotSpec

if TYPE_CHECKING:  # pragma: no cover
    from ..agent.providers.base import BaseProvider
    from ..sir.schema import PanoramaSceneGraph

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一位 360° 全景视频的剪辑导演。

你会收到一段全景视频的结构化描述（包含若干「轨道」= 画面中的主体），
以及用户的一句剪辑指令。你的任务是把它翻译成一份剪辑方案。

# 你能决定的事
- 拍**谁**：通过 track_id 引用轨道（subject 字段）
- 什么**景别**：extreme_wide / wide / medium / close_up / extreme_close_up
- 什么**运镜**：static / follow / lead / orbit / push_in / pull_out / pan
- 什么**节奏**：每个镜头的起止时间

# 你**不能**做的事（重要）
- **不要输出任何角度、坐标、位置数值**。相机轨迹由几何优化器求解，
  你只需表达意图（拍谁、什么景别），不需要也不应该指定相机朝向。
- 不要引用描述中不存在的 track_id。

# 输出格式
只输出一个 JSON 对象，结构如下（不要有任何额外文字）：
{
  "shots": [
    {"t_start": 0.0, "t_end": 5.0, "subject": 3,
     "shot_type": "wide", "camera_motion": "static", "fov_deg": 85.0}
  ],
  "style": "cinematic",
  "constraints": {"rule_of_thirds": true, "max_angular_vel_dps": 45.0,
                  "keep_subject_in_frame": true, "max_fov_rate_dps": 20.0,
                  "smoothness_weight": 1.0},
  "audio_sync": false
}

# 剪辑常识
- 开头用一个较宽的镜头建立环境，再推进到主体
- 主体运动快时用 follow，静态主体可用 push_in 制造张力
- 相邻镜头的 FOV 不宜突变（观感跳跃）
- 各个 track_id 的 highlight_score 越高越值得拍
"""


@dataclass
class PlannerConfig:
    """规划器配置。"""

    model: str = "deepseek-chat"
    max_retries: int = 3
    temperature: float = 0.3
    """略高于 0 —— 剪辑方案需要一点多样性，但不应大到失控。"""

    max_tokens: int = 2000
    capture_reasoning: bool = True
    """是否请求并保存模型的推理过程（deepseek-flash 会返回 reasoning_content）。

    这对本项目有特殊价值：**可解释的自动运镜** —— 推理链可以直接
    作为技术报告里"Agent 决策可解释"的证据。
    """


@dataclass
class PlanResult:
    """规划结果。"""

    dsl: EditDSL
    attempts: int = 1
    used_fallback: bool = False
    reasoning: str = ""
    """模型的推理过程（若可得）。"""

    errors: list[str] = field(default_factory=list)
    """历次尝试的失败原因，便于事后分析模型的典型错误模式。"""

    @property
    def is_fallback(self) -> bool:
        return self.used_fallback

    def __repr__(self) -> str:
        tag = " [降级]" if self.used_fallback else ""
        return f"<PlanResult {self.dsl!r} 尝试 {self.attempts} 次{tag}>"


class EditPlanner:
    """把自然语言指令翻译成 Edit DSL 的规划器。"""

    def __init__(
        self,
        provider: BaseProvider,
        config: PlannerConfig | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or PlannerConfig()

    # ------------------------------------------------------------ 主入口

    def plan(
        self,
        psg: PanoramaSceneGraph,
        instruction: str,
        *,
        target_duration_s: float | None = None,
    ) -> PlanResult:
        """生成剪辑方案。

        Args:
            psg: 全景场景图（提供符号层视图给 LLM）。
            instruction: 用户的自然语言指令。
            target_duration_s: 目标成片时长。None 则让 LLM 自行决定。

        Returns:
            PlanResult。**永不抛异常** —— 最坏情况下降级到纯优化器方案。
        """
        valid_ids = psg.track_ids
        errors: list[str] = []
        reasoning = ""

        if not valid_ids:
            logger.warning("PSG 中没有轨迹，直接使用降级方案")
            return PlanResult(
                dsl=self._fallback_dsl(psg, target_duration_s),
                used_fallback=True,
                errors=["PSG 中没有任何轨迹"],
            )

        messages = self._build_messages(psg, instruction, target_duration_s)

        for attempt in range(1, self.config.max_retries + 1):
            try:
                result = self.provider.chat(
                    messages,
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    json_mode=True,
                )
            except Exception as exc:  # noqa: BLE001 - 网络/API 故障不应中断流程
                msg = f"第 {attempt} 次调用失败: {type(exc).__name__}: {exc}"
                logger.warning(msg)
                errors.append(msg)
                continue

            if self.config.capture_reasoning:
                reasoning = getattr(result, "reasoning", "") or reasoning

            # ---- 解析与校验 ----
            try:
                payload = json.loads(result.text)
                dsl = EditDSL.model_validate(payload)
            except json.JSONDecodeError as exc:
                msg = f"第 {attempt} 次：输出不是合法 JSON —— {exc}"
                errors.append(msg)
                messages.append(self._assistant_message(result.text))
                messages.append(
                    self._user_message(
                        f"你的输出不是合法 JSON（{exc}）。请只输出一个 JSON 对象，"
                        f"不要包含 markdown 代码块标记或任何解释文字。"
                    )
                )
                continue
            except ValidationError as exc:
                msg = f"第 {attempt} 次：schema 校验失败 —— {exc.error_count()} 处问题"
                errors.append(msg)
                messages.append(self._assistant_message(result.text))
                messages.append(self._user_message(self._format_validation_error(exc)))
                continue

            # ---- 幻觉引用检查 ----
            problems = dsl.validate_against_tracks(valid_ids)
            if problems:
                msg = f"第 {attempt} 次：引用了不存在的轨道"
                errors.append(msg)
                messages.append(self._assistant_message(result.text))
                messages.append(
                    self._user_message(
                        "以下引用无效：\n"
                        + "\n".join(f"- {p}" for p in problems)
                        + "\n请只用上面列出的 track_id 重新生成。"
                    )
                )
                continue

            # ---- FOV 一致性（软检查，只记录不阻断）----
            for w in dsl.validate_fov_consistency():
                logger.info("FOV 提示: %s", w)

            return PlanResult(
                dsl=dsl,
                attempts=attempt,
                used_fallback=False,
                reasoning=reasoning,
                errors=errors,
            )

        # ---- 全部重试失败，降级 ----
        logger.warning("规划失败 %d 次，降级到纯优化器方案", self.config.max_retries)
        return PlanResult(
            dsl=self._fallback_dsl(psg, target_duration_s),
            attempts=self.config.max_retries,
            used_fallback=True,
            reasoning=reasoning,
            errors=errors,
        )

    # ------------------------------------------------------------ 消息构造

    def _build_messages(
        self,
        psg: PanoramaSceneGraph,
        instruction: str,
        target_duration_s: float | None,
    ) -> list:
        from .providers.base import ChatMessage

        # 只把**符号层**交给 LLM —— to_llm_view 保证其中不含坐标
        view = psg.to_llm_view(max_tracks=20)

        parts = [
            "# 视频描述",
            json.dumps(view, ensure_ascii=False, indent=2),
        ]

        if target_duration_s is not None:
            parts.append(f"\n# 目标成片时长\n约 {target_duration_s:.0f} 秒")

        parts.append(f"\n# 剪辑指令\n{instruction}")

        return [
            ChatMessage.system(SYSTEM_PROMPT),
            ChatMessage.user("\n".join(parts)),
        ]

    @staticmethod
    def _assistant_message(text: str):
        from .providers.base import ChatMessage

        return ChatMessage.assistant(text)

    @staticmethod
    def _user_message(text: str):
        from .providers.base import ChatMessage

        return ChatMessage.user(text)

    @staticmethod
    def _format_validation_error(exc: ValidationError) -> str:
        """把 pydantic 的报错整理成 LLM 能理解的重试提示。"""
        lines = ["你的输出不符合 schema，具体问题："]
        for err in exc.errors()[:8]:  # 最多列 8 条，避免 prompt 过长
            loc = ".".join(str(p) for p in err.get("loc", ()))
            lines.append(f"- 字段 `{loc}`: {err.get('msg', '')}")
        lines.append("\n请修正后重新输出完整的 JSON 对象。")
        return "\n".join(lines)

    # ------------------------------------------------------------ 降级方案

    def _fallback_dsl(
        self,
        psg: PanoramaSceneGraph,
        target_duration_s: float | None,
    ) -> EditDSL:
        """不依赖 LLM 的兜底方案。

        策略：按 highlight_score 挑出最重要的若干轨道，给每个分配一个
        "跟随"镜头，开头加一个建立镜头。

        **这同时是消融实验的对照组**：计划中列出的三个关键消融之一就是
        「有 LLM 规划 vs 纯优化器」，本方法提供了后者的输入。
        """
        duration = target_duration_s or min(psg.meta.duration_s, 15.0)

        # 按高光分数排序挑选主体
        ranked = sorted(
            psg.tracks.values(), key=lambda n: -n.stats.highlight_score
        )
        # 过滤掉过短的轨迹
        subjects = [n for n in ranked if n.stats.duration_s >= 0.5][:3]

        shots: list[ShotSpec] = []

        # 开场：建立镜头，不指定主体（若轨道太多时也避免开场就聚焦）
        establish = min(3.0, duration * 0.25)
        if len(subjects) >= 2 and establish > 0.5:
            shots.append(
                ShotSpec(
                    t_start=0.0,
                    t_end=establish,
                    subject=None,
                    shot_type="wide",
                    camera_motion="static",
                    fov_deg=95.0,
                )
            )
            remaining_start = establish
        else:
            remaining_start = 0.0

        # 主体镜头：平分剩余时长
        n = max(len(subjects), 1)
        each = max((duration - remaining_start) / n, 0.5)

        for i, node in enumerate(subjects or [None]):
            t0 = remaining_start + i * each
            t1 = min(t0 + each, duration)
            if t1 <= t0:
                break

            # 运动快的主体用 follow，否则用 push_in 制造推进感
            motion = (
                "follow"
                if node is not None and node.stats.angular_velocity_mean_dps > 20.0
                else "push_in"
            )
            shots.append(
                ShotSpec(
                    t_start=t0,
                    t_end=t1,
                    subject=node.track_id if node is not None else None,
                    shot_type="medium",
                    camera_motion=motion,
                    fov_deg=70.0,
                )
            )

        if not shots:
            shots.append(
                ShotSpec(
                    t_start=0.0,
                    t_end=max(duration, 1.0),
                    subject=None,
                    shot_type="wide",
                    camera_motion="static",
                    fov_deg=90.0,
                )
            )

        return EditDSL(
            shots=shots,
            style="documentary",
            constraints=GlobalConstraints(rule_of_thirds=False),
            audio_sync=False,
        )
