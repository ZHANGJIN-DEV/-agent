"""LLM 规划器的测试。

用**脚本化的假 provider** 构造各种失败场景：
真实 LLM 不会按我们需要的方式出错，而重试与降级逻辑恰恰只在
那些边界上才被执行到。

这组测试守住的核心性质：**plan() 永不抛异常**。
演示时不能因为模型抽风就整个 pipeline 挂掉。
"""

from __future__ import annotations

import json

import pytest

from pano_agent.agent.planner import EditPlanner, PlannerConfig
from pano_agent.agent.providers.base import ChatResult, ChatUsage
from pano_agent.agent.providers.mock import MockProvider
from pano_agent.sir import (
    EventNode,
    PanoramaSceneGraph,
    ShotNode,
    TrackNode,
    TrackStats,
    VideoMeta,
)


class ScriptedProvider(MockProvider):
    """按脚本依次返回预设响应的假 provider。

    用于精确控制"第几次调用返回什么"，从而触发重试与降级路径。
    """

    def __init__(self, responses: list[str], *, raise_on_call: set[int] | None = None):
        super().__init__()
        self.responses = list(responses)
        self.raise_on_call = raise_on_call or set()

    def chat(self, *args, **kwargs) -> ChatResult:
        self.call_count += 1
        if self.call_count in self.raise_on_call:
            raise RuntimeError(f"模拟第 {self.call_count} 次调用故障")
        idx = min(self.call_count - 1, len(self.responses) - 1)
        return ChatResult(
            text=self.responses[idx],
            model="scripted",
            usage=ChatUsage(),
            provider="scripted",
        )


VALID_DSL_JSON = json.dumps({
    "shots": [
        {"t_start": 0.0, "t_end": 5.0, "subject": 1, "shot_type": "wide",
         "camera_motion": "static", "fov_deg": 85.0},
        {"t_start": 5.0, "t_end": 12.0, "subject": 2, "shot_type": "medium",
         "camera_motion": "follow", "fov_deg": 65.0},
    ],
    "style": "cinematic",
    "constraints": {"rule_of_thirds": True, "max_angular_vel_dps": 45.0,
                    "keep_subject_in_frame": True, "max_fov_rate_dps": 20.0,
                    "smoothness_weight": 1.0},
    "audio_sync": False,
})


def make_psg(*, n_tracks: int = 3) -> PanoramaSceneGraph:
    tracks = {}
    for i in range(1, n_tracks + 1):
        tracks[i] = TrackNode(
            track_id=i,
            category="person" if i % 2 else "car",
            caption=f"轨道 {i}",
            stats=TrackStats(
                frame_range=(0, 300),
                duration_s=8.0 - i * 0.5,
                angular_size_mean_deg=12.0,
                angular_velocity_mean_dps=30.0 if i == 1 else 5.0,
                angular_velocity_max_dps=60.0,
                latitude_span_deg=(-10.0, 10.0),
                saliency_mean=0.6,
                highlight_score=0.8 - i * 0.1,
            ),
        )
    return PanoramaSceneGraph(
        meta=VideoMeta(fps=30.0, duration_s=20.0, erp_width=2048, erp_height=1024),
        tracks=tracks,
        shots=[ShotNode(shot_id=0, t_start=0.0, t_end=20.0)],
        events=[
            EventNode(event_id=0, t_start=1.0, t_end=4.0,
                      kind="action_peak", involved_tracks=[1], score=0.9)
        ],
    )


def make_planner(responses: list[str], **cfg) -> tuple[EditPlanner, ScriptedProvider]:
    provider = ScriptedProvider(responses)
    config = PlannerConfig(max_retries=3, **cfg)
    return EditPlanner(provider, config), provider


# ============================================================== 正常路径


class TestSuccessfulPlanning:
    def test_valid_response_produces_dsl(self):
        planner, provider = make_planner([VALID_DSL_JSON])
        result = planner.plan(make_psg(), "剪一个 12 秒的片段")

        assert not result.used_fallback
        assert result.attempts == 1
        assert len(result.dsl.shots) == 2
        assert result.dsl.referenced_tracks == {1, 2}
        assert provider.call_count == 1

    def test_view_sent_to_llm_has_no_coordinates(self):
        """★ 送给 LLM 的内容里不能有坐标 —— 否则它会开始填坐标。"""
        captured: list[str] = []

        class CapturingProvider(ScriptedProvider):
            def chat(self, messages, **kwargs):
                for m in messages:
                    captured.append(m.text_only())
                return super().chat(messages, **kwargs)

        provider = CapturingProvider([VALID_DSL_JSON])
        EditPlanner(provider, PlannerConfig()).plan(make_psg(), "任意指令")

        prompt_text = "\n".join(captured)
        for field in ("yaw_deg", "pitch_deg", "numeric_ref", ".npz"):
            assert field not in prompt_text, f"prompt 中泄漏了 {field}"

    def test_system_prompt_forbids_coordinates(self):
        """系统提示必须显式禁止 LLM 输出坐标。"""
        from pano_agent.agent.planner import SYSTEM_PROMPT

        assert "不要输出任何角度" in SYSTEM_PROMPT or "不要输出" in SYSTEM_PROMPT
        assert "track_id" in SYSTEM_PROMPT


# ============================================================== 重试路径


class TestRetryOnMalformedOutput:
    def test_retries_on_invalid_json(self):
        planner, provider = make_planner(["这不是 JSON", VALID_DSL_JSON])
        result = planner.plan(make_psg(), "指令")

        assert result.attempts == 2
        assert not result.used_fallback
        assert provider.call_count == 2

    def test_retries_on_markdown_wrapped_json(self):
        """LLM 常把 JSON 包在 ```json ... ``` 里。"""
        wrapped = f"```json\n{VALID_DSL_JSON}\n```"
        planner, _ = make_planner([wrapped, VALID_DSL_JSON])
        result = planner.plan(make_psg(), "指令")
        assert result.attempts == 2

    def test_retry_prompt_includes_error(self):
        """重试时应把错误信息回灌给模型，而不是简单重发。"""
        seen: list[str] = []

        class CapturingProvider(ScriptedProvider):
            def chat(self, messages, **kwargs):
                seen.append(messages[-1].text_only())
                return super().chat(messages, **kwargs)

        provider = CapturingProvider(["坏 JSON", VALID_DSL_JSON])
        EditPlanner(provider, PlannerConfig(max_retries=3)).plan(make_psg(), "指令")

        # 第二次调用的最后一条消息应当指出问题
        assert "JSON" in seen[1]


class TestRetryOnSchemaViolation:
    def test_retries_on_missing_required_field(self):
        bad = json.dumps({"style": "cinematic"})  # 缺 shots
        planner, _ = make_planner([bad, VALID_DSL_JSON])
        result = planner.plan(make_psg(), "指令")
        assert result.attempts == 2
        assert not result.used_fallback

    def test_retries_on_inverted_time_range(self):
        bad = json.dumps({
            "shots": [{"t_start": 10.0, "t_end": 2.0, "shot_type": "wide"}]
        })
        planner, _ = make_planner([bad, VALID_DSL_JSON])
        assert planner.plan(make_psg(), "指令").attempts == 2

    def test_retries_on_coordinate_field_injection(self):
        """★ LLM 若试图输出坐标字段，必须被拒绝并重试。

        这是「类型隔离」在运行时的体现：schema 里没有的字段，
        模型塞进来也会被 `extra="forbid"` 挡住。
        """
        bad = json.dumps({
            "shots": [{"t_start": 0.0, "t_end": 5.0, "yaw_deg": 45.0,
                       "shot_type": "wide"}]
        })
        planner, _ = make_planner([bad, VALID_DSL_JSON])
        result = planner.plan(make_psg(), "指令")

        assert result.attempts == 2, "含坐标字段的输出应被拒绝并重试"
        assert not result.used_fallback

    def test_retries_on_unknown_shot_type(self):
        bad = json.dumps({
            "shots": [{"t_start": 0.0, "t_end": 5.0, "shot_type": "super_zoom"}]
        })
        planner, _ = make_planner([bad, VALID_DSL_JSON])
        assert planner.plan(make_psg(), "指令").attempts == 2


class TestRetryOnHallucinatedTrackId:
    def test_retries_when_referencing_missing_track(self):
        """★ LLM 引用了不存在的 track_id 时必须重试。"""
        hallucinated = json.dumps({
            "shots": [{"t_start": 0.0, "t_end": 5.0, "subject": 999,
                       "shot_type": "wide", "fov_deg": 85.0}]
        })
        planner, _ = make_planner([hallucinated, VALID_DSL_JSON])
        result = planner.plan(make_psg(), "指令")

        assert result.attempts == 2
        assert not result.used_fallback

    def test_retry_prompt_lists_valid_ids(self):
        """重试提示中应给出可用 track_id，帮模型自我纠正。"""
        seen: list[str] = []

        class CapturingProvider(ScriptedProvider):
            def chat(self, messages, **kwargs):
                seen.append(messages[-1].text_only())
                return super().chat(messages, **kwargs)

        hallucinated = json.dumps({
            "shots": [{"t_start": 0.0, "t_end": 5.0, "subject": 999,
                       "shot_type": "wide", "fov_deg": 85.0}]
        })
        provider = CapturingProvider([hallucinated, VALID_DSL_JSON])
        EditPlanner(provider, PlannerConfig(max_retries=3)).plan(make_psg(), "指令")

        assert "999" in seen[1]
        assert "1" in seen[1] and "2" in seen[1], "重试提示应列出可用 id"


# ============================================================== 降级


class TestFallback:
    def test_falls_back_after_exhausting_retries(self):
        planner, provider = make_planner(["坏", "还是坏", "依然坏"])
        result = planner.plan(make_psg(), "指令")

        assert result.used_fallback
        assert result.attempts == 3
        assert provider.call_count == 3
        assert len(result.dsl.shots) >= 1, "降级方案也必须产出可用的 DSL"

    def test_fallback_on_empty_psg(self):
        """PSG 里没有轨迹时不应调用 LLM，直接降级。"""
        empty = PanoramaSceneGraph(
            meta=VideoMeta(fps=30.0, duration_s=10.0, erp_width=2048, erp_height=1024),
            tracks={},
        )
        planner, provider = make_planner([VALID_DSL_JSON])
        result = planner.plan(empty, "指令")

        assert result.used_fallback
        assert provider.call_count == 0, "没有轨道时不该浪费一次 API 调用"

    def test_fallback_dsl_is_valid(self):
        """★ 降级方案产出的 DSL 必须能通过完整校验。

        降级路径是最后的安全网，它自己出错就真的没救了。
        """
        planner, _ = make_planner(["坏"] * 3)
        result = planner.plan(make_psg(), "指令", target_duration_s=12.0)

        dsl = result.dsl
        assert len(dsl.shots) >= 1
        assert dsl.total_duration_s <= 12.0 + 1e-6
        # 引用的 track_id 必须真实存在
        assert dsl.validate_against_tracks(make_psg().track_ids) == []

    def test_fallback_prefers_high_highlight_tracks(self):
        """降级方案应按 highlight_score 挑主体。"""
        planner, _ = make_planner(["坏"] * 3)
        psg = make_psg()
        result = planner.plan(psg, "指令")

        # 轨道 1 的 highlight_score 最高
        assert 1 in result.dsl.referenced_tracks

    def test_fallback_adds_establishing_shot(self):
        """多个主体时，降级方案应加一个开场建立镜头。"""
        planner, _ = make_planner(["坏"] * 3)
        result = planner.plan(make_psg(n_tracks=3), "指令", target_duration_s=15.0)

        assert result.dsl.shots[0].subject is None, "开场应是无主体的建立镜头"
        assert result.dsl.shots[0].shot_type == "wide"

    def test_fallback_respects_target_duration(self):
        for target in (5.0, 10.0, 30.0):
            planner, _ = make_planner(["坏"] * 3)
            result = planner.plan(make_psg(), "指令", target_duration_s=target)
            assert result.dsl.total_duration_s <= target + 1e-6


class TestNetworkFailure:
    def test_survives_api_exception(self):
        """★ API 抛异常时也必须降级而非崩溃。"""
        provider = ScriptedProvider([VALID_DSL_JSON], raise_on_call={1, 2, 3})
        planner = EditPlanner(provider, PlannerConfig(max_retries=3))
        result = planner.plan(make_psg(), "指令")

        assert result.used_fallback
        assert len(result.errors) == 3

    def test_recovers_if_api_works_on_retry(self):
        provider = ScriptedProvider([VALID_DSL_JSON, VALID_DSL_JSON],
                                    raise_on_call={1})
        planner = EditPlanner(provider, PlannerConfig(max_retries=3))
        result = planner.plan(make_psg(), "指令")

        assert not result.used_fallback
        assert result.attempts == 2


# ============================================================== 结果对象


class TestPlanResult:
    def test_records_error_history(self):
        planner, _ = make_planner(["坏1", "坏2", "坏3"])
        result = planner.plan(make_psg(), "指令")

        assert len(result.errors) == 3, "应记录每次失败的原因，便于分析错误模式"

    def test_repr_marks_fallback(self):
        planner, _ = make_planner(["坏"] * 3)
        assert "降级" in repr(planner.plan(make_psg(), "指令"))

    def test_never_raises(self):
        """★ 无论 provider 怎么抽风，plan() 都不应抛异常。"""
        for responses in ([], ["坏"], ["{", "}", "[]"]):
            provider = ScriptedProvider(responses or ["坏"])
            planner = EditPlanner(provider, PlannerConfig(max_retries=2))
            planner.plan(make_psg(), "指令")  # 不应抛

    def test_captures_reasoning_when_available(self):
        """若 provider 返回推理过程，应被保存下来（可解释性证据）。"""
        class ReasoningProvider(ScriptedProvider):
            def chat(self, messages, **kwargs):
                r = super().chat(messages, **kwargs)
                object.__setattr__(r, "reasoning", "我选择轨道 1 因为它运动最快")
                return r

        provider = ReasoningProvider([VALID_DSL_JSON])
        result = EditPlanner(provider, PlannerConfig()).plan(make_psg(), "指令")
        assert "轨道 1" in result.reasoning
