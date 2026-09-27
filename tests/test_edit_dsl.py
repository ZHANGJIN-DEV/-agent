"""Edit DSL 的测试。

核心是守住一条不变量：**schema 里不能存在任何坐标字段**。

这不是风格偏好。一旦 LLM 有了可以填坐标的地方，它就会去填，
而它输出的坐标既无物理意义又不可自动校验 —— 整个
「LLM 输出约束而非解」的立论基础就塌了。

因此这组测试是**架构级的回归防线**，不是普通的功能测试。
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pano_agent.agent.dsl import (
    FORBIDDEN_COORDINATE_FIELDS,
    SHOT_TYPE_FOV_RANGE,
    CameraMotion,
    EditDSL,
    GlobalConstraints,
    ShotSpec,
    get_edit_dsl_json_schema,
)


def make_shot(**kw) -> ShotSpec:
    base = {"t_start": 0.0, "t_end": 5.0, "subject": 1, "shot_type": "medium"}
    base.update(kw)
    return ShotSpec(**base)


def make_dsl(shots=None, **kw) -> EditDSL:
    return EditDSL(shots=shots or [make_shot()], **kw)


# ============================================================== ★ 不变量


class TestNoCoordinateFields:
    """架构级不变量：DSL 中不得出现坐标。"""

    def test_json_schema_contains_no_coordinate_fields(self):
        """★ JSON Schema（送给 LLM 的）里不能有任何坐标字段名。

        这是最强的一条断言：schema 是 LLM 实际看到的契约，
        它里面没有的东西，LLM 就填不出来。
        """
        schema_text = json.dumps(get_edit_dsl_json_schema(), ensure_ascii=False)
        leaked = [f for f in FORBIDDEN_COORDINATE_FIELDS if f'"{f}"' in schema_text]
        assert not leaked, (
            f"EditDSL schema 中出现了坐标字段: {leaked}。"
            f"这会让 LLM 开始输出几何坐标 —— 见 dsl.py 模块文档。"
        )

    def test_pydantic_fields_contain_no_coordinates(self):
        """从模型定义本身再查一遍（以防 JSON Schema 导出遗漏）。"""
        names = set(ShotSpec.model_fields) | set(EditDSL.model_fields)
        names |= set(GlobalConstraints.model_fields)
        leaked = names & FORBIDDEN_COORDINATE_FIELDS
        assert not leaked, f"模型字段中出现坐标: {leaked}"

    def test_extra_coordinate_field_is_rejected(self):
        """★ LLM 若自作主张加一个 yaw_deg，必须报错而非默默忽略。

        `extra="forbid"` 是这条防线能生效的前提 ——
        默认的 `ignore` 会让非法字段静默消失，问题被掩盖。
        """
        with pytest.raises(ValidationError):
            EditDSL(
                shots=[
                    {"t_start": 0.0, "t_end": 5.0, "yaw_deg": 45.0}
                ]
            )

    def test_extra_field_on_shot_rejected(self):
        with pytest.raises(ValidationError):
            make_shot(pitch_deg=12.0)

    def test_extra_field_on_constraints_rejected(self):
        with pytest.raises(ValidationError):
            GlobalConstraints(waypoints=[1, 2, 3])

    def test_subject_is_an_id_not_a_coordinate(self):
        """`subject` 是 track_id（整数），不是角度 —— 类型本身即是约束。"""
        field = ShotSpec.model_fields["subject"]
        assert field.annotation == (int | None)

    def test_fov_is_allowed_but_is_not_a_position(self):
        """fov_deg 是景别的取值（约束参数），不是相机位置（解）。

        它允许存在是有意为之：景别本身有主观空间，
        让 LLM 给出具体 FOV 需要的是语义知识而非几何知识。
        """
        assert "fov_deg" in ShotSpec.model_fields
        assert "fov_deg" not in FORBIDDEN_COORDINATE_FIELDS


# ============================================================== ShotSpec


class TestShotSpec:
    def test_basic_construction(self):
        s = make_shot(t_start=1.0, t_end=4.0, subject=3, shot_type="close_up")
        assert s.duration_s == pytest.approx(3.0)
        assert s.subject == 3

    def test_subject_can_be_none(self):
        """无主体的镜头（纯风光）应合法。"""
        assert make_shot(subject=None).subject is None

    def test_rejects_inverted_time(self):
        with pytest.raises(ValidationError, match="起止时间非法"):
            make_shot(t_start=5.0, t_end=2.0)

    def test_rejects_zero_duration(self):
        with pytest.raises(ValidationError, match="起止时间非法"):
            make_shot(t_start=3.0, t_end=3.0)

    def test_rejects_negative_start(self):
        with pytest.raises(ValidationError):
            make_shot(t_start=-1.0, t_end=5.0)

    def test_rejects_out_of_range_fov(self):
        with pytest.raises(ValidationError):
            make_shot(fov_deg=5.0)
        with pytest.raises(ValidationError):
            make_shot(fov_deg=170.0)

    def test_rejects_unknown_shot_type(self):
        with pytest.raises(ValidationError):
            make_shot(shot_type="super_zoom")

    def test_rejects_unknown_camera_motion(self):
        with pytest.raises(ValidationError):
            make_shot(camera_motion="teleport")

    def test_all_declared_shot_types_valid(self):
        for st in SHOT_TYPE_FOV_RANGE:
            assert make_shot(shot_type=st).shot_type == st

    def test_all_camera_motions_valid(self):
        from typing import get_args

        for motion in get_args(CameraMotion):
            assert make_shot(camera_motion=motion).camera_motion == motion


# ============================================================== EditDSL


class TestEditDSL:
    def test_basic_construction(self):
        dsl = make_dsl([make_shot(t_start=0, t_end=5), make_shot(t_start=5, t_end=10)])
        assert len(dsl.shots) == 2
        assert dsl.total_duration_s == pytest.approx(10.0)

    def test_shots_sorted_by_time(self):
        """乱序输入应被自动排序。"""
        dsl = make_dsl([make_shot(t_start=5, t_end=10), make_shot(t_start=0, t_end=5)])
        assert dsl.shots[0].t_start == pytest.approx(0.0)

    def test_rejects_empty_shots(self):
        with pytest.raises(ValidationError):
            EditDSL(shots=[])

    def test_rejects_overlapping_shots(self):
        with pytest.raises(ValidationError, match="重叠"):
            make_dsl([make_shot(t_start=0, t_end=6), make_shot(t_start=5, t_end=10)])

    def test_adjacent_shots_allowed(self):
        """首尾相接（不重叠）应合法。"""
        dsl = make_dsl([make_shot(t_start=0, t_end=5), make_shot(t_start=5, t_end=10)])
        assert len(dsl.shots) == 2

    def test_referenced_tracks(self):
        dsl = make_dsl(
            [
                make_shot(t_start=0, t_end=3, subject=1),
                make_shot(t_start=3, t_end=6, subject=2),
                make_shot(t_start=6, t_end=9, subject=1),
            ]
        )
        assert dsl.referenced_tracks == {1, 2}

    def test_referenced_tracks_excludes_none(self):
        dsl = make_dsl([make_shot(subject=None)])
        assert dsl.referenced_tracks == set()

    def test_default_style_and_constraints(self):
        dsl = make_dsl()
        assert dsl.style == "cinematic"
        assert dsl.constraints.keep_subject_in_frame is True

    def test_to_prompt_dict_serializable(self):
        dsl = make_dsl([make_shot(subject=2)])
        d = dsl.to_prompt_dict()
        json.dumps(d)  # 不应抛异常
        assert d["shots"][0]["subject"] == 2

    def test_to_prompt_dict_has_no_coordinates(self):
        dsl = make_dsl()
        text = json.dumps(dsl.to_prompt_dict())
        leaked = [f for f in FORBIDDEN_COORDINATE_FIELDS if f'"{f}"' in text]
        assert not leaked

    def test_repr_is_informative(self):
        dsl = make_dsl([make_shot(subject=7)])
        assert "EditDSL" in repr(dsl) and "7" in repr(dsl)


# ============================================================== 校验


class TestTrackIdValidation:
    def test_detects_hallucinated_ids(self):
        """★ LLM 幻觉出的 track_id 必须能被检出。"""
        dsl = make_dsl([make_shot(subject=99)])
        problems = dsl.validate_against_tracks({1, 2, 3})
        assert len(problems) == 1
        assert "99" in problems[0]

    def test_passes_for_valid_ids(self):
        dsl = make_dsl([make_shot(subject=2)])
        assert dsl.validate_against_tracks({1, 2, 3}) == []

    def test_ignores_none_subject(self):
        dsl = make_dsl([make_shot(subject=None)])
        assert dsl.validate_against_tracks({1}) == []

    def test_reports_all_problems(self):
        dsl = make_dsl(
            [
                make_shot(t_start=0, t_end=3, subject=98),
                make_shot(t_start=3, t_end=6, subject=99),
            ]
        )
        assert len(dsl.validate_against_tracks({1, 2})) == 2

    def test_error_message_lists_available_ids(self):
        """错误信息要能直接回灌给 LLM 作为重试提示。"""
        problems = make_dsl([make_shot(subject=99)]).validate_against_tracks({1, 5, 9})
        assert "1" in problems[0] and "5" in problems[0] and "9" in problems[0]


class TestFovConsistency:
    def test_flags_mismatched_fov(self):
        """特写却给了 110° 的广角 FOV —— 应给出提示（软检查）。"""
        dsl = make_dsl([make_shot(shot_type="close_up", fov_deg=110.0)])
        warnings = dsl.validate_fov_consistency()
        assert len(warnings) == 1
        assert "close_up" in warnings[0]

    def test_passes_for_consistent_fov(self):
        dsl = make_dsl([make_shot(shot_type="medium", fov_deg=60.0)])
        assert dsl.validate_fov_consistency() == []

    def test_is_a_warning_not_an_error(self):
        """FOV 与景别不一致不应阻断流程 —— 景别本就有主观空间。"""
        dsl = make_dsl([make_shot(shot_type="close_up", fov_deg=110.0)])
        assert isinstance(dsl.validate_fov_consistency(), list)  # 构造未失败


class TestGlobalConstraints:
    def test_defaults_are_sane(self):
        c = GlobalConstraints()
        assert c.max_angular_vel_dps == pytest.approx(45.0)
        assert c.rule_of_thirds is False

    def test_rejects_absurd_angular_velocity(self):
        with pytest.raises(ValidationError):
            GlobalConstraints(max_angular_vel_dps=0.0)
        with pytest.raises(ValidationError):
            GlobalConstraints(max_angular_vel_dps=500.0)

    def test_rejects_negative_smoothness_weight(self):
        with pytest.raises(ValidationError):
            GlobalConstraints(smoothness_weight=-1.0)


class TestJsonSchemaExport:
    def test_schema_is_valid_json_schema(self):
        schema = get_edit_dsl_json_schema()
        assert schema.get("type") == "object"
        assert "shots" in schema.get("properties", {})

    def test_schema_declares_event_kinds(self):
        """枚举值应体现在 schema 中，让 LLM 无法发明新的景别。"""
        text = json.dumps(get_edit_dsl_json_schema())
        assert "close_up" in text
        assert "orbit" in text
