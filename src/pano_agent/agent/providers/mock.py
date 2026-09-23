"""离线 Mock 提供商 —— 不消耗额度、不发网络请求。

存在的意义（而非仅是测试补丁）:
1. **全流程离线可跑**：没有 API key 时，pipeline、Web Demo、CI 依然能端到端运行。
2. **单元测试确定性**：真实 LLM 的输出不可复现，无法断言。Mock 让
   `test_e2e_mock.py` 能稳定验证「感知 → PSG → 规划 → 渲染」的接线是否正确。
3. **成本控制**：调试编排逻辑时不该烧 token。

返回内容是**结构合法**的假数据（合法 JSON、合法 track_id），
这样下游的 pydantic 校验能真实通过，避免"mock 通过了但真实调用挂"。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import BaseProvider, ChatMessage, ChatResult, ChatUsage


class MockProvider(BaseProvider):
    """确定性假响应提供商。"""

    name = "mock"
    supports_vision = True

    def __init__(self, canned_response: str | None = None) -> None:
        self._canned = canned_response
        self.call_count = 0

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> ChatResult:
        self.call_count += 1

        if self._canned is not None:
            text = self._canned
        elif json_mode:
            text = json.dumps(self._synthesize_dsl(messages), ensure_ascii=False)
        else:
            text = self._synthesize_text(messages)

        return ChatResult(
            text=text,
            model=model or "mock-1",
            usage=ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            cached=False,
            provider=self.name,
        )

    # ------------------------------------------------------------ 假数据生成

    def _synthesize_text(self, messages: list[ChatMessage]) -> str:
        """VLM 场景描述请求：返回类人描述，且尽量引用提示中出现的轨道号。"""
        prompt = " ".join(m.text_only() for m in messages)
        ids = re.findall(r"\btrack[_\s]*(\d+)\b", prompt, flags=re.IGNORECASE)
        if ids:
            return f"轨道 {ids[0]}：画面中的一个主体，正在移动，位于画面中央区域。"
        return "画面中有一个主体，正在移动。"

    def _synthesize_dsl(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """规划请求：返回一个结构合法的 EditDSL。

        刻意引用提示中出现过的 track_id，模拟"模型正确理解上下文"的行为；
        找不到就返回 subject=None（纯全景镜头），仍然合法。
        """
        prompt = " ".join(m.text_only() for m in messages)
        ids = re.findall(r'"track_id"\s*:\s*(\d+)', prompt)
        subject = int(ids[0]) if ids else None

        return {
            "shots": [
                {
                    "t_start": 0.0,
                    "t_end": 5.0,
                    "subject": None,
                    "shot_type": "wide",
                    "camera_motion": "static",
                    "fov_deg": 90.0,
                },
                {
                    "t_start": 5.0,
                    "t_end": 15.0,
                    "subject": subject,
                    "shot_type": "medium",
                    "camera_motion": "follow",
                    "fov_deg": 70.0,
                },
            ],
            "style": "cinematic",
            "constraints": {
                "rule_of_thirds": True,
                "max_angular_vel_dps": 45.0,
                "keep_subject_in_frame": True,
            },
            "audio_sync": False,
        }
