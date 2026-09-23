"""LLM / VLM 提供商的统一接口。

三家国内平台（DeepSeek / 通义 DashScope / 智谱 GLM）均提供 OpenAI 兼容的
`/v1/chat/completions` 接口，因此共用一个客户端实现，只改 base_url 与 model 名。

设计要点:
- 用 dataclass 而非 pydantic 表示消息：这一层要与外部 API 边界对齐，
  不需要校验能力，保持轻量。
- MockProvider 是一等公民而非测试补丁：它让全流程在没有密钥时也能跑通，
  是 CI 与单元测试的默认后端。
"""

from __future__ import annotations

import base64
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


# ---------------------------------------------------------------- 消息内容


@dataclass
class TextPart:
    text: str

    def to_openai(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass
class ImagePart:
    """一张待理解的图像，以本地路径或已有 data URI 表示。"""

    path: Path | str | None = None
    data_uri: str | None = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.data_uri is None):
            raise ValueError("ImagePart 必须且只能提供 path 或 data_uri 之一")

    def resolve_data_uri(self) -> str:
        if self.data_uri is not None:
            return self.data_uri
        path = Path(self.path)  # type: ignore[arg-type]
        if not path.is_file():
            raise FileNotFoundError(f"图像不存在: {path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def to_openai(self) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": self.resolve_data_uri()}}


ContentPart = TextPart | ImagePart


@dataclass
class ChatMessage:
    role: Role
    content: str | list[ContentPart]

    @classmethod
    def system(cls, text: str) -> ChatMessage:
        return cls("system", text)

    @classmethod
    def user(cls, content: str | list[ContentPart]) -> ChatMessage:
        return cls("user", content)

    @classmethod
    def assistant(cls, text: str) -> ChatMessage:
        return cls("assistant", text)

    def has_image(self) -> bool:
        return isinstance(self.content, list) and any(
            isinstance(p, ImagePart) for p in self.content
        )

    def to_openai(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {"role": self.role, "content": [p.to_openai() for p in self.content]}

    def text_only(self) -> str:
        """用于日志与缓存 key —— 不含图像字节，避免缓存键过大。"""
        if isinstance(self.content, str):
            return self.content
        return " ".join(p.text for p in self.content if isinstance(p, TextPart))


# ---------------------------------------------------------------- 返回结构


@dataclass
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_openai(cls, raw: dict[str, Any] | None) -> ChatUsage:
        raw = raw or {}
        return cls(
            prompt_tokens=int(raw.get("prompt_tokens", 0)),
            completion_tokens=int(raw.get("completion_tokens", 0)),
            total_tokens=int(raw.get("total_tokens", 0)),
        )


@dataclass
class ChatResult:
    text: str
    model: str
    usage: ChatUsage = field(default_factory=ChatUsage)
    cached: bool = False
    provider: str = ""


class ProviderError(RuntimeError):
    """提供商调用失败（网络、鉴权、限流、响应格式异常）。"""


# ---------------------------------------------------------------- 抽象基类


class BaseProvider(ABC):
    """LLM / VLM 提供商的公共接口。"""

    name: str = "base"
    supports_vision: bool = False

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> ChatResult:
        """发送一轮对话并返回结果。

        Args:
            json_mode: 要求模型返回严格 JSON（三家平台均支持 response_format）。
                这是 Agent 层可靠性的关键 —— 规划器的输出必须是可解析的 JSON。
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} vision={self.supports_vision}>"
