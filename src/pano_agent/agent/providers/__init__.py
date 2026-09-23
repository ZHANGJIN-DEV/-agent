"""提供商工厂：按名字构造 LLM / VLM 客户端。

用法:
    llm = create_provider("llm")            # 按 .env 配置的默认提供商
    vlm = create_provider("vlm", "qwen")    # 显式指定
    mock = create_provider("llm", "mock")   # 离线
"""

from __future__ import annotations

from typing import Literal

from ...config import ProviderName, Settings, get_settings
from .base import (
    BaseProvider,
    ChatMessage,
    ChatResult,
    ChatUsage,
    ImagePart,
    ProviderError,
    TextPart,
)
from .mock import MockProvider
from .openai_compat import OpenAICompatProvider

ProviderKind = Literal["llm", "vlm"]

__all__ = [
    "BaseProvider",
    "ChatMessage",
    "ChatResult",
    "ChatUsage",
    "ImagePart",
    "MockProvider",
    "OpenAICompatProvider",
    "ProviderError",
    "TextPart",
    "create_provider",
]


def _resolve_spec(kind: ProviderKind, name: ProviderName, settings: Settings) -> dict:
    """把 (用途, 提供商名) 解析成 OpenAICompatProvider 的构造参数。"""
    spec: dict[str, dict] = {
        "deepseek": {
            "base_url": settings.deepseek_base_url,
            "model": settings.deepseek_model,
            "api_key": settings.deepseek_api_key.get_secret_value(),
            "supports_vision": False,  # DeepSeek 无视觉模型
        },
        "qwen": {
            "base_url": settings.dashscope_base_url,
            "model": settings.qwen_vl_model if kind == "vlm" else settings.qwen_llm_model,
            "api_key": settings.dashscope_api_key.get_secret_value(),
            "supports_vision": True,
        },
        "zhipu": {
            "base_url": settings.zhipu_base_url,
            "model": settings.glm_vl_model if kind == "vlm" else settings.glm_llm_model,
            "api_key": settings.zhipu_api_key.get_secret_value(),
            "supports_vision": True,
        },
        "mock": {},  # 单独处理
    }

    if name not in spec:
        raise ValueError(f"未知提供商: {name!r}（可选: {', '.join(spec)}）")

    resolved = spec[name]
    if name != "mock" and kind == "vlm" and not resolved["supports_vision"]:
        raise ProviderError(
            f"提供商 {name!r} 不支持视觉输入，无法用作 VLM。"
            f"请在 .env 中设置 PANODIRECTOR_VLM_PROVIDER=qwen 或 zhipu。"
        )
    return resolved


def create_provider(
    kind: ProviderKind,
    name: ProviderName | None = None,
    settings: Settings | None = None,
) -> BaseProvider:
    """构造提供商实例。

    Args:
        kind: 'llm' 用于规划，'vlm' 用于图像理解与反思。
        name: 提供商名；None 时取 .env 中对应 kind 的默认值。
        settings: 配置；None 时用全局单例。
    """
    settings = settings or get_settings()
    resolved_name = name or (
        settings.llm_provider if kind == "llm" else settings.vlm_provider
    )

    if resolved_name == "mock":
        return MockProvider()

    spec = _resolve_spec(kind, resolved_name, settings)
    return OpenAICompatProvider(
        name=resolved_name,
        api_key=spec["api_key"],
        base_url=spec["base_url"],
        default_model=spec["model"],
        supports_vision=spec["supports_vision"],
    )
