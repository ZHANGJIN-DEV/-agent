"""OpenAI 兼容的 LLM / VLM 客户端。

DeepSeek、通义 DashScope（compatible-mode）、智谱 GLM 都实现同一套
`POST {base_url}/chat/completions` 协议，因此这里只维护一个实现，
差异通过 base_url / model / api_key 注入。

重试策略：指数退避处理 429 / 5xx / 网络抖动；4xx 鉴权与参数错误不重试
（重试没有意义，只会浪费额度）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseProvider, ChatMessage, ChatResult, ChatUsage, ProviderError

logger = logging.getLogger(__name__)

# 可重试的瞬时故障
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class _RetryableAPIError(ProviderError):
    """标记为可重试的 API 错误。"""


class OpenAICompatProvider(BaseProvider):
    """通过 OpenAI 兼容协议访问国内大模型平台。"""

    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str,
        default_model: str,
        supports_vision: bool = False,
        timeout_s: float = 120.0,
    ) -> None:
        if not api_key:
            raise ProviderError(f"{name} 未配置 API key，请在 .env 中填写")
        self.name = name
        self.supports_vision = supports_vision
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._client = httpx.Client(timeout=timeout_s)

    # ------------------------------------------------------------ 内部

    def _endpoint(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            # 三家平台都支持 response_format，返回严格 JSON —— Agent 规划器的可靠性基础
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """单次请求。瞬时错误抛 _RetryableAPIError 交给 tenacity。"""
        try:
            resp = self._client.post(
                self._endpoint(), headers=self._headers(), json=payload
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise _RetryableAPIError(f"{self.name} 网络异常: {exc}") from exc

        if resp.status_code in _RETRYABLE_STATUS:
            raise _RetryableAPIError(
                f"{self.name} 返回 {resp.status_code}: {resp.text[:300]}"
            )
        if resp.status_code >= 400:
            # 鉴权/参数错误，重试无意义
            raise ProviderError(
                f"{self.name} 请求失败 ({resp.status_code}): {resp.text[:500]}"
            )

        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.name} 返回非 JSON 响应: {resp.text[:300]}") from exc

    # ------------------------------------------------------------ 公开接口

    @retry(
        retry=retry_if_exception_type(_RetryableAPIError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(payload)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> ChatResult:
        if any(m.has_image() for m in messages) and not self.supports_vision:
            raise ProviderError(
                f"{self.name} 不支持视觉输入。VLM 请用 qwen 或 zhipu provider。"
            )

        resolved_model = model or self._default_model
        payload = self._build_payload(
            messages, resolved_model, temperature, max_tokens, json_mode
        )
        data = self._post_with_retry(payload)

        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name} 响应结构异常: {str(data)[:300]}") from exc

        return ChatResult(
            text=text,
            model=data.get("model", resolved_model),
            usage=ChatUsage.from_openai(data.get("usage")),
            provider=self.name,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
