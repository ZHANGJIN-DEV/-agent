"""全局配置：从 .env 加载，统一管理密钥、路径与设备选择。

设计原则:
1. 密钥一律用 SecretStr 包装，避免日志或异常回溯时意外打印明文。
2. 路径统一从 PANO_DATA_ROOT 派生，业务代码里不出现硬编码绝对路径。
   （数据放 C 盘而非 D 盘，因为 D 盘仅剩 58G，放不下 4K 全景视频。）
3. provider 默认 mock，保证没有密钥时测试与 CI 仍能跑通全流程。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（src/pano_agent/config.py -> 上溯三级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ProviderName = Literal["deepseek", "qwen", "zhipu", "mock"]


class Settings(BaseSettings):
    """运行期配置。字段名小写，对应 .env 中的大写环境变量。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM（Agent 规划层）----
    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # ---- VLM（语义描述 + 反思层）----
    dashscope_api_key: SecretStr = SecretStr("")
    qwen_vl_model: str = "qwen-vl-plus"
    qwen_llm_model: str = "qwen-plus"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    zhipu_api_key: SecretStr = SecretStr("")
    glm_vl_model: str = "glm-4v-flash"
    glm_llm_model: str = "glm-4.5-flash"
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"

    # ---- provider 选择 ----
    llm_provider: ProviderName = Field(default="mock", alias="PANODIRECTOR_LLM_PROVIDER")
    vlm_provider: ProviderName = Field(default="mock", alias="PANODIRECTOR_VLM_PROVIDER")

    # ---- 路径 ----
    pano_data_root: Path = Path("C:/pano_data")
    pano_cache_root: Path = Path("C:/pano_data/cache")

    # ---- 运行设备 ----
    pano_device: Literal["auto", "cuda", "cpu"] = "auto"

    # ---- 派生路径 ----
    @property
    def datasets_dir(self) -> Path:
        return self.pano_data_root / "datasets"

    @property
    def work_dir(self) -> Path:
        """2K H.265 中间格式，主力读取目录。"""
        return self.pano_data_root / "work"

    @property
    def fast_dir(self) -> Path:
        """512x256 快扫版，供光流与显著性使用。"""
        return self.pano_data_root / "fast"

    def ensure_dirs(self) -> None:
        """创建所有数据目录（幂等）。"""
        for path in (
            self.pano_data_root,
            self.pano_cache_root,
            self.datasets_dir,
            self.work_dir,
            self.fast_dir,
            self.pano_data_root / "raw",
        ):
            path.mkdir(parents=True, exist_ok=True)

    # ---- 设备解析 ----
    def resolve_device(self) -> str:
        """把 'auto' 解析为实际设备。

        注意：不依赖 torch，因为几何层等模块是纯 numpy 的，
        不应为了查设备而引入重型依赖。
        """
        if self.pano_device != "auto":
            return self.pano_device
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible.strip() in {"", "-1"}:
            return "cpu"
        try:  # pragma: no cover - 依赖运行环境
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ---- 密钥可用性检查 ----
    def has_llm_credentials(self, provider: ProviderName | None = None) -> bool:
        name = provider or self.llm_provider
        return {
            "deepseek": bool(self.deepseek_api_key.get_secret_value()),
            "qwen": bool(self.dashscope_api_key.get_secret_value()),
            "zhipu": bool(self.zhipu_api_key.get_secret_value()),
            "mock": True,
        }[name]

    def has_vlm_credentials(self, provider: ProviderName | None = None) -> bool:
        name = provider or self.vlm_provider
        return {
            "deepseek": False,  # DeepSeek 暂无视觉模型
            "qwen": bool(self.dashscope_api_key.get_secret_value()),
            "zhipu": bool(self.zhipu_api_key.get_secret_value()),
            "mock": True,
        }[name]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例（带缓存，避免重复解析 .env）。"""
    return Settings()
