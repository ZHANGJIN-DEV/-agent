"""用 VLM 为轨迹生成语义描述（`TrackNode.caption`）。

==============================================================================
为什么 caption 是必需品而非装饰
==============================================================================

PSG 的符号层要让 LLM 理解"画面里有什么"。但仅有 `track_id` 和统计量
（角速度 45°/秒、时长 3.2 秒）不足以支撑剪辑决策 —— LLM 需要知道
这是**一个人**还是**一辆车**，是**在滑板**还是**在散步**。

caption 就是这座桥：把轨迹的视觉内容翻译成语言。

用户指令「剪一个突出滑板者的片段」要能匹配到某条轨迹，
靠的正是 caption 里的语义。

==============================================================================
关键帧选择策略
==============================================================================

一条轨迹可能有几百帧。全部送 VLM 既昂贵又无必要 —— 相邻帧高度冗余。

策略：均匀取 **N 帧**（默认 3 帧），覆盖目标出现 / 发展 / 消失。

为什么不取更多：caption 需要的是"这是什么"，不是"逐帧发生了什么"。
3 帧足以让 VLM 判断主体类别与大致行为，成本却是 1/100。

==============================================================================
内容寻址缓存
==============================================================================

VLM 调用要花钱。调试时同一张图可能被重复调用几十次。

因此所有调用都经过**内容寻址缓存**：

    key = sha1(图像字节 + prompt + 模型名)

同一输入永远只调一次。开发期这能省 5-10 倍成本 ——
这是工程素养的直接体现（计划里明确要求的一点）。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    from ..agent.providers.base import BaseProvider

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "这是从 360° 全景视频中裁出的同一个目标的多帧画面（按时间顺序）。\n"
    "请用**一句中文**描述：这是什么主体、正在做什么。\n"
    "要求：\n"
    "- 只描述你确实看到的，不要推测画外信息\n"
    "- 直接给出描述，不要任何前缀或解释\n"
    "- 控制在 30 字以内"
)


@dataclass
class CaptionConfig:
    """caption 生成配置。"""

    model: str = "deepseek-flash"
    """带视觉的模型。实测 deepseek-flash / deepseek-chat 均支持图像输入。"""

    num_frames: int = 3
    """送多少帧给 VLM。3 帧是信息量与成本的折中（见模块文档）。"""

    max_tokens: int = 120
    temperature: float = 0.0
    """描述任务不需要创造性，用 0 保证可复现。"""

    prompt: str = DEFAULT_PROMPT
    cache_dir: Path | None = None
    """内容寻址缓存目录。None 则用 `config.settings.pano_cache_root/vlm`。"""

    enabled: bool = True


# ============================================================== 缓存


def _frame_digest(frames: list[NDArray], prompt: str, model: str) -> str:
    """计算内容寻址键。

    基于图像**字节**而非路径 —— 同一张图换了个文件名仍是同一个 key，
    而内容变了则 key 必变。这正是"内容寻址"的含义。
    """
    h = hashlib.sha1()
    for frame in frames:
        arr = np.ascontiguousarray(frame)
        h.update(arr.tobytes())
        h.update(str(arr.shape).encode())
    h.update(prompt.encode("utf-8"))
    h.update(model.encode("utf-8"))
    return h.hexdigest()


class CaptionCache:
    """内容寻址的 caption 缓存。"""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        path = self.cache_dir / f"{key}.json"
        if path.is_file():
            try:
                self.hits += 1
                return json.loads(path.read_text(encoding="utf-8"))["caption"]
            except (json.JSONDecodeError, KeyError):
                return None  # 缓存损坏，当作未命中
        self.misses += 1
        return None

    def put(self, key: str, caption: str) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(
            json.dumps({"caption": caption}, ensure_ascii=False), encoding="utf-8"
        )

    @property
    def stats(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total if total else 0.0
        return f"缓存命中 {self.hits}/{total}（{rate:.0%}）"


# ============================================================== 图像编码


def _encode_png_data_uri(frame: NDArray) -> str:
    """把一帧编码成 PNG 的 data URI。

    转 PNG 而非直接用原始数组：API 只接受常见图像格式，
    且 PNG 无损 —— caption 任务里不该引入有损压缩的干扰。
    """
    import io

    from PIL import Image

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float64)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = (a - lo) / (hi - lo) * 255.0 if hi - lo > 1e-9 else np.zeros_like(a)
        arr = a.astype(np.uint8)

    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def select_keyframes(
    frames: list[NDArray], num: int = 3
) -> list[NDArray]:
    """从一条轨迹的帧序列中均匀选取代表帧。

    均匀采样而非取前 N 帧：目标可能在轨迹后段才展现特征。
    """
    if not frames:
        return []
    if len(frames) <= num:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, num).round().astype(int)
    return [frames[i] for i in idx]


# ============================================================== 主入口


def caption_track(
    frames: list[NDArray],
    provider: BaseProvider,
    *,
    config: CaptionConfig | None = None,
    cache: CaptionCache | None = None,
) -> str:
    """为一条轨迹生成语义描述。

    Args:
        frames: 该轨迹的关键帧（应已按时间排序）。
            调用方通常用 `select_keyframes` 从完整序列中抽取。
        provider: VLM 提供商。
        config: 配置。
        cache: 缓存实例。None 则用 config 中的 cache_dir 构造。

    Returns:
        一句中文描述。调用失败时返回空串（**不抛异常**）——
        caption 是增强信息，它的缺失不应让整条 pipeline 中断。
    """
    config = config or CaptionConfig()

    if not config.enabled:
        return ""
    if not frames:
        return ""

    if not provider.supports_vision:
        logger.warning("提供商 %s 不支持视觉，跳过 caption", provider.name)
        return ""

    if cache is None and config.cache_dir is not None:
        cache = CaptionCache(config.cache_dir)

    key = _frame_digest(frames, config.prompt, config.model)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    try:
        from ..agent.providers.base import ChatMessage, ImagePart, TextPart

        parts = [TextPart(config.prompt)]
        for frame in frames:
            parts.append(ImagePart(data_uri=_encode_png_data_uri(frame)))

        result = provider.chat(
            [ChatMessage.user(parts)],
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        caption = result.text.strip()

    except Exception as exc:  # noqa: BLE001 - caption 失败不应中断 pipeline
        logger.warning("caption 生成失败（已跳过）: %s: %s", type(exc).__name__, exc)
        return ""

    if cache is not None and caption:
        cache.put(key, caption)

    return caption


def caption_tracks(
    tracks_frames: dict[int, list[NDArray]],
    provider: BaseProvider,
    *,
    config: CaptionConfig | None = None,
) -> dict[int, str]:
    """批量生成 caption。

    单条失败不影响其他条 —— 逐条捕获异常。
    """
    config = config or CaptionConfig()
    cache = CaptionCache(config.cache_dir) if config.cache_dir else None

    results: dict[int, str] = {}
    for track_id, frames in tracks_frames.items():
        keyframes = select_keyframes(frames, config.num_frames)
        results[track_id] = caption_track(
            keyframes, provider, config=config, cache=cache
        )

    if cache is not None:
        logger.info("caption 批量完成，%s", cache.stats)

    return results
