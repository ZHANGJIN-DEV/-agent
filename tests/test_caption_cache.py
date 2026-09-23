"""caption 模块的测试。

用 MockProvider 验证通路 —— 不发真实 API 请求、不花钱、结果确定。
重点是**缓存正确性**与**失败容错**：caption 是增强信息，
它的缺失绝不该让整条 pipeline 中断。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pano_agent.agent.providers import MockProvider
from pano_agent.sir.caption import (
    CaptionCache,
    CaptionConfig,
    _frame_digest,
    caption_track,
    caption_tracks,
    select_keyframes,
)


def make_frame(value: int = 128, h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# ============================================================== 关键帧


class TestKeyframeSelection:
    def test_fewer_than_requested_returns_all(self):
        frames = [make_frame(i) for i in range(2)]
        assert len(select_keyframes(frames, num=3)) == 2

    def test_selects_evenly_spaced(self):
        """均匀采样而非取前 N 帧 —— 目标可能在轨迹后段才展现特征。"""
        frames = [make_frame(i * 10) for i in range(10)]
        picked = select_keyframes(frames, num=3)

        assert len(picked) == 3
        # 应覆盖首、中、末
        assert np.array_equal(picked[0], frames[0])
        assert np.array_equal(picked[-1], frames[-1])

    def test_empty_input(self):
        assert select_keyframes([], num=3) == []

    def test_num_larger_than_available(self):
        frames = [make_frame(i) for i in range(5)]
        assert len(select_keyframes(frames, num=10)) == 5


# ============================================================== 缓存


class TestCaptionCache:
    def test_put_and_get(self, tmp_path):
        cache = CaptionCache(tmp_path)
        cache.put("abc", "一个人正在滑板")
        assert cache.get("abc") == "一个人正在滑板"

    def test_miss_returns_none(self, tmp_path):
        cache = CaptionCache(tmp_path)
        assert cache.get("nonexistent") is None

    def test_corrupted_entry_treated_as_miss(self, tmp_path):
        """缓存文件损坏时应当作未命中，而不是抛异常。"""
        cache = CaptionCache(tmp_path)
        (tmp_path / "bad.json").write_text("这不是 JSON", encoding="utf-8")
        assert cache.get("bad") is None

    def test_tracks_hit_rate(self, tmp_path):
        cache = CaptionCache(tmp_path)
        cache.put("k", "v")
        cache.get("k")   # 命中
        cache.get("x")   # 未命中

        assert cache.hits == 1
        assert cache.misses == 1
        assert "50%" in cache.stats

    def test_creates_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        CaptionCache(nested)
        assert nested.is_dir()


# ============================================================== 内容寻址


class TestContentAddressedKey:
    """key 必须由内容决定，而不是路径或调用顺序。"""

    def test_same_input_same_key(self):
        frames = [make_frame(100)]
        k1 = _frame_digest(frames, "prompt", "model")
        k2 = _frame_digest([make_frame(100)], "prompt", "model")
        assert k1 == k2

    def test_different_pixels_different_key(self):
        k1 = _frame_digest([make_frame(100)], "p", "m")
        k2 = _frame_digest([make_frame(101)], "p", "m")
        assert k1 != k2

    def test_different_shape_different_key(self):
        k1 = _frame_digest([make_frame(100, h=32, w=32)], "p", "m")
        k2 = _frame_digest([make_frame(100, h=64, w=64)], "p", "m")
        assert k1 != k2

    def test_different_prompt_different_key(self):
        frames = [make_frame()]
        assert _frame_digest(frames, "a", "m") != _frame_digest(frames, "b", "m")

    def test_different_model_different_key(self):
        frames = [make_frame()]
        assert _frame_digest(frames, "p", "m1") != _frame_digest(frames, "p", "m2")

    def test_frame_order_matters(self):
        """帧序不同则应视为不同输入 —— caption 依赖时间顺序。"""
        a, b = make_frame(50), make_frame(200)
        assert _frame_digest([a, b], "p", "m") != _frame_digest([b, a], "p", "m")


# ============================================================== 端到端


class TestCaptionTrack:
    def test_produces_caption_with_mock(self, tmp_path):
        provider = MockProvider()
        config = CaptionConfig(cache_dir=tmp_path / "cache")

        caption = caption_track([make_frame()], provider, config=config)

        assert isinstance(caption, str)
        assert len(caption) > 0
        assert provider.call_count == 1

    def test_second_call_hits_cache(self, tmp_path):
        """★ 相同输入第二次调用必须命中缓存，不再请求 API。"""
        provider = MockProvider()
        config = CaptionConfig(cache_dir=tmp_path / "cache")
        frames = [make_frame(77)]

        first = caption_track(frames, provider, config=config)
        calls_after_first = provider.call_count

        second = caption_track([make_frame(77)], provider, config=config)

        assert second == first
        assert provider.call_count == calls_after_first, "第二次调用不应再请求 API"

    def test_different_frames_bypass_cache(self, tmp_path):
        provider = MockProvider()
        config = CaptionConfig(cache_dir=tmp_path / "cache")

        caption_track([make_frame(10)], provider, config=config)
        caption_track([make_frame(20)], provider, config=config)

        assert provider.call_count == 2, "不同图像不应命中同一缓存"

    def test_empty_frames_returns_empty(self, tmp_path):
        provider = MockProvider()
        assert caption_track([], provider, config=CaptionConfig(cache_dir=tmp_path)) == ""
        assert provider.call_count == 0

    def test_disabled_returns_empty(self):
        provider = MockProvider()
        config = CaptionConfig(enabled=False)
        assert caption_track([make_frame()], provider, config=config) == ""
        assert provider.call_count == 0

    def test_failure_is_swallowed(self, tmp_path):
        """★ caption 失败不能中断 pipeline —— 它只是增强信息。"""

        class FailingProvider(MockProvider):
            supports_vision = True

            def chat(self, *args, **kwargs):
                raise RuntimeError("模拟 API 故障")

        caption = caption_track(
            [make_frame()], FailingProvider(), config=CaptionConfig(cache_dir=tmp_path)
        )
        assert caption == "", "失败时应返回空串而非抛异常"

    def test_provider_without_vision_returns_empty(self, tmp_path):
        provider = MockProvider()
        provider.supports_vision = False

        caption = caption_track(
            [make_frame()], provider, config=CaptionConfig(cache_dir=tmp_path)
        )
        assert caption == ""
        assert provider.call_count == 0


class TestCaptionTracks:
    def test_batch_returns_dict(self, tmp_path):
        provider = MockProvider()
        config = CaptionConfig(cache_dir=tmp_path / "cache")

        result = caption_tracks(
            {1: [make_frame(10)], 2: [make_frame(20)]}, provider, config=config
        )

        assert set(result) == {1, 2}
        assert all(isinstance(v, str) for v in result.values())

    def test_batch_shares_cache_across_tracks(self, tmp_path):
        """同一批内的重复图像应共享缓存。"""
        provider = MockProvider()
        config = CaptionConfig(cache_dir=tmp_path / "cache")

        # 两条轨迹用完全相同的帧
        caption_tracks(
            {1: [make_frame(42)], 2: [make_frame(42)]}, provider, config=config
        )

        assert provider.call_count == 1, "相同图像应命中同一条缓存"

    def test_batch_continues_after_failure(self, tmp_path):
        """单条失败不应影响其他条。"""

        class SometimesFailing(MockProvider):
            supports_vision = True

            def chat(self, messages, **kwargs):
                # 注意 MockProvider.chat 成功时会自增 call_count，
                # 因此「第一次调用」对应 call_count == 0
                if self.call_count == 0:
                    self.call_count += 1
                    raise RuntimeError("第一条失败")
                return super().chat(messages, **kwargs)

        provider = SometimesFailing()
        config = CaptionConfig(cache_dir=tmp_path / "cache")

        result = caption_tracks(
            {1: [make_frame(1)], 2: [make_frame(2)]}, provider, config=config
        )

        assert len(result) == 2
        assert result[1] == "", "失败的应为空串"
        assert result[2] != "", "后续的应正常完成"
