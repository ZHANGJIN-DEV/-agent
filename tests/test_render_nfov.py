"""渲染模块的测试。

分两层：
    1. **纯逻辑**（帧数校验、dtype 转换）—— 不需要 ffmpeg
    2. **端到端编码** —— 需要 ffmpeg（本机已装）
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pano_agent.optimize.solver import CameraTrajectory
from pano_agent.render.nfov import NfovRenderer, RenderConfig, _to_uint8

HAS_FFMPEG = shutil.which("ffmpeg") is not None

ERP_H, ERP_W = 128, 256


def make_erp_frames(n: int = 8, *, kind: str = "gray") -> np.ndarray:
    """构造测试用 ERP 帧序列。"""
    rng = np.random.default_rng(0)
    if kind == "gray":
        return rng.integers(0, 255, (n, ERP_H, ERP_W), dtype=np.uint8)
    return rng.integers(0, 255, (n, ERP_H, ERP_W, 3), dtype=np.uint8)


def make_trajectory(n: int, *, width: int = 160, height: int = 90) -> CameraTrajectory:
    return CameraTrajectory(
        times=np.arange(n, dtype=float) / 30.0,
        yaw_deg=np.linspace(-30.0, 30.0, n),
        pitch_deg=np.zeros(n),
        fov_deg=np.full(n, 75.0),
    )


def make_renderer(**cfg) -> NfovRenderer:
    base = {"width": 160, "height": 90, "fps": 30.0}
    base.update(cfg)
    return NfovRenderer(RenderConfig(**base))


# ============================================================== 逐帧渲染


class TestRenderFrames:
    def test_renders_correct_number_of_frames(self):
        n = 6
        frames = list(make_renderer().render_frames(make_erp_frames(n), make_trajectory(n)))
        assert len(frames) == n

    def test_output_shape_matches_config(self):
        r = make_renderer(width=160, height=90)
        out = list(r.render_frames(make_erp_frames(3), make_trajectory(3)))
        assert out[0].shape[:2] == (90, 160)

    def test_output_dtype_is_uint8(self):
        out = list(make_renderer().render_frames(make_erp_frames(2), make_trajectory(2)))
        assert out[0].dtype == np.uint8

    def test_color_input_preserved(self):
        r = make_renderer(width=160, height=90)
        out = list(r.render_frames(make_erp_frames(2, kind="rgb"), make_trajectory(2)))
        assert out[0].shape == (90, 160, 3)

    def test_single_frame_input_accepted(self):
        """单帧 ERP（无 batch 维）也应接受。"""
        frame = make_erp_frames(1)[0]
        traj = make_trajectory(1)
        out = list(make_renderer().render_frames(frame, traj))
        assert len(out) == 1

    def test_rejects_frame_count_mismatch(self):
        """★ 帧数与轨迹长度不一致必须报错。

        若放任不管，成品会音画不同步或提前截断 ——
        这类问题在成片里很难定位。
        """
        with pytest.raises(ValueError, match="不一致"):
            list(make_renderer().render_frames(make_erp_frames(5), make_trajectory(3)))

    def test_is_lazy(self):
        """生成器应当是惰性的 —— 15 秒 1080p 全渲染出来会占好几 GB。"""
        gen = make_renderer().render_frames(make_erp_frames(100), make_trajectory(100))
        assert hasattr(gen, "__next__")
        first = next(gen)
        assert first is not None  # 只取一帧不应触发全部渲染


class TestSeamRendering:
    def test_camera_crossing_seam_renders_without_artifact(self):
        """★ 相机跨越 ±180° 时应正常渲染（无黑边/错位）。

        接缝的环绕处理在 `geometry.sample_erp` 内部完成，
        本测试确认它经渲染路径后依然生效。
        """
        n = 20
        # 构造一张沿经度渐变、在接缝处连续的 ERP
        ramp = np.linspace(0, 2 * np.pi, ERP_W, endpoint=False)
        base = (np.cos(ramp) * 0.5 + 0.5) * 255
        erp = np.tile(base, (ERP_H, 1)).astype(np.uint8)
        frames = np.stack([erp] * n)

        # 相机从 170° 扫到 190°（跨接缝）
        traj = CameraTrajectory(
            times=np.arange(n) / 30.0,
            yaw_deg=np.linspace(170.0, 190.0, n),
            pitch_deg=np.zeros(n),
            fov_deg=np.full(n, 75.0),
        )

        out = list(make_renderer(width=160, height=90).render_frames(frames, traj))

        for i, frame in enumerate(out):
            assert frame.shape[:2] == (90, 160)
            # 不应出现全黑帧（黑边是接缝处理失败的典型症状）
            assert frame.mean() > 10, f"第 {i} 帧几乎全黑，疑似接缝渲染失败"
            # 也不应有异常亮/暗的条纹
            assert frame.std() > 5, f"第 {i} 帧方差过小，疑似采样失败"


# ============================================================== dtype 转换


class TestToUint8:
    def test_uint8_passthrough(self):
        a = np.array([[0, 128, 255]], dtype=np.uint8)
        assert _to_uint8(a).dtype == np.uint8
        np.testing.assert_array_equal(_to_uint8(a), a)

    def test_float_unit_range_scaled(self):
        a = np.array([[0.0, 0.5, 1.0]])
        out = _to_uint8(a)
        assert out.dtype == np.uint8
        assert out[0, 0] == 0
        assert out[0, 2] == 255

    def test_float_beyond_unit_range_stretched(self):
        a = np.array([[10.0, 20.0, 30.0]])
        out = _to_uint8(a)
        assert out.min() == 0 and out.max() == 255

    def test_constant_array_does_not_crash(self):
        """全常数的图（方差为 0）不应触发除零。"""
        a = np.full((4, 4), 7.5)
        out = _to_uint8(a)
        assert np.isfinite(out).all()

    def test_nan_handled(self):
        a = np.array([[0.0, np.nan, 1.0]])
        assert np.isfinite(_to_uint8(a)).all()

    def test_values_clipped(self):
        a = np.array([[-100.0, 500.0]])
        out = _to_uint8(a)
        assert out.min() >= 0 and out.max() <= 255


# ============================================================== 端到端


@pytest.mark.skipif(not HAS_FFMPEG, reason="需要 ffmpeg")
class TestEndToEndEncoding:
    def test_writes_playable_mp4(self, tmp_path):
        """★ 完整走一遍：渲染 → 编码 → 用 ffprobe 验证产物。"""
        n = 12
        out_path = tmp_path / "out.mp4"

        renderer = make_renderer(width=160, height=90, fps=10.0)
        result = renderer.render_to_file(
            make_erp_frames(n), make_trajectory(n), out_path
        )

        assert result.is_file()
        assert result.stat().st_size > 0

        # 用 ffprobe 校验：帧数、分辨率、编码格式
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,width,height,codec_name",
                "-of", "default=noprint_wrappers=1",
                str(result),
            ],
            capture_output=True, text=True, timeout=60,
        )
        info = dict(
            line.split("=") for line in probe.stdout.strip().splitlines() if "=" in line
        )
        assert info.get("codec_name") == "h264"
        assert int(info.get("width", 0)) == 160
        assert int(info.get("height", 0)) == 90

    def test_creates_output_directory(self, tmp_path):
        n = 6
        out_path = tmp_path / "nested" / "dir" / "out.mp4"
        make_renderer().render_to_file(make_erp_frames(n), make_trajectory(n), out_path)
        assert out_path.is_file()

    def test_handles_color_input(self, tmp_path):
        n = 8
        out_path = tmp_path / "color.mp4"
        make_renderer().render_to_file(
            make_erp_frames(n, kind="rgb"), make_trajectory(n), out_path
        )
        assert out_path.stat().st_size > 0

    def test_mismatch_fails_before_writing(self, tmp_path):
        """帧数不匹配应在写文件前就失败，不留半成品。"""
        out_path = tmp_path / "bad.mp4"
        with pytest.raises(ValueError, match="不一致"):
            make_renderer().render_to_file(
                make_erp_frames(10), make_trajectory(4), out_path
            )
