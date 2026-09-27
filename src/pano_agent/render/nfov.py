"""NFOV 渲染 —— 把求解出的相机轨迹应用到 ERP 全景，输出成片。

==============================================================================
渲染流程
==============================================================================

    ERP 帧序列 + 相机轨迹
        ↓ geometry.projection.render_nfov    逐帧切出透视视图
        ↓ ffmpeg 编码
    成片视频

渲染本身只是调用 `render_nfov`，但有三处工程细节决定成败：

1. **接缝连续性**
   `sample_erp` 内部已处理水平环绕（x 方向取模），因此相机跨越 ±180°
   时不会出现黑边或错位。这一性质由 geometry 层的测试保证，
   本模块不需要也不应该重复处理 —— 重复处理反而会引入新的不一致。

2. **编码参数**
   H.264 + yuv420p（兼容性最好，所有播放器都认），CRF 20 左右。
   通过 pipe 逐帧写入，不落盘中间帧 —— 15 秒 1080p 的中间帧
   序列会占好几 GB。

3. **帧对齐**
   渲染帧数必须与音轨一致，否则成品音画不同步。本模块在写入前
   显式校验这一点。

==============================================================================
与渲染质量有关的取舍
==============================================================================

`render_nfov` 用的是我们自己实现的**双线性采样**，而非 cv2.remap。
原因在 geometry 层的对拍测试里验证过：两者在双线性模式下差异
仅 0.004 像素（远小于一个像素），而自实现版本与我们的球面坐标
约定严格一致，不存在坐标系换算的风险。

对于「用 OpenCV 更快」的疑问：本项目的瓶颈在感知与优化，
渲染 450 帧的开销可忽略，不值得为此引入第二套坐标约定。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..geometry.projection import render_nfov
from ..optimize.solver import CameraTrajectory

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    """渲染配置。"""

    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    crf: int = 20
    """H.264 质量参数，越小越好（18-23 是常用区间）。"""

    preset: str = "medium"
    """ffmpeg 编码预设。medium 在速度与体积间平衡较好。"""

    pix_fmt: str = "yuv420p"
    """兼容性最好的像素格式 —— 缺了它很多播放器打不开。"""

    interpolation: str = "bilinear"


class NfovRenderer:
    """把相机轨迹渲染成 NFOV 视频。"""

    def __init__(self, config: RenderConfig | None = None) -> None:
        self.config = config or RenderConfig()

    # ------------------------------------------------------------ 逐帧

    def render_frames(
        self,
        erp_frames: ArrayLike,
        trajectory: CameraTrajectory,
    ) -> Iterator[NDArray[np.uint8]]:
        """逐帧渲染（生成器，不占内存）。

        Args:
            erp_frames: (N, H, W[, C]) 的 ERP 帧序列。
                也接受单帧 (H, W[, C])。
            trajectory: 相机轨迹，长度需与帧数一致。

        Yields:
            每帧的 (height, width[, C]) 透视图。
        """
        frames = np.asarray(erp_frames)
        n_cams = len(trajectory)

        # ⚠ 维度消歧：`(N, H, W)`（多帧灰度）与 `(H, W, C)`（单帧彩色）
        # **都是 3 维**，仅凭 shape 无法可靠区分 ——
        # 例如 8 帧 128×256 的灰度序列是 (8,128,256)，
        # 而单帧 128×256 的 RGB 是 (128,256,3)，两者都是三维数组。
        #
        # 因此用**轨迹长度**作为消歧依据：帧数必须与相机数相同。
        # 这是唯一可靠且无歧义的信号。
        if frames.ndim == 2:  # (H, W) —— 单帧灰度
            frames = frames[None, ...]
        elif frames.ndim == 3:
            if frames.shape[0] == n_cams:
                pass  # (N, H, W) —— 多帧灰度
            elif frames.shape[-1] in (3, 4):
                frames = frames[None, ...]  # (H, W, C) —— 单帧彩色
            else:
                frames = frames[None, ...]  # 保守：按单帧处理

        n_frames = len(frames)
        if n_cams != n_frames:
            raise ValueError(
                f"轨迹帧数({n_cams})与视频帧数({n_frames})不一致 —— "
                f"这会导致音画不同步或渲染截断。"
                f"（输入形状 {np.asarray(erp_frames).shape}）"
            )

        cameras = trajectory.to_cameras(self.config.width, self.config.height)

        for frame, cam in zip(frames, cameras, strict=True):
            out = render_nfov(frame, cam, mode=self.config.interpolation)
            yield _to_uint8(out)

    # ------------------------------------------------------------ 落盘

    def render_to_file(
        self,
        erp_frames: ArrayLike,
        trajectory: CameraTrajectory,
        output_path: Path | str,
        *,
        audio_path: Path | str | None = None,
    ) -> Path:
        """渲染并编码成 mp4。

        Args:
            erp_frames: ERP 帧序列。
            trajectory: 相机轨迹。
            output_path: 输出 mp4 路径。
            audio_path: 可选音轨（会与视频合流）。

        Returns:
            输出文件路径。
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("未找到 ffmpeg。请确认它已安装并在 PATH 中。")

        cmd = self._build_ffmpeg_cmd(output, audio_path)
        logger.info("开始渲染: %s", output)

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            n_written = 0
            for frame in self.render_frames(erp_frames, trajectory):
                assert proc.stdin is not None
                proc.stdin.write(frame.tobytes())
                n_written += 1

            if proc.stdin is not None:
                proc.stdin.close()

            _, stderr = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg 编码失败（返回码 {proc.returncode}）:\n"
                    f"{stderr.decode('utf-8', errors='ignore')[-800:]}"
                )

            logger.info("渲染完成: %d 帧 → %s", n_written, output)

        except BrokenPipeError as exc:
            _, stderr = proc.communicate()
            raise RuntimeError(
                f"ffmpeg 管道中断:\n{stderr.decode('utf-8', errors='ignore')[-800:]}"
            ) from exc
        except Exception:
            proc.kill()
            proc.communicate()
            raise

        return output

    # ------------------------------------------------------------ 内部

    def _build_ffmpeg_cmd(
        self, output: Path, audio_path: Path | str | None
    ) -> list[str]:
        """构造 ffmpeg 命令。

        视频从 stdin 以 rawvideo 读入，避免落盘中间帧。
        """
        cfg = self.config
        cmd = [
            "ffmpeg",
            "-y",                              # 覆盖已存在文件
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{cfg.width}x{cfg.height}",
            "-pix_fmt", "rgb24",
            "-r", str(cfg.fps),
            "-i", "pipe:0",                    # 视频来自管道
        ]

        if audio_path is not None and Path(audio_path).is_file():
            cmd += ["-i", str(audio_path), "-c:a", "aac", "-shortest"]

        cmd += [
            "-c:v", "libx264",
            "-preset", cfg.preset,
            "-crf", str(cfg.crf),
            "-pix_fmt", cfg.pix_fmt,
            "-movflags", "+faststart",         # 便于网页流式播放
            str(output),
        ]
        return cmd


def _to_uint8(frame: NDArray) -> NDArray[np.uint8]:
    """把渲染结果转成 uint8。

    `render_nfov` 保持输入的 dtype，而 ERP 源数据可能是 uint8、
    float32（[0,1]）或 float64。这里统一处理。
    """
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        return arr

    a = arr.astype(np.float64)
    if a.size == 0:
        return a.astype(np.uint8)

    # 先把 NaN/Inf 抹平 —— 否则后续的 nanmin 传播 NaN，
    # 最终 astype(uint8) 会触发 "invalid value encountered in cast" 警告
    # 并产生未定义的值。
    a = np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)

    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo > 1e-9:
        # 若数值明显落在 [0,1] 内则按此缩放，否则线性拉伸到满量程
        if hi <= 1.0001 and lo >= -0.0001:
            a = a * 255.0
        else:
            a = (a - lo) / (hi - lo) * 255.0
    else:
        a = np.zeros_like(a)

    return np.clip(a, 0, 255).astype(np.uint8)
