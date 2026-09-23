"""实测显存占用，用于校准 configs/vram_budget.yaml。

为什么必须实测而不能看标称值:
    RTX 3050 Laptop 标称 4GB 显存，但 Windows WDDM 下桌面合成（DWM）
    常驻占用约 1.5GB，**实际可用仅约 2.5GB**。所有模型选型与输入
    分辨率必须按实测值设计。

    这张实测表同时是技术报告里「端侧落地能力」的直接证据。

用法:
    python scripts/vram_probe.py                    # 基础显存报告
    python scripts/vram_probe.py --measure yolo11n  # 实测单模型峰值
    python scripts/vram_probe.py --json             # 机器可读输出
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field

MiB = 1024 * 1024


@dataclass
class GpuInfo:
    name: str = "unknown"
    total_mb: int = 0
    used_mb: int = 0
    free_mb: int = 0
    driver: str = ""
    source: str = "nvidia-smi"

    @property
    def available_mb(self) -> int:
        """扣除桌面合成等系统占用后可用的显存。"""
        return self.free_mb


@dataclass
class MeasureResult:
    model: str
    device: str
    input_shape: list[int]
    precision: str
    peak_mb: float = 0.0
    baseline_mb: float = 0.0
    delta_mb: float = 0.0
    latency_ms: float = 0.0
    note: str = ""
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 基础探测


def probe_nvidia_smi() -> GpuInfo:
    """用 nvidia-smi 拿显存信息（不依赖 torch，永远可用）。"""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return GpuInfo(source="unavailable")

    line = out.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return GpuInfo(source="parse-error")
    return GpuInfo(
        name=parts[0],
        total_mb=int(parts[1]),
        used_mb=int(parts[2]),
        free_mb=int(parts[3]),
        driver=parts[4],
        source="nvidia-smi",
    )


def probe_torch() -> tuple[GpuInfo | None, str]:
    """用 torch 拿更精确的显存信息。返回 (info, 说明)。"""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None, "torch 未安装"

    if not torch.cuda.is_available():
        return None, f"CUDA 不可用（torch {torch.__version__}，可能是 CPU 版）"

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(0)
    return (
        GpuInfo(
            name=props.name,
            total_mb=int(total_bytes / MiB),
            used_mb=int((total_bytes - free_bytes) / MiB),
            free_mb=int(free_bytes / MiB),
            driver=f"torch {torch.__version__} / CUDA {torch.version.cuda}",
            source="torch",
        ),
        "",
    )


# ---------------------------------------------------------------- 模型测量

# 每个模型如何构造 → (加载函数, 前向输入构造)。延迟导入避免脚本启动就吃显存。
def _measure_yolo(weights: str, imgsz: tuple[int, int], device: str) -> tuple[float, float]:
    """返回 (峰值显存 MB, 单帧延迟 ms)。"""
    import time

    import numpy as np
    import torch
    from ultralytics import YOLO

    model = YOLO(weights)
    h, w = imgsz
    dummy = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    # 预热
    model.predict(dummy, imgsz=(h, w), device=device, verbose=False, half=(device == "cuda"))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    n_runs = 5
    for _ in range(n_runs):
        model.predict(dummy, imgsz=(h, w), device=device, verbose=False, half=(device == "cuda"))
    latency = (time.perf_counter() - t0) / n_runs * 1000

    peak = torch.cuda.max_memory_allocated() / MiB
    del model
    torch.cuda.empty_cache()
    return peak, latency


MEASURABLE = {
    "yolo11n": ("yolo11n.pt", (640, 1280)),
    "yolo11s": ("yolo11s.pt", (640, 1280)),
    "yolov8n": ("yolov8n.pt", (640, 1280)),
}


def measure_model(name: str, device: str) -> MeasureResult:
    result = MeasureResult(model=name, device=device, input_shape=[], precision="fp16")
    if name not in MEASURABLE:
        result.errors.append(
            f"未知模型 {name!r}。已登记: {', '.join(MEASURABLE)}"
        )
        return result

    weights, imgsz = MEASURABLE[name]
    result.input_shape = list(imgsz)

    if device != "cuda":
        result.note = "CPU 模式，显存测量无意义"
        return result

    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated() / MiB
        result.baseline_mb = round(baseline, 1)

        peak, latency = _measure_yolo(weights, imgsz, device)
        result.peak_mb = round(peak, 1)
        result.delta_mb = round(peak - baseline, 1)
        result.latency_ms = round(latency, 2)
        result.precision = "fp16"
    except ImportError as exc:
        result.errors.append(f"依赖缺失: {exc}（先 pip install -e '.[dev]' 与 ultralytics）")
    except Exception as exc:  # noqa: BLE001 - 探测脚本需要容错
        result.errors.append(f"{type(exc).__name__}: {exc}")
    return result


# ---------------------------------------------------------------- 输出


def render_basic(info: GpuInfo, budget_mb: int = 2500) -> str:
    lines = [
        "",
        "=" * 62,
        "  显存基础报告",
        "=" * 62,
        f"  GPU        : {info.name}",
    ]
    if info.source == "unavailable":
        lines.append("  ✗ 未检测到 nvidia-smi，无法获取显存信息")
        lines.append("=" * 62)
        return "\n".join(lines)

    lines += [
        f"  驱动       : {info.driver}",
        f"  总显存     : {info.total_mb:>6} MiB",
        f"  系统占用   : {info.used_mb:>6} MiB   ← 桌面合成(DWM)等",
        f"  实际可用   : {info.free_mb:>6} MiB",
        "-" * 62,
    ]

    if info.free_mb < budget_mb:
        lines.append(
            f"  ⚠ 可用显存 {info.free_mb} MiB 低于预算 {budget_mb} MiB"
        )
        lines.append("    请关闭占用 GPU 的程序（浏览器硬件加速、游戏、其他训练任务）")
    else:
        lines.append(f"  ✓ 可用显存满足预算 {budget_mb} MiB")

    lines += [
        "=" * 62,
        "",
        "  提示：模型选型请以「实际可用」为准，不要以「总显存」为准。",
        "",
    ]
    return "\n".join(lines)


def render_measures(results: list[MeasureResult], budget_mb: int = 2500) -> str:
    if not results:
        return ""
    lines = [
        "=" * 78,
        "  模型实测",
        "=" * 78,
        f"  {'模型':<14}{'设备':<8}{'输入':<14}{'峰值(MiB)':<12}{'延迟(ms)':<12}{'判定':<8}",
        "-" * 78,
    ]
    for r in results:
        if r.errors:
            lines.append(f"  {r.model:<14}{r.device:<8}{'-':<14}{'ERROR':<12}{'-':<12}")
            for err in r.errors:
                lines.append(f"      └ {err}")
            continue
        shape = "x".join(str(s) for s in r.input_shape)
        verdict = "OK" if r.peak_mb < budget_mb else "超预算"
        lines.append(
            f"  {r.model:<14}{r.device:<8}{shape:<14}"
            f"{r.peak_mb:<12.1f}{r.latency_ms:<12.2f}{verdict:<8}"
        )
    lines += ["=" * 78, ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="显存探测与预算校准")
    parser.add_argument("--measure", action="append", default=[], metavar="MODEL",
                        help=f"实测模型峰值。可选: {', '.join(MEASURABLE)}")
    parser.add_argument("--all", action="store_true", help="实测全部已登记模型")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--budget", type=int, default=2500, help="显存预算 MiB")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    info = probe_nvidia_smi()
    torch_info, torch_note = probe_torch()

    targets = list(MEASURABLE) if args.all else args.measure
    results = [measure_model(m, args.device) for m in targets]

    if args.json:
        print(json.dumps(
            {
                "gpu": asdict(info),
                "torch": asdict(torch_info) if torch_info else None,
                "torch_note": torch_note,
                "budget_mb": args.budget,
                "measurements": [asdict(r) for r in results],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    print(render_basic(info, args.budget))
    if torch_note:
        print(f"  torch 备注: {torch_note}\n")
    print(render_measures(results, args.budget))
    return 0


if __name__ == "__main__":
    sys.exit(main())
