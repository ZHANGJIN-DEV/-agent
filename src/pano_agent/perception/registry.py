"""模型调度：子进程隔离 + 显存预算管理。

==============================================================================
为什么必须用子进程隔离模型
==============================================================================

RTX 3050 Laptop 标称 4GB 显存，但 Windows WDDM 下桌面合成（DWM）
常驻占用约 1.5GB，**实际可用仅约 2.5GB**。在这个预算下同时加载多个
模型几乎必然 OOM。

而更麻烦的是这一条：

    **torch.cuda.empty_cache() 并不可靠。**

它只把 PyTorch 缓存分配器中「未被引用」的块标记为可复用，并不保证
把这些显存归还给驱动。于是连续调用多个模型时，显存占用只增不减 ——
即使每个模型都已 `del` 并调用了 `empty_cache()`。

**唯一可靠的显存释放方式是进程退出。**

因此本模块把每个中型以上的模型放在独立子进程里，用完即退出，显存随之
干净归还。代价是进程启动开销（spawn + 模型加载约 1-3 秒），换来的是
可预测、可预算的显存行为。

==============================================================================
Windows 平台的额外约束
==============================================================================

Windows 只支持 spawn 启动方式（没有 fork），由此带来两个硬性限制：

1. 子进程的目标函数必须是**模块级顶层函数** —— 不能用闭包或 lambda；
2. 模型工厂用 `"module:attr"` 字符串表示，在子进程内动态导入，
   避免把模型对象本身 pickle 过去。

本实现严格遵守这两点。

==============================================================================
与 vram_budget.yaml 的关系
==============================================================================

`configs/vram_budget.yaml` 声明每个模型的**实测**峰值显存
（由 `scripts/vram_probe.py` 回填）。本模块读取该文件作为调度依据，
因此预算表与调度逻辑是数据驱动的 —— 换一张显卡只需改 yaml。
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Windows 必须使用 spawn；显式指定以免在不同平台行为不一致
_MP_CONTEXT = "spawn"


# ============================================================== 规格与预算


@dataclass
class ModelSpec:
    """单个模型的声明式规格。"""

    name: str
    factory: str
    """模型工厂的导入路径，格式 "package.module:function"。

    必须是**模块级顶层函数** —— spawn 模式下无法 pickle 闭包。
    """

    peak_mb: int
    """实测峰值显存（MiB）。由 scripts/vram_probe.py 测得后回填 yaml。"""

    can_coexist: bool = True
    """是否允许与其他模型同时常驻。大模型（RAFT / Depth）应设为 False。"""

    device: str = "cuda"
    fallback_device: str = "cpu"
    kwargs: dict[str, Any] = field(default_factory=dict)
    note: str = ""


class VramBudget:
    """显存预算管理器。

    追踪已加载模型占用的显存，并据此判断新模型能否加载。
    这是纯逻辑，不依赖 torch —— 因此可以独立单元测试。
    """

    def __init__(self, total_mb: int = 4096, reserved_mb: int = 1500) -> None:
        self.total_mb = int(total_mb)
        self.reserved_mb = int(reserved_mb)
        self._loaded: dict[str, int] = {}

    @property
    def budget_mb(self) -> int:
        """可用于模型的显存 = 总显存 - 系统保留。"""
        return max(self.total_mb - self.reserved_mb, 0)

    @property
    def used_mb(self) -> int:
        return sum(self._loaded.values())

    @property
    def free_mb(self) -> int:
        return self.budget_mb - self.used_mb

    @property
    def loaded_models(self) -> dict[str, int]:
        return dict(self._loaded)

    def can_load(self, spec: ModelSpec) -> tuple[bool, str]:
        """判断能否加载该模型。

        Returns:
            (是否可以, 原因说明)
        """
        if not spec.can_coexist and self._loaded:
            others = ", ".join(self._loaded)
            return False, f"{spec.name} 需要独占显存，但已加载: {others}"

        if spec.peak_mb > self.free_mb:
            return False, (
                f"{spec.name} 需要 {spec.peak_mb}MB，"
                f"当前可用 {self.free_mb}MB（预算 {self.budget_mb}MB，"
                f"已用 {self.used_mb}MB）"
            )
        return True, ""

    def acquire(self, spec: ModelSpec) -> None:
        """登记一个已加载的模型。

        模型不可共存时，会先自动卸载其他所有模型 —— 这模拟了
        「启动一个独占模型」的真实行为。
        """
        if not spec.can_coexist and self._loaded:
            logger.info("模型 %s 需要独占，卸载 %s", spec.name, list(self._loaded))
            self._loaded.clear()
        self._loaded[spec.name] = spec.peak_mb

    def release(self, name: str) -> None:
        self._loaded.pop(name, None)

    def clear(self) -> None:
        self._loaded.clear()

    def summary(self) -> str:
        lines = [
            f"显存预算: {self.budget_mb}MB "
            f"(总 {self.total_mb}MB - 系统保留 {self.reserved_mb}MB)",
            f"已用: {self.used_mb}MB / 可用: {self.free_mb}MB",
        ]
        for name, mb in self._loaded.items():
            lines.append(f"  - {name}: {mb}MB")
        return "\n".join(lines)

    @classmethod
    def from_yaml(cls, path: Path | str) -> tuple[VramBudget, dict[str, ModelSpec]]:
        """从 configs/vram_budget.yaml 构造预算与模型规格表。

        yaml 缺失时回退到保守默认值 —— 保证没有配置文件也能跑起来。
        """
        path = Path(path)
        if not path.is_file():
            logger.warning("未找到显存预算配置 %s，使用默认值", path)
            return cls(), {}

        import yaml  # 延迟导入：核心几何测试不需要 pyyaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        hw = raw.get("hardware", {})
        budget = cls(
            total_mb=int(hw.get("total_mb", 4096)),
            reserved_mb=int(hw.get("reserved_display_mb", 1500)),
        )

        specs: dict[str, ModelSpec] = {}
        for name, cfg in (raw.get("models") or {}).items():
            cfg = cfg or {}
            specs[name] = ModelSpec(
                name=name,
                factory=cfg.get("factory", "") or f"pano_agent.perception.detect:build_{name}",
                peak_mb=int(cfg.get("peak_mb", 0)),
                can_coexist=bool(cfg.get("can_coexist", True)),
                device=cfg.get("device", "cuda"),
                fallback_device=cfg.get("fallback_device", "cpu"),
                kwargs=cfg.get("kwargs", {}) or {},
                note=cfg.get("note", ""),
            )
        return budget, specs


# ============================================================== 子进程 worker


def _import_factory(dotted: str):
    """把 "module:attr" 解析成实际对象。"""
    if ":" not in dotted:
        raise ValueError(f"工厂路径必须形如 'module:attr'，实际为 {dotted!r}")
    module_path, attr = dotted.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _worker_main(
    conn,
    factory_path: str,
    kwargs: dict[str, Any],
) -> None:  # pragma: no cover - 需在子进程中运行
    """子进程入口。

    ⚠ **必须是模块级顶层函数** —— Windows 的 spawn 模式无法 pickle 闭包。
    这也是本模块把工厂表示为字符串而非直接传对象的原因。
    """
    try:
        factory = _import_factory(factory_path)
        instance = factory(**kwargs)
        conn.send(("ready", None))

        while True:
            msg = conn.recv()
            if msg is None:  # 关闭信号
                break
            method_name, args, kw = msg
            try:
                result = getattr(instance, method_name)(*args, **kw)
                conn.send(("ok", result))
            except Exception as exc:  # noqa: BLE001 - 需把任何异常回传给父进程
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
    except Exception as exc:  # noqa: BLE001
        try:
            conn.send(("error", f"启动失败: {type(exc).__name__}: {exc}"))
        except Exception:  # noqa: BLE001 - 管道可能已断开
            pass
    finally:
        conn.close()


class SubprocessModel:
    """在独立子进程中运行的模型代理。

    对调用方而言它像普通对象一样暴露方法；实际上每次调用都通过管道
    转发到子进程执行。进程退出时，其占用的显存被彻底归还给驱动。

    Usage:
        model = SubprocessModel(spec)
        model.start()
        result = model.call("detect", erp_image)
        model.stop()          # 显存在此刻真正释放
    """

    def __init__(self, spec: ModelSpec, *, startup_timeout_s: float = 180.0) -> None:
        self.spec = spec
        self.startup_timeout_s = startup_timeout_s
        self._process: mp.process.BaseProcess | None = None
        self._parent_conn = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        """启动子进程并等待模型加载完成。"""
        if self.is_running:
            return

        ctx = mp.get_context(_MP_CONTEXT)
        parent_conn, child_conn = ctx.Pipe()
        self._process = ctx.Process(
            target=_worker_main,
            args=(child_conn, self.spec.factory, self.spec.kwargs),
            name=f"pano-{self.spec.name}",
            daemon=True,
        )
        self._process.start()
        child_conn.close()  # 父进程只保留自己这一端

        status, payload = self._recv_with_timeout(parent_conn, self.startup_timeout_s)

        if status != "ready":
            self._parent_conn = parent_conn
            self.stop()
            from .types import SphericalDetection  # noqa: F401  仅确保模块可导入

            raise RuntimeError(f"模型 {self.spec.name} 启动失败: {payload}")

        self._parent_conn = parent_conn
        logger.info("模型 %s 已在子进程中就绪", self.spec.name)

    def call(self, method: str, *args, **kwargs) -> Any:
        """在子进程中调用方法并返回结果。

        注意：args / kwargs 需要能被 pickle 跨进程传输。
        """
        if not self.is_running or self._parent_conn is None:
            raise RuntimeError(f"模型 {self.spec.name} 未运行")

        self._parent_conn.send((method, args, kwargs))
        status, payload = self._recv_with_timeout(self._parent_conn, None)

        if status != "ok":
            raise RuntimeError(f"{self.spec.name}.{method} 执行失败: {payload}")
        return payload

    def stop(self, timeout_s: float = 10.0) -> None:
        """停止子进程 —— **显存在此刻才真正释放**。"""
        if self._parent_conn is not None:
            try:
                self._parent_conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            self._parent_conn.close()
            self._parent_conn = None

        if self._process is not None:
            self._process.join(timeout=timeout_s)
            if self._process.is_alive():
                logger.warning("模型 %s 未在超时内退出，强制终止", self.spec.name)
                self._process.terminate()
                self._process.join(timeout=2.0)
            self._process = None

    @staticmethod
    def _recv_with_timeout(conn, timeout_s: float | None) -> tuple[str, Any]:
        if timeout_s is None:
            return conn.recv()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if conn.poll(0.2):
                return conn.recv()
        return "error", f"等待超时（{timeout_s:.0f}s）"

    def __enter__(self) -> SubprocessModel:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


# ============================================================== 注册表


class ModelRegistry:
    """按显存预算调度多个模型的启停。

    提供两种使用方式：
    - `lease(name)`：上下文管理器，进入时启动、退出时**立即释放显存**；
    - `run(name, method, *args)`：一次性调用的便捷封装。
    """

    def __init__(
        self,
        budget: VramBudget,
        specs: dict[str, ModelSpec],
        *,
        on_oom: str = "cuda_to_cpu",
    ) -> None:
        self.budget = budget
        self.specs = specs
        self.on_oom = on_oom
        self._active: dict[str, SubprocessModel] = {}

    def register(self, spec: ModelSpec) -> None:
        self.specs[spec.name] = spec

    def lease(self, name: str) -> _ModelLease:
        """以上下文管理器方式使用模型。

        Usage:
            with registry.lease("yolo11n") as model:
                dets = model.call("detect", frame)
            # 到这里显存已归还
        """
        if name not in self.specs:
            raise KeyError(f"未注册的模型 {name!r}，可用: {list(self.specs)}")
        return _ModelLease(self, self.specs[name])

    def run(self, name: str, method: str, *args, **kwargs) -> Any:
        """一次性调用：启动 → 调用 → 停止。"""
        with self.lease(name) as model:
            return model.call(method, *args, **kwargs)

    # ------------------------------------------------------------ 内部

    def _acquire(self, spec: ModelSpec) -> SubprocessModel:
        if spec.name in self._active:
            return self._active[spec.name]

        ok, reason = self.budget.can_load(spec)
        if not ok:
            # 独占模型：先清空再试
            if not spec.can_coexist:
                self._evict_all()
                ok, reason = self.budget.can_load(spec)

        if not ok:
            raise MemoryError(f"显存不足，无法加载 {spec.name}: {reason}")

        model = SubprocessModel(spec)
        model.start()
        self.budget.acquire(spec)
        self._active[spec.name] = model
        return model

    def _release(self, name: str) -> None:
        model = self._active.pop(name, None)
        if model is not None:
            model.stop()
            self.budget.release(name)

    def _evict_all(self) -> None:
        for name in list(self._active):
            self._release(name)

    def shutdown(self) -> None:
        """释放所有模型。**必须调用**，否则子进程可能在程序退出后残留。"""
        self._evict_all()
        self.budget.clear()

    def __enter__(self) -> ModelRegistry:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()


class _ModelLease:
    """`ModelRegistry.lease` 返回的上下文管理器。"""

    def __init__(self, registry: ModelRegistry, spec: ModelSpec) -> None:
        self._registry = registry
        self._spec = spec
        self._model: SubprocessModel | None = None

    def __enter__(self) -> SubprocessModel:
        self._model = self._registry._acquire(self._spec)
        return self._model

    def __exit__(self, *exc_info: object) -> None:
        # 每次 lease 结束都释放 —— 这正是「显存只在进程退出时归还」的落点
        self._registry._release(self._spec.name)


# ============================================================== 自检模型


class EchoModel:
    """用于验证调度通路的最小模型（不加载任何真实权重）。

    存在的意义：子进程调度涉及管道通信、异常传递、超时、生命周期，
    这些逻辑与「模型本身是什么」无关。用一个零成本的假模型来测它们，
    测试可以既快又稳定，不必依赖 CUDA 或模型权重。

    同时也是排查环境问题的第一个探针 —— 当真实模型启动失败时，
    先跑一次 EchoModel 就能区分「调度器坏了」还是「模型加载坏了」。
    """

    def __init__(self, tag: str = "default", **_ignored: Any) -> None:
        self.tag = tag
        self.call_count = 0

    def echo(self, value: Any) -> dict[str, Any]:
        self.call_count += 1
        return {"tag": self.tag, "value": value, "count": self.call_count}

    def add(self, a: float, b: float) -> float:
        return a + b

    def fail(self) -> None:
        """故意抛异常，用于验证子进程异常能否被父进程看到。"""
        raise ValueError("故意失败：验证异常回传")


def build_echo_model(**kwargs: Any) -> EchoModel:
    """EchoModel 的顶层工厂 —— spawn 模式要求目标可 pickle。"""
    return EchoModel(**kwargs)


def echo_spec(peak_mb: int = 10, **kwargs: Any) -> ModelSpec:
    """构造一个指向 EchoModel 的规格，便于测试与自检。"""
    return ModelSpec(
        name="echo",
        factory=f"{__name__}:build_echo_model",
        peak_mb=peak_mb,
        kwargs=kwargs,
    )
