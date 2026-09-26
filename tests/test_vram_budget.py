"""显存预算管理与子进程模型调度的测试。

预算部分是纯逻辑，完全独立可测；
子进程部分用 `EchoModel`（零成本假模型）验证通路 ——
管道通信、异常回传、生命周期这些逻辑与"模型本身是什么"无关，
用假模型测既快又稳定，且不依赖 CUDA。
"""

from __future__ import annotations

import pytest

from pano_agent.perception.registry import (
    EchoModel,
    ModelRegistry,
    ModelSpec,
    SubprocessModel,
    VramBudget,
    echo_spec,
)


def spec(name: str, peak_mb: int, *, coexist: bool = True) -> ModelSpec:
    return ModelSpec(name=name, factory="pano_agent.perception.registry:build_echo_model",
                     peak_mb=peak_mb, can_coexist=coexist)


# ============================================================== 预算


class TestVramBudget:
    def test_budget_excludes_reserved_display(self):
        """预算必须扣掉桌面合成占用 —— 这是 4GB 卡上的真实约束。"""
        b = VramBudget(total_mb=4096, reserved_mb=1500)
        assert b.budget_mb == 2596
        assert b.free_mb == 2596
        assert b.used_mb == 0

    def test_budget_never_negative(self):
        b = VramBudget(total_mb=1024, reserved_mb=1500)
        assert b.budget_mb == 0

    def test_can_load_within_budget(self):
        b = VramBudget()
        ok, reason = b.can_load(spec("small", 400))
        assert ok and reason == ""

    def test_cannot_load_over_budget(self):
        b = VramBudget()
        ok, reason = b.can_load(spec("huge", 5000))
        assert not ok
        assert "需要" in reason and "可用" in reason

    def test_second_model_blocked_when_budget_exhausted(self):
        b = VramBudget(total_mb=4096, reserved_mb=1500)  # 预算 2596
        b.acquire(spec("a", 2000))
        ok, _ = b.can_load(spec("b", 1000))
        assert not ok, "剩余 596MB 不应能加载 1000MB 的模型"

    def test_exclusive_model_blocked_by_others(self):
        b = VramBudget()
        b.acquire(spec("a", 300))
        ok, reason = b.can_load(spec("big", 300, coexist=False))
        assert not ok
        assert "独占" in reason

    def test_acquire_exclusive_evicts_others(self):
        """加载独占模型时应自动清空其他模型。"""
        b = VramBudget()
        b.acquire(spec("a", 300))
        b.acquire(spec("b", 400))
        assert b.used_mb == 700

        b.acquire(spec("raft", 1100, coexist=False))
        assert b.loaded_models == {"raft": 1100}, "独占模型应清空其他模型"

    def test_release_frees_budget(self):
        b = VramBudget()
        b.acquire(spec("a", 800))
        assert b.free_mb == b.budget_mb - 800
        b.release("a")
        assert b.free_mb == b.budget_mb

    def test_release_unknown_is_noop(self):
        b = VramBudget()
        b.release("nonexistent")
        assert b.used_mb == 0

    def test_realistic_scenario_from_vram_budget_yaml(self):
        """复现 configs/vram_budget.yaml 里的真实组合。"""
        b = VramBudget(total_mb=4096, reserved_mb=1500)

        # 三个常驻小模型可以共存
        for name, mb in [("yolo11n", 400), ("unet_saliency", 350), ("clip", 320)]:
            ok, _ = b.can_load(spec(name, mb))
            assert ok, f"{name} 应能加载"
            b.acquire(spec(name, mb))

        assert b.used_mb == 1070

        # RAFT 是独占模型，应能驱逐它们
        b.acquire(spec("raft_small", 1100, coexist=False))
        assert b.loaded_models == {"raft_small": 1100}

    def test_summary_is_readable(self):
        b = VramBudget()
        b.acquire(spec("a", 400))
        text = b.summary()
        assert "2596MB" in text and "a" in text


class TestBudgetFromYaml:
    def test_loads_real_config(self):
        from pathlib import Path

        cfg = Path(__file__).resolve().parents[1] / "configs" / "vram_budget.yaml"
        if not cfg.is_file():
            pytest.skip("vram_budget.yaml 不存在")

        budget, specs = VramBudget.from_yaml(cfg)

        assert budget.total_mb == 4096
        # 实测（torch 装好后）：mem_get_info 报告可用 3306MB，
        # 故非 CUDA 占用约 800MB —— 与 nvidia-smi 的 1474MB 口径不同
        assert budget.reserved_mb == 800
        assert budget.budget_mb == 3296
        # 配置里应包含这些关键模型
        for name in ("yolo11n", "raft_small", "depth_anything_v2_small"):
            assert name in specs, f"{name} 应在预算表中"

        # 大模型必须标记为不可共存
        assert specs["raft_small"].can_coexist is False
        assert specs["yolo11n"].can_coexist is True

    def test_missing_file_falls_back_to_defaults(self, tmp_path):
        budget, specs = VramBudget.from_yaml(tmp_path / "nope.yaml")
        assert budget.total_mb == 4096
        assert specs == {}


# ============================================================== 子进程


class TestSubprocessModel:
    def test_start_call_stop_roundtrip(self):
        with SubprocessModel(echo_spec()) as model:
            assert model.is_running
            assert model.call("add", 2.0, 3.0) == pytest.approx(5.0)
        assert not model.is_running, "退出上下文后子进程应已终止"

    def test_state_persists_within_process(self):
        """子进程内状态应当保持 —— 证明是同一个实例在处理多次调用。"""
        with SubprocessModel(echo_spec()) as model:
            assert model.call("echo", "a")["count"] == 1
            assert model.call("echo", "b")["count"] == 2

    def test_kwargs_reach_the_model(self):
        with SubprocessModel(echo_spec(tag="custom")) as model:
            assert model.call("echo", 1)["tag"] == "custom"

    def test_exception_propagates_to_parent(self):
        """子进程内的异常必须能被父进程感知，而不是静默丢失。"""
        with SubprocessModel(echo_spec()) as model:
            with pytest.raises(RuntimeError, match="故意失败"):
                model.call("fail")

    def test_call_before_start_raises(self):
        model = SubprocessModel(echo_spec())
        with pytest.raises(RuntimeError, match="未运行"):
            model.call("add", 1, 2)

    def test_bad_factory_raises_on_start(self):
        bad = ModelSpec(name="bad", factory="no.such.module:factory", peak_mb=10)
        model = SubprocessModel(bad, startup_timeout_s=15)
        with pytest.raises(RuntimeError, match="启动失败"):
            model.start()
        model.stop()

    def test_malformed_factory_path_raises(self):
        bad = ModelSpec(name="bad", factory="missing_colon", peak_mb=10)
        model = SubprocessModel(bad, startup_timeout_s=15)
        with pytest.raises(RuntimeError):
            model.start()
        model.stop()

    def test_stop_is_idempotent(self):
        model = SubprocessModel(echo_spec())
        model.start()
        model.stop()
        model.stop()  # 再次调用不应抛异常
        assert not model.is_running


# ============================================================== 注册表


class TestModelRegistry:
    def test_run_convenience_releases_after(self):
        budget = VramBudget()
        registry = ModelRegistry(budget, {"echo": echo_spec(peak_mb=100)})

        assert registry.run("echo", "add", 1, 2) == pytest.approx(3.0)
        assert budget.used_mb == 0, "一次性调用结束后显存应释放"

        registry.shutdown()

    def test_lease_releases_on_exit(self):
        budget = VramBudget()
        registry = ModelRegistry(budget, {"echo": echo_spec(peak_mb=100)})

        with registry.lease("echo") as model:
            assert budget.used_mb == 100
            assert model.call("add", 1, 1) == pytest.approx(2.0)

        assert budget.used_mb == 0, "lease 退出后显存必须释放"
        registry.shutdown()

    def test_specs_registered_via_budget_yaml_style(self):
        """从 yaml 读入的真实规格也应能被调度。"""
        from pathlib import Path

        cfg = Path(__file__).resolve().parents[1] / "configs" / "vram_budget.yaml"
        if not cfg.is_file():
            pytest.skip("vram_budget.yaml 不存在")

        budget, specs = VramBudget.from_yaml(cfg)
        # 把 yolo11n 的工厂替换成 echo，验证调度逻辑本身
        specs["yolo11n"] = echo_spec(peak_mb=specs["yolo11n"].peak_mb)

        registry = ModelRegistry(budget, specs)
        assert registry.run("yolo11n", "add", 1, 2) == pytest.approx(3.0)
        registry.shutdown()

    def test_unknown_model_raises_keyerror(self):
        registry = ModelRegistry(VramBudget(), {})
        with pytest.raises(KeyError, match="未注册"):
            registry.lease("nope")

    def test_shutdown_clears_everything(self):
        budget = VramBudget()
        registry = ModelRegistry(budget, {"echo": echo_spec(peak_mb=200)})
        with registry.lease("echo"):
            pass
        registry.shutdown()
        assert budget.used_mb == 0
        assert budget.loaded_models == {}

    def test_echo_model_is_usable_directly(self):
        """EchoModel 本身应可独立使用（不经过子进程），便于快速自检。"""
        model = EchoModel(tag="direct")
        assert model.echo("x")["tag"] == "direct"
        assert model.add(2, 3) == 5
