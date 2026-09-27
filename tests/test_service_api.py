"""服务层测试。

用 `tmp_path` 构造最小假数据集，避免依赖那 12GB 的真实数据 ——
测试应当能被任何人在任何机器上跑起来。

唯一需要真实数据的是端到端那条，它会在数据缺失时自动跳过。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pano_agent.service.api import create_app
from pano_agent.service.pipeline import PipelineConfig, run_pipeline

REAL_DATA = Path("C:/pano_data/datasets/nyc_instance/nyc_instance_panorama")

# ------------------------------------------------------------------ 测试夹具


def make_fake_instance_map(path: Path, *, seed: int = 0) -> None:
    """构造一张最小的实例分割图（格式与真实数据一致）。

    A 通道 = 类别 ID（24 = building）
    RGB    = 实例彩色编码
    """
    from PIL import Image

    h, w = 256, 512
    arr = np.zeros((h, w, 4), dtype=np.uint8)

    # 三个建筑实例
    for i, x in enumerate([40, 200, 360]):
        arr[60:160, x : x + 90, :3] = [80 + i * 50, 120, 200 - i * 40]
        arr[60:160, x : x + 90, 3] = 24
    # 一个路灯
    arr[200:240, 250:270, :3] = [200, 200, 60]
    arr[200:240, 250:270, 3] = 10

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def make_fake_raw(path: Path) -> None:
    """构造一张最小的真实 RGB 全景图。"""
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = rng.integers(40, 220, (256, 512, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """构造一个最小的假数据目录。"""
    inst_dir = tmp_path / "datasets" / "nyc_instance" / "nyc_instance_panorama"
    raw_dir = tmp_path / "datasets" / "nyc_raw" / "nyc_Raw"
    make_fake_instance_map(inst_dir / "test_a.png")
    make_fake_instance_map(inst_dir / "test_b.png", seed=1)
    make_fake_raw(raw_dir / "test_a.png")
    return tmp_path


@pytest.fixture
def client(data_root: Path, tmp_path: Path) -> TestClient:
    app = create_app(data_root=data_root, output_root=tmp_path / "outputs")
    return TestClient(app)


# ============================================================== 基础接口


class TestBasicEndpoints:
    def test_health(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "ok"
        assert d["instance_dir_available"] is True

    def test_health_reports_missing_data(self, tmp_path: Path):
        """数据目录不存在时也应有响应（而非 500）。"""
        app = create_app(data_root=tmp_path / "nonexistent", output_root=tmp_path / "out")
        r = TestClient(app).get("/health")
        assert r.status_code == 200
        assert r.json()["instance_dir_available"] is False

    def test_examples_lists_files(self, client: TestClient):
        r = client.get("/api/examples")
        assert r.status_code == 200
        names = {e["name"] for e in r.json()["examples"]}
        assert "test_a.png" in names

    def test_examples_reports_raw_availability(self, client: TestClient):
        """有对应 RGB 的应标出来 —— 前端据此决定用哪个素材。"""
        examples = {e["name"]: e for e in client.get("/api/examples").json()["examples"]}
        assert examples["test_a.png"]["has_raw"] is True
        assert examples["test_b.png"]["has_raw"] is False

    def test_examples_handles_missing_dir(self, tmp_path: Path):
        app = create_app(data_root=tmp_path / "nope", output_root=tmp_path / "out")
        r = TestClient(app).get("/api/examples")
        assert r.status_code == 200
        assert r.json()["examples"] == []


# ============================================================== 任务提交


class TestJobSubmission:
    def test_create_job_returns_id(self, client: TestClient):
        r = client.post("/api/jobs", json={"instance_name": "test_a.png"})
        assert r.status_code == 202
        assert "job_id" in r.json()

    def test_rejects_unknown_instance(self, client: TestClient):
        r = client.post("/api/jobs", json={"instance_name": "no_such_file.png"})
        assert r.status_code == 404
        assert "不存在" in r.json()["detail"]

    def test_rejects_invalid_params(self, client: TestClient):
        """参数越界应被 pydantic 拦下。"""
        for bad in (
            {"num_frames": 5},        # < 10
            {"num_frames": 9999},     # > 300
            {"fps": 0},               # <= 0
            {"shift_px": -1},         # < 1
        ):
            r = client.post("/api/jobs", json={"instance_name": "test_a.png", **bad})
            assert r.status_code == 422, f"参数 {bad} 应被拒绝"

    def test_rejects_overlong_instruction(self, client: TestClient):
        r = client.post(
            "/api/jobs", json={"instance_name": "test_a.png", "instruction": "x" * 600}
        )
        assert r.status_code == 422

    def test_unknown_job_returns_404(self, client: TestClient):
        assert client.get("/api/jobs/nonexistent").status_code == 404
        assert client.get("/api/jobs/nonexistent/result").status_code == 404


# ============================================================== 任务生命周期


class TestJobLifecycle:
    def _wait(self, client: TestClient, job_id: str, timeout_s: float = 90.0) -> dict:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            d = client.get(f"/api/jobs/{job_id}").json()
            if d["status"] in ("done", "failed"):
                return d
            time.sleep(0.3)
        raise AssertionError(f"任务超时未完成: {job_id}")

    def test_job_runs_to_completion(self, client: TestClient):
        """★ 完整跑一个任务：提交 → 轮询 → 完成。"""
        job_id = client.post(
            "/api/jobs",
            json={"instance_name": "test_a.png", "num_frames": 10, "fps": 10.0,
                  "width": 320, "height": 180},
        ).json()["job_id"]

        d = self._wait(client, job_id)
        assert d["status"] == "done", f"任务失败: {d.get('error')}"

        result = d["result"]
        assert result["render_frames"] > 0
        assert result["num_tracks"] > 0
        assert len(result["stages"]) == 6, "应记录全部 6 个阶段"
        assert Path(result["output_path"]).is_file()

    def test_progress_is_reported(self, client: TestClient):
        """验证进度机制：既有实时推送，也有完整的阶段记录。

        注意不要断言"轮询一定能看到某个中间阶段" —— 小任务的执行时间
        可能短于轮询间隔，那种断言会随机失败（本质是在测时序而非测功能）。

        可靠的验证分两处：
            1. 完成后的 `result.stages` —— 一定包含全部阶段的耗时记录
            2. 轮询期间至少观察到过一次阶段变化 —— 证明实时推送在工作
        """
        job_id = client.post(
            "/api/jobs",
            json={"instance_name": "test_a.png", "num_frames": 10, "fps": 10.0,
                  "width": 320, "height": 180},
        ).json()["job_id"]

        seen_stages: set[str] = set()
        deadline = time.time() + 90
        while time.time() < deadline:
            d = client.get(f"/api/jobs/{job_id}").json()
            if d.get("stage"):
                seen_stages.add(d["stage"])
            if d["status"] in ("done", "failed"):
                break
            time.sleep(0.05)

        final = client.get(f"/api/jobs/{job_id}").json()
        assert final["status"] == "done", f"任务失败: {final.get('error')}"

        # (1) 完整阶段记录
        stage_names = {s["stage"] for s in final["result"]["stages"]}
        assert "实例提取" in stage_names
        assert "渲染成片" in stage_names
        assert all(s["duration_s"] >= 0 for s in final["result"]["stages"])

        # (2) 实时推送确实发生过
        assert len(seen_stages) >= 1, "轮询期间未观察到任何阶段信息"

    def test_result_downloadable(self, client: TestClient):
        job_id = client.post(
            "/api/jobs",
            json={"instance_name": "test_a.png", "num_frames": 10, "fps": 10.0,
                  "width": 320, "height": 180},
        ).json()["job_id"]
        self._wait(client, job_id)

        r = client.get(f"/api/jobs/{job_id}/result")
        assert r.status_code == 200
        assert r.headers["content-type"] == "video/mp4"
        assert len(r.content) > 1000, "产物不应为空"

    def test_result_before_completion_returns_409(self, client: TestClient):
        """未完成时请求产物应返回 409 而非 500。"""
        job_id = client.post(
            "/api/jobs",
            json={"instance_name": "test_a.png", "num_frames": 30, "fps": 30.0},
        ).json()["job_id"]

        r = client.get(f"/api/jobs/{job_id}/result")
        # 可能在极短时间内已完成，两种都接受，但不能是 5xx
        assert r.status_code in (200, 409)

    def test_jobs_list(self, client: TestClient):
        client.post("/api/jobs", json={"instance_name": "test_a.png"})
        r = client.get("/api/jobs")
        assert r.status_code == 200
        assert len(r.json()["jobs"]) >= 1


# ============================================================== 失败处理


class TestFailureHandling:
    def test_failure_is_recorded_not_swallowed(self, client: TestClient, tmp_path: Path):
        """★ 任务失败必须记入状态。

        若异常冒泡出 BackgroundTasks，会被静默吞掉 ——
        用户看到的是任务永远卡在 running，这是最难排查的一类问题。
        """
        # 构造一张没有前景实例的图（全背景类）
        from PIL import Image

        inst_dir = tmp_path / "datasets" / "nyc_instance" / "nyc_instance_panorama"
        arr = np.zeros((256, 512, 4), dtype=np.uint8)
        arr[..., :3] = [100, 150, 200]
        arr[..., 3] = 3  # sky —— 会被当作背景过滤掉
        Image.fromarray(arr).save(inst_dir / "empty.png")

        job_id = client.post(
            "/api/jobs", json={"instance_name": "empty.png", "num_frames": 10}
        ).json()["job_id"]

        deadline = time.time() + 60
        d = {}
        while time.time() < deadline:
            d = client.get(f"/api/jobs/{job_id}").json()
            if d["status"] in ("done", "failed"):
                break
            time.sleep(0.2)

        assert d["status"] == "failed", "没有实例时应失败而非卡住"
        assert d.get("error"), "失败必须带错误信息"


# ============================================================== pipeline 函数


class TestPipelineFunction:
    def test_runs_standalone(self, tmp_path: Path):
        """pipeline 应能脱离 HTTP 独立调用（CLI/测试都需要）。"""
        inst = tmp_path / "inst.png"
        make_fake_instance_map(inst)

        result = run_pipeline(
            inst, None, tmp_path / "out.mp4",
            config=PipelineConfig(num_frames=10, fps=10.0, width=320, height=180),
        )

        assert result.output_path is not None
        assert result.output_path.is_file()
        assert result.render_frames == 10
        assert result.num_tracks > 0

    def test_progress_callback_invoked(self, tmp_path: Path):
        inst = tmp_path / "inst.png"
        make_fake_instance_map(inst)

        calls: list[tuple[str, float]] = []
        run_pipeline(
            inst, None, tmp_path / "out.mp4",
            config=PipelineConfig(num_frames=8, fps=10.0, width=320, height=180),
            on_progress=lambda s, p, d: calls.append((s, p)),
        )

        stages = {s for s, _ in calls}
        assert "实例提取" in stages
        assert "渲染成片" in stages

    def test_config_is_dataclass_not_argparse(self, tmp_path: Path):
        """配置应当是普通 dataclass —— 服务层不该依赖 argparse。"""
        import dataclasses

        assert dataclasses.is_dataclass(PipelineConfig)
        assert dataclasses.is_dataclass(PipelineConfig(num_frames=5).__class__)

    def test_raises_on_empty_instances(self, tmp_path: Path):
        from PIL import Image

        arr = np.zeros((256, 512, 4), dtype=np.uint8)
        arr[..., 3] = 3
        inst = tmp_path / "empty.png"
        Image.fromarray(arr).save(inst)

        with pytest.raises(ValueError, match="没有可用的前景实例"):
            run_pipeline(inst, None, tmp_path / "out.mp4",
                         config=PipelineConfig(num_frames=5))


# ============================================================== 真实数据（可选）


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="真实数据集不可用")
class TestWithRealData:
    def test_real_instance_produces_video(self, tmp_path: Path):
        """用真实实例数据跑一次 —— 这是最有价值的集成测试。"""
        real = REAL_DATA / "panorama_3518.png"
        if not real.is_file():
            pytest.skip("示例文件不存在")

        result = run_pipeline(
            real, None, tmp_path / "real.mp4",
            config=PipelineConfig(num_frames=15, fps=10.0, width=480, height=270),
        )

        assert result.output_path is not None and result.output_path.is_file()
        assert result.num_tracks > 10, "真实数据应提取出较多轨迹"
        assert result.output_path.stat().st_size > 5000
