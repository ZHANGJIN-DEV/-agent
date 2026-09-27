"""FastAPI 服务 —— 把 pipeline 暴露成 HTTP 接口。

==============================================================================
接口
==============================================================================

    GET  /health                  健康检查
    GET  /api/examples            列出可用示例素材
    POST /api/jobs                提交任务 → {job_id}
    GET  /api/jobs/{job_id}       查询状态与进度
    GET  /api/jobs/{job_id}/result   下载成片

==============================================================================
两个"刻意不用"的设计决策
==============================================================================

**1. 不用 Celery / Redis**
本任务是单机串行的（一块 GPU、一次一个视频）。Celery 会引入 Redis 服务、
worker 进程、任务序列化，换来的分布式能力在此毫无用武之地。
FastAPI 的 `BackgroundTasks` 足够承担"提交后异步跑、期间可查状态"。

**2. 不用数据库**
任务状态存在内存字典里，重启即清空。对单机 Demo 来说这完全可以接受；
引入 SQLAlchemy + Alembic 迁移会让部署复杂度陡增，而收益为零。

两处都是为了**让 Demo 能被真正跑起来** —— 一个需要先起三个服务才能
演示的系统，实际使用率会远低于一个 `uvicorn` 就能启动的系统。

（若日后要支持多用户/持久化，再评估引入这些组件，
那时会带着真实约束去选型，而不是凭"生产环境都这么做"的想象。）
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .pipeline import PipelineConfig, PipelineResult, run_pipeline

JobStatus = Literal["pending", "running", "done", "failed"]

# 默认数据路径
DEFAULT_DATA_ROOT = Path("C:/pano_data")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "assets" / "jobs"


# ============================================================== 任务状态


@dataclass
class Job:
    """一个任务的状态。"""

    job_id: str
    status: JobStatus = "pending"
    stage: str = ""
    progress: float = 0.0
    detail: str = ""
    error: str = ""
    created_at: str = ""
    result: PipelineResult | None = None

    def to_dict(self) -> dict:
        data: dict = {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "detail": self.detail,
            "created_at": self.created_at,
        }
        if self.error:
            data["error"] = self.error
        if self.status == "done" and self.result is not None:
            data["result"] = self.result.to_dict()
            data["download_url"] = f"/api/jobs/{self.job_id}/result"
        return data


class JobStore:
    """内存任务表（线程安全）。

    用锁保护是因为 pipeline 跑在 `BackgroundTasks` 的线程里，
    而查询请求来自事件循环线程 —— 两者会并发访问。
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)

    def all_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())


# ============================================================== 请求模型


class CreateJobRequest(BaseModel):
    """提交任务的请求体。"""

    instance_name: str = Field(
        default="panorama_3518.png",
        description="实例分割图文件名（见 /api/examples）",
    )
    instruction: str = Field(
        default="剪一个突出画面中最显眼主体的短片",
        max_length=500,
    )
    num_frames: int = Field(default=45, ge=10, le=300)
    fps: float = Field(default=15.0, gt=0, le=60)
    width: int = Field(default=960, ge=320, le=1920)
    height: int = Field(default=540, ge=180, le=1080)
    shift_px: int = Field(default=14, ge=1, le=100)
    use_llm: bool = False


# ============================================================== 应用


def create_app(
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> FastAPI:
    """构造 FastAPI 应用。

    用工厂函数而非模块级单例，便于测试时注入不同的数据目录。
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    store = JobStore()
    app = FastAPI(
        title="PanoDirector API",
        description="360° 全景视频自动运镜 Agent",
        version="0.1.0",
    )

    instance_dir = data_root / "datasets" / "nyc_instance" / "nyc_instance_panorama"
    raw_dir = data_root / "datasets" / "nyc_raw" / "nyc_Raw"

    # ---------------------------------------------------------- 基础

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "instance_dir_available": instance_dir.is_dir(),
            "raw_dir_available": raw_dir.is_dir(),
            "jobs_total": len(store.all_jobs()),
        }

    @app.get("/api/examples")
    def list_examples(limit: int = 20) -> dict:
        """列出可用示例素材。"""
        if not instance_dir.is_dir():
            return {"examples": [], "note": f"数据目录不存在: {instance_dir}"}

        files = sorted(instance_dir.glob("*.png"))
        # 均匀取样，避免总是返回前 N 个
        step = max(len(files) // max(limit, 1), 1)
        picked = files[::step][:limit]

        return {
            "examples": [
                {
                    "name": f.name,
                    "has_raw": (raw_dir / f.name).is_file(),
                }
                for f in picked
            ],
            "total_available": len(files),
        }

    # ---------------------------------------------------------- 任务

    @app.post("/api/jobs", status_code=202)
    def create_job(
        req: CreateJobRequest, background: BackgroundTasks
    ) -> dict:
        """提交一个渲染任务，立即返回 job_id。"""
        instance_path = instance_dir / req.instance_name
        if not instance_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"实例图不存在: {req.instance_name}",
            )

        raw_path = raw_dir / req.instance_name
        job = store.create()
        out_path = output_root / f"{job.job_id}.mp4"

        config = PipelineConfig(
            fps=req.fps,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            shift_px=req.shift_px,
            use_llm=req.use_llm,
            instruction=req.instruction,
        )

        background.add_task(
            _run_job,
            store,
            job.job_id,
            instance_path,
            raw_path if raw_path.is_file() else None,
            out_path,
            config,
        )

        return {"job_id": job.job_id, "status": "pending"}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/result")
    def get_result(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"任务尚未完成（{job.status}）")
        if job.result is None or job.result.output_path is None:
            raise HTTPException(status_code=500, detail="任务完成但没有产物")

        path = Path(job.result.output_path)
        if not path.is_file():
            raise HTTPException(status_code=410, detail="产物文件已不存在")
        return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")

    @app.get("/api/jobs")
    def list_jobs(limit: int = 20) -> dict:
        jobs = sorted(store.all_jobs(), key=lambda j: j.created_at, reverse=True)
        return {"jobs": [j.to_dict() for j in jobs[:limit]]}

    # ---------------------------------------------------------- 静态前端
    # 单文件 HTML，无需构建步骤 —— 打开 http://localhost:8000 即可使用。
    #
    # ⚠ 必须**最后**挂载：StaticFiles 挂在 "/" 上会兜住所有未匹配路径，
    # 若放在 API 路由之前，那些路由就永远不会被命中。
    web_dir = Path(__file__).resolve().parents[3] / "web"
    if web_dir.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app


def _run_job(
    store: JobStore,
    job_id: str,
    instance_path: Path,
    raw_path: Path | None,
    output_path: Path,
    config: PipelineConfig,
) -> None:
    """在后台线程里执行任务。

    **任何异常都必须被捕获并记入任务状态** —— 若让它冒泡，
    BackgroundTasks 会静默吞掉，用户看到的是任务永远卡在 running。
    """

    def on_progress(stage: str, pct: float, detail: str) -> None:
        store.update(job_id, stage=stage, progress=pct, detail=detail)

    store.update(job_id, status="running", stage="启动", progress=0.0)

    try:
        result = run_pipeline(
            instance_path,
            raw_path,
            output_path,
            config=config,
            on_progress=on_progress,
        )
        store.update(
            job_id,
            status="done",
            stage="完成",
            progress=1.0,
            detail=f"生成 {result.render_frames} 帧",
            result=result,
        )
    except Exception as exc:  # noqa: BLE001 - 必须捕获所有异常
        store.update(
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            detail="任务失败",
        )


# 供 uvicorn 直接引用：uvicorn pano_agent.service.api:app
app = create_app()
