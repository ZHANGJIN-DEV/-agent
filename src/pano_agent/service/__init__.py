"""服务层：把完整 pipeline 暴露成 HTTP 接口。

设计取舍见 `api` 模块文档 —— 刻意不引入 Celery/Redis 与数据库，
让 Demo 能被真正跑起来（一条 `uvicorn` 命令启动）。
"""

from .pipeline import PipelineConfig, PipelineResult, StageReport, run_pipeline

__all__ = ["PipelineConfig", "PipelineResult", "StageReport", "run_pipeline"]
