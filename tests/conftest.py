"""pytest 公共配置。

把 src/ 加入 sys.path，使测试可以在 `pip install -e .` 之前直接运行 ——
几何层是纯 numpy 的，不应为了跑测试而先装一整套深度学习依赖。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
