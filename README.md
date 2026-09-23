# PanoDirector · 360° 全景视频自动运镜 Agent

> 给一段等距柱状投影（ERP）全景视频和一句自然语言指令，输出一段电影感的 NFOV 成片，
> 并给出可解释的运镜决策。

```
输入：全景视频.mp4 + "剪一个突出滑板者的 15 秒高能片段"
输出：16:9 成片.mp4 + 轨迹曲线 + 决策理由
```

---

## 核心洞察

整个项目建立在一句第一性原理上：

> **在 ERP 图像上做的任何像素级运算（IoU、距离、池化、速度）都是错的，
> 因为 ERP 是球面的非均匀采样。所有几何量必须以「球面角度」为单位存储和计算。**

ERP 把球面按经度、纬度线性映射到图像坐标，导致 x 方向每像素对应的球面弧长正比于 `cos φ`：
赤道处 10 像素 ≈ 0.3°，高纬处 10 像素 ≈ 3°。面积畸变因子 `1/cos φ` 在极点发散——
极点这个**单点**被展开成了整条边。

这条原理贯穿代码、评测和技术报告的每一层：

| 层面 | 常规做法（错） | 本项目做法 |
|---|---|---|
| 跟踪关联 | 像素欧氏距离 | great-circle 距离 |
| yaw 滤波 | 在 [-180,180) 上直接滤波 → 边界跳变 | `np.unwrap` lift 到 ℝ 后滤波 |
| 检测框 IoU | 像素 IoU | 球面 IoU（Fibonacci 采样 + 立体角加权） |
| 显著性损失 | 逐像素 MSE | `cos φ` 加权 |
| 轨迹求解 | 在 S¹ 上优化 → 跨越接缝时跳变 | lift 到 ℝ 后 L-BFGS-B |

---

## 架构

```
L6 服务层   FastAPI(async) + Celery + Redis + React18/TS Web Demo + Docker Compose
L5 Agent 层 Planner(LLM) ⇄ Tool Registry ⇄ Reflector(VLM) → Edit DSL (pydantic)
L4 优化层   Trajectory Compiler → L-BFGS-B on unwrapped yaw/pitch/fov → 轨迹曲线
L3 表示层 ★ Panorama Scene Graph (PSG)：符号层(给LLM) ⇄ track_id 对齐 ⇄ 数值层(给优化器)
L2 感知层   YOLO11n → 球面Kalman跟踪(跨边界) → 显著性UNet → 光流 DIS/RAFT
            调度：VRAM Budget Scheduler + 子进程模型隔离
L1 几何层 ★ sphere / projection / region / camera（地基，最先写）
L0 输入层   ERP视频 → ffmpeg(nvdec硬解) → 2K中间格式 → 抽帧
```

### 技术核心：PSG 的双视图设计

LLM 需要少 token、语义化、可读；优化器需要全精度、逐帧、可微。两者需求矛盾，
通过**一个图、两种视图、`track_id` 严格对齐**解决：

- **符号层** `to_llm_view()` → 纯 JSON：`track_id / caption / 属性标签 / 球面统计量`（30s 视频约 2-4k token）
- **数值层** `numeric_ref` → `.npz`（mmap 零拷贝）：逐帧 `yaw/pitch/w/h/rot/conf`，全为角度单位

**强制类型隔离**：`EditDSL` 的 pydantic schema 里**根本没有 float 坐标字段**，
LLM 只能引用 `track_id`。这是「LLM 输出**约束**（可校验、可拒绝、可兜底）而非**解**（不可校验）」
的立论基础。

---

## 环境搭建

### 关键约束

| 项 | 实测值 | 说明 |
|---|---|---|
| GPU | RTX 3050 Laptop 4GB | Ampere sm_86 |
| **可用显存** | **约 2.5GB** | 桌面合成已占 1474MB，**按 2.5GB 设计而非 4GB** |
| 数据目录 | `C:\pano_data` | D 盘仅剩 58G，放不下 4K 全景视频 |
| 代码目录 | `D:\pano-agent` | — |

### 安装

```bash
# 1. 创建环境（不要用 Python 3.13/3.14，生态轮子不全）
conda create -n pano python=3.11 -y
conda activate pano

# 2. 配置国内镜像（否则装包超时）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 3. 安装依赖（torch 必须先装，否则 ultralytics 会拉 CPU 版）
pip install -r requirements.txt

# 4. 以可编辑模式安装本项目
pip install -e ".[dev]"

# 5. 配置密钥
cp .env.example .env   # 然后填入 API key
```

### 验证安装

```bash
pytest tests/ -v                    # 单元测试
python scripts/vram_probe.py        # 实测各模型显存峰值
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 项目状态

| 阶段 | 状态 | 产出 |
|---|---|---|
| **W0** 地基 | 🚧 进行中 | 环境 + git + 数据通路 |
| W1-2 L1 几何层 | ⬜ 未开始 | 球面几何库 + 对拍测试 |
| W3-4 L2 感知层 | ⬜ 未开始 | 跨边界球面跟踪 |
| W5-6 L3 PSG | ⬜ 未开始 | 场景图 + 时间轴查看器 |
| W7-8 L5 Agent 层 | ⬜ 未开始 | 端到端自动剪辑 |
| W9-10 评测 | ⬜ 未开始 | benchmark + 消融 |
| W11 工程化 | ⬜ 未开始 | Docker Demo + 量化 |
| W12 报告 | ⬜ 未开始 | 技术报告 + 简历文案 |

---

## 数据集

| 数据集 | 用途 | 许可 |
|---|---|---|
| [360Motion-Dataset](https://huggingface.co/datasets/KlingTeam/360Motion-Dataset) | 主力开发数据（36000 段 + 6DoF 轨迹） | Apache-2.0 |
| [360VOT](https://github.com/HuajianUP/360VOT) | 跟踪评测基准（120 序列 4K，20 属性） | 学术研究用 |
| [Pano2Vid](https://vision.cs.utexas.edu/projects/Pano2Vid/) | 人工 NFOV 运镜轨迹 GT | 学术研究用 |
| [D-SAV360](https://graphics.unizar.es/projects/D-SAV360/dataset_index.html) | gaze scanpath + 音频能量 | 需申请 |

详见 [docs/datasets.md](docs/datasets.md)（含完整许可声明）。

---

## 许可

Apache-2.0
