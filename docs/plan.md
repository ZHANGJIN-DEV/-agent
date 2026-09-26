# PanoDirector：360° 全景视频自动运镜 Agent

## Context

**为什么做这个项目**：投递 Insta360（影石创新）2027 届实习。调研发现其 2026-08 发布的自研端侧全景多模态模型 **PanoMind**，三大能力正是「高光识别 + 虚拟相机轨迹生成 + 叙事逻辑判断」；配套 Moments Pro 是自然语言驱动的 AI 剪辑 Agent；研究院还在扩招 VLM + 具身 Agent（万兆亮团队）。本项目直接对标这条技术线。

**为什么是现在**：用户技术画像为「工程强、CV 课程级、Agent 零经验」。项目必须同时做到：① 发挥全栈优势（FastAPI/Celery/React 可直接复用）；② 补上 git/测试/CUDA/Agent 四个空白；③ 在 4GB 显存的笔记本上能真实跑通。

**预期产出**：一个可演示、可讲 30 分钟的开源项目 + 技术报告 + 简历文案，覆盖视觉算法 / 研究院 Agent / AI 应用三类岗位。

---

## 一、已实测的环境事实（三条修正原假设）

| 项 | 实测值 | 影响 |
|---|---|---|
| **可用显存** | **2.5GB**（4GB 中桌面合成已占 1474MB） | 所有显存预算按 2.5GB 算，不按 4GB |
| **磁盘** | D: 58G 可用（88% 已用）/ **C: 244G 可用** | **数据放 `C:\pano_data`，代码放 `D:\pano-agent`** |
| ffmpeg | 8.1.1 full，**已编译 nvdec/nvenc/cuda-llvm** | 4K 全景可硬解，瓶颈消除 |
| git | 2.54.0 已装 | W0 可直接 `git init` |
| pip | **无任何镜像配置** | W0 必须先配清华源，否则装包超时 |
| node/npm | v24.15.0 / 11.12.1，registry 已是 npmmirror | Web Demo 无障碍 |
| conda | base(py3.13,torch CPU) / annotation(py3.12) / py27 / brat | 都不适合，**新建专用 env** |

---

## 二、架构

**第一性原理**（贯穿代码、报告、面试）：

> 在 ERP 图像上做的任何像素级运算（IoU、距离、池化、速度）都是错的，因为 ERP 是球面的非均匀采样。所有几何量必须以**球面角度**为单位存储和计算。

```
L6 服务层   FastAPI(async) + Celery + Redis + React18/TS Web Demo + Docker Compose
L5 Agent 层 Planner(LLM) ⇄ Tool Registry ⇄ Reflector(VLM) → Edit DSL(pydantic)
L4 优化层   Trajectory Compiler → L-BFGS-B on unwrapped yaw/pitch/fov → 轨迹曲线
L3 表示层 ★ Panorama Scene Graph (PSG)：符号层(给LLM) ⇄ track_id 对齐 ⇄ 数值层(给优化器)
L2 感知层   YOLO11n → 球面Kalman跟踪(跨边界) → 显著性UNet → 光流DIS/RAFT
            调度: VRAM Budget Scheduler + 子进程模型隔离
L1 几何层 ★sphere.py / projection.py / region.py / camera.py（地基，最先写）
L0 输入层   ERP视频 → ffmpeg(nvdec硬解) → 2K中间格式 → 抽帧
```

### 技术核心：PSG 的双视图设计

LLM 要少 token、语义化；优化器要全精度、逐帧。**矛盾通过「一个图、两种视图、`track_id` 严格对齐」解决**：

- **符号层**（`to_llm_view()` → 纯 JSON）：`track_id / caption / 属性标签 / 球面统计量`。30s 视频约 2-4k token。
- **数值层**（`numeric_ref` → `.npz`，mmap 零拷贝）：逐帧 `yaw/pitch/w/h/rot/conf`，全为角度单位。

**强制类型隔离**：`EditDSL` 的 pydantic schema 里**根本没有 float 坐标字段**，LLM 只能引用 `track_id`。这是「LLM 输出约束（可校验/可拒绝/可兜底）而非解（不可校验）」的立论基础。

### 两个关键几何技巧

1. **yaw 在 S¹ 上，求解前先 lift 到 R**（`np.unwrap`）。跨 ±180° 时轨迹自然连续——这是 DA360「消除 ERP 接缝」思路在轨迹空间的同构物。
2. **三分法是球面投影问题**。需走完整虚拟相机模型（`camera.py`：由 `(yaw,pitch,roll,fov)` 构造 R/K），不是线性映射。

---

## 三、技术选型（严格适配 2.5GB）

**本地能跑**（FP16，输入 ≤640p，单模型独占）：YOLO11n(400MB) / YOLO11s(800MB) / 自训 UNet-base32(350MB) / OSNet-x0.25(180MB) / CLIP ViT-B/32(320MB) / MobileSAM(550MB) / OpenCV DIS 光流(CPU)
**独立子进程**：RAFT-small(1.1GB) / Depth-Anything-V2-Small(1.1GB)
**必须走 API**：VLM、LLM
**明确跑不了**：任何本地 VLM（3B FP16 需 6.2GB）、扩散模型、4K 原生推理（单帧 FP32 就 88MB）、ViT-L 级 backbone、任何 >0.5B 的微调

**工程手法：子进程模型隔离 + 显存预算调度。** `torch.cuda.empty_cache()` 在 PyTorch 里不可靠，唯一可靠的释放是**进程退出**。配 `configs/vram_budget.yaml` 声明峰值 + `scripts/vram_probe.py` 实测，并设 `fallback.on_oom: cuda_to_cpu` 保证 demo 永不挂。**这张实测表本身就是「端侧落地」能力的证明。**

**API 选型**（预算 <¥50）：
| 用途 | 首选 | 备选 |
|---|---|---|
| LLM 规划 | DeepSeek-V3 | GLM-4.5-Flash（免费） |
| VLM 语义描述 | Qwen-VL-Plus | GLM-4V-Flash（免费） |
| 兜底 | **MockProvider**（离线可跑全流程 + CI 用） | — |

所有 API 调用走**内容寻址缓存**（key = sha1(prompt+图片 hash+model)），调试期省 5-10 倍成本。

**Python 环境**：`conda create -n pano python=3.11`（**不用 3.13/3.14**，sam2/torchreid 轮子不全）→ 清华源 → `torch==2.7.1+cu126`（3050 是 sm_86，完全兼容；不用最新版，2.7 是稳定甜点）+ `numpy<2.3`。

---

## 四、数据方案

| 优先级 | 数据集 | 用途 | 许可 |
|---|---|---|---|
| **P0 必拿** | **360Motion-Dataset** (KlingTeam) | 主力开发数据：36000 段全景 + 6DoF 轨迹 + HDRI | **Apache 2.0** ✅ 唯一零风险，W0 必须拿到 |
| **P0 必拿** | **360VOT** (ICCV 2023) | 跟踪评测基准：120 序列/4K，20 属性含 7 个 360 特有挑战 | 学术用，走百度网盘 |
| P1 尽力 | **Pano2Vid** (ACCV 2016) | **唯一人工 NFOV 运镜轨迹 GT** → 轨迹质量黄金标准 | 链接可能失效 |
| P1 尽力 | **D-SAV360** (TVCG 2023) | 4596 gaze scanpath + 音频能量 | 需邮件申请，**W0 就发** |
| P2 加分 | 360VOTS / Wild360 / VR-EyeTracking | 像素级 mask / 眼动热图 | 各有约束，只做定性展示 |
| **禁用** | ~~SoccerNet-360~~ | **该数据集不存在** | — |

**存储铁律**：① 360VOT 只下 30 个序列（覆盖全部 7 个属性）；② `ffmpeg -hwaccel cuda -vf scale=2048:1024 -c:v libx265 -crf 24` 转码降 4-6 倍；③ 转码幂等可断点续跑。预估 40-80GB，C 盘够。

**降级方案**：Pano2Vid 拿不到 → **自建人工运镜数据集**（Streamlit 标注工具 + 5 位同学 × 10 段），并做标注者间一致性分析。这反而变成加分项。

---

## 五、分阶段计划（12 周 / 每个里程碑都可演示）

| 周 | 阶段 | 关键产出 | 验收标准 |
|---|---|---|---|
| **W0** (3天) | 地基 | `git init` + 清华源 + `pano` env + 360Motion 子集 + API 连通 | `git log` 有 3+ commits；ffprobe 读出 `ProjectionType=equirectangular` |
| **W1-2** | **L1 几何层** ★ | sphere/projection/region/camera + **对拍测试** | 与 `py360convert.e2c` 逐像素误差 <1e-4；球面 IoU 在经度平移 360° 后不变；覆盖率 >80% |
| **W3-4** | L2 感知层 | YOLO11n + **球面 Kalman 跟踪** + 显著性 UNet + vram_probe | 360VOT 跨边界序列轨迹无跳变；实测峰值 <2.5GB；AUC-Judd >0.75 |
| **W5-6** | **L3 PSG** ★ | `scene_graph.json` + 数值层 npz + Streamlit 时间轴查看器 | `to_llm_view()` <4k token；npz mmap 加载 <50ms；非法 DSL 被正确拒绝 |
| **W7-8** | L5 Agent 层 | **一条 15s 自动剪辑成片** + Web 初版 | 输入自然语言 → 输出 mp4，主体始终在画面内；MockProvider 离线可跑；`git tag v0.1-demo` |
| **W9-10** | 评测 | `benchmark.md` + 6-8 张报告图 + 4 baseline + 3 消融 | 球面 Success 较 naïve-ERP +≥5%，**跨边界子集 +≥15%**；jerk -≥50%；A/B 胜率 >60% 且 CI 不含 50% |
| **W11** | 工程化 | Docker 一键起 Demo + ONNX INT8 量化 + CI | `docker compose up` 起全栈；2K ERP 感知 <30ms/帧；量化模型 <10MB；CI 绿 |
| **W12** | 报告 | 技术报告 ≥15 页 + README + 简历文案 + 面经 | 陌生人能按 README 复现 |

**止损点**：
- **W4**：感知慢 → 立即砍 Depth-Anything 和 SAM
- **W6**：PSG 没通 → 砍掉 W9-10 全部加分项，只保核心链路

**砍功能顺序**：深度 → SAM → ONNX 量化 → VLM 反思 → 节拍对齐。**VLM 单轮反思务必保留**（成本低，但「perception→planning→reflection」的叙事完整度提升大）。

**git 与测试的自然引入**（用户这两项为零，不说教）：
- W0 第一件事 `git init`，理由是"要写 12 周，中途肯定改烂，先 init 才敢随便改"
- W1 第一行几何代码配第一个测试，理由是"球面投影错一个符号，后面 10 周全错且极难 debug"
- 每个 Milestone 打 tag，README 贴对应 demo 动图

**Agent 学习曲线应对**：**不要上 LangGraph/AutoGen**。第一版就是 30 行 while 循环 + JSON tool calling。跑通后再评估是否重构——报告里可写「评估了 LangGraph，但本任务是单链路 + 需严格控 token，自研极简编排更合适」，这是工程判断力的体现。

---

## 六、评测方法论（技术报告的分水岭）

**跟踪**：ERP 上普通 IoU 是错的（赤道 10px≈0.3°，高纬 10px≈3°）。用 **球面 Success**（Fibonacci 球面 10000 点采样 + 立体角加权）+ **球面 Precision**（great-circle 距离 <20°）。**分属性报告 7 个 360 特有挑战**。

**最重要的消融**（证明增益来自几何建模而非其他）：
| 配置 | 全部序列 | 含 CROSS_BOUNDARY | 不含边界 |
|---|---|---|---|
| naïve ERP (ByteTrack 原版) | 基线 | 基线 | 基线 |
| + 球面距离关联 | ↑ | ↑↑ | ≈持平 |
| + unwrapped yaw Kalman | ↑ | ↑↑↑ | ≈持平 |
| + circular padding | ↑ | ↑↑↑ | ≈持平 |

**轨迹**：Pano2Vid 协议 —— **DTW 对齐**（节奏不同，直接逐帧比是错的）+ 多参考取 min + MedAE(度) + AUC@30°/60° + FOV Wasserstein 距离。另测 jerk 平滑度、主体居中率、三分法命中率、跨边界连续性、极点退化。

**高光**：诚实声明无公开 GT，用 AUC-Judd/CC/KLD/SIM（球面加权）+ 人工排序 Spearman ρ，**明确标注为启发式弱监督，不作为主贡献**。

**端到端**：20 段视频 × 10 被试 A/B 偏好测试（bootstrap 95% CI）+ 30 条指令的遵循率 + 同指令 5 次的稳定性方差。

**7 个 baseline**：CenterCrop / Saliency-Greedy / Saliency+Smooth / AutoFrame 启发式复现 / **Ours w/o LLM** / **Ours** / (加分) Ours w/o 球面几何 —— 让每一层增益都能被单独归因。

---

## 七、风险与应对（Top 6）

| 风险 | 概率 | 应对 |
|---|---|---|
| **可用显存仅 2.5GB**，OOM 频发 | 高 | 按 2.5GB 预算；子进程隔离；全 FP16 + ≤640p；`vram_probe` 实测；**OOM 一键切 CPU** 保证 demo 不挂 |
| **球面几何实现错误且难察觉** | 高 | **致命风险**。W1 就对拍测试 + 恒等性测试（IoU=1 / 经度平移不变）+ ERP/Cube 双视图可视化互验 |
| **VLM 在 ERP 原图上理解差** | 高 | **这不是 bug 是 discovery** —— 设计「VLM 看 ERP vs 看 NFOV」对照实验写进报告，直接论证"为什么需要投影" |
| LLM 输出非法 JSON / 幻觉 track_id | 高 | pydantic 强校验 + tenacity 重试 + **track_id 白名单** + 连续失败降级为纯优化器基线 |
| Windows 多进程坑（spawn 无法 pickle 闭包） | 高 | DataLoader `num_workers=0`；子进程统一 `get_context('spawn')` + 顶层函数 |
| 12 周做不完 | 中 | 每里程碑独立可交付；**W8 = 最小完整故事**；W4/W6 两个明确止损点 |

---

## 八、验证方式

**每个阶段都能独立验证，不依赖后续阶段：**

1. **W1 几何层**：`pytest tests/test_geometry_vs_py360convert.py`（对拍误差 <1e-4）；CLI `python -m pano_agent.cli render --erp x.jpg --yaw 90 --pitch 20 --fov 75 -o out.png` 目视无接缝
2. **W3 感知层**：`python scripts/vram_probe.py` 输出显存实测表；跟踪结果叠加可视化 mp4，跨边界处无跳变
3. **W5 PSG**：`streamlit run src/pano_agent/sir/viewer.py` 拖时间轴看全景帧 + 轨道 + 显著图
4. **W7 端到端**：输入「剪一个突出滑板者的 15 秒高能片段」→ 输出 mp4；`pytest tests/test_e2e_mock.py`（MockProvider 离线全流程）
5. **W9 评测**：`python scripts/run_benchmark.py` → `docs/benchmark.md` + 报告图
6. **W11 服务**：`docker compose up` → 浏览器上传全景视频 + 输指令 → 并排播放原片与成片

---

## 九、代码结构

```
D:\pano-agent\                          ← 代码（D 盘，小）
├── src\pano_agent\
│   ├── geometry\   sphere/projection/region/camera      ★ 地基，最先写
│   ├── io\         video.py(nvdec解码) / cache.py(内容寻址)
│   ├── perception\ detect/track_spherical/saliency/flow/depth/registry ★
│   ├── sir\        schema/build/query                    ★★ 技术核心
│   ├── agent\      dsl/planner/reflector/tools/providers/{deepseek,qwen,zhipu,mock}
│   ├── optimize\   compiler/solver/objectives
│   ├── render\     nfov/transition
│   ├── eval\       metrics_spherical/metrics_traj/metrics_saliency/protocols ★
│   └── service\    FastAPI + Celery
├── tests\      test_geometry_vs_py360convert / test_spherical_iou / test_yaw_wrap
│               / test_dsl_schema / test_e2e_mock
├── scripts\    prepare_data / vram_probe / run_benchmark / make_report_figs
├── configs\    models.yaml / vram_budget.yaml ★ / agent.yaml
├── web\        React18 + TS + Vite + Tailwind + Zustand
└── docs\       report.md / metrics.md / datasets.md(许可清单) ★

C:\pano_data\   raw / work / fast / cache / datasets\{360vot,360motion,pano2vid,dsav360}
```

**最先创建、最关键的 5 个文件**：
1. `src\pano_agent\sir\schema.py` — PSG 接口契约，全项目技术核心
2. `src\pano_agent\geometry\projection.py` — 所有几何正确性的地基
3. `src\pano_agent\agent\dsl.py` — Edit DSL，LLM 层与优化层的解耦点
4. `src\pano_agent\perception\registry.py` — 子进程隔离 + 显存预算 + OOM 降级
5. `src\pano_agent\eval\protocols.py` — 球面指标 + Pano2Vid DTW 协议

---

## 十、批准后立即执行的第一步（W0）

1. `conda create -n pano python=3.11` + 清华 pip 镜像 + `torch==2.7.1+cu126`
2. `D:\pano-agent\` 骨架 + `git init` + `.gitignore`（含 `.env`）+ 首次 commit
3. `C:\pano_data\` 目录 + `HF_ENDPOINT=https://hf-mirror.com` 拉 360Motion-Dataset 子集
4. 百度网盘取 360VOT 3 个序列验证通路
5. 给 D-SAV360 / Pano2Vid 作者发申请邮件
6. 写 30 行 `hello_vlm.py` 调通一次 VLM API（建立信心 + 验证 key）

**需要你提供**：DeepSeek / 通义 / 智谱 三家的 API key（至少一个 LLM + 一个 VLM），写入 `D:\pano-agent\.env`（已 gitignore）。

**需要你确认的项目名**：暂定仓库名 `pano-agent`，展示名「PanoDirector · 全景视频自动运镜 Agent」。
