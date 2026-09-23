# PanoDirector · 开发说明

> 360° 全景视频的自动运镜 Agent。对标 Insta360 的 **PanoMind**
> （高光识别 + 虚拟相机轨迹生成 + 叙事逻辑判断）与 **Moments Pro**
> （自然语言驱动的 AI 剪辑）。

**这是给 Claude Code 的项目说明。** 新会话开始时读本文件即可恢复上下文 ——
每完成一个阶段请更新「当前进度」与「下一步」两节。

---

## 快速上手

```bash
conda activate pano                        # Python 3.11

# 跑测试（几何层与感知层的纯逻辑部分不需要 torch）
python -m pytest tests/ -q

# 渲染演示
python scripts/demo_geometry.py

# API 连通性自检
python scripts/hello_llm.py --vision

# 显存实测
python scripts/vram_probe.py
```

---

## 当前进度

| 阶段 | 状态 | 说明 |
|---|---|---|
| **W1 几何层** | ✅ 完成 | 4 模块 + 165 测试，含与 py360convert 的对拍 |
| **W3-4 感知层** | ✅ 完成 | 7 模块 + 269 测试 |
| **W5-6 PSG** | 🔶 进行中 | schema + build 已完成；时间轴 viewer 待做 |
| W7-8 Agent 层 | ⬜ 未开始 | LLM 规划 + 轨迹优化 + 渲染 |
| W9-10 评测 | ⬜ 未开始 | baseline + 消融 |
| W11 工程化 | ⬜ 未开始 | Docker Demo + 量化 |
| W12 报告 | ⬜ 未开始 | 技术报告 + 简历文案 |

**最后一次提交**：`7531206 feat(sir): PSG 构建`

**测试总数：327 项全通过**（`python -m pytest tests/ -q`）

### W5-6 剩余工作

- [ ] `sir/viewer.py` —— Streamlit 时间轴查看器（拖时间轴看全景帧 +
      轨道叠加 + 显著性热图）
- [ ] 接 VLM 生成 `TrackNode.caption`（DeepSeek `deepseek-flash` 支持视觉）
- [ ] 接真实感知输出（需 torch + YOLO，等环境就绪）

### 下一步：W5-6 PSG（Panorama Scene Graph）

项目技术核心。设计目标：**一个图，两种视图，通过 `track_id` 严格对齐**。

- **符号层**（`to_llm_view()` → 纯 JSON）：`track_id / caption / 属性标签 /
  球面统计量`。30 秒视频约 2-4k token
- **数值层**（`numeric_ref` → `.npz`，mmap 零拷贝）：逐帧
  `yaw/pitch/w/h/rot/conf`，全为角度单位

关键约束：**`EditDSL` 的 pydantic schema 里不能有 float 坐标字段** ——
LLM 只能引用 `track_id`。这是「LLM 输出约束（可校验、可拒绝、可兜底）
而非解（不可校验）」的立论基础。

**可以先用合成数据开发** —— `perception.types.FrameTracks` 是纯数据结构，
不需要 torch。等 torch 就绪后再切真实检测输出。

---

## 核心设计约定（修改前务必先读）

### 1. 所有几何量以球面角度为单位

> ERP 是球面的非均匀采样。赤道处 10 像素 ≈ 0.3°，高纬处 10 像素 ≈ 3°。

因此**任何像素级的距离、面积、IoU、速度计算都是错的**。
代码中出现 `*_deg` 的字段一律是角度；`*_dps` 是角速度（度/秒）。

### 2. 周期量的处理发生在数据进入系统的边界

这是全项目的统一原则，四个模块用不同工具实现同一件事：

| 场景 | 工具 |
|---|---|
| yaw 滤波 | `sphere.unwrap_yaw` 提升 S¹ → ℝ |
| 跨接缝检测框 | 环形 padding + 接缝连续坐标合并 |
| 光流 | 环形 padding 后计算 |
| 跨接缝框的球面中心 | 圆统计（circular mean），**不能取端点平均** |

**不要在每个使用者处分别处理周期性** —— 在边界处理一次，之后各层
都可以当普通线性量。

### 3. 像素中心约定：整数索引即像素中心

`sph_to_erp_pixel` 内部的 `-0.5` 偏移是全局唯一的一处。
`projection.sample_erp` 不得重复施加。

### 4. 相机采用端点采样约定

第 0 列与最后一列精确对应 `±tan(fov_x/2)`，与 py360convert 一致。
改这个约定会同时破坏 `test_geometry_vs_py360convert.py` 的全部测试。

### 5. 任何球面标量聚合都要 cos φ 加权

用 `saliency.weighted_mean` 或 `sphere.erp_row_solid_angles`，
不要直接 `.mean()`。

---

## 环境事实（实测，不是推测）

| 项 | 值 | 影响 |
|---|---|---|
| GPU | RTX 3050 Laptop 标称 4GB | — |
| **实际可用显存** | **约 2.5GB** | 桌面合成已占 1.47GB，**所有选型按 2.5GB 算** |
| 数据目录 | `C:\pano_data` | D 盘仅剩 58G，放不下 4K 全景 |
| 代码目录 | `D:\pano-agent` | — |
| ffmpeg | 8.1.1，已编译 nvdec | 4K 可硬解 |
| **网络** | GitHub / HuggingFace **直连不通** | 用 `hf-mirror.com`；代码托管用 Gitee |

### 关键认知：`torch.cuda.empty_cache()` 不可靠

它只把缓存分配器中未引用的块标记为可复用，不保证归还驱动。
**唯一可靠的显存释放是进程退出** —— 这是 `perception/registry.py`
用子进程隔离模型的原因。

---

## 数据源

**Insta360 官方 Omni360-X**（CVPR 2026 AirSim360 配套数据集）。

- 位置：`C:\pano_data\datasets\nyc_seg\`（全景点集）
- 格式：**ERP 2048×1024，比例精确 2.0**，5179 张
- 语义类别存在 **alpha 通道**（R/G/B 全 0），值域 [3, 28]
- 类别表：`C:\pano_data\raw\semantic_lists_nyc.txt`
- 许可：CC-BY-NC-4.0（学术可用，**非商用**）

**注意**：`nyc_Raw.zip`（真实 RGB，11.36GB）可能仍在后台下载中，
检查 `C:\pano_data\raw\nyc_Raw.zip` 的大小。

### 曾用数据源及弃用原因（避免重复踩坑）

- **360Motion-Dataset**：文件名与 `Hemi12_transforms.json` 证实它是
  **12 个透视相机视角，不是 ERP**，且 `Desert.zip` 有 26.4GB。**已弃用**。
- **SoccerNet-360**：**该数据集不存在**，不要引用。

---

## API

DeepSeek 一个平台覆盖 LLM + VLM（实测 2026-09）：

| 模型 | 视觉 | 用途 |
|---|---|---|
| `deepseek-chat` | ✅ | 文本规划 |
| `deepseek-flash` | ✅ | 视觉理解，**返回 `reasoning_content`** |
| `deepseek-v4-pro` | ❌ | 不要用于 VLM |

`deepseek-flash` 的推理链可用于展示 Agent 的决策依据 ——
契合本项目「可解释的自动运镜」定位，报告里可以放。

密钥在 `.env`（已 gitignore）。无密钥时 `MockProvider` 接管全流程。

---

## 待办与限制

- [ ] torch 装完后：跑 `tests/test_saliency_spherical.py` 中跳过的
      torch 部分；用真实数据训练显著性 UNet
- [ ] 下载完 `nyc_Raw.zip` 后：验证真实 RGB，用真实图像做一次端到端
- [ ] `detect.py` 的极点分块（W4 止损点优先裁剪项）
- [ ] 计划文档尚未更新（数据源变更、网络方案、DeepSeek 双能力）
- 无真实眼动 GT，显著性采用语义 mask 弱监督，**报告中须声明**
