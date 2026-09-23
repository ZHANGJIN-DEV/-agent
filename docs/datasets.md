# 数据集清单与许可声明

本文档记录本项目使用的所有数据集、其许可条款与本项目中的使用范围。

> **合规原则**：非商用许可（CC-BY-NC-SA 等）或条款不明的数据集，
> 其衍生结果**仅用于学术研究与技术报告，不用于任何商业用途**；
> 公开 Demo 与截图只使用 Apache-2.0 等宽松许可的素材。

---

## 主力数据集（P0）

### 360Motion-Dataset

| 项 | 内容 |
|---|---|
| **来源** | [HuggingFace: KlingTeam/360Motion-Dataset](https://huggingface.co/datasets/KlingTeam/360Motion-Dataset) |
| **规模** | 72,000 段全景视频（公开 36,000），50 类实体、6 个 UE 场景、121 条轨迹模板 |
| **规格** | 480×720 / 99 帧 / 3.3s / 30fps，含 6DoF 相机轨迹 JSON + HDRI + 12 相机外参 |
| **许可** | **Apache-2.0** ✅ 最宽松，可自由使用与再分发 |
| **用途** | 主力开发数据；轨迹生成的预训练与几何合法性验证 |
| **获取** | `HF_ENDPOINT=https://hf-mirror.com` + `huggingface-cli download` |
| **风险** | 低 |

> 合成数据的好处：**有真实 6DoF 轨迹 GT**，且无隐私与版权争议。
> 这是 W0 唯一"零风险必拿"的数据，保证项目有数据可开工。

---

### 360VOT

| 项 | 内容 |
|---|---|
| **来源** | [GitHub: HuajianUP/360VOT](https://github.com/HuajianUP/360VOT)（ICCV 2023） |
| **规模** | 120 序列 / ~113K 帧 / 3840×1920 4K，32 类目标 |
| **标注** | BBox / rBBox / BFoV / rBFoV 四种稠密标注 |
| **许可** | 学术研究用（需引用原论文） |
| **用途** | **跟踪模块的核心评测基准**。20 个属性中含 7 个 360° 特有挑战 |
| **获取** | GitHub README 中的网盘链接（Google Drive 在国内不可达，走百度网盘） |
| **风险** | 中 —— 只下 30 个序列子集（覆盖全部 7 个 360° 属性） |

**7 个 360° 特有属性**（本项目消融实验的分组依据）：

`CROSS_BOUNDARY`（跨 ERP 边界）、`LARGE_DISTORTION`（大畸变）、
`STITCH_ARTIFACT`（拼接伪影）、`SPHERICAL_FAST_MOTION`（球面快速运动）、
`LATITUDE_CHANGE`（纬度变化）、`LARGE_FOV`（大视场）、`HIGH_LATITUDE`（高纬度）

**引用**：
```bibtex
@inproceedings{huang2023_360vot,
  title={360VOT: A New Benchmark Dataset for Omnidirectional Visual Object Tracking},
  author={Huang, Huajian and others},
  booktitle={ICCV},
  year={2023}
}
```

---

## 重要数据集（P1）

### Pano2Vid

| 项 | 内容 |
|---|---|
| **来源** | [vision.cs.utexas.edu/projects/Pano2Vid](https://vision.cs.utexas.edu/projects/Pano2Vid/)（ACCV 2016, UT Austin） |
| **规模** | 86 段 360° 视频 / 7.3 小时 + **人工 NFOV 运镜轨迹标注**（3.4 小时） |
| **类别** | Soccer / Mountain Climbing / Parade / Hiking |
| **许可** | 未明确声明（素材源自 YouTube，学术用途） |
| **用途** | **唯一的人工运镜轨迹 GT** → 轨迹质量评测的黄金标准（DTW 对齐协议） |
| **获取** | 项目页直链 / Web Archive 兜底 |
| **风险** | **中高** —— 2016 年的项目页，链接可能已失效 |

**降级方案**：若链接失效，自建人工运镜数据集
（Streamlit 标注工具 + 5 位同学 × 10 段视频），并做标注者间一致性分析。
**这反而是加分项** —— 证明有独立构建评测基准的能力。

**引用**：
```bibtex
@inproceedings{su2016_pano2vid,
  title={Pano2Vid: Automatic Cinematography for Watching 360° Videos},
  author={Su, Yu-Chuan and Grauman, Kristen},
  booktitle={ACCV},
  year={2016}
}
```

---

### D-SAV360

| 项 | 内容 |
|---|---|
| **来源** | [graphics.unizar.es/projects/D-SAV360](https://graphics.unizar.es/projects/D-SAV360/dataset_index.html)（IEEE TVCG 2023） |
| **规模** | 85 段 4K/60fps 全景视频（50 立体 + 35 单目），4,596 条 gaze/head scanpath，87 名被试 |
| **附加** | 逐帧显著性图、RAFT 光流、深度图、**Ambisonic 音频能量图** |
| **许可** | 研究用，**需邮件申请并签署协议** |
| **用途** | 显著性预测的 GT；音视频联合高光检测 |
| **获取** | 邮件申请，周期 2-4 周 → **W0 就要发出申请** |
| **风险** | 中 —— 若 W5 前未获批则砍掉该模块 |

---

## 加分数据集（P2，时间充裕才用）

| 数据集 | 规模 | 用途 | 许可 | 风险 |
|---|---|---|---|---|
| [360VOTS](https://huggingface.co/datasets/xuyzshaun/360VOTS) | 290 序列，像素级 mask | 球面视频目标分割 | CC-BY-NC-SA-4.0（**需 HF 登录同意条款**） | 中 |
| [Wild360](https://github.com/hsientzucheng/CP-360-Weakly-Supervised-Saliency) | 85 段 / 55,000 帧 + 30 人眼动热图 | 显著性交叉验证 | **未明确** → 只做定性展示 | 中 |
| [VR-EyeTracking](https://github.com/xuyanyu-shh/VR-EyeTracking) | 208 段 4K + 45 被试 | 显著性 | 学术 | 低 |
| [Sports-360](https://github.com/vhchuong/Saliency-prediction-for-360-degree-video) | 104 段运动视频 + 显著图 | 运动场景显著性 | 学术 | 低 |

---

## ⚠️ 注意

**不要引用 "SoccerNet-360"** —— 经调研核实，**该数据集不存在**。
SoccerNet 系列是透视广播足球视频，非全景。最接近的足球全景数据集是
Football360 / Calib360（VISAPP 2023，全景相机标定），但与本项目无关。

---

## 存储与体积控制

**数据根目录**：`C:\pano_data`（D 盘仅剩 58G，4K 全景视频放不下）

| 子目录 | 内容 | 说明 |
|---|---|---|
| `raw/` | 原始下载 | 转码后即删 |
| `work/` | 2K H.265 中间格式 | **主力读取目录** |
| `fast/` | 512×256 快扫版 | 供光流与显著性使用 |
| `cache/` | 内容寻址缓存（API 响应、中间结果） | — |
| `datasets/` | 各数据集 | — |

**三条铁律**：

1. **只下子集** —— 360VOT 只取 30 个序列（覆盖全部 7 个 360° 属性），不贪全
2. **转码即压缩** —— `ffmpeg -hwaccel cuda -vf scale=2048:1024 -c:v libx265 -crf 24`
   使 4K→2K 体积降 4-6 倍
3. **转码幂等** —— `scripts/prepare_data.py` 检查产物存在即跳过，支持断点续跑

预估总占用：30 序列 × 5min × 2K H.265 ≈ **40-80GB**

---

## 致谢

本项目的实现离不开以上数据集的作者。所有被使用的数据集将在技术报告中
逐一引用，并在 README 中致谢。
