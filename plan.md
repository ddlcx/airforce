# 羽毛球场地视频分析系统 — 项目总览

## 1. 项目概述

本系统针对手机摄像头拍摄的羽毛球比赛视频，实现两个核心功能：

- **模块1（球场检测与渲染）**：基于 YOLO pose 模型检测球场26个关键点，计算 Homography 矩阵，将标准球场线条叠加到视频画面
- **模块2（相机标定）**：通过 YOLO 检测球网 4 个关键点，结合地面关键点求解相机投影矩阵 P = K[R|t]

整个流水线串行执行，每个环节设置独立的验证门控，防止误差累积传播。

---

## 2. 项目结构

```
airforce/
├── config/
│   └── court_config.py              # 标准球场尺寸、26关键点坐标、线段定义
├── module1/                         # 模块1：球场检测与渲染（在线推理）
│   ├── __init__.py
│   ├── yolo_detector.py             # YOLO pose 推理封装
│   ├── homography.py                # Homography 计算与验证
│   ├── court_renderer.py            # 球场线条叠加渲染
│   └── net_overlay.py               # 球网地面线投影
├── module2/                         # 模块2：球网检测与相机标定（在线推理）
│   ├── __init__.py
│   ├── net_top_detector_yolo.py     # 球网关键点提取
│   ├── camera_calibration.py        # 相机标定（DLT + PnP）
│   └── projection_validation.py     # 投影矩阵验证
├── training/                        # YOLO 模型训练（离线）
│   ├── config.py                    # 训练超参数与硬件预设
│   ├── train.py                     # 主训练入口
│   ├── split_dataset.py             # 数据合并与随机拆分
│   ├── prepare_data.py              # 数据集验证与 flip_idx 修正
│   ├── keypoint_mapping.py          # 球场关键点排序映射
│   └── keypoint_mapping_net.py      # 球网关键点排序映射
├── yolo/                            # YOLO 相关数据与产物
│   ├── datasets/                    # 训练数据集
│   │   ├── court/                   #   球场22关键点数据集 (Roboflow)
│   │   └── net/                     #   球网4关键点数据集 (Roboflow)
│   ├── runs/                        # 训练过程产物（曲线图、日志、checkpoint）
│   │   ├── court/                   #   球场模型训练结果
│   │   └── net/                     #   球网模型训练结果
│   ├── weights/                     # 推理用最终权重
│   │   ├── court_yolo.pt            #   球场关键点模型
│   │   └── net_yolo.pt              #   球网关键点模型
│   └── pretrained/                  # YOLO 预训练基础模型
│       ├── yolov8n-pose.pt
│       └── yolov8s-pose.pt
├── utils/
│   ├── geometry.py                  # 直线交点、齐次坐标投影
│   ├── metrics.py                   # 重投影误差、质量统计
│   └── temporal_filter.py           # 时间域平滑（EMA + 标定锁定）
├── scripts/
│   ├── run_module1.py               # 模块1独立运行入口
│   ├── run_module2.py               # 模块2独立运行入口
│   ├── run_pipeline.py              # 全流程入口
│   └── sample_inference.py          # YOLO 模型抽样推理验证
├── tests/
├── data/                            # 其他静态资源
└── docs/                            # 设计文档
```

**依赖**：
```
ultralytics>=8.1.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pytest>=7.4.0
```

---

## 3. YOLO Pose 模型

本项目使用**两个独立的 YOLO Pose 模型**，分别检测球场地面关键点和球网关键点。两个模型各自训练、各自推理，在下游通过全局编号体系统一管理。

### 两个模型概览

| | 球场关键点模型 | 球网关键点模型 |
|------|---------------|---------------|
| **目标** | 检测22个地面关键点 | 检测4个球网角点 |
| **数据集** | `yolo/datasets/court/` | `yolo/datasets/net/` |
| **图片数** | 1,194 张 | 495 张 |
| **kpt_shape** | [22, 3] | [4, 3] |
| **推理权重** | `yolo/weights/court_yolo.pt` | `yolo/weights/net_yolo.pt` |
| **下游消费者** | 1.2 Homography、2.4 端点推断 | 2.2 YOLO球网检测 |

### 训练与推理的关系

训练和推理是两个独立阶段，由不同的代码模块负责：

```
离线：训练阶段                          在线：推理阶段
┌─────────────────────────┐            ┌─────────────────────────┐
│  training/ 模块          │            │  module1/yolo_detector.py│
│                         │            │                         │
│  数据集: yolo/datasets/ │ ─→ .pt ─→  │  权重: yolo/weights/    │
│  训练产物: yolo/runs/   │   权重文件  │  输出: 26个关键点坐标    │
│  预训练: yolo/pretrained│            │  + 置信度               │
└─────────────────────────┘            └─────────────────────────┘
```

- **训练阶段**（离线，开发时执行）：从 Roboflow 标注数据集出发，通过 `training/` 模块训练 YOLO 模型，产出 `.pt` 权重文件，存放在 `yolo/weights/`
- **推理阶段**（在线，运行时执行）：`module1/yolo_detector.py` 加载训练好的权重，对视频帧进行关键点检测，输出 `CourtDetectionResult`

### 详细文档分工

| 文档 | 覆盖阶段 | 内容 |
|------|---------|------|
| [`docs/training_guide.md`](docs/training_guide.md) | 训练阶段 | 数据集详情、关键点排序映射、flip_idx、训练超参数、数据增强、训练命令、调优建议 |
| [`docs/yolo_pose_detector.md`](docs/yolo_pose_detector.md) | 推理阶段 | 推理接口设计、输入输出数据结构、算法伪代码、辅助方法、测试方案 |

### 训练脚本

| 文件 | 功能 |
|------|------|
| `training/train.py` | 主训练入口（`--model-type court\|net` 切换） |
| `training/config.py` | 训练超参数与硬件预设 |
| `training/split_dataset.py` | 数据合并与随机拆分 |
| `training/prepare_data.py` | 数据集验证与 flip_idx 修正 |
| `training/keypoint_mapping.py` | 球场关键点排序映射（数据集索引 ↔ 全局索引） |
| `training/keypoint_mapping_net.py` | 球网关键点排序映射（数据集索引 ↔ 全局索引） |

### 快速命令

```bash
# 球场模型训练
python -m training.train --dataset-dir yolo/datasets/court

# 球网模型训练
python -m training.train --dataset-dir yolo/datasets/net --model-type net

# 训练产物 → 推理权重（手动复制 best.pt）
cp yolo/runs/court/weights/best.pt yolo/weights/court_yolo.pt
cp yolo/runs/net/weights/best.pt   yolo/weights/net_yolo.pt
```

---

## 4. 基础定义

坐标系、关键点、线段等基础数据定义，所有模块共享。

→ 详见 [`docs/base_definitions.md`](docs/base_definitions.md)

**要点速览**：
- 世界坐标系原点在球场中心（球网地面线中点），X沿球网、Y沿球场长度、Z竖直向上
- 26个关键点：22个地面点（Z=0，覆盖所有线交叉点）+ 4个球网点（Z>0）
- `COURT_LINE_SEGMENTS` 定义13条需渲染的球场线段

---

## 5. 模块1：球场检测与渲染

从视频帧中检测球场关键点，计算 Homography，渲染标准球场线条。

| 编号 | 子模块 | 功能 | 代码文件 | 文档 |
|------|--------|------|----------|------|
| 1.1 | YOLO Pose 检测器 | 加载训练好的模型，检测26个关键点的像素坐标和置信度 | `module1/yolo_detector.py` | [`yolo_pose_detector.md`](docs/yolo_pose_detector.md)（推理设计） |
| 1.2 | Homography 计算 | 从地面关键点(0-21)计算 H 矩阵 + 5项验证门控 | `module1/homography.py` | [`homography.md`](docs/homography.md) |
| 1.3 | 球场线条渲染 | 利用 H 将标准球场线投影到视频帧；投影球网底部坐标 | `module1/court_renderer.py` `module1/net_overlay.py` | [`court_renderer.md`](docs/court_renderer.md) |

**模块1输出**：`HomographyResult`（H矩阵）+ `CourtDetectionResult`（26关键点检测）+ 叠加画面

**验证门控**：H 必须通过5项验证（重投影误差 < 5px、内点比例 > 70%、行列式正、条件数 < 1e6、奇异值比 < 1e4）才能进入模块2。

---

## 6. 模块2：球网检测与相机标定

从球网定位中推断相机完整3D投影参数。

| 编号 | 子模块 | 功能 | 代码文件 | 文档 |
|------|--------|------|----------|------|
| 2.2 | 球网关键点提取 | 提取 kp22-25 | `module2/net_top_detector_yolo.py` | [`net_detector_yolo.md`](docs/net_detector_yolo.md) |
| 2.5 | 相机标定 | DLT(固定机位) / calibrateCamera(移动) / IAC / PnP | `module2/camera_calibration.py` | [`camera_calibration.md`](docs/camera_calibration.md) |
| 2.6 | 投影矩阵验证 | 重投影误差、相机位置、H-P一致性等6项检查 | `module2/projection_validation.py` | [`projection_validation.md`](docs/projection_validation.md) |

**模块2输出**：`CameraCalibrationResult`（含 K、R、t、P = K[R|t]）

**标定策略**：固定机位用「多帧平均 → DLT → PnP精化 + IAC交叉验证」；移动机位用「calibrateCamera → PnP」。

---

## 7. 辅助模块

| 模块 | 功能 | 文档 |
|------|------|------|
| 时间域平滑 | 关键点EMA平滑(α=0.4) + 标定锁定策略，消除视频抖动和闪烁 | [`temporal_filter.md`](docs/temporal_filter.md) |
| 误差传播管理 | 各环节误差预算、7项缓解策略、质量监控指标 | [`error_management.md`](docs/error_management.md) |
| 实施指南 | 4阶段实施顺序、12项设计决策、OpenCV接口速查表 | [`implementation_guide.md`](docs/implementation_guide.md) |

---

## 8. 模块间依赖关系

```
┌──────────────────────────────────────────────────┐
│                  基础定义层                        │
│   坐标系 + 26关键点 + 线段定义 (base_definitions) │
└──────────────────────┬───────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    ↓                  ↓                  ↓
┌─────────┐     ┌───────────┐     ┌───────────┐
│ 1.1     │     │ 1.2       │     │ 1.3       │
│ YOLO    │────→│ Homography│────→│ 渲染+投影 │
│ 检测器  │     │           │     │           │
└────┬────┘     └─────┬─────┘     └───────────┘
     │                │
     │    ┌───────────┼───────────────────┐
     │    │           │                   │
     ↓    │           │                   │
┌─────────┐           │                   │
│ 2.2     │           │                   │
│ 球网    │           │                   │
│ 关键点  │           │                   │
│ 提取    │           │                   │
└────┬────┘           │                   │
     │    │           │                   │
     ↓    ↓           ↓                   │
┌─────────────────────────┐               │
│ 2.5 相机标定             │               │
│ DLT / PnP / IAC         │               │
└────────────┬────────────┘               │
             ↓                            │
┌─────────────────────────┐               │
│ 2.6 投影验证             │←──────────────┘
│ (H-P一致性)             │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│ 时间域平滑 & 锁定        │
│ (仅视频模式)             │
└─────────────────────────┘
```

**依赖总结表**：

| 子模块 | 依赖的上游输出 | 为谁提供输入 |
|--------|--------------|-------------|
| 1.1 YOLO检测 | 无（入口） | 1.2, 2.2, 2.5 |
| 1.2 Homography | 1.1 的地面关键点 | 1.3, 2.5, 2.6 |
| 1.3 渲染 | 1.2 的H矩阵 | 画面叠加 |
| 2.2 球网关键点提取 | 1.1 的球网关键点 kp22-25 | 2.5 |
| 2.5 相机标定 | 1.1 地面点 + 2.2 球网关键点 + 1.2 的H(IAC验证) | 2.6 |
| 2.6 投影验证 | 2.5 的P + 1.2 的H | 最终输出 |

---

## 9. 核心数据结构速览

各子模块间传递的关键数据结构：

| 数据结构 | 产生者 | 消费者 | 核心字段 |
|----------|--------|--------|----------|
| `CourtDetectionResult` | 1.1 YOLO | 1.2, 2.2 | `keypoints[26]`(pixel_xy, confidence, visible), `bbox`, `num_visible` |
| `HomographyResult` | 1.2 Homography | 1.3, 2.2, 2.5, 2.6 | `H`(3×3), `H_inv`, `reprojection_error`, `inlier_mask` |
| `NetKeypointResult` | 2.2 球网关键点提取 | 2.5 | `left/right_top/base_pixel`(2,), `*_3d`(3,) |
| `CameraCalibrationResult` | 2.5 标定 | 2.6 | `K`(3×3), `R`(3×3), `tvec`(3,1), `P`(3×4), `reprojection_error` |

---

## 10. 全流程编排

### 入口脚本

`scripts/run_pipeline.py` — 全流程统一入口，串联模块1和模块2的所有步骤。

```bash
# 单张图片
python -m scripts.run_pipeline --input frame.jpg --output result.jpg

# 视频
python -m scripts.run_pipeline --input video.mp4 --output output.mp4

# 显示关键点和球网标注（调试用）
python -m scripts.run_pipeline --input frame.jpg --output result.jpg --show-keypoints --show-net

# 指定设备
python -m scripts.run_pipeline --input video.mp4 --output output.mp4 --device mps
```

参数列表：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | (必填) | 输入图片或视频路径 |
| `--output` | (必填) | 输出图片或视频路径 |
| `--court-model` | `yolo/weights/court_yolo.pt` | 球场模型权重 |
| `--net-model` | `yolo/weights/net_yolo.pt` | 球网模型权重 |
| `--min-conf` | 0.5 | 关键点最低置信度 |
| `--device` | 自动 | 推理设备 (cpu / mps / 0) |
| `--color` | 0,255,0 | 球场线渲染颜色 (BGR) |
| `--show-keypoints` | 关 | 标注检测到的关键点 |
| `--show-net` | 关 | 标注球网顶线和端点 |

### 单帧处理流程（7步）

```
输入: BGR 视频帧
  │
  ├─ Step 1: YOLO检测 (1.1) → CourtDetectionResult (26关键点)
  │     失败 → 返回原始帧
  │
  ├─ Step 2: Homography计算 (1.2) → HomographyResult (H矩阵)
  │     失败 → 返回原始帧（关键点不足4个）
  │
  ├─ Step 3: H验证门控 (5项检查)
  │     失败 → 中止模块2，仅输出球场线叠加
  │
  ├─ Step 4: 球场线渲染 (1.3) → 叠加画面
  │
  ├─ Step 5: 球网关键点提取 (2.2) → NetKeypointResult
  │     失败 → 中止（kp22/kp24置信度不足）
  │
  ├─ Step 6: 相机标定 (2.5) → CameraCalibrationResult (K, R, t, P)
  │     DLT → clean_intrinsics → PnP精化 → IAC交叉验证
  │     失败 → 中止
  │
  └─ Step 7: 投影验证 (2.6) → 6项检查 + overall_ok
        通过 → P 可用于下游
        未通过 → P 不可信，记录失败项
  │
输出: FrameResult {detection, h_result, h_metrics, net_kpts, calibration, proj_metrics}
     + 渲染后的帧图像
```

每一步失败都会立即中止后续步骤，`FrameResult.failed_step` 记录中止位置。

### 视频模式

视频逐帧执行上述 7 步流水线，每 100 帧输出进度日志，结束时输出汇总统计：

```
视频处理完成
  总帧数: 3000
  H验证通过: 2850 (95.0%)
  标定完成: 2700 (90.0%)
  投影验证通过: 180 (6.0%)
  中止分布:
    Step3:H验证: 150 (5.0%)
    Step5:球网关键点: 120 (4.0%)
    Step6:相机标定: 30 (1.0%)
```

> 当前视频模式尚未集成时间域平滑。未来可在 Step 1 之后插入 EMA 关键点平滑，在 Step 7 之后插入标定锁定策略。

---

## 11. 核心技术选型

| 类别 | 选型 | 说明 |
|------|------|------|
| 关键点检测 | YOLOv8/YOLO11 Pose | 26关键点，单类检测 |
| 平面映射 | `cv2.findHomography` (RANSAC) | 地面22点 → H |
| 球网检测 | YOLO 关键点 kp22-25 | 提取 4 关键点，直接使用 YOLO 模型原始值 |
| 内参估计（固定机位） | DLT → `cv2.decomposeProjectionMatrix` | 无需 K 先验假设 |
| 内参估计（移动机位） | `cv2.calibrateCamera` (Zhang法) | 需多帧不同视角 |
| 外参求解 | `cv2.solvePnPRansac` + LM精化 | 地面+球网非共面点 |
| 时间域平滑 | EMA (α=0.4) + 标定锁定 | 消除抖动和闪烁 |

---

## 12. 实施阶段概览

| 阶段 | 目标 | 涉及模块 |
|------|------|----------|
| 阶段1：基础设施 | 项目骨架、配置文件、工具函数 | config, utils |
| 阶段2：模块1 | 关键点检测 + Homography + 球场渲染 | 1.1, 1.2, 1.3 |
| 阶段3：模块2 | 球网检测 + 标定 | 2.2 ~ 2.6 |
| 阶段4：集成验证 | 全流程串联 + 时间域平滑 + 端到端测试 | pipeline, temporal_filter |

→ 详细步骤见 [`docs/implementation_guide.md`](docs/implementation_guide.md)
