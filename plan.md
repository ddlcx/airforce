# 羽毛球场地视频分析系统 — 项目总览

## 1. 项目概述

本系统针对手机摄像头拍摄的羽毛球比赛视频，实现两个核心功能：

- **模块1（球场检测与渲染）**：基于 YOLO pose 模型检测球场26个关键点，计算 Homography 矩阵，将标准球场线条叠加到视频画面
- **模块2（相机标定）**：通过 YOLO + Hough 双路检测球网顶部线，推断球网端点3D位置，结合地面关键点求解相机投影矩阵 P = K[R|t]

整个流水线串行执行，每个环节设置独立的验证门控，防止误差累积传播。

---

## 2. 项目结构

```
badminton_plan/
├── config/
│   └── court_config.py              # 标准球场尺寸、26关键点坐标、线段定义
├── module1/
│   ├── __init__.py
│   ├── yolo_detector.py             # YOLO pose 推理封装
│   ├── homography.py                # Homography 计算与验证
│   ├── court_renderer.py            # 球场线条叠加渲染
│   └── net_overlay.py               # 球网地面线投影
├── module2/
│   ├── __init__.py
│   ├── net_top_detector_hough.py    # Hough 线检测球网顶部
│   ├── net_top_detector_yolo.py     # YOLO 关键点提取球网顶线
│   ├── net_top_fusion.py            # 双路融合与交叉验证
│   ├── net_endpoint_inference.py    # 球网端点推断
│   ├── camera_calibration.py        # 相机标定（DLT + PnP）
│   └── projection_matrix.py        # 投影矩阵验证
├── utils/
│   ├── __init__.py
│   ├── geometry.py                  # 直线交点、齐次坐标投影
│   ├── metrics.py                   # 重投影误差、质量统计
│   └── temporal_filter.py           # 时间域平滑（EMA + 标定锁定）
├── tests/
│   ├── test_court_config.py
│   ├── test_homography.py
│   ├── test_net_detection.py
│   ├── test_camera_calibration.py
│   └── test_pipeline.py
├── scripts/
│   ├── run_module1.py               # 模块1独立运行入口
│   ├── run_module2.py               # 模块2独立运行入口
│   └── run_pipeline.py              # 全流程入口
├── data/
│   ├── yolo_model/                  # 训练好的 .pt 模型权重
│   ├── sample_frames/               # 测试图片
│   └── dataset/                     # YOLO 训练数据集
└── requirements.txt
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

## 3. YOLO Pose 模型训练

模块1的核心依赖——YOLO Pose 关键点检测模型，需要在 `BadmintonCourtDetection.yolov8` 数据集上训练。

→ 完整训练指南见 [`docs/training_guide.md`](docs/training_guide.md)

### 数据集

| 项目 | 值 |
|------|-----|
| 来源 | Roboflow (CC BY 4.0) |
| 格式 | YOLOv8 Pose |
| 类别 | 1 (badminton_court) |
| 关键点 | 22 个地面点 |
| 总量 | 1,194 张 |

### 数据管理

所有原始数据统一存放在 `BadmintonCourtDetection.yolov8/all/` 目录中。通过 `training/split_dataset.py` 按可配置比例随机拆分，生成 `train.txt` / `val.txt` / `test.txt` 文件列表供 YOLO 读取，数据文件本身不做任何移动或复制。

```bash
# 拆分数据集（默认 train=77%, valid=17%, test=6%）
python -m training.split_dataset --dataset-dir BadmintonCourtDetection.yolov8

# 训练（small 模型，自动检测设备）
python -m training.train --dataset-dir BadmintonCourtDetection.yolov8

# 拆分 + 训练一步完成
python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 --split
```

### 训练脚本

| 文件 | 功能 |
|------|------|
| `training/split_dataset.py` | 数据合并与随机拆分 |
| `training/prepare_data.py` | 数据集验证与 flip_idx 修正 |
| `training/config.py` | 训练超参数与硬件预设 |
| `training/train.py` | 主训练入口 |
| `training/keypoint_mapping.py` | 关键点定义与排序映射 |

### 训练输出

模型权重保存在 `runs/pose/badminton_court/weights/best.pt`，部署时需将其放入 `data/yolo_model/` 供模块1推理使用。

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
| 1.1 | YOLO Pose 检测器 | 检测26个关键点的像素坐标和置信度 | `module1/yolo_detector.py` | [`yolo_pose_detector.md`](docs/yolo_pose_detector.md) |
| 1.2 | Homography 计算 | 从地面关键点(0-21)计算 H 矩阵 + 5项验证门控 | `module1/homography.py` | [`homography.md`](docs/homography.md) |
| 1.3 | 球场线条渲染 | 利用 H 将标准球场线投影到视频帧；投影球网底部坐标 | `module1/court_renderer.py` `module1/net_overlay.py` | [`court_renderer.md`](docs/court_renderer.md) |

**模块1输出**：`HomographyResult`（H矩阵）+ `CourtDetectionResult`（26关键点检测）+ 叠加画面

**验证门控**：H 必须通过5项验证（重投影误差 < 5px、内点比例 > 70%、行列式正、条件数 < 1e6、奇异值比 < 1e4）才能进入模块2。

---

## 6. 模块2：球网检测与相机标定

从球网定位中推断相机完整3D投影参数。

| 编号 | 子模块 | 功能 | 代码文件 | 文档 |
|------|--------|------|----------|------|
| 2.1 | Hough 球网检测 | ROI内 Canny+HoughLinesP 检测球网顶线 | `module2/net_top_detector_hough.py` | [`net_detector_hough.md`](docs/net_detector_hough.md) |
| 2.2 | YOLO 球网检测 | 从关键点22-24拟合球网顶线 | `module2/net_top_detector_yolo.py` | [`net_detector_yolo.md`](docs/net_detector_yolo.md) |
| 2.3 | 双路融合 | 交叉验证 + 加权融合/冲突解决 | `module2/net_top_fusion.py` | [`net_fusion.md`](docs/net_fusion.md) |
| 2.4 | 球网端点推断 | 垂直线与顶线交点 → 球网端点像素坐标 | `module2/net_endpoint_inference.py` | [`net_endpoint_inference.md`](docs/net_endpoint_inference.md) |
| 2.5 | 相机标定 | DLT(固定机位) / calibrateCamera(移动) / IAC / PnP | `module2/camera_calibration.py` | [`camera_calibration.md`](docs/camera_calibration.md) |
| 2.6 | 投影矩阵验证 | 重投影误差、相机位置、H-P一致性等6项检查 | `module2/projection_matrix.py` | [`projection_validation.md`](docs/projection_validation.md) |

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
┌─────────────────────────────────────────────────────┐
│                  基础定义层                            │
│   坐标系 + 26关键点 + 线段定义 (base_definitions.md)  │
└───────────────────────┬─────────────────────────────┘
                        │
    ┌───────────────────┼───────────────────┐
    ↓                   ↓                   ↓
┌────────┐      ┌──────────┐        ┌──────────┐
│1.1 YOLO│      │1.2 Homo- │        │1.3 渲染  │
│ 检测器  │─────→│ graphy   │───────→│ + 投影   │
└───┬────┘      └────┬─────┘        └──────────┘
    │                │
    │     ┌──────────┼──────────────────────┐
    │     │          │                      │
    ↓     │          ↓                      ↓
┌────────┐│   ┌──────────┐          ┌──────────┐
│2.2 YOLO││   │2.1 Hough │          │2.4 端点  │
│球网检测 ││   │球网检测   │          │ 推断     │
└───┬────┘│   └────┬─────┘          └────┬─────┘
    │     │        │                     │
    ↓     │        ↓                     │
┌─────────┴────────────┐                │
│2.3 双路融合+交叉验证  │────────────────→│
└──────────────────────┘                │
                                        ↓
                                ┌──────────────┐
                                │2.5 相机标定   │
                                │ DLT/PnP/IAC  │
                                └──────┬───────┘
                                       ↓
                                ┌──────────────┐
                                │2.6 投影验证   │
                                │ (H-P一致性)  │
                                └──────┬───────┘
                                       ↓
                              ┌──────────────────┐
                              │ 时间域平滑 & 锁定 │
                              │ (仅视频模式)      │
                              └──────────────────┘
```

**依赖总结表**：

| 子模块 | 依赖的上游输出 | 为谁提供输入 |
|--------|--------------|-------------|
| 1.1 YOLO检测 | 无（入口） | 1.2, 2.2, 2.4 |
| 1.2 Homography | 1.1 的地面关键点 | 1.3, 2.1, 2.4, 2.5, 2.6 |
| 1.3 渲染 | 1.2 的H矩阵 | 2.4（球网底部投影） |
| 2.1 Hough检测 | 1.2 的H矩阵 | 2.3 |
| 2.2 YOLO检测 | 1.1 的球网关键点 | 2.3 |
| 2.3 双路融合 | 2.1 + 2.2 结果 | 2.4 |
| 2.4 端点推断 | 1.2的H + 2.3融合线 + 1.1的关键点22/23 | 2.5 |
| 2.5 相机标定 | 1.1地面点 + 2.4球网端点 + 1.2的H(IAC验证) | 2.6 |
| 2.6 投影验证 | 2.5的P + 1.2的H | 最终输出 |

---

## 9. 核心数据结构速览

各子模块间传递的关键数据结构：

| 数据结构 | 产生者 | 消费者 | 核心字段 |
|----------|--------|--------|----------|
| `CourtDetectionResult` | 1.1 YOLO | 1.2, 2.2, 2.4 | `keypoints[26]`(pixel_xy, confidence, visible), `bbox`, `num_visible` |
| `HomographyResult` | 1.2 Homography | 1.3, 2.1, 2.4, 2.5, 2.6 | `H`(3×3), `H_inv`, `reprojection_error`, `inlier_mask` |
| `NetTopLineResult` | 2.1/2.2 球网检测 | 2.3 | `endpoint1/2`(2,), `line_coeffs`(3,), `confidence`, `source` |
| `FusionResult` | 2.3 融合 | 2.4 | `fused_line`, `consistency_level`, `method_used` |
| `NetEndpointResult` | 2.4 端点推断 | 2.5 | `left/right_top_pixel`(2,), `left/right_top_3d`(3,), `method` |
| `CameraCalibrationResult` | 2.5 标定 | 2.6 | `K`(3×3), `R`(3×3), `tvec`(3,1), `P`(3×4), `reprojection_error` |

---

## 10. 全流程编排

### 单帧处理（9步）

```
输入: BGR视频帧
  │
  ├─ Step 1: YOLO检测 (1.1) → CourtDetectionResult (26关键点)
  ├─ Step 2: Homography计算 (1.2) → HomographyResult (H矩阵)
  ├─ Step 3: H验证门控 → 不通过则跳过模块2
  ├─ Step 4: 球场线渲染 (1.3) → 叠加画面
  ├─ Step 5: 双路球网检测 (2.1 + 2.2) → Hough结果 + YOLO结果
  ├─ Step 6: 双路融合 (2.3) → FusionResult
  ├─ Step 7: 球网端点推断 (2.4) → NetEndpointResult
  ├─ Step 8: 相机标定 (2.5) → CameraCalibrationResult (P矩阵)
  └─ Step 9: 投影验证 (2.6) → 验证指标 + overall_ok
  │
输出: {H, P, 叠加画面, 验证指标}
```

### 视频处理（在单帧处理外层包裹平滑和锁定）

```
逐帧循环:
  ├─ Step A: YOLO检测 (1.1)
  ├─ Step B: EMA关键点平滑 (时间域平滑第一层)
  ├─ Step C: 单帧处理 (上述9步)
  └─ Step D: 标定锁定/解锁 (时间域平滑第二层)
```

> 单帧处理支持独立调用（不含平滑）。视频模式默认开启平滑（`enable_smoothing=True`）。

---

## 11. 核心技术选型

| 类别 | 选型 | 说明 |
|------|------|------|
| 关键点检测 | YOLOv8/YOLO11 Pose | 26关键点，单类检测 |
| 平面映射 | `cv2.findHomography` (RANSAC) | 地面22点 → H |
| 球网检测 | Hough + YOLO 双路融合 | 互为冗余，交叉验证 |
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
| 阶段3：模块2 | 球网检测 + 标定 | 2.1 ~ 2.6 |
| 阶段4：集成验证 | 全流程串联 + 时间域平滑 + 端到端测试 | pipeline, temporal_filter |

→ 详细步骤见 [`docs/implementation_guide.md`](docs/implementation_guide.md)
