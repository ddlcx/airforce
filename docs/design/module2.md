# 模块2实现计划：球网检测与相机标定

## Context

模块1（球场检测与渲染）已完成：YOLO 推理封装输出 26 关键点的 `CourtDetectionResult`，Homography 计算输出 `HomographyResult`（含 H 矩阵），投影工具函数 `project_points_batch` 可用。

模块2 从球网关键点定位中推断相机完整 3D 投影参数。包含 3 个子模块（2.2, 2.5, 2.6）。

## 需要创建/修改的文件

```
config/
└── court_config.py              # 修改：新增 COURT_KEYPOINTS_3D (22,3)

module2/
├── __init__.py                  # 空
├── net_top_detector_yolo.py     # 2.2 球网关键点提取
├── camera_calibration.py        # 2.5 相机标定
└── projection_validation.py     # 2.6 投影矩阵验证

scripts/
└── run_module2.py               # 独立运行入口
```

---

## Step 0: 修改 `config/court_config.py`

**新增内容**：
```python
# 22个地面关键点的3D坐标 (Z=0)，供相机标定使用
COURT_KEYPOINTS_3D = np.hstack([
    COURT_KEYPOINTS_2D,
    np.zeros((NUM_GROUND_KEYPOINTS, 1))
])  # shape (22, 3)

```

---

## Step 1: `module2/net_top_detector_yolo.py` — 球网关键点提取

从 YOLO 球网模型提取球网 4 个关键点（kp22-25），直接使用 YOLO 模型原始值。

**数据结构**：`NetKeypointResult`
**主函数**：`extract_net_keypoints(detection, min_conf)`

---

## Step 2: `module2/camera_calibration.py` — 相机标定

**策略**：DLT → PnP 精化 → IAC 交叉验证，fallback 到 IAC → PnP 或 FOV=55° → PnP。

---

## Step 3: `module2/projection_validation.py` — 投影矩阵验证

6 项检查：重投影误差、相机位置、R正交性、内参合理性、球场中心投影、H-P一致性。

---

## Step 4: `scripts/run_module2.py` — 独立运行入口

---

## 关键设计决策

1. **直接使用 YOLO 原始值**：YOLO 球网模型精度高，直接使用原始关键点值，`extract_net_keypoints` 不依赖 H 矩阵，降低模块间耦合
2. **聚焦单帧标定**：核心单帧 DLT → PnP 流程
3. **复用已有代码**：`project_points_batch`、`line_intersection`、`HomographyResult` 等

## 模块间数据流

```
CourtDetectionResult (1.1)
    ├── get_ground_keypoints() → HomographyResult (1.2) → H
    ├── kp22-25 → extract_net_keypoints (2.2) → NetKeypointResult
    │
    ├── ground_pts + net_kpts → calibrate_frame (2.5) → CameraCalibrationResult
    │
    └── calibration + H → validate_projection (2.6) → metrics dict
```
