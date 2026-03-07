# 子模块1.1：YOLO Pose 检测器

## 功能概述

采用 Ultralytics YOLOv8-pose 或 YOLO11-pose 架构，对输入视频帧检测羽毛球场的26个关键点（22个地面 + 4个球网），输出每个关键点的像素坐标和置信度。这是整个流水线的第一步，为后续所有模块提供原始检测数据。

本模块封装**两个独立模型**的推理逻辑：
- **球场模型**：检测22个地面关键点（球场线交叉点）
- **球网模型**：检测4个球网角点（左上、右上、左下、右下）

两个模型的推理结果合并为统一的26关键点输出，通过 `keypoint_mapping` 模块将数据集索引转换为全局索引。

## 依赖关系

- **上游**：无（流水线入口）
- **下游**：Homography计算(1.2)、球网关键点提取(2.2)、相机标定(2.5)

## 对应代码文件

`module1/yolo_detector.py`

---

## 模型来源

推理所需的模型权重由 `training/` 模块离线训练产出，存放在 `yolo/weights/` 目录：

| 模型 | 权重文件 | 关键点数 | 数据集 |
|------|---------|---------|--------|
| 球场关键点 | `yolo/weights/court_yolo.pt` | 22 | `yolo/datasets/court/` |
| 球网关键点 | `yolo/weights/net_yolo.pt` | 4 | `yolo/datasets/net/` |

> 训练流程、数据集详情、关键点排序映射、超参数配置等详见 [`docs/training_guide.md`](training_guide.md)。

---

## 输入/输出数据结构

**输入**：BGR 图像帧（numpy array, H×W×3）

**输出**：`CourtDetectionResult`

```
CourtDetectionResult:
    keypoints: list[KeypointDetection]    # 26个关键点
    bbox: ndarray (4,)                    # 检测框 [x1,y1,x2,y2]
    bbox_confidence: float
    num_visible: int
    smoothed: bool = False                # 是否经过时间域平滑

KeypointDetection:
    index: int                    # 0-25（全局索引）
    pixel_xy: ndarray (2,)       # 像素坐标
    confidence: float            # 0-1
    visible: bool                # confidence >= threshold
```

---

## 算法详解

```
function detect(frame):
    results = model(frame, conf=0.25)
    if no detections: return None

    best = argmax(results.boxes.conf)         # 取最高置信度检测
    kpts_data = results.keypoints.data[best]  # shape (N, 3) → (x, y, conf)，N=22(球场)/4(球网)

    keypoints = []
    for i in range(26):
        x, y, conf = kpts_data[i]
        visible = (conf >= 0.5)
        keypoints.append(KeypointDetection(i, [x,y], conf, visible))

    return CourtDetectionResult(keypoints, bbox, bbox_conf, num_visible)
```

**关键点索引转换**：模型输出使用数据集原始排序，需通过 `training/keypoint_mapping.py`（球场）和 `training/keypoint_mapping_net.py`（球网）中的 `DATASET_TO_PLAN_INDEX` 映射表转换为全局索引后，再传递给下游模块。

---

## 辅助方法

- `get_visible_keypoints(min_conf=0.5)`：返回置信度达标的关键点列表
- `to_pixel_array(min_conf=0.5)`：返回 `(indices, pixel_coords)` 两个数组，方便下游使用
- `get_ground_keypoints(min_conf=0.5)`：仅返回地面关键点（编号0-21）
- `get_net_keypoints(min_conf=0.5)`：仅返回球网关键点（编号22-25）

---

## 测试方案

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| 模型加载 | 加载 .pt 文件并推理一张图 | 无异常，输出shape正确 |
| 检测精度 | 在10+标注测试图上运行 | OKS > 0.75 |
| 关键点范围 | 检查输出像素坐标在图像范围内 | 所有可见点在 [0,W]×[0,H] 内 |
| 批量推理 | 连续处理100帧 | 无内存泄漏，帧率稳定 |
