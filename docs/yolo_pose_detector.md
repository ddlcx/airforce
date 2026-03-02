# 子模块1.1：YOLO Pose 检测器

## 功能概述

采用 Ultralytics YOLOv8-pose 或 YOLO11-pose 架构，对输入视频帧检测羽毛球场的26个关键点（22个地面 + 4个球网），输出每个关键点的像素坐标和置信度。这是整个流水线的第一步，为后续所有模块提供原始检测数据。

## 依赖关系

- **上游**：无（流水线入口）
- **下游**：Homography计算(1.2)、YOLO球网检测(2.2)、球网端点推断(2.4)

## 对应代码文件

`module1/yolo_detector.py`

---

## 模型配置

**数据集 YAML**（`data/dataset/badminton_court.yaml`）：
```yaml
path: /path/to/data/dataset
train: images/train
val: images/val

kpt_shape: [26, 3]      # 26关键点，每个 (x, y, visibility)
flip_idx: [1,0, 3,2, 5,4, 7,6, 9,8, 11,10, 13,12, 15,14, 17,16, 18,19, 20,21, 23,22, 24,25]
names:
  0: badminton_court
```

配置要点说明：
- `kpt_shape: [26, 3]`：26个关键点（22地面 + 4球网），每个标注为 (x_normalized, y_normalized, visibility_flag)
- visibility 标注规则：0 = 不在画面中且未标注；1 = 在画面中但被遮挡（标注了位置）；2 = 可见
- `flip_idx`：水平翻转时的关键点索引映射（详见 [base_definitions.md](base_definitions.md)）

## 标注格式

每张图片的标签文件（`.txt`）包含一行：
```
class_id  cx  cy  w  h  x0 y0 v0  x1 y1 v1  ...  x25 y25 v25
```
- `class_id`：固定为 0（badminton_court）
- `cx, cy, w, h`：整个可见球场的 bounding box（归一化）
- `xi, yi`：关键点坐标（归一化至 [0,1]）
- `vi`：visibility flag（0/1/2）

## 数据标注与准备

**标注平台**：推荐使用 Roboflow 免费层进行标注（支持关键点标注），完成后导出为 **"YOLOv8 Pose"** 格式，下载到本地。

**不需要在 Roboflow 上付费训练**。本地训练只需：
```bash
pip install ultralytics
```

标注导出后的目录结构应为：
```
data/dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── badminton_court.yaml
```

## 本地训练配置

```python
from ultralytics import YOLO

model = YOLO("yolo11m-pose.yaml")    # 从头开始
# 或 model = YOLO("yolo11m-pose.pt") # 从预训练权重开始（推荐）

model.train(
    data="data/dataset/badminton_court.yaml",
    epochs=200,
    imgsz=640,
    batch=16,
    lr0=0.01,
    lrf=0.001,
    mosaic=0.5,           # 降低 mosaic 比例，避免破坏球场几何一致性
    degrees=5.0,           # 轻微旋转增强
    translate=0.1,
    scale=0.3,
    flipud=0.0,            # 禁止上下翻转（球场有方向性）
    fliplr=0.5,            # 水平翻转配合 flip_idx
    perspective=0.0005,    # 轻微透视增强
    device="0",            # 使用 GPU 0
)
```

> 训练完成后模型保存在 `runs/pose/train/weights/best.pt`。有显卡的本地环境训练比 Roboflow 付费训练更灵活，可随时调参。

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
    index: int                    # 0-25
    pixel_xy: ndarray (2,)       # 像素坐标
    confidence: float            # 0-1
    visible: bool                # confidence >= threshold
```

## 算法详解

```
function detect(frame):
    results = model(frame, conf=0.25)
    if no detections: return None

    best = argmax(results.boxes.conf)         # 取最高置信度检测
    kpts_data = results.keypoints.data[best]  # shape (26, 3) → (x, y, conf)

    keypoints = []
    for i in range(26):
        x, y, conf = kpts_data[i]
        visible = (conf >= 0.5)
        keypoints.append(KeypointDetection(i, [x,y], conf, visible))

    return CourtDetectionResult(keypoints, bbox, bbox_conf, num_visible)
```

**辅助方法**：
- `get_visible_keypoints(min_conf=0.5)`：返回置信度达标的关键点列表
- `to_pixel_array(min_conf=0.5)`：返回 `(indices, pixel_coords)` 两个数组，方便下游使用
- `get_ground_keypoints(min_conf=0.5)`：仅返回地面关键点（编号0-21）
- `get_net_keypoints(min_conf=0.5)`：仅返回球网关键点（编号22-25）

## 测试方案

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| 模型加载 | 加载 .pt 文件并推理一张图 | 无异常，输出shape正确 |
| 检测精度 | 在10+标注测试图上运行 | OKS > 0.75 |
| 关键点范围 | 检查输出像素坐标在图像范围内 | 所有可见点在 [0,W]×[0,H] 内 |
| 批量推理 | 连续处理100帧 | 无内存泄漏，帧率稳定 |
