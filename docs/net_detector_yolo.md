# 子模块2.2：球网关键点提取

## 功能概述

从 YOLO 检测结果提取球网 4 个关键点（kp22-25），直接使用 YOLO 模型原始值。

## 依赖关系

- **上游**：YOLO Pose 检测器(1.1) → `CourtDetectionResult`（关键点 22-25）
- **下游**：相机标定(2.5)

## 对应代码文件

`module2/net_top_detector_yolo.py`

---

## 输入/输出数据结构

**输入**：
- `CourtDetectionResult`（关键点 22-25）

**输出**：
```
NetKeypointResult:
    left_top_pixel: ndarray (2,)      # kp22 像素坐标
    right_top_pixel: ndarray (2,)     # kp24 像素坐标
    left_base_pixel: ndarray (2,) | None  # kp23 像素坐标
    right_base_pixel: ndarray (2,) | None # kp25 像素坐标
    left_top_3d: ndarray (3,)         # (-3.05, 0, 1.55) 固定值
    right_top_3d: ndarray (3,)        # (+3.05, 0, 1.55) 固定值
    left_base_3d: ndarray (3,)        # (-3.05, 0, 0) 固定值
    right_base_3d: ndarray (3,)       # (+3.05, 0, 0) 固定值
```

## 算法详解

### 主函数 `extract_net_keypoints`

```
function extract_net_keypoints(detection, min_conf=0.5):
    # Step 1: 提取顶部关键点
    kp22 = detection.keypoints[22]  # 网左端顶部
    kp24 = detection.keypoints[24]  # 网右端顶部
    if kp22.confidence < min_conf or kp24.confidence < min_conf:
        return None

    # Step 2: 提取底部关键点
    kp23 = detection.keypoints[23]  # 网左端底部
    kp25 = detection.keypoints[25]  # 网右端底部
    left_base = kp23.pixel_xy if kp23.confidence >= min_conf else None
    right_base = kp25.pixel_xy if kp25.confidence >= min_conf else None

    return NetKeypointResult(...)
```

## 设计要点

- 至少需要 kp22 和 kp24（顶部两点）可见，否则返回 None
- 底部关键点直接从 YOLO 球网模型提取（kp23/kp25）
- 3D 坐标为固定值，来自标准羽毛球场尺寸
- 直接使用 YOLO 模型原始值，不依赖 Homography 矩阵

## 测试方案

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| 顶部关键点提取 | kp22/kp24 均可见 | 返回有效 NetKeypointResult |
| 顶部不足 | kp22 或 kp24 不可见 | 返回 None |
| 底部关键点提取 | kp23/kp25 均可见 | 返回有效底部坐标 |
