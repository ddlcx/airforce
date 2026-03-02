# 子模块2.2：YOLO 球网顶线检测

## 功能概述

直接利用 YOLO 检测结果中的球网关键点（编号22-24）拟合球网顶部线。这是球网检测的两条独立路径之一，与 Hough 路径(2.1)互为冗余。相比 Hough 路径，YOLO 路径更直接但依赖模型对球网关键点的检测质量。

## 依赖关系

- **上游**：YOLO Pose 检测器(1.1) → `CourtDetectionResult`（关键点22-24）
- **下游**：双路融合(2.3)

## 对应代码文件

`module2/net_top_detector_yolo.py`

---

## 输入/输出数据结构

**输入**：`CourtDetectionResult`（仅使用关键点22、23、24）

**输出**：
```
NetTopLineResult:
    endpoint1: ndarray (2,)      # 像素坐标
    endpoint2: ndarray (2,)
    line_coeffs: ndarray (3,)    # ax+by+c=0（归一化）
    confidence: float            # 球网关键点平均置信度
    roi_bbox: None               # YOLO路径不使用ROI
    source: str                  # "yolo"
```

## 算法详解

```
function detect_net_top_yolo(detection):
    # 提取球网顶部关键点 (编号22, 23, 24)
    net_kpts = []
    for idx in [22, 23, 24]:
        kp = detection.keypoints[idx]
        if kp.visible and kp.confidence >= 0.5:
            net_kpts.append(kp)

    if len(net_kpts) < 2:
        return None    # 至少需要2个点才能定义直线

    # 提取像素坐标
    points = array([kp.pixel_xy for kp in net_kpts])    # shape (M, 2)

    if len(points) == 2:
        # 两点直接连线
        ep1, ep2 = points[0], points[1]
    else:
        # 三点及以上：使用 cv2.fitLine 进行鲁棒直线拟合
        vx, vy, x0, y0 = cv2.fitLine(
            points.astype(np.float32),
            distType=cv2.DIST_L2,  # 最小二乘距离
            param=0, reps=0.01, aeps=0.01
        )
        vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

        # 用拟合的方向向量和基准点，求出端点
        x_min, x_max = min(points[:,0]), max(points[:,0])
        if abs(vx) > 1e-10:
            t_min = (x_min - x0) / vx
            t_max = (x_max - x0) / vx
            ep1 = [x0 + vx * t_min, y0 + vy * t_min]
            ep2 = [x0 + vx * t_max, y0 + vy * t_max]
        else:
            ep1, ep2 = points[0], points[-1]

    # 计算归一化直线方程 ax+by+c=0
    dx, dy = ep2 - ep1
    a, b, c = -dy, dx, -((-dy)*ep1[0] + dx*ep1[1])
    norm = sqrt(a^2 + b^2)
    line_coeffs = [a/norm, b/norm, c/norm]

    # 置信度取球网关键点的平均置信度
    avg_conf = mean([kp.confidence for kp in net_kpts])

    return NetTopLineResult(ep1, ep2, line_coeffs, avg_conf, None, "yolo")
```

## 设计要点

- 使用 `cv2.fitLine`（L2距离）进行直线拟合，替代手写 SVD
- 两点时直接连线，三点时拟合
- 置信度直接取自 YOLO 检测结果，反映模型对球网区域的检测质量

## 测试方案

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| 3点共线 | 输入精确共线的3点 | 拟合误差 = 0 |
| 3点有噪声 | 加2px高斯噪声 | 拟合误差 < 1px |
| 仅2点 | 只提供关键点22和23 | 直线方向正确 |
| 1点或0点 | 不足2个可见球网点 | 返回None |
