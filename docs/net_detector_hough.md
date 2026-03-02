# 子模块2.1：Hough 球网顶线检测

## 功能概述

利用 Homography 定义球网区域的 ROI，在 ROI 内用传统计算机视觉方法（Canny 边缘检测 + HoughLinesP 直线检测）检测球网顶部边缘线。这是球网检测的两条独立路径之一，与 YOLO 路径(2.2)互为冗余。

## 依赖关系

- **上游**：Homography 计算(1.2) → `HomographyResult.H`；球场渲染(1.3) → `project_points_batch`
- **下游**：双路融合(2.3)

## 对应代码文件

`module2/net_top_detector_hough.py`

---

## 输入/输出数据结构

**输入**：BGR帧图像 + Homography H

**输出**：
```
NetTopLineResult:
    endpoint1: ndarray (2,)      # 像素坐标
    endpoint2: ndarray (2,)
    line_coeffs: ndarray (3,)    # ax+by+c=0（归一化）
    confidence: float
    roi_bbox: ndarray (4,)       # ROI框 [x1,y1,x2,y2]
    source: str                  # "hough"
```

## 算法详解

```
function detect_net_top_hough(frame, H):
    # Step 1: 定义 ROI（使用 cv2.perspectiveTransform 批量投影）
    roi_corners_court = np.array([[-3.55,-2], [+3.55,-2], [+3.55,+2], [-3.55,+2]])
    roi_corners_pixel = project_points_batch(H, roi_corners_court)
    roi_bbox = bounding_box(roi_corners_pixel) + margin(20px, 50px)
    roi_bbox = clip_to_image(roi_bbox, frame.shape)

    # Step 2: 提取 ROI 并预处理
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), sigma=1.5)
    edges = cv2.Canny(blurred, 50, 150)

    # Step 3: Hough 线检测
    lines = cv2.HoughLinesP(edges, rho=1, theta=π/180,
                             threshold=80,
                             minLineLength=100,
                             maxLineGap=20)
    if lines is None: return None

    # Step 4: 计算参考角度（球网地面线在图像中的倾斜角）
    net_bases_px = project_points_batch(H, np.array([[-3.05,0], [+3.05,0]]))
    net_left_px, net_right_px = net_bases_px[0], net_bases_px[1]
    expected_angle = atan2(net_right_px[1]-net_left_px[1],
                           net_right_px[0]-net_left_px[0])

    # Step 5: 筛选与评分
    best_line, best_score = None, -1
    for line in lines:
        lx1, ly1, lx2, ly2 = line (in ROI coords)
        # 转换到帧坐标
        lx1_f, ly1_f = lx1 + x1, ly1 + y1
        lx2_f, ly2_f = lx2 + x1, ly2 + y1

        # 筛选a: 角度约束
        angle = atan2(ly2-ly1, lx2-lx1)
        angle_diff = angular_difference(angle, expected_angle)
        if degrees(angle_diff) > 15: continue

        # 筛选b: 位置约束 — 球网顶线应在地面线之上（y值更小）
        net_ground_y = (net_left_px[1] + net_right_px[1]) / 2
        line_center_y = (ly1_f + ly2_f) / 2
        if line_center_y > net_ground_y: continue   # 在下方，跳过

        # 评分
        length = sqrt((lx2-lx1)^2 + (ly2-ly1)^2)
        angle_score = 1.0 - degrees(angle_diff) / 15.0
        score = length * angle_score
        if score > best_score:
            best_score = score
            best_line = ([lx1_f,ly1_f], [lx2_f,ly2_f])

    if best_line is None: return None

    # Step 6: 计算直线方程 ax+by+c=0
    ep1, ep2 = best_line
    dx, dy = ep2 - ep1
    a, b, c = -dy, dx, -((-dy)*ep1[0] + dx*ep1[1])
    norm = sqrt(a^2 + b^2)
    line_coeffs = [a/norm, b/norm, c/norm]

    return NetTopLineResult(ep1, ep2, line_coeffs, score/1000, roi_bbox, "hough")
```

## 设计要点

- ROI 由 Homography 定义，自适应不同拍摄角度
- 角度约束使用球网地面线的投影角度作为参考（容差15°）
- 位置约束要求球网顶线在地面线之上（y值更小）
- 评分公式：线段长度 × 角度匹配度

## 测试方案

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| ROI覆盖 | 在真实帧上可视化ROI区域 | ROI完全包含球网 |
| 合成检测 | 在纯色背景上画已知角度的线 | 检测角度误差 < 1° |
| 真实帧检测 | 在10+帧上运行，与手工标注比较 | 角度误差 < 5°, 位置误差 < 5px |
| 无球网场景 | 输入无球网的图像 | 返回None |
