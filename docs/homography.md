# 子模块1.2：Homography 计算与验证

## 功能概述

从 YOLO 检测到的地面关键点（编号0-21，Z=0平面）计算 Homography 矩阵 H，建立标准球场2D坐标到图像像素坐标的映射关系，并通过5项验证确保 H 的质量。H 是后续球场渲染和模块2的核心输入。

## 依赖关系

- **上游**：YOLO Pose 检测器(1.1) → `CourtDetectionResult`
- **下游**：球场线条渲染(1.3)、Hough球网检测(2.1)、球网端点推断(2.4)、相机标定(2.5)、投影矩阵验证(2.6)
- **基础数据**：[base_definitions.md](base_definitions.md) 中的 `COURT_KEYPOINTS_2D`

## 对应代码文件

`module1/homography.py`

---

## 算法说明

Homography 矩阵 H 描述标准球场地面平面（2D世界坐标，单位米）到图像像素坐标的映射：

```
s × [u, v, 1]^T = H × [X_court, Y_court, 1]^T
```

其中 s 是齐次因子。**仅使用22个地面关键点（编号0-21）** 计算 Homography，球网关键点（编号22-25）不参与Homography计算（它们不在Z=0平面上）。

## 输入/输出数据结构

**输入**：
- `CourtDetectionResult`（YOLO检测结果）
- `COURT_KEYPOINTS_2D`（22×2 标准球场坐标数组）
- `min_confidence`：关键点置信度阈值（默认0.5）
- `ransac_threshold`：RANSAC重投影阈值（默认5.0px）

**输出**：
```
HomographyResult:
    H: ndarray (3,3)          # 球场坐标 → 像素坐标
    H_inv: ndarray (3,3)      # 像素坐标 → 球场坐标
    inlier_mask: ndarray (N,)  # 布尔数组，标记内点
    num_inliers: int
    num_correspondences: int
    reprojection_error: float  # 内点平均重投影误差(px)
    used_indices: ndarray      # 使用的关键点编号
    used_court_pts: ndarray    # 使用的球场坐标
    used_pixel_pts: ndarray    # 使用的像素坐标
```

## 算法详解

```
function compute_homography(detection, court_keypoints_2d, min_conf=0.5, ransac_thresh=5.0):
    # 1. 提取高置信度的地面关键点
    indices, pixel_pts = detection.get_ground_keypoints(min_conf)

    if len(indices) < 4:
        return None    # 至少需要4个点

    # 2. 查表获取对应的标准球场坐标
    court_pts = court_keypoints_2d[indices]    # shape (N, 2)

    # 3. RANSAC 求解 Homography
    H, mask = cv2.findHomography(
        srcPoints = court_pts,     # 源：球场坐标(米)
        dstPoints = pixel_pts,     # 目标：像素坐标
        method = cv2.RANSAC,
        ransacReprojThreshold = ransac_thresh
    )

    if H is None:
        return None

    # 4. 计算重投影误差（使用 cv2.perspectiveTransform）
    court_pts_input = court_pts.reshape(1, -1, 2).astype(np.float32)
    projected_2d = cv2.perspectiveTransform(court_pts_input, H.astype(np.float32))
    projected_2d = projected_2d.reshape(-1, 2)
    errors = ||projected_2d - pixel_pts||_2 per row
    mean_error = mean(errors[inliers])

    # 5. 计算逆矩阵
    H_inv = inv(H)

    return HomographyResult(H, H_inv, mask, ...)
```

## Homography 验证（`validate_homography`）

对计算得到的 H 进行5项质量检查：

```
function validate_homography(result):
    metrics = {}

    # 检查1：重投影误差
    metrics['reprojection_error'] = result.reprojection_error
    metrics['reproj_ok'] = result.reprojection_error < 5.0   # 像素

    # 检查2：内点比例
    inlier_ratio = result.num_inliers / result.num_correspondences
    metrics['inlier_ratio'] = inlier_ratio
    metrics['inlier_ok'] = inlier_ratio > 0.7

    # 检查3：行列式符号（应保持方向，det > 0）
    det_H = det(result.H)
    metrics['det_ok'] = det_H > 0

    # 检查4：条件数（不应过大）
    cond = cond(result.H)
    metrics['cond_ok'] = cond < 1e6

    # 检查5：奇异值比（最大/最小）
    U, S, Vt = svd(result.H)
    sv_ratio = S[0] / S[-1]
    metrics['sv_ok'] = sv_ratio < 1e4

    metrics['overall_ok'] = all checks pass
    return metrics
```

**验证门控**：如果 `overall_ok = False`，则不进入模块2，输出警告日志。

## 设计要点

- 使用 `cv2.findHomography` 内置 RANSAC，无需手写
- 仅使用地面关键点（Z=0），保证单平面映射的数学正确性
- 5项验证覆盖了数值稳定性（条件数、奇异值比）和几何合理性（行列式、内点比例、重投影）

## 测试方案

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| 合成恢复 | 用已知H投影22个点，加高斯噪声(σ=2px)，恢复H | 重投影 < 3px |
| 最少4点 | 仅用4个关键点计算H | H不为None且重投影 < 10px |
| 遮挡降级 | 随机移除50%关键点 | 重投影 < 8px |
| 验证函数 | 对合成数据运行validate_homography | 全部检查通过 |
| 退化检测 | 输入共线3点 | 返回None或验证失败 |
