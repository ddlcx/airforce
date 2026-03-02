# 实施指南

## 功能概述

本文档提供项目的实施顺序、关键设计决策总结和 OpenCV 接口速查表，帮助开发者了解整体实施路线和技术选型。

---

## 实施顺序

### 阶段1：基础设施
1. 创建项目目录结构和所有 `__init__.py`
2. 实现 `config/court_config.py`（关键点坐标、线段定义、枚举类）
3. 实现 `utils/geometry.py`（直线交点、齐次投影）
4. 编写关键点坐标的对称性单元测试

### 阶段2：模块1 — 检测与 Homography
1. 实现 `module1/yolo_detector.py`
2. 实现 `module1/homography.py`
3. 实现 `module1/court_renderer.py` 和 `module1/net_overlay.py`
4. 编写合成 Homography 测试
5. 标注训练数据，训练 YOLO pose 模型
6. 在真实帧上验证

### 阶段3：模块2 — 球网检测与标定
1. 实现 `module2/net_top_detector_hough.py`
2. 实现 `module2/net_top_detector_yolo.py`
3. 实现 `module2/net_top_fusion.py`
4. 实现 `module2/net_endpoint_inference.py`
5. 实现 `module2/camera_calibration.py`（DLT + PnP）
6. 实现 `module2/projection_matrix.py`
7. 编写各子模块的合成测试

### 阶段4：集成与验证
1. 实现 `scripts/run_pipeline.py`
2. 实现 `utils/metrics.py`
3. 实现 `utils/temporal_filter.py`（KeypointSmoother + CalibrationLocker）
4. 在 `run_pipeline.py` 中集成时间域平滑
5. 端到端集成测试
6. 视频输出稳定性对比测试（开启/关闭平滑）
7. 参数调优
8. 误差分析

---

## 关键设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| 关键点数量 | 24 (20地面+4球网) | 地面20点覆盖全部线交叉，4球网点支持双路检测 |
| Homography | `cv2.findHomography` | OpenCV 成熟接口，RANSAC内置 |
| 点投影 | `cv2.perspectiveTransform` | 批量操作，数值稳定，替代手写齐次乘法 |
| 直线拟合 | `cv2.fitLine` | 支持多种距离度量，替代手写SVD |
| 重投影计算 | `cv2.projectPoints` | 支持畸变校正，替代手写 P@X |
| 标定主路径（固定机位） | 多帧平均 → DLT → K → PnP 精化 | 固定机位可多帧平均消噪 + DLT 无需 K 假设 |
| 标定路径（移动机位） | cv2.calibrateCamera → K+dist → PnP | 标准 Zhang 方法，需相机移动 |
| IAC 定位 | 交叉验证（非主路径） | 闭式近似，无 OpenCV 库，假设多，用于验证 DLT 的 K |
| EXIF 依赖 | 完全不依赖 | 视频容器通常无 EXIF 焦距信息 |
| 球网检测 | Hough + YOLO 双路 | 互为冗余，交叉验证提高鲁棒性 |
| 垂直假设 | 初始假设垂直，可迭代改进 | 手机拍摄通常近似水平，迭代可处理倾斜 |
| 最少点数 | H需4个, PnP需4个, DLT需6个非共面 | 工程冗余：通常可用10+个地面点 + 2个球网点 |
| 时间域平滑 | EMA（α=0.4）+ 标定锁定 | EMA 实现简单、对静态目标效果好；锁定策略消除渲染闪烁并提供遮挡容忍 |

---

## OpenCV 关键接口速查表

| 功能 | OpenCV 函数 | 方案中使用位置 |
|------|------------|--------------|
| Homography 计算 | `cv2.findHomography(src, dst, RANSAC, thresh)` | 模块1: homography.py |
| 2D点批量投影(Homography) | `cv2.perspectiveTransform(pts, H)` | 模块1: court_renderer.py, 模块2各处 |
| 直线拟合 | `cv2.fitLine(pts, DIST_L2, 0, 0.01, 0.01)` | 模块2: net_top_detector_yolo.py |
| 边缘检测 | `cv2.Canny(img, low, high)` | 模块2: net_top_detector_hough.py |
| 直线检测 | `cv2.HoughLinesP(edges, rho, theta, thresh)` | 模块2: net_top_detector_hough.py |
| PnP求解(RANSAC) | `cv2.solvePnPRansac(obj, img, K, dist)` | 模块2: camera_calibration.py |
| PnP精化(LM) | `cv2.solvePnPRefineLM(obj, img, K, dist, r, t)` | 模块2: camera_calibration.py |
| 3D→2D投影 | `cv2.projectPoints(obj, rvec, tvec, K, dist)` | 模块2: 验证+误差计算 |
| Rodrigues 转换 | `cv2.Rodrigues(vec_or_mat)` | 模块2: camera_calibration.py |
| 多帧相机标定 | `cv2.calibrateCamera(obj, img, size, K, dist)` | 模块2: 视频模式内参+畸变估计 |
| 畸变校正 | `cv2.undistortPoints(pts, K, dist)` | 模块2: 配合 calibrateCamera 使用 |
| P矩阵分解 | `cv2.decomposeProjectionMatrix(P)` | 模块2: DLT 标定（固定机位主路径） |
| 视频读取 | `cv2.VideoCapture(path)` | scripts/run_pipeline.py |

> 本方案中手写矩阵求解代码包括：DLT（固定机位主路径）和 IAC 焦距提取（交叉验证用）。`cv2.calibrateCamera` 在移动机位时使用，实现完整的 Zhang 标定法。
