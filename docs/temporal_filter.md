# 子模块3.1：时间域平滑（EMA + 标定锁定）

## 功能概述

解决 YOLO 关键点逐帧检测时 2-5px 随机抖动导致的球场线条闪烁问题。采用两层策略：第一层对关键点坐标进行 EMA 低通滤波，第二层在标定连续成功后锁定 H/P 矩阵。仅在视频模式下启用，单帧处理不使用。

## 依赖关系

- **上游**：YOLO检测(1.1) → `CourtDetectionResult`（第一层EMA平滑输入）；Homography(1.2) + 相机标定(2.5) → H/P（第二层锁定输入）
- **下游**：替换原始检测结果供后续所有模块使用

## 对应代码文件

`utils/temporal_filter.py`

---

## 问题背景

YOLO 关键点检测在逐帧处理时存在 2-5px 的随机抖动（jitter）。虽然 RANSAC 能处理离群点，但无法消除内点的微小波动。这导致：
- 叠加的球场线条在视频播放时产生视觉闪烁
- Homography 矩阵逐帧微跳，球场渲染不稳定
- 投影矩阵 P 不必要的波动

球场是**静态目标**，其关键点的真实像素位置变化仅来自相机运动，通常是平滑的。因此可以利用时间连续性来抑制检测噪声。

---

## 第一层：关键点级 EMA 平滑

**数据结构**：
```
class KeypointSmoother:
    alpha: float = 0.4              # 平滑系数（0-1，越小越平滑）
    smoothed: ndarray (24, 2)       # 平滑后的24个关键点坐标
    initialized: bool = False
    prev_confidences: ndarray (24,) # 上一帧各关键点的置信度
```

**算法伪代码**：
```
function smooth(detection: CourtDetectionResult) -> CourtDetectionResult:
    if not initialized:
        smoothed = detection.all_pixel_coords
        prev_confidences = detection.all_confidences
        initialized = True
        return detection

    for i in range(24):
        kp = detection.keypoints[i]
        if kp.visible:
            if prev_confidences[i] < 0.5:
                # 关键点从不可见变为可见 → 直接采用新值
                smoothed[i] = kp.pixel_xy
            else:
                # 正常 EMA 平滑
                smoothed[i] = alpha * kp.pixel_xy + (1 - alpha) * smoothed[i]

    prev_confidences = detection.all_confidences

    smoothed_detection = detection.copy_with_coords(smoothed)
    smoothed_detection.smoothed = True
    return smoothed_detection
```

> **设计说明**：
> - `alpha=0.4`：对于30fps视频，等效时间常数约 1/(0.4×30) ≈ 0.08秒（~2.5帧）。稳态下可将标准差从 σ 降至 0.5σ
> - 关键点从不可见变为可见时不做平滑：避免从旧位置缓慢滑动
> - 选择 EMA 而非卡尔曼滤波：球场关键点运动简单，EMA 足够且实现最简

---

## 第二层：标定锁定策略

**数据结构**：
```
class CalibrationLocker:
    lock_threshold: int = 10            # 连续成功帧数达到此值时锁定
    unlock_deviation: float = 15.0      # 锁定后偏差超过此值(px)时解锁
    consecutive_success: int = 0
    locked: bool = False
    locked_H: ndarray (3,3)
    locked_P: ndarray (3,4)
```

**算法伪代码**：
```
function update(frame_result) -> frame_result:
    if not frame_result.success:
        consecutive_success = 0
        if locked:
            # 标定失败但已锁定 → 沿用锁定值（短暂遮挡容忍）
            frame_result.homography.H = locked_H
            frame_result.calibration.P = locked_P
            frame_result.success = True
            frame_result.source = "locked"
        return frame_result

    if not locked:
        consecutive_success += 1
        if consecutive_success >= lock_threshold:
            locked = True
            locked_H = frame_result.homography.H
            locked_P = frame_result.calibration.P
        return frame_result

    else:  # locked == True
        deviation = compute_h_deviation(frame_result.homography.H, locked_H)
        if deviation > unlock_deviation:
            locked = False
            consecutive_success = 1
            return frame_result
        else:
            # 偏差小 → 缓慢适应
            locked_H = 0.95 * locked_H + 0.05 * frame_result.homography.H
            locked_H = locked_H / locked_H[2,2]
            locked_P = 0.95 * locked_P + 0.05 * frame_result.calibration.P
            locked_P = locked_P / np.linalg.norm(locked_P[2,:3])
            frame_result.homography.H = locked_H
            frame_result.calibration.P = locked_P
            return frame_result
```

**偏差度量**：
```
function compute_h_deviation(H_new, H_locked):
    corners = np.array([[-3.05,-6.70], [3.05,-6.70], [-3.05,6.70], [3.05,6.70]])
    pts_new = project_points_batch(H_new, corners)
    pts_locked = project_points_batch(H_locked, corners)
    return np.max(np.linalg.norm(pts_new - pts_locked, axis=1))
```

## 设计要点

- 锁定需连续 10 帧成功（≈0.33秒@30fps），防止偶然一帧成功就锁定
- 锁定后检测失败时沿用锁定值，实现短暂遮挡容忍
- 解锁使用球场角点最大偏差（像素级，直观易理解）
- 锁定后仍以 5% 权重缓慢吸收新帧的 H/P，适应微小晃动

## 测试方案

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| EMA 收敛 | 输入恒定坐标+2px高斯噪声，运行100帧 | 平滑后标准差 < 0.8px |
| EMA 响应速度 | 在第50帧突然偏移20px | 5帧内跟上新位置（误差<2px） |
| 可见性切换 | 关键点第10帧消失第20帧出现在新位置 | 第20帧直接跳到新位置，无漂移 |
| 锁定触发 | 连续输入稳定的标定结果 | 第10帧后 locked=True |
| 遮挡容忍 | 锁定后输入3帧检测失败 | 输出沿用锁定值，success=True |
| 相机移动解锁 | 锁定后 H 偏差突然 > 15px | 自动解锁并重新标定 |
| 视频稳定性对比 | 同一视频分别开启/关闭平滑，对比叠加渲染 | 开启平滑后线条无闪烁 |
