# 子模块2.6：投影矩阵验证

## 功能概述

对相机标定输出的投影矩阵 P 进行6项质量检查，确保 P 在物理上合理且与模块1的 Homography H 一致。验证不通过则不输出最终结果。

## 依赖关系

- **上游**：相机标定(2.5) → `CameraCalibrationResult`；Homography(1.2) → H（用于 H-P 一致性检查）
- **下游**：最终输出门控

## 对应代码文件

`module2/projection_matrix.py`

---

## 输入/输出数据结构

**输入**：P, K, R, rvec, tvec, dist_coeffs, 3D/2D对应点, H

**输出**：验证指标字典（含 `overall_ok` 布尔值）

## 算法详解

```
function validate_projection_matrix(P, obj_pts, img_pts, K, R, rvec, tvec, dist_coeffs):
    metrics = {}

    # 1. 重投影误差（使用 cv2.projectPoints）
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist_coeffs)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - img_pts, axis=1)
    metrics['mean_reproj'] = mean(errors)
    metrics['max_reproj'] = max(errors)
    metrics['reproj_ok'] = mean(errors) < 5.0

    # 2. 相机位置合理性
    camera_pos = -R.T @ tvec
    cam_height = camera_pos[2]
    cam_dist = norm(camera_pos)
    metrics['cam_height'] = cam_height
    metrics['cam_dist'] = cam_dist
    metrics['cam_ok'] = cam_height > 0 and 2 < cam_height < 15 and 3 < cam_dist < 60

    # 3. 旋转矩阵正交性
    orth_error = norm(R @ R.T - eye(3))
    metrics['orth_ok'] = orth_error < 1e-6

    # 4. 内参合理性
    fx, fy = K[0,0], K[1,1]
    metrics['fx'] = fx
    metrics['fy'] = fy
    metrics['intrinsics_ok'] = fx > 0 and fy > 0 and abs(fx/fy - 1) < 0.5

    # 5. 球场中心投影（使用 cv2.projectPoints）
    center_proj, _ = cv2.projectPoints(
        np.array([[0.0, 0.0, 0.0]]), rvec, tvec, K, dist_coeffs)
    metrics['center_px'] = center_proj.reshape(2)

    # 6. H-P 一致性
    H_from_P = P[:, [0, 1, 3]]
    H_from_P_norm = H_from_P / norm(H_from_P)
    H_norm = H / norm(H)
    consistency = min(norm(H_norm - H_from_P_norm), norm(H_norm + H_from_P_norm))
    metrics['hp_consistency'] = consistency
    metrics['hp_ok'] = consistency < 0.05

    metrics['overall_ok'] = all checks pass
    return metrics
```

### H-P一致性检查说明

当 Z=0 时，投影矩阵 P 退化为：
```
P @ [X, Y, 0, 1]^T = [P[:,0], P[:,1], P[:,3]] @ [X, Y, 1]^T
```

因此 `H_from_P = P[:, [0,1,3]]` 应与模块1计算的 Homography H 成比例。这是一个强有力的端到端一致性验证。

## 设计要点

- 6项检查覆盖数值正确性（重投影）、物理合理性（相机位置、内参）和端到端一致性（H-P）
- 全部使用 OpenCV 接口计算，保持与标定模块一致
- `overall_ok` 作为最终门控，决定是否输出结果

## 测试方案

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| 合成数据全通过 | 已知K,R,t生成合成数据 | 6项检查全部通过 |
| 异常K检测 | 输入fx=0的K | intrinsics_ok = False |
| 异常相机位置 | 相机高度<0 | cam_ok = False |
| H-P一致性 | 使用同一组点的H和P | hp_consistency < 0.01 |
| H-P不一致 | 人工构造偏差较大的P | hp_ok = False |
