# 子模块2.5：相机标定

## 功能概述

从3D-2D对应点（地面关键点Z=0 + 球网端点Z=1.55）求解相机完整参数：内参矩阵 K、外参 R/t、投影矩阵 P = K[R|t]。根据相机是固定机位还是移动机位，采用不同标定策略。这是整个系统中算法最复杂的子模块，包含 DLT、PnP、calibrateCamera、IAC 四种方法。

## 依赖关系

- **上游**：YOLO检测(1.1) → 地面关键点；球网端点推断(2.4) → `NetEndpointResult`；Homography(1.2) → H（用于IAC交叉验证）
- **下游**：投影矩阵验证(2.6)
- **基础数据**：[base_definitions.md](base_definitions.md) 中的 `COURT_KEYPOINTS_3D`

## 对应代码文件

`module2/camera_calibration.py`

---

## 关键点使用总览

| 步骤 | 输入关键点 | 说明 |
|------|-----------|------|
| Homography H 计算 | 地面关键点 0-21（Z=0），≥4个可见 | 仅平面点，已在模块1完成 |
| 多帧关键点平均（固定机位） | 多帧地面+球网关键点 | 固定机位独有优势：N帧平均 → σ/√N |
| DLT → P → 分解 K（**固定机位主路径**） | 地面点 + 球网端点，≥6个非共面点 | 无需任何内参假设，直接求出 K, R, t |
| IAC 从 H 估计 f（交叉验证） | 不需要额外关键点，直接从 H 矩阵提取 | 验证 DLT 的 K 是否合理 |
| cv2.calibrateCamera（**仅移动机位**） | 多帧地面关键点 0-21（Z=0），每帧≥6个 | 需要相机移动，固定机位不可用 |
| PnP 求外参 | **全部**可用的 3D-2D 对应：地面点(0-21) + 球网顶部端点(22,23) | 使用 DLT 或 calibrateCamera 得到的 K |

## 输出数据结构

```
CameraCalibrationResult:
    K: ndarray (3,3)              # 内参矩阵
    dist_coeffs: ndarray (5,)     # 畸变系数（calibrateCamera 输出，否则为 zeros）
    rvec: ndarray (3,1)           # Rodrigues 旋转向量
    tvec: ndarray (3,1)           # 平移向量
    R: ndarray (3,3)              # 旋转矩阵
    P: ndarray (3,4)              # 投影矩阵 P = K @ [R|t]
    reprojection_error: float     # 平均重投影误差(px)
    num_points_used: int
    method: str                   # "dlt_pnp" / "dlt" / "calibrate_camera" / "iac_pnp" / "fallback_pnp"
    k_estimation_method: str      # "dlt" / "calibrate_camera" / "iac_consistent" / "iac_single" / "fallback_fov"
    iac_cross_check: dict|None    # {"f_iac": float, "f_dlt": float, "consistent": bool}
```

---

## 算法详解

### 1. 3D-2D 对应关系构建

```
function build_3d_2d_correspondences(detection, court_keypoints_3d, net_endpoints):
    # 地面关键点（Z=0）
    indices, pixel_pts = detection.get_ground_keypoints(min_conf=0.5)
    ground_3d = court_keypoints_3d[indices]      # (M, 3), Z列全为0
    ground_2d = pixel_pts                         # (M, 2)

    # 球网顶部端点（Z=1.55）
    net_3d = [net_endpoints.left_top_3d, net_endpoints.right_top_3d]
    net_2d = [net_endpoints.left_top_pixel, net_endpoints.right_top_pixel]

    # 合并
    object_points = vstack([ground_3d, net_3d])    # (M+2, 3)
    image_points = vstack([ground_2d, net_2d])      # (M+2, 2)

    return object_points, image_points
```

> **关键**：地面点全在 Z=0 平面，加入 Z=1.55 的球网点使点集变为非共面。这对 DLT 至关重要——纯共面点导致 DLT 矩阵 rank deficient。PnP 也受益于非共面点。

### 2. 多帧关键点平均（固定机位核心步骤）

固定机位的**独特优势**：同一场景的多帧观测可以平均，大幅降低 YOLO 关键点检测噪声。

```
function average_keypoints_fixed_camera(frame_detections, n_frames=50):
    """固定机位：对多帧关键点坐标取中位数，降低检测噪声。
    50帧中位数：σ=3px → σ_median ≈ 0.53px
    """
    keypoint_coords = defaultdict(list)

    for detection in frame_detections[:n_frames]:
        for kp_id, (x, y, conf) in detection.keypoints.items():
            if conf > 0.5:
                keypoint_coords[kp_id].append((x, y))

    averaged = {}
    for kp_id, coords in keypoint_coords.items():
        if len(coords) >= n_frames * 0.5:
            coords = np.array(coords)
            averaged[kp_id] = np.median(coords, axis=0)

    return averaged
```

> 使用**中位数**而非均值，对 YOLO 偶尔的外点更鲁棒。50帧约1.7秒@30fps。

### 3. DLT（固定机位主路径）

**定位**：固定机位的**推荐主路径**。直接从非共面 3D-2D 对应点求出 3×4 投影矩阵 P，然后用 `cv2.decomposeProjectionMatrix` 分解出 K, R, t，**无需任何关于内参 K 的先验假设**。

```
function calibrate_dlt(object_points_3d, image_points_2d):
    N = len(object_points_3d)
    assert N >= 6

    # 构建 DLT 矩阵 A (2N × 12)
    A = zeros(2*N, 12)
    for i in range(N):
        X, Y, Z = object_points_3d[i]
        u, v = image_points_2d[i]
        A[2*i]   = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]

    _, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    P = P / np.linalg.norm(P[2, :3])
    if np.linalg.det(P[:, :3]) < 0:
        P = -P

    K, R, t_h, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    K = K / K[2,2]
    t = (t_h[:3] / t_h[3]).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)

    projected, _ = cv2.projectPoints(
        object_points_3d.astype(np.float64), rvec, t, K, np.zeros(5))
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points_2d, axis=1)

    return CameraCalibrationResult(K, zeros(5), rvec, t, R, P,
                                    mean(errors), N, "dlt")
```

**DLT 后置处理：内参清洁化**

```
function clean_intrinsics(K):
    """强制 DLT 分解出的 K 满足物理约束：fx=fy, s=0。"""
    f = (K[0,0] + K[1,1]) / 2.0
    cx, cy = K[0,2], K[1,2]
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
```

### 4. IAC 焦距估计（交叉验证用）

**定位**：主要用于**交叉验证 DLT 结果**。当 DLT 不可用时也可作为 fallback。

**核心思想**：利用 H = K[r1|r2|t] 中旋转列 r1⊥r2 且 ‖r1‖=‖r2‖ 的约束，从 H 反推焦距 f。

**简化假设**（对手机摄像头合理）：fx = fy = f，s = 0，cx = W/2, cy = H/2

```
function estimate_intrinsics_from_homography(H, image_width, image_height):
    cx = image_width / 2.0
    cy = image_height / 2.0
    h1 = H[:, 0]
    h2 = H[:, 1]

    # 约束1：正交性 → f²_ortho
    numerator_ortho = (h1[0]*h2[0] + h1[1]*h2[1]
                       - cx*(h1[0]*h2[2] + h1[2]*h2[0])
                       - cy*(h1[1]*h2[2] + h1[2]*h2[1])
                       + (cx**2 + cy**2) * h1[2]*h2[2])
    denominator_ortho = h1[2] * h2[2]

    if abs(denominator_ortho) < 1e-10:
        return None

    f_sq_ortho = -numerator_ortho / denominator_ortho

    # 约束2：等模性 → f²_norm
    def compute_h_omega_h(h):
        return (h[0]**2 + h[1]**2
                - 2*cx*h[0]*h[2] - 2*cy*h[1]*h[2]
                + (cx**2 + cy**2) * h[2]**2)

    lhs = compute_h_omega_h(h1)
    rhs = compute_h_omega_h(h2)
    denom_norm = h1[2]**2 - h2[2]**2

    f_sq_norm = None if abs(denom_norm) < 1e-10 else -(lhs - rhs) / denom_norm

    # 综合两个约束
    f_sq_candidates = []
    if f_sq_ortho > 0: f_sq_candidates.append(f_sq_ortho)
    if f_sq_norm is not None and f_sq_norm > 0: f_sq_candidates.append(f_sq_norm)

    if len(f_sq_candidates) == 0:
        log.warning("IAC约束求解失败，回退到FOV=55°估计")
        f = image_width / (2 * tan(radians(55 / 2)))
        method = "fallback_fov"
    elif len(f_sq_candidates) == 1:
        f = sqrt(f_sq_candidates[0])
        method = "iac_single"
    else:
        f1 = sqrt(f_sq_candidates[0])
        f2 = sqrt(f_sq_candidates[1])
        relative_diff = abs(f1 - f2) / max(f1, f2)
        if relative_diff < 0.15:
            f = (f1 + f2) / 2.0
            method = "iac_consistent"
        else:
            f = f1
            method = "iac_ortho_preferred"
            log.warning(f"IAC两约束不一致: f_ortho={f1:.1f}, f_norm={f2:.1f}")

    # 合理性验证
    f_min = 0.5 * image_width
    f_max = 2.5 * image_width
    if not (f_min < f < f_max):
        log.warning(f"估计焦距 f={f:.1f} 超出合理范围 [{f_min:.0f}, {f_max:.0f}]")
        f = np.clip(f, f_min, f_max)

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    return IntrinsicsEstimate(K=K, f=f, cx=cx, cy=cy, method=method)
```

### 5. cv2.calibrateCamera（仅移动机位）

> **重要**：大多数羽毛球视频为固定机位拍摄，此方法不可用。仅当检测到相机有明显移动时才启用。

```
function estimate_intrinsics_calibrate(frame_detections, court_keypoints_2d, image_size):
    object_points_list = []
    image_points_list = []

    for detection in frame_detections:
        indices, pixel_pts = detection.get_ground_keypoints(min_conf=0.5)
        if len(indices) < 6: continue

        obj_pts_3d = np.zeros((len(indices), 3), dtype=np.float32)
        obj_pts_3d[:, :2] = court_keypoints_2d[indices].astype(np.float32)

        object_points_list.append(obj_pts_3d)
        image_points_list.append(pixel_pts.reshape(-1, 1, 2).astype(np.float32))

    if len(object_points_list) < 3:
        return None, None

    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list, image_points_list, image_size, None, None,
        flags=cv2.CALIB_FIX_ASPECT_RATIO
    )

    if ret > 2.0:
        log.warning(f"calibrateCamera 重投影误差偏大: {ret:.2f}px")

    return K, dist_coeffs
```

### 6. PnP 外参求解

```
function calibrate_pnp(object_points_3d, image_points_2d, K, dist_coeffs=None,
                        initial_rvec=None, initial_tvec=None):
    if dist_coeffs is None:
        dist_coeffs = zeros(5)

    if initial_rvec is not None and initial_tvec is not None:
        # 模式A：有初始猜测（DLT 联动）
        success, rvec, tvec = cv2.solvePnP(
            object_points_3d.astype(np.float64),
            image_points_2d.reshape(-1, 1, 2).astype(np.float64),
            K.astype(np.float64), dist_coeffs.astype(np.float64),
            rvec=initial_rvec.copy(), tvec=initial_tvec.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success: raise RuntimeError("solvePnP with initial guess failed")
    else:
        # 模式B：无初始猜测（通用）
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points_3d.astype(np.float64),
            imagePoints=image_points_2d.reshape(-1, 1, 2).astype(np.float64),
            cameraMatrix=K.astype(np.float64), distCoeffs=dist_coeffs.astype(np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=8.0, confidence=0.99
        )
        if not success: raise RuntimeError("solvePnPRansac failed")

    # LM 精化
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points_3d.astype(np.float64),
        image_points_2d.reshape(-1, 1, 2).astype(np.float64),
        K, dist_coeffs, rvec, tvec
    )

    R, _ = cv2.Rodrigues(rvec)
    P = K @ np.hstack([R, tvec])

    projected, _ = cv2.projectPoints(
        object_points_3d.astype(np.float64), rvec, tvec, K, dist_coeffs)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points_2d, axis=1)

    return CameraCalibrationResult(K, dist_coeffs, rvec, tvec, R, P,
                                    mean(errors), len(object_points_3d), "pnp")
```

### 7. 推荐标定策略（主入口）

```
function calibrate_video(frame_results, image_size):
    w, h = image_size

    # Step 0: 判断相机是否有移动
    H_list = [r.homography.H for r in frame_results if r.success]
    h_diversity = compute_h_diversity(H_list)
    is_fixed_camera = h_diversity < threshold

    # ═══ 路径A：移动机位 → calibrateCamera ═══
    if not is_fixed_camera:
        K, dist = estimate_intrinsics_calibrate(...)
        if K is not None:
            for r in frame_results:
                if r.success:
                    obj_pts, img_pts = build_3d_2d_correspondences(...)
                    r.calibration = calibrate_pnp(obj_pts, img_pts, K, dist)
            return CalibrationResult(K=K, method="calibrate_camera")

    # ═══ 路径B：固定机位（大多数情况）═══

    # Step 1: 多帧平均
    avg_keypoints = average_keypoints_fixed_camera(...)

    # Step 2: 构建高精度 3D-2D 对应
    obj_pts, img_pts = build_3d_2d_from_averaged(...)

    # Step 3: DLT → P → K, R, t
    if len(obj_pts) >= 6 and has_noncoplanar_points(obj_pts):
        dlt_result = calibrate_dlt(obj_pts, img_pts)

        # Step 4: DLT→PnP 联动
        K_clean = clean_intrinsics(dlt_result.K)
        pnp_refined = calibrate_pnp(obj_pts, img_pts, K_clean,
                                     initial_rvec=dlt_result.rvec,
                                     initial_tvec=dlt_result.tvec)
        best = min(dlt_result, pnp_refined, key=lambda r: r.reprojection_error)

        # Step 5: IAC 交叉验证
        iac_result = estimate_intrinsics_from_homography(H_avg, w, h)
        if iac_result is not None:
            f_dlt = dlt_result.K[0, 0]
            f_iac = iac_result.f
            relative_diff = abs(f_dlt - f_iac) / max(f_dlt, f_iac)
            if relative_diff > 0.20:
                log.warning(f"DLT与IAC焦距不一致: f_dlt={f_dlt:.1f}, f_iac={f_iac:.1f}")
            best.iac_cross_check = {"f_iac": f_iac, "f_dlt": f_dlt,
                                     "consistent": relative_diff < 0.20}
        return best

    # Step 6: IAC fallback
    K_iac = estimate_intrinsics_from_homography(H, w, h)
    if K_iac is not None:
        return calibrate_pnp(obj_pts, img_pts, K_iac.K)

    # Step 7: FOV=55° 最终兜底
    K_fallback = estimate_intrinsics_fallback(w, h, fov_deg=55)
    return calibrate_pnp(obj_pts, img_pts, K_fallback)
```

> **策略总结**：
>
> | 场景 | 策略 | 优先级 |
> |------|------|--------|
> | **固定机位（大多数）** | 多帧平均 → DLT → K → PnP 精化 + IAC 交叉验证 | **主路径** |
> | 移动机位（少数） | cv2.calibrateCamera → K + dist → PnP | 次选 |
> | 非共面点不足 | IAC → K → PnP | fallback |
> | IAC 也失败 | FOV=55° → K → PnP | 最终兜底 |

## 设计要点

- DLT 无需任何关于 K 的假设，多帧平均补偿其噪声敏感性
- IAC 作为独立交叉验证，检查 DLT 结果一致性
- calibrateCamera 仅用于移动机位，固定机位会退化失败
- PnP 精化是通用的最后一步，使用全部非共面点
- EXIF 完全不依赖

## 测试方案

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| 多帧平均降噪 | 50帧合成数据(σ=3px) → 中位数 | 平均后 σ < 0.6px |
| DLT 合成恢复 | 已知K,R,t → 投影3D点 → DLT恢复 | 重投影 < 2px, K相对误差 < 5% |
| 固定机位完整流程 | 多帧平均 → DLT → PnP精化 | 重投影 < 2px |
| DLT-IAC 交叉验证 | 合成数据，比较 f_dlt 和 f_iac | 相对差 < 10% |
| 相机移动检测 | 固定/移动各5组测试视频 | 正确分类率 100% |
| IAC 焦距恢复 | 已知K生成合成H → IAC求解 → 比较f | 相对误差 < 5% |
| IAC 两约束一致性 | 合成数据+噪声 | f_ortho 和 f_norm 相对差 < 10% |
| FOV fallback 触发 | 构造使 IAC 失败的退化 H | 正确回退到 FOV=55° |
| calibrateCamera | 多帧合成数据（不同视角） | K 各参数误差 < 3% |
| PnP 使用正确点集 | 检查输入的 3D 点是否含非共面点 | Z 列不全为0 |
| 最少点数 | DLT用6点，PnP用4点 | 能成功求解 |
