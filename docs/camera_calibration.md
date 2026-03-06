# 子模块2.5：相机标定

## 功能概述

从3D-2D对应点（地面关键点Z=0 + 球网端点Z=1.55）求解相机完整参数：内参矩阵 K、外参 R/t、投影矩阵 P = K[R|t]。当前实现聚焦单帧标定（固定机位主路径），包含 DLT、PnP、IAC 三种方法。

## 依赖关系

- **上游**：YOLO检测(1.1) → 地面关键点；球网关键点提取(2.2) → `NetKeypointResult`；Homography(1.2) → H（用于IAC交叉验证）
- **下游**：投影矩阵验证(2.6)
- **基础数据**：`config/court_config.py` 中的 `COURT_KEYPOINTS_3D`

## 对应代码文件

`module2/camera_calibration.py`

---

## 关键点使用总览

| 步骤 | 输入关键点 | 说明 |
|------|-----------|------|
| Homography H 计算 | 地面关键点 0-21（Z=0），≥4个可见 | 仅平面点，已在模块1完成 |
| DLT → P → 分解 K（**主路径**） | 地面点 + 球网端点，≥6个非共面点 | 无需任何内参假设，直接求出 K, R, t |
| IAC 从 H 估计 f（交叉验证） | 不需要额外关键点，直接从 H 矩阵提取 | 验证 DLT 的 K 是否合理 |
| PnP 求外参 | 全部可用的 3D-2D 对应：地面点(0-21) + 球网顶部端点(22,24) | 使用 DLT 得到的 K |

## 输出数据结构

```
CameraCalibrationResult:
    K: ndarray (3,3)              # 内参矩阵
    dist_coeffs: ndarray (5,)     # 畸变系数（当前为 zeros）
    rvec: ndarray (3,1)           # Rodrigues 旋转向量
    tvec: ndarray (3,1)           # 平移向量
    R: ndarray (3,3)              # 旋转矩阵
    P: ndarray (3,4)              # 投影矩阵 P = K @ [R|t]
    reprojection_error: float     # 平均重投影误差(px)
    num_points_used: int
    method: str                   # "dlt" / "dlt_pnp" / "iac_pnp" / "fallback_pnp" / "pnp"
    k_estimation_method: str      # "dlt" / "iac_consistent" / "iac_single" / "iac_ortho_preferred" / "fallback_fov" / ""
    iac_cross_check: dict|None    # {"f_iac": float, "f_dlt": float, "consistent": bool, ...}
```

---

## 算法详解

### 1. 3D-2D 对应关系构建

```
function build_3d_2d_correspondences(detection, net_kpts, min_conf=0.5):
    # 地面关键点（Z=0）
    indices, pixel_pts = detection.get_ground_keypoints(min_conf)
    ground_3d = COURT_KEYPOINTS_3D[indices]   # (M, 3), Z列全为0
    ground_2d = pixel_pts                      # (M, 2)

    # 球网顶部端点（Z=1.55）
    net_3d = [net_kpts.left_top_3d, net_kpts.right_top_3d]
    net_2d = [net_kpts.left_top_pixel, net_kpts.right_top_pixel]

    # 合并
    object_points = vstack([ground_3d, net_3d])   # (M+2, 3)
    image_points = vstack([ground_2d, net_2d])     # (M+2, 2)

    return object_points, image_points
```

> **关键**：地面点全在 Z=0 平面，加入 Z=1.55 的球网点使点集变为非共面。这对 DLT 至关重要——纯共面点导致 DLT 矩阵 rank deficient。PnP 也受益于非共面点。

### 2. DLT（主路径）

**定位**：单帧标定的**推荐主路径**。直接从非共面 3D-2D 对应点求出 3x4 投影矩阵 P，然后用 `cv2.decomposeProjectionMatrix` 分解出 K, R, t，**无需任何关于内参 K 的先验假设**。

**原理**：投影关系 `s * [u, v, 1]^T = P * [X, Y, Z, 1]^T` 消去齐次因子 s 后，每对对应点产生 2 个线性方程，P 有 12 个未知数（差一个尺度因子，有效自由度 11），故需至少 6 个对应点。将所有方程堆叠为 `A * p = 0`，通过 SVD 取最小奇异值对应的右奇异向量作为解。

```
function calibrate_dlt(object_points_3d, image_points_2d):
    N = len(object_points_3d)
    assert N >= 6

    # 构建 DLT 矩阵 A (2N x 12)
    A = zeros(2*N, 12)
    for i in range(N):
        X, Y, Z = object_points_3d[i]
        u, v = image_points_2d[i]
        A[2*i]   = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]

    _, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    # 归一化：令 P 第三行前3列的模为1，使 P[2,:3] 代表单位方向
    P = P / np.linalg.norm(P[2, :3])
    # 符号校正：保证 det(P[:,:3]) > 0（右手系）
    if np.linalg.det(P[:, :3]) < 0:
        P = -P

    K, R, t_h, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    K = K / K[2,2]
    t = (t_h[:3] / t_h[3]).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)

    return CameraCalibrationResult(K, zeros(5), rvec, t, R, P,
                                    mean(errors), N, "dlt", "dlt")
```

**DLT 后置处理：内参清洁化**

DLT 通过 SVD 求解线性方程组，再用 RQ 分解提取 K。由于关键点检测噪声和数值误差，分解出的 K 通常存在以下问题：

- `fx != fy`：物理上手机摄像头的像素是方形的，fx 和 fy 应相等
- `K[0,1] != 0`：skew 参数，现代相机中应为 0

清洁化将 DLT 分解出的 K 强制投影到满足物理约束的空间，为后续 PnP 精化提供更合理的初始内参。

```
function clean_intrinsics(K):
    f = (K[0,0] + K[1,1]) / 2.0    # fx, fy 取均值
    cx, cy = K[0,2], K[1,2]        # 主点保留
    return [[f, 0, cx], [0, f, cy], [0, 0, 1]]
```

### 3. IAC 焦距估计

**定位**：在主路径中作为 DLT 结果的**交叉验证**；当 DLT 不可用时作为 fallback 提供 K 的初始估计。

#### 3.1 数学原理

Homography H 描述地面平面（Z=0）到图像的映射。相机模型下，H 和内参 K 的关系为：

```
H = K * [r1 | r2 | t]
```

其中 r1, r2 是旋转矩阵 R 的前两列。由于 R 是正交矩阵，r1 和 r2 满足两个约束：

- **正交约束**：`r1^T * r2 = 0`（两列向量正交）
- **等模约束**：`||r1|| = ||r2||`（两列向量模相等）

从 `H = K * [r1|r2|t]` 可得 `r1 = K^{-1} * h1`, `r2 = K^{-1} * h2`（h1, h2 是 H 的前两列）。代入约束：

- 正交约束 → `h1^T * K^{-T} * K^{-1} * h2 = 0`
- 等模约束 → `h1^T * K^{-T} * K^{-1} * h1 = h2^T * K^{-T} * K^{-1} * h2`

其中 `omega = K^{-T} * K^{-1}` 称为 Image of the Absolute Conic (IAC)。

#### 3.2 简化假设

对手机摄像头做合理假设：`fx = fy = f`，skew `s = 0`，主点 `cx = W/2, cy = H/2`。此时 K 只有一个未知数 f，两个约束各产生一个关于 `f^2` 的方程：

| 约束 | 方程 | 输出 |
|------|------|------|
| 正交约束 | `f^2_ortho = -numerator_ortho / (h1[2] * h2[2])` | 从 r1 和 r2 的正交性推导 |
| 等模约束 | `f^2_norm = -(lhs - rhs) / (h1[2]^2 - h2[2]^2)` | 从 r1 和 r2 的等长性推导 |

#### 3.3 综合策略与 method 字段

两个约束独立求解后，通过比较结果判断可信度：

| 条件 | method 值 | 处理策略 | 可信度 |
|------|-----------|----------|--------|
| 两约束均有效，相对差 < 15% | `"iac_consistent"` | 取两者均值 | 高 |
| 仅一个约束有效（另一个 f^2 <= 0） | `"iac_single"` | 使用有效的那个 | 中 |
| 两约束均有效但相对差 >= 15% | `"iac_ortho_preferred"` | 优先使用正交约束 | 中低 |
| 两约束均无效（f^2 <= 0） | `"fallback_fov"` | 回退到 FOV=55 度估计 | 低 |

> 正交约束优先的原因：等模约束的分母 `h1[2]^2 - h2[2]^2` 在相机接近正对球场时趋近于零，数值不稳定。

#### 3.4 焦距合理性范围

估计出的焦距 f 需满足 `0.5 * W < f < 2.5 * W`（W 为图像宽度），对应水平 FOV 约 22 度到 90 度。超出此范围说明 H 质量差或场景退化，此时将 f 截断到边界值。

### 4. PnP 外参求解

**定位**：在已知内参 K 的前提下，求解外参 R 和 t。PnP 是非线性优化，比 DLT 的线性解更精确。

两种使用模式：

| 模式 | 触发条件 | 求解方法 | 说明 |
|------|----------|----------|------|
| 模式A：有初始猜测 | DLT 成功后联动 | `cv2.solvePnP` + ITERATIVE + `useExtrinsicGuess=True` | 从 DLT 的解出发迭代精化 |
| 模式B：无初始猜测 | fallback 路径 | `cv2.solvePnPRansac` | RANSAC 自动处理可能的异常点 |

两种模式之后均通过 `cv2.solvePnPRefineLM`（Levenberg-Marquardt）进一步精化。

```
function calibrate_pnp(object_points_3d, image_points_2d, K, dist_coeffs=None,
                        initial_rvec=None, initial_tvec=None):
    if initial_rvec is not None:
        success, rvec, tvec = cv2.solvePnP(..., useExtrinsicGuess=True)
    else:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(...)

    # LM 精化（try/except 保护）
    rvec, tvec = cv2.solvePnPRefineLM(...)

    R, _ = cv2.Rodrigues(rvec)
    P = K @ np.hstack([R, tvec])

    return CameraCalibrationResult(K, dist_coeffs, rvec, tvec, R, P,
                                    mean(errors), N, "pnp", "")
```

### 5. 单帧标定主入口

```
function calibrate_frame(detection, h_result, net_kpts, image_size):
    # Step 1: 构建 3D-2D 对应
    obj_pts, img_pts = build_3d_2d_correspondences(detection, net_kpts)

    if len(obj_pts) < 6 or 没有非共面点:
        return _fallback_calibration(obj_pts, img_pts, H, w, h)

    # Step 2: DLT
    dlt_result = calibrate_dlt(obj_pts, img_pts)

    if dlt_result is not None:
        # Step 3: DLT → PnP 精化
        K_clean = clean_intrinsics(dlt_result.K)
        pnp_result = calibrate_pnp(obj_pts, img_pts, K_clean,
                                     initial_rvec=dlt_result.rvec,
                                     initial_tvec=dlt_result.tvec)
        # 取误差更小者
        best = min(dlt_result, pnp_result, key=reprojection_error)
        best.method = "dlt_pnp"   # 若 PnP 精化更好

        # Step 4: IAC 交叉验证
        iac_result = estimate_intrinsics_from_homography(H, w, h)
        if iac_result is not None:
            f_dlt, f_iac = best.K[0,0], iac_result["f"]
            best.iac_cross_check = {"f_dlt": ..., "f_iac": ..., "consistent": ...}

        return best

    # DLT 失败 → fallback
    return _fallback_calibration(obj_pts, img_pts, H, w, h)


function _fallback_calibration(obj_pts, img_pts, H, w, h):
    # 尝试 IAC → PnP
    iac_result = estimate_intrinsics_from_homography(H, w, h)
    if iac_result is not None:
        pnp = calibrate_pnp(obj_pts, img_pts, iac_result["K"])
        pnp.method = "iac_pnp"
        return pnp

    # FOV=55 度最终兜底
    K_fallback = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
    pnp = calibrate_pnp(obj_pts, img_pts, K_fallback)
    pnp.method = "fallback_pnp"
    return pnp
```

---

## IAC 交叉验证与投影验证的关系

本模块（2.5）中的 IAC 交叉验证与下游模块（2.6）中的投影矩阵验证是**两个不同层级**的质量检查，各有明确分工：

| 维度 | IAC 交叉验证（2.5 内部） | 投影矩阵验证（2.6） |
|------|-------------------------|---------------------|
| 执行位置 | 标定过程中，DLT 完成后 | 标定完成后，作为独立的下游模块 |
| 检查范围 | 仅验证焦距 f 这一个参数 | 验证整个标定结果的 6 个维度 |
| 检查方法 | 用 IAC（独立算法）重新估计 f，与 DLT 的 f 比较 | 重投影误差、相机位置、旋转矩阵、内参、球场中心、H-P 一致性 |
| 结果性质 | **信息性**：写入 `iac_cross_check` 字段，不阻断标定流程 | **门控性**：`overall_ok = False` 时拒绝输出 |
| 价值 | 早期预警 DLT 分解不稳定 | 端到端确认 P 可用 |

**为什么 IAC 验证不阻断流程**：IAC 自身有较强的简化假设（主点在图像中心、无畸变），即使 IAC 的 f 与 DLT 的 f 不一致，DLT+PnP 的结果仍然可能是准确的。真正决定 P 是否可用的是下游的 6 项投影验证。

**IAC 交叉验证的阈值**：当 `|f_dlt - f_iac| / max(f_dlt, f_iac) < 20%` 时判定为一致。此阈值考虑了 IAC 简化假设带来的固有偏差，以及 YOLO 检测噪声的传播。

---

## 标定策略总览

| 场景 | 策略 | method 字段 | 优先级 |
|------|------|-------------|--------|
| **非共面点 >= 6（大多数情况）** | DLT → clean K → PnP 精化 + IAC 交叉验证 | `"dlt"` 或 `"dlt_pnp"` | **主路径** |
| 非共面点不足或 DLT 失败 | IAC 估计 K → PnP 求外参 | `"iac_pnp"` | fallback |
| IAC 也失败 | FOV=55 度估计 K → PnP 求外参 | `"fallback_pnp"` | 最终兜底 |

## 设计要点

- DLT 无需任何关于 K 的假设，直接从非共面点求解 P
- clean_intrinsics 将 DLT 的 K 投影到物理合理空间（fx=fy, s=0），为 PnP 提供更好的初始内参
- IAC 作为独立交叉验证，利用完全不同的数学路径（从 H 推 f）检查 DLT 的 f 是否合理
- PnP 精化是通用的最后一步，通过非线性优化进一步降低重投影误差
- 当前实现为单帧标定，未来可扩展为多帧平均（固定机位）或 `cv2.calibrateCamera`（移动机位）
- EXIF 完全不依赖

## 测试方案

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| DLT 合成恢复 | 已知K,R,t → 投影3D点 → DLT恢复 | 重投影 < 2px, K相对误差 < 5% |
| DLT-PnP 联动 | DLT → PnP精化 | PnP 误差 <= DLT 误差 |
| DLT-IAC 交叉验证 | 合成数据，比较 f_dlt 和 f_iac | 相对差 < 10% |
| IAC 焦距恢复 | 已知K生成合成H → IAC求解 → 比较f | 相对误差 < 5% |
| IAC 两约束一致性 | 合成数据+噪声 | f_ortho 和 f_norm 相对差 < 10% |
| FOV fallback 触发 | 构造使 IAC 失败的退化 H | 正确回退到 FOV=55 度 |
| PnP 使用正确点集 | 检查输入的 3D 点是否含非共面点 | Z 列不全为0 |
| 最少点数 | DLT用6点，PnP用4点 | 能成功求解 |
