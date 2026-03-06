"""
相机标定 (子模块 2.5)。

从 3D-2D 对应点（地面关键点 Z=0 + 球网端点 Z=1.55）求解相机完整参数：
内参矩阵 K、外参 R/t、投影矩阵 P = K[R|t]。

当前实现聚焦单帧标定（固定机位主路径）：
    DLT → PnP 精化 → IAC 交叉验证
    fallback: IAC → PnP → FOV=55° → PnP
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from config.court_config import COURT_KEYPOINTS_3D, NET_KEYPOINTS_3D
from module1.homography import HomographyResult
from module1.yolo_detector import CourtDetectionResult
from module2.net_top_detector_yolo import NetKeypointResult

logger = logging.getLogger(__name__)


@dataclass
class CameraCalibrationResult:
    """相机标定结果。"""

    K: np.ndarray               # (3, 3) 内参矩阵
    dist_coeffs: np.ndarray     # (5,) 畸变系数
    rvec: np.ndarray            # (3, 1) Rodrigues 旋转向量
    tvec: np.ndarray            # (3, 1) 平移向量
    R: np.ndarray               # (3, 3) 旋转矩阵
    P: np.ndarray               # (3, 4) 投影矩阵
    reprojection_error: float
    num_points_used: int
    method: str                 # "dlt_pnp" / "dlt" / "iac_pnp" / "fallback_pnp"
    k_estimation_method: str    # "dlt" / "iac_consistent" / "iac_single" / "fallback_fov"
    iac_cross_check: dict | None = field(default=None)


def build_3d_2d_correspondences(
    detection: CourtDetectionResult,
    net_kpts: NetKeypointResult,
    min_conf: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """构建 3D-2D 对应关系（地面点 + 球网端点）。

    Returns:
        (object_points_3d, image_points_2d)
        object_points_3d: shape (N, 3)
        image_points_2d: shape (N, 2)
    """
    # 地面关键点 (Z=0)
    indices, pixel_pts = detection.get_ground_keypoints(min_conf)
    if len(indices) > 0:
        ground_3d = COURT_KEYPOINTS_3D[indices]  # (M, 3)
        ground_2d = pixel_pts                     # (M, 2)
    else:
        ground_3d = np.zeros((0, 3), dtype=np.float64)
        ground_2d = np.zeros((0, 2), dtype=np.float64)

    # 球网顶部端点 (Z=1.55)
    net_3d = np.array([
        net_kpts.left_top_3d,
        net_kpts.right_top_3d,
    ], dtype=np.float64)  # (2, 3)

    net_2d = np.array([
        net_kpts.left_top_pixel,
        net_kpts.right_top_pixel,
    ], dtype=np.float64)  # (2, 2)

    # 合并
    object_points = np.vstack([ground_3d, net_3d])
    image_points = np.vstack([ground_2d, net_2d])

    return object_points, image_points


def calibrate_dlt(
    object_points_3d: np.ndarray,
    image_points_2d: np.ndarray,
) -> CameraCalibrationResult | None:
    """使用 DLT（Direct Linear Transform）求解投影矩阵 P。

    Args:
        object_points_3d: (N, 3) 世界坐标。
        image_points_2d: (N, 2) 像素坐标。

    Returns:
        CameraCalibrationResult 或 None。
    """
    n = len(object_points_3d)
    if n < 6:
        logger.warning("DLT 需要至少 6 个点, 当前 %d 个", n)
        return None

    # 构建 DLT 矩阵 A (2N × 12)
    A = np.zeros((2 * n, 12), dtype=np.float64)
    for i in range(n):
        X, Y, Z = object_points_3d[i]
        u, v = image_points_2d[i]
        A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]

    # SVD 求解
    _, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    # 归一化和符号校正
    scale = np.linalg.norm(P[2, :3])
    if scale < 1e-10:
        logger.warning("DLT 投影矩阵第三行接近零")
        return None
    P = P / scale

    if np.linalg.det(P[:, :3]) < 0:
        P = -P

    # 分解 P → K, R, t
    try:
        K, R, t_h, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    except cv2.error as e:
        logger.warning("decomposeProjectionMatrix 失败: %s", e)
        return None

    K = K / K[2, 2]
    t = (t_h[:3] / t_h[3]).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)

    # 计算重投影误差
    projected, _ = cv2.projectPoints(
        object_points_3d.astype(np.float64),
        rvec, t, K, np.zeros(5),
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points_2d, axis=1)
    mean_error = float(np.mean(errors))

    return CameraCalibrationResult(
        K=K,
        dist_coeffs=np.zeros(5, dtype=np.float64),
        rvec=rvec,
        tvec=t,
        R=R,
        P=P,
        reprojection_error=mean_error,
        num_points_used=n,
        method="dlt",
        k_estimation_method="dlt",
    )


def _clean_intrinsics(K: np.ndarray) -> np.ndarray:
    """强制 DLT 分解出的 K 满足物理约束：fx=fy, s=0。"""
    f = (K[0, 0] + K[1, 1]) / 2.0
    cx, cy = K[0, 2], K[1, 2]
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)


def estimate_intrinsics_from_homography(
    H: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict | None:
    """从 Homography 矩阵利用 IAC 约束估计焦距。

    Args:
        H: (3, 3) Homography 矩阵。
        image_width: 图像宽度。
        image_height: 图像高度。

    Returns:
        {"K": ndarray, "f": float, "cx": float, "cy": float, "method": str}
        或 None。
    """
    cx = image_width / 2.0
    cy = image_height / 2.0
    h1 = H[:, 0]
    h2 = H[:, 1]

    # 约束1: 正交性 → f²_ortho
    numerator_ortho = (
        h1[0] * h2[0] + h1[1] * h2[1]
        - cx * (h1[0] * h2[2] + h1[2] * h2[0])
        - cy * (h1[1] * h2[2] + h1[2] * h2[1])
        + (cx ** 2 + cy ** 2) * h1[2] * h2[2]
    )
    denominator_ortho = h1[2] * h2[2]

    if abs(denominator_ortho) < 1e-10:
        logger.warning("IAC 正交约束分母接近零")
        return None

    f_sq_ortho = -numerator_ortho / denominator_ortho

    # 约束2: 等模性 → f²_norm
    def _h_omega_h(h: np.ndarray) -> float:
        return (
            h[0] ** 2 + h[1] ** 2
            - 2 * cx * h[0] * h[2]
            - 2 * cy * h[1] * h[2]
            + (cx ** 2 + cy ** 2) * h[2] ** 2
        )

    lhs = _h_omega_h(h1)
    rhs = _h_omega_h(h2)
    denom_norm = h1[2] ** 2 - h2[2] ** 2

    f_sq_norm = None
    if abs(denom_norm) >= 1e-10:
        f_sq_norm = -(lhs - rhs) / denom_norm

    # 综合两个约束
    f_sq_candidates = []
    if f_sq_ortho > 0:
        f_sq_candidates.append(f_sq_ortho)
    if f_sq_norm is not None and f_sq_norm > 0:
        f_sq_candidates.append(f_sq_norm)

    if len(f_sq_candidates) == 0:
        logger.warning("IAC 约束求解失败，回退到 FOV=55° 估计")
        f = image_width / (2 * math.tan(math.radians(55 / 2)))
        method = "fallback_fov"
    elif len(f_sq_candidates) == 1:
        f = math.sqrt(f_sq_candidates[0])
        method = "iac_single"
    else:
        f1 = math.sqrt(f_sq_candidates[0])
        f2 = math.sqrt(f_sq_candidates[1])
        relative_diff = abs(f1 - f2) / max(f1, f2)
        if relative_diff < 0.15:
            f = (f1 + f2) / 2.0
            method = "iac_consistent"
        else:
            f = f1  # 正交约束优先
            method = "iac_ortho_preferred"
            logger.warning("IAC 两约束不一致: f_ortho=%.1f, f_norm=%.1f", f1, f2)

    # 合理性验证
    f_min = 0.5 * image_width
    f_max = 2.5 * image_width
    if not (f_min < f < f_max):
        logger.warning("估计焦距 f=%.1f 超出合理范围 [%.0f, %.0f]", f, f_min, f_max)
        f = float(np.clip(f, f_min, f_max))

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    return {"K": K, "f": f, "cx": cx, "cy": cy, "method": method}


def calibrate_pnp(
    object_points_3d: np.ndarray,
    image_points_2d: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    initial_rvec: np.ndarray | None = None,
    initial_tvec: np.ndarray | None = None,
) -> CameraCalibrationResult | None:
    """使用 PnP 求解外参。

    Args:
        object_points_3d: (N, 3) 世界坐标。
        image_points_2d: (N, 2) 像素坐标。
        K: (3, 3) 内参矩阵。
        dist_coeffs: (5,) 畸变系数。
        initial_rvec: 初始旋转向量（有时可加速收敛）。
        initial_tvec: 初始平移向量。

    Returns:
        CameraCalibrationResult 或 None。
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    obj = object_points_3d.astype(np.float64)
    img = image_points_2d.reshape(-1, 1, 2).astype(np.float64)

    try:
        if initial_rvec is not None and initial_tvec is not None:
            success, rvec, tvec = cv2.solvePnP(
                obj, img, K.astype(np.float64), dist_coeffs.astype(np.float64),
                rvec=initial_rvec.copy(), tvec=initial_tvec.copy(),
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=obj, imagePoints=img,
                cameraMatrix=K.astype(np.float64),
                distCoeffs=dist_coeffs.astype(np.float64),
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=8.0, confidence=0.99,
            )
    except cv2.error as e:
        logger.warning("solvePnP 失败: %s", e)
        return None

    if not success:
        logger.warning("solvePnP 返回失败")
        return None

    # LM 精化
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            obj, img, K.astype(np.float64), dist_coeffs.astype(np.float64),
            rvec, tvec,
        )
    except cv2.error:
        pass  # 精化失败时保留原结果

    R, _ = cv2.Rodrigues(rvec)
    P = K @ np.hstack([R, tvec])

    # 计算重投影误差
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist_coeffs)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points_2d, axis=1)
    mean_error = float(np.mean(errors))

    return CameraCalibrationResult(
        K=K.copy(),
        dist_coeffs=dist_coeffs.copy(),
        rvec=rvec,
        tvec=tvec,
        R=R,
        P=P,
        reprojection_error=mean_error,
        num_points_used=len(object_points_3d),
        method="pnp",
        k_estimation_method="",
    )


def calibrate_frame(
    detection: CourtDetectionResult,
    h_result: HomographyResult,
    net_kpts: NetKeypointResult,
    image_size: tuple[int, int],
) -> CameraCalibrationResult | None:
    """单帧相机标定主入口。

    标定策略（固定机位）：
        1. 构建 3D-2D 对应
        2. DLT → clean_intrinsics → PnP 精化
        3. IAC 交叉验证
        4. fallback: IAC → PnP 或 FOV=55° → PnP

    Args:
        detection: YOLO 检测结果。
        h_result: Homography 结果。
        net_kpts: 球网端点推断结果。
        image_size: (width, height)。

    Returns:
        CameraCalibrationResult 或 None。
    """
    w, h = image_size

    # Step 1: 构建 3D-2D 对应
    obj_pts, img_pts = build_3d_2d_correspondences(detection, net_kpts)

    if len(obj_pts) < 6:
        logger.warning("3D-2D 对应点不足: %d 个 (需 ≥ 6)", len(obj_pts))
        # 尝试 IAC fallback
        return _fallback_calibration(obj_pts, img_pts, h_result.H, w, h)

    # 检查是否有非共面点
    z_values = obj_pts[:, 2]
    has_noncoplanar = np.any(z_values != 0.0)

    if not has_noncoplanar:
        logger.warning("所有点共面 (Z=0), DLT 不可用, 尝试 fallback")
        return _fallback_calibration(obj_pts, img_pts, h_result.H, w, h)

    # Step 2: DLT
    dlt_result = calibrate_dlt(obj_pts, img_pts)

    best = None

    if dlt_result is not None:
        logger.info(
            "DLT 成功: 重投影误差 %.2f px, %d 点",
            dlt_result.reprojection_error, dlt_result.num_points_used,
        )

        # Step 3: DLT → PnP 精化
        K_clean = _clean_intrinsics(dlt_result.K)
        pnp_result = calibrate_pnp(
            obj_pts, img_pts, K_clean,
            initial_rvec=dlt_result.rvec,
            initial_tvec=dlt_result.tvec,
        )

        if pnp_result is not None:
            pnp_result.method = "dlt_pnp"
            pnp_result.k_estimation_method = "dlt"
            logger.info(
                "DLT+PnP: 重投影误差 %.2f px", pnp_result.reprojection_error
            )
            # 取误差更小者
            if pnp_result.reprojection_error < dlt_result.reprojection_error:
                best = pnp_result
            else:
                best = dlt_result
        else:
            best = dlt_result

        # Step 4: IAC 交叉验证
        iac_result = estimate_intrinsics_from_homography(h_result.H, w, h)
        if iac_result is not None and best is not None:
            f_dlt = best.K[0, 0]
            f_iac = iac_result["f"]
            relative_diff = abs(f_dlt - f_iac) / max(f_dlt, f_iac)
            best.iac_cross_check = {
                "f_iac": f_iac,
                "f_dlt": f_dlt,
                "relative_diff": relative_diff,
                "consistent": relative_diff < 0.20,
                "iac_method": iac_result["method"],
            }
            if relative_diff > 0.20:
                logger.warning(
                    "DLT 与 IAC 焦距不一致: f_dlt=%.1f, f_iac=%.1f (差 %.1f%%)",
                    f_dlt, f_iac, relative_diff * 100,
                )
            else:
                logger.info(
                    "DLT-IAC 交叉验证通过: f_dlt=%.1f, f_iac=%.1f",
                    f_dlt, f_iac,
                )

        return best

    # DLT 失败 → fallback
    logger.warning("DLT 失败, 尝试 IAC fallback")
    return _fallback_calibration(obj_pts, img_pts, h_result.H, w, h)


def _fallback_calibration(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    H: np.ndarray,
    image_width: int,
    image_height: int,
) -> CameraCalibrationResult | None:
    """当 DLT 不可用时的 fallback 标定策略。"""
    if len(obj_pts) < 4:
        logger.warning("点数不足 (%d), 无法标定", len(obj_pts))
        return None

    # 尝试 IAC
    iac_result = estimate_intrinsics_from_homography(H, image_width, image_height)
    if iac_result is not None:
        K = iac_result["K"]
        pnp_result = calibrate_pnp(obj_pts, img_pts, K)
        if pnp_result is not None:
            pnp_result.method = "iac_pnp"
            pnp_result.k_estimation_method = iac_result["method"]
            logger.info(
                "IAC+PnP fallback: 重投影误差 %.2f px (K method: %s)",
                pnp_result.reprojection_error, iac_result["method"],
            )
            return pnp_result

    # FOV=55° 最终兜底
    f = image_width / (2 * math.tan(math.radians(55 / 2)))
    cx = image_width / 2.0
    cy = image_height / 2.0
    K_fallback = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    pnp_result = calibrate_pnp(obj_pts, img_pts, K_fallback)
    if pnp_result is not None:
        pnp_result.method = "fallback_pnp"
        pnp_result.k_estimation_method = "fallback_fov"
        logger.info(
            "FOV fallback: 重投影误差 %.2f px", pnp_result.reprojection_error
        )
        return pnp_result

    logger.error("所有标定策略均失败")
    return None
