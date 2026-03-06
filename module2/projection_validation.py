"""
投影矩阵验证 (子模块 2.6)。

对相机标定输出的投影矩阵 P 进行 6 项质量检查，确保 P 在物理上合理
且与模块1的 Homography H 一致。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from module2.camera_calibration import CameraCalibrationResult

logger = logging.getLogger(__name__)


def validate_projection(
    calibration: CameraCalibrationResult,
    object_points_3d: np.ndarray,
    image_points_2d: np.ndarray,
    H: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> dict:
    """对投影矩阵执行 6 项质量检查。

    Args:
        calibration: 相机标定结果。
        object_points_3d: (N, 3) 世界坐标。
        image_points_2d: (N, 2) 像素坐标。
        H: (3, 3) Homography 矩阵。
        image_size: (width, height)，用于检查投影是否在画面内。

    Returns:
        dict，含各项检查结果和 'overall_ok' 布尔值。
    """
    metrics = {}
    K = calibration.K
    R = calibration.R
    rvec = calibration.rvec
    tvec = calibration.tvec
    P = calibration.P
    dist = calibration.dist_coeffs

    # ─── 检查 1: 重投影误差 ───
    projected, _ = cv2.projectPoints(
        object_points_3d.astype(np.float64),
        rvec, tvec, K, dist,
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points_2d, axis=1)
    metrics["mean_reproj"] = float(np.mean(errors))
    metrics["max_reproj"] = float(np.max(errors))
    metrics["reproj_ok"] = metrics["mean_reproj"] < 5.0

    # ─── 检查 2: 相机位置合理性 ───
    camera_pos = -R.T @ tvec  # (3, 1)
    cam_height = float(camera_pos[2, 0])
    cam_dist = float(np.linalg.norm(camera_pos))
    metrics["cam_height"] = cam_height
    metrics["cam_dist"] = cam_dist
    metrics["cam_ok"] = (
        cam_height > 0
        and 2.0 < cam_height < 15.0
        and 3.0 < cam_dist < 60.0
    )

    # ─── 检查 3: 旋转矩阵正交性 ───
    orth_error = float(np.linalg.norm(R @ R.T - np.eye(3)))
    metrics["orth_error"] = orth_error
    metrics["orth_ok"] = orth_error < 1e-6

    # ─── 检查 4: 内参合理性 ───
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    metrics["fx"] = fx
    metrics["fy"] = fy
    metrics["intrinsics_ok"] = fx > 0 and fy > 0 and abs(fx / fy - 1) < 0.5

    # ─── 检查 5: 球场中心投影 ───
    center_proj, _ = cv2.projectPoints(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        rvec, tvec, K, dist,
    )
    center_px = center_proj.reshape(2)
    metrics["center_px"] = center_px

    if image_size is not None:
        w, h = image_size
        metrics["center_in_frame"] = (
            0 <= center_px[0] <= w and 0 <= center_px[1] <= h
        )
    else:
        metrics["center_in_frame"] = True  # 无法验证时默认通过
    metrics["center_ok"] = metrics["center_in_frame"]

    # ─── 检查 6: H-P 一致性 ───
    # P[:, [0, 1, 3]] 应与 H 成比例
    H_from_P = P[:, [0, 1, 3]]
    H_from_P_norm = H_from_P / np.linalg.norm(H_from_P)
    H_norm = H / np.linalg.norm(H)
    # 考虑符号不确定性
    consistency = min(
        float(np.linalg.norm(H_norm - H_from_P_norm)),
        float(np.linalg.norm(H_norm + H_from_P_norm)),
    )
    metrics["hp_consistency"] = consistency
    metrics["hp_ok"] = consistency < 0.05

    # ─── 总体判定 ───
    metrics["overall_ok"] = all([
        metrics["reproj_ok"],
        metrics["cam_ok"],
        metrics["orth_ok"],
        metrics["intrinsics_ok"],
        metrics["center_ok"],
        metrics["hp_ok"],
    ])

    if not metrics["overall_ok"]:
        failed = [
            k for k in ["reproj_ok", "cam_ok", "orth_ok",
                         "intrinsics_ok", "center_ok", "hp_ok"]
            if not metrics[k]
        ]
        logger.warning("投影矩阵验证失败: %s", failed)

    return metrics
