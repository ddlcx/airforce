"""
Homography 计算与验证。

使用地面关键点（0-21, Z=0 平面）计算标准球场坐标 → 像素坐标的 Homography 矩阵 H，
并通过 5 项验证确保 H 的质量。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from module1.yolo_detector import CourtDetectionResult

logger = logging.getLogger(__name__)


@dataclass
class HomographyResult:
    """Homography 计算结果。"""

    H: np.ndarray              # (3, 3) 球场坐标 → 像素坐标
    H_inv: np.ndarray          # (3, 3) 像素坐标 → 球场坐标
    inlier_mask: np.ndarray    # (N,) boolean
    num_inliers: int
    num_correspondences: int
    reprojection_error: float  # 内点平均重投影误差 (px)
    used_indices: np.ndarray   # (N,) 使用的关键点全局索引
    used_court_pts: np.ndarray # (N, 2) 使用的球场坐标
    used_pixel_pts: np.ndarray # (N, 2) 使用的像素坐标


def compute_homography(
    detection: CourtDetectionResult,
    court_keypoints_2d: np.ndarray,
    min_confidence: float = 0.5,
    ransac_threshold: float = 5.0,
) -> HomographyResult | None:
    """从检测结果计算 Homography。

    Args:
        detection: YOLO 检测结果。
        court_keypoints_2d: 标准球场 22 个地面关键点坐标, shape (22, 2)。
        min_confidence: 关键点最低置信度。
        ransac_threshold: RANSAC 重投影阈值 (px)。

    Returns:
        HomographyResult 或 None（点数不足或计算失败）。
    """
    # 1. 提取高置信度地面关键点
    indices, pixel_pts = detection.get_ground_keypoints(min_confidence)

    if len(indices) < 4:
        logger.warning(
            "Homography 计算失败：可用地面关键点 %d 个，不足 4 个", len(indices)
        )
        return None

    # 2. 查表获取对应的标准球场坐标
    court_pts = court_keypoints_2d[indices]  # shape (N, 2)

    # 3. 求解 Homography
    src = court_pts.astype(np.float64)
    dst = pixel_pts.astype(np.float64)

    if len(indices) >= 5:
        H, mask = cv2.findHomography(
            srcPoints=src,
            dstPoints=dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
        )
    else:
        # 恰好 4 点：最小二乘法（无 RANSAC）
        H, mask = cv2.findHomography(
            srcPoints=src,
            dstPoints=dst,
            method=0,
        )

    if H is None:
        logger.warning("cv2.findHomography 返回 None")
        return None

    # 4. 计算重投影误差
    if mask is not None:
        inlier_mask = mask.ravel().astype(bool)
    else:
        inlier_mask = np.ones(len(indices), dtype=bool)

    court_pts_input = court_pts.reshape(1, -1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(court_pts_input, H.astype(np.float64))
    projected = projected.reshape(-1, 2)

    errors = np.linalg.norm(projected - pixel_pts, axis=1)
    inlier_errors = errors[inlier_mask]
    mean_error = (
        float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else float("inf")
    )

    # 5. 逆矩阵
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        logger.warning("Homography 矩阵不可逆")
        return None

    return HomographyResult(
        H=H,
        H_inv=H_inv,
        inlier_mask=inlier_mask,
        num_inliers=int(np.sum(inlier_mask)),
        num_correspondences=len(indices),
        reprojection_error=mean_error,
        used_indices=indices,
        used_court_pts=court_pts,
        used_pixel_pts=pixel_pts,
    )


def validate_homography(result: HomographyResult) -> dict:
    """对 Homography 结果执行 5 项质量检查。

    Returns:
        dict，包含各项检查结果和 'overall_ok' 布尔值。
    """
    metrics = {}

    # 检查 1: 重投影误差 < 5.0 px
    metrics["reprojection_error"] = result.reprojection_error
    metrics["reproj_ok"] = result.reprojection_error < 5.0

    # 检查 2: 内点比例 > 0.7
    inlier_ratio = result.num_inliers / max(result.num_correspondences, 1)
    metrics["inlier_ratio"] = inlier_ratio
    metrics["inlier_ok"] = inlier_ratio > 0.7

    # 检查 3: 行列式非零（非退化）
    # 注意：球场坐标 Y 向上，像素坐标 Y 向下，det(H) < 0 是正常现象
    det_H = float(np.linalg.det(result.H))
    metrics["det_H"] = det_H
    metrics["det_ok"] = abs(det_H) > 1e-10

    # 检查 4: 条件数 < 1e6
    cond = float(np.linalg.cond(result.H))
    metrics["condition_number"] = cond
    metrics["cond_ok"] = cond < 1e6

    # 检查 5: 奇异值比 (max/min) < 1e5
    # 真实透视变换的 SV 比可达 1e4 量级，阈值适度放宽
    _, S, _ = np.linalg.svd(result.H)
    sv_ratio = float(S[0] / max(S[-1], 1e-15))
    metrics["singular_value_ratio"] = sv_ratio
    metrics["sv_ok"] = sv_ratio < 1e5

    metrics["overall_ok"] = all([
        metrics["reproj_ok"],
        metrics["inlier_ok"],
        metrics["det_ok"],
        metrics["cond_ok"],
        metrics["sv_ok"],
    ])

    if not metrics["overall_ok"]:
        failed = [
            k for k in ["reproj_ok", "inlier_ok", "det_ok", "cond_ok", "sv_ok"]
            if not metrics[k]
        ]
        logger.warning("Homography 验证失败: %s", failed)

    return metrics
