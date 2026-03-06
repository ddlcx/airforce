"""
球场线条叠加渲染。

使用 Homography 矩阵 H 将标准球场线段投影到视频帧上并绘制。
提供 project_points_batch 作为项目通用投影工具函数。
"""

from __future__ import annotations

import cv2
import numpy as np

from module1.yolo_detector import CourtDetectionResult


def project_points_batch(H: np.ndarray, court_pts: np.ndarray) -> np.ndarray:
    """批量投影球场坐标到像素坐标。

    Args:
        H: (3, 3) Homography 矩阵（球场坐标 → 像素坐标）。
        court_pts: (N, 2) 球场世界坐标。

    Returns:
        (N, 2) 像素坐标。
    """
    pts = court_pts.reshape(1, -1, 2).astype(np.float32)
    result = cv2.perspectiveTransform(pts, H.astype(np.float64))
    return result.reshape(-1, 2)


def project_point(H: np.ndarray, court_pt: np.ndarray) -> np.ndarray:
    """单点投影球场坐标到像素坐标。

    Args:
        H: (3, 3) Homography 矩阵。
        court_pt: (2,) 球场世界坐标。

    Returns:
        (2,) 像素坐标。
    """
    return project_points_batch(H, court_pt.reshape(1, 2))[0]


def draw_court_overlay(
    frame: np.ndarray,
    H: np.ndarray,
    keypoints: np.ndarray,
    segments: list[tuple[int, int]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    num_samples: int = 21,
) -> np.ndarray:
    """在帧上渲染球场线条叠加层。

    Args:
        frame: BGR 图像, shape (H, W, 3)。
        H: (3, 3) Homography 矩阵（球场坐标 → 像素坐标）。
        keypoints: (22, 2) 标准球场地面关键点坐标（世界坐标, 米）。
        segments: 线段列表, 每个元素是 (i, j) 关键点索引对。
        color: BGR 颜色。
        thickness: 线条粗细 (px)。
        num_samples: 每条线段的采样点数。

    Returns:
        叠加球场线条的帧图像（新副本）。
    """
    overlay = frame.copy()
    t_values = np.linspace(0.0, 1.0, num_samples)

    for i, j in segments:
        pt_a = keypoints[i]   # (2,)
        pt_b = keypoints[j]   # (2,)

        # 沿线段均匀采样
        sample_pts = (
            np.outer(1.0 - t_values, pt_a) + np.outer(t_values, pt_b)
        )  # (num_samples, 2)

        # 批量投影
        pixel_pts = project_points_batch(H, sample_pts)  # (num_samples, 2)

        # 绘制连续线段
        pts_int = pixel_pts.astype(np.int32)
        for k in range(num_samples - 1):
            p0 = tuple(pts_int[k])
            p1 = tuple(pts_int[k + 1])
            cv2.line(overlay, p0, p1, color, thickness, cv2.LINE_AA)

    return overlay


def draw_keypoint_markers(
    frame: np.ndarray,
    detection: CourtDetectionResult,
    min_conf: float = 0.5,
    color: tuple[int, int, int] = (0, 0, 255),
    radius: int = 5,
    ground_only: bool = False,
) -> np.ndarray:
    """在帧上标注检测到的关键点位置（用于调试）。

    Args:
        frame: BGR 图像。
        detection: CourtDetectionResult。
        min_conf: 最低置信度。
        color: BGR 颜色。
        radius: 圆点半径。
        ground_only: 仅标注地面关键点 (0-21)，不标注球网关键点 (22-25)。

    Returns:
        标注后的帧图像（新副本）。
    """
    from config.court_config import NUM_GROUND_KEYPOINTS

    overlay = frame.copy()
    for kp in detection.keypoints:
        if kp.confidence < min_conf:
            continue
        if ground_only and kp.index >= NUM_GROUND_KEYPOINTS:
            continue
        center = tuple(kp.pixel_xy.astype(int))
        cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(kp.index),
            (center[0] + 8, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay
