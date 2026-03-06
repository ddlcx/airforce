"""
YOLO 球网关键点提取 (子模块 2.2)。

从 YOLO 球网模型检测结果提取球网 4 个关键点 (kp22-25)，
直接使用 YOLO 模型原始值，无额外矫正。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from module1.yolo_detector import CourtDetectionResult

logger = logging.getLogger(__name__)

# 球网端点固定 3D 坐标
_LEFT_TOP_3D = np.array([-3.05, 0.0, 1.55], dtype=np.float64)
_RIGHT_TOP_3D = np.array([+3.05, 0.0, 1.55], dtype=np.float64)
_LEFT_BASE_3D = np.array([-3.05, 0.0, 0.0], dtype=np.float64)
_RIGHT_BASE_3D = np.array([+3.05, 0.0, 0.0], dtype=np.float64)


@dataclass
class NetKeypointResult:
    """球网关键点提取结果。"""

    left_top_pixel: np.ndarray       # (2,) kp22 像素坐标
    right_top_pixel: np.ndarray      # (2,) kp24 像素坐标
    left_base_pixel: np.ndarray | None  # (2,) kp23 像素坐标
    right_base_pixel: np.ndarray | None # (2,) kp25 像素坐标
    left_top_3d: np.ndarray          # (-3.05, 0, 1.55)
    right_top_3d: np.ndarray         # (+3.05, 0, 1.55)
    left_base_3d: np.ndarray         # (-3.05, 0, 0)
    right_base_3d: np.ndarray        # (+3.05, 0, 0)


def extract_net_keypoints(
    detection: CourtDetectionResult,
    min_conf: float = 0.5,
) -> NetKeypointResult | None:
    """从 YOLO 球网模型检测结果提取关键点。

    直接使用球网模型的 kp22-25，不与球场模型的 kp8/kp9 融合。

    Args:
        detection: 26 关键点检测结果。
        min_conf: 关键点最低置信度。

    Returns:
        NetKeypointResult 或 None（kp22/kp24 不足时）。
    """
    # ── Step 1: 提取顶部关键点 ──
    kp22 = detection.keypoints[22]  # 网左端顶部
    kp24 = detection.keypoints[24]  # 网右端顶部

    if kp22.confidence < min_conf or kp24.confidence < min_conf:
        logger.debug(
            "球网顶部关键点不足: kp22=%.2f, kp24=%.2f (阈值 %.2f)",
            kp22.confidence, kp24.confidence, min_conf,
        )
        return None

    left_top_px = kp22.pixel_xy.copy()
    right_top_px = kp24.pixel_xy.copy()

    # ── Step 2: 提取底部关键点 ──
    kp23 = detection.keypoints[23]  # 网左端底部
    kp25 = detection.keypoints[25]  # 网右端底部

    left_base_px = kp23.pixel_xy.copy() if kp23.confidence >= min_conf else None
    right_base_px = kp25.pixel_xy.copy() if kp25.confidence >= min_conf else None

    return NetKeypointResult(
        left_top_pixel=left_top_px,
        right_top_pixel=right_top_px,
        left_base_pixel=left_base_px,
        right_base_pixel=right_base_px,
        left_top_3d=_LEFT_TOP_3D.copy(),
        right_top_3d=_RIGHT_TOP_3D.copy(),
        left_base_3d=_LEFT_BASE_3D.copy(),
        right_base_3d=_RIGHT_BASE_3D.copy(),
    )
