"""
几何工具函数。
"""

from __future__ import annotations

import numpy as np


def line_intersection(
    line1_coeffs: tuple[float, float, float],
    line2_coeffs: tuple[float, float, float],
) -> np.ndarray | None:
    """求两条直线 a*x + b*y + c = 0 的交点。

    Args:
        line1_coeffs: (a1, b1, c1)。
        line2_coeffs: (a2, b2, c2)。

    Returns:
        交点坐标 ndarray shape (2,)，平行线返回 None。
    """
    a1, b1, c1 = line1_coeffs
    a2, b2, c2 = line2_coeffs
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return np.array([x, y], dtype=np.float64)
