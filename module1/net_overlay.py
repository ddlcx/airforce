"""
球网地面线投影工具。

投影球网立柱底部到像素空间，供模块2端点推断使用。
"""

from __future__ import annotations

import numpy as np

from module1.court_renderer import project_points_batch

# 球网立柱底部世界坐标 (Z=0 平面)
_NET_POST_BASES = np.array([
    [-3.05, 0.0],   # 左侧立柱底部
    [+3.05, 0.0],   # 右侧立柱底部
], dtype=np.float64)


def get_net_post_base_pixels(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """投影球网立柱底部到像素坐标。

    Args:
        H: (3, 3) Homography 矩阵（球场坐标 → 像素坐标）。

    Returns:
        (left_base_px, right_base_px) 各 shape (2,)。
    """
    pixels = project_points_batch(H, _NET_POST_BASES)  # (2, 2)
    return pixels[0], pixels[1]
