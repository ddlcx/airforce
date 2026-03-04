"""
球网关键点索引映射模块。

数据集（Roboflow）的球网关键点排序为：左上、右上、左下、右下，
而 base_definitions.md 的全局编号为：22(左上)、23(左下)、24(右上)、25(右下)。
本模块定义两者之间的双向映射，以及正确的 flip_idx。
"""

import numpy as np

# ──────────────────────────────────────────────────────────────
# 数据集关键点数量
# ──────────────────────────────────────────────────────────────
NUM_KEYPOINTS = 4

# ──────────────────────────────────────────────────────────────
# 数据集排序下的关键点名称
# 排序依据：通过分析 Roboflow 导出标注的坐标空间分布确认
#   ds0: 小x 小y → 左上
#   ds1: 大x 小y → 右上
#   ds2: 小x 大y → 左下
#   ds3: 大x 大y → 右下
# ──────────────────────────────────────────────────────────────
DATASET_KEYPOINT_NAMES = {
    0: "网左端顶部",
    1: "网右端顶部",
    2: "网左端底部",
    3: "网右端底部",
}

# ──────────────────────────────────────────────────────────────
# 数据集索引 → base_definitions.md 全局索引 的映射
# ──────────────────────────────────────────────────────────────
DATASET_TO_PLAN_INDEX = {
    0: 22,   # 网左端顶部
    1: 24,   # 网右端顶部
    2: 23,   # 网左端底部
    3: 25,   # 网右端底部
}

# 全局索引 → 数据集索引 的反向映射
PLAN_TO_DATASET_INDEX = {v: k for k, v in DATASET_TO_PLAN_INDEX.items()}

# ──────────────────────────────────────────────────────────────
# 正确的 flip_idx（基于数据集排序）
# 水平翻转时，左右对称的关键点互换索引：
#   ds0(左上) ↔ ds1(右上)
#   ds2(左下) ↔ ds3(右下)
# ──────────────────────────────────────────────────────────────
DATASET_FLIP_IDX = [1, 0, 3, 2]

# ──────────────────────────────────────────────────────────────
# 数据集排序下的世界坐标 (X, Y, Z)，单位：米
# 原点在球场中心（球网地面线中点）
# ──────────────────────────────────────────────────────────────
DATASET_WORLD_COORDS = {
    0: (-3.05, 0.00, 1.55),   # 网左端顶部
    1: (+3.05, 0.00, 1.55),   # 网右端顶部
    2: (-3.05, 0.00, 0.00),   # 网左端底部
    3: (+3.05, 0.00, 0.00),   # 网右端底部
}


def remap_keypoints(keypoints, source_to_target):
    """将关键点数组按映射表重新排序。

    Args:
        keypoints: shape (N, ...) 的数组，N >= NUM_KEYPOINTS。
                   每个关键点可以是 (x, y)、(x, y, conf) 等。
        source_to_target: dict，source_index → target_index。

    Returns:
        重排后的数组，shape 与输入相同。
    """
    keypoints = np.asarray(keypoints)
    result = np.zeros_like(keypoints)
    for src_idx, tgt_idx in source_to_target.items():
        result[tgt_idx] = keypoints[src_idx]
    return result


def dataset_to_plan(keypoints):
    """将数据集排序的关键点转换为 base_definitions.md 全局排序。

    输入: 4个关键点 (ds0-ds3)
    输出: 4个关键点，索引对应全局编号 22-25（偏移后从0开始）
    """
    return remap_keypoints(keypoints, DATASET_TO_PLAN_INDEX)


def plan_to_dataset(keypoints):
    """将 base_definitions.md 全局排序的关键点转换为数据集排序。"""
    return remap_keypoints(keypoints, PLAN_TO_DATASET_INDEX)
