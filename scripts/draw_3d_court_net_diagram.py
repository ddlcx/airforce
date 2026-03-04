"""
绘制标准羽毛球场3D视图，同时展示22个地面关键点和4个球网关键点。
坐标系：原点在球场中心（球网地面线中点），X沿球网，Y沿场地长度，Z竖直向上。
"""

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── 22 个地面关键点 (X, Y, Z=0) ──
GROUND_KEYPOINTS = {
    0:  (-3.05, +6.70, 0.0),
    1:  (+3.05, +6.70, 0.0),
    2:  (-3.05, -6.70, 0.0),
    3:  (+3.05, -6.70, 0.0),
    4:  (-2.59, +6.70, 0.0),
    5:  (+2.59, +6.70, 0.0),
    6:  (-2.59, -6.70, 0.0),
    7:  (+2.59, -6.70, 0.0),
    8:  (-3.05,  0.00, 0.0),
    9:  (+3.05,  0.00, 0.0),
    10: (-3.05, +1.98, 0.0),
    11: (+3.05, +1.98, 0.0),
    12: (-3.05, -1.98, 0.0),
    13: (+3.05, -1.98, 0.0),
    14: (-3.05, +5.94, 0.0),
    15: (+3.05, +5.94, 0.0),
    16: (-3.05, -5.94, 0.0),
    17: (+3.05, -5.94, 0.0),
    18: ( 0.00, +1.98, 0.0),
    19: ( 0.00, -1.98, 0.0),
    20: ( 0.00, +6.70, 0.0),
    21: ( 0.00, -6.70, 0.0),
}

# ── 4 个球网关键点 (X, Y, Z) ──
NET_KEYPOINTS = {
    22: (-3.05, 0.00, 1.55),
    23: (-3.05, 0.00, 0.00),
    24: (+3.05, 0.00, 1.55),
    25: (+3.05, 0.00, 0.00),
}

ALL_KEYPOINTS = {**GROUND_KEYPOINTS, **NET_KEYPOINTS}

# ── 关键点名称 ──
KEYPOINT_NAMES = {
    0:  "远端左双打角",      1:  "远端右双打角",
    2:  "近端左双打角",      3:  "近端右双打角",
    4:  "远端左单打角",      5:  "远端右单打角",
    6:  "近端左单打角",      7:  "近端右单打角",
    8:  "网左端底部",        9:  "网右端底部",
    10: "远端左前发球线端",  11: "远端右前发球线端",
    12: "近端左前发球线端",  13: "近端右前发球线端",
    14: "远端左后发球角",    15: "远端右后发球角",
    16: "近端左后发球角",    17: "近端右后发球角",
    18: "远端中线交点",      19: "近端中线交点",
    20: "远端底线中点",      21: "近端底线中点",
    22: "网左端顶部",        23: "网左端底部",
    24: "网右端顶部",        25: "网右端底部",
}

# ── 球场线段 ──
COURT_LINE_SEGMENTS = [
    (0, 1), (1, 3), (3, 2), (2, 0),    # 双打外边界
    (4, 6), (5, 7),                      # 单打边线
    (8, 9),                              # 球网地面线
    (10, 11), (12, 13),                  # 前发球线
    (14, 15), (16, 17),                  # 双打后发球线
    (18, 20), (19, 21),                  # 中线
]

# ── 球网线段 ──
NET_LINE_SEGMENTS = [
    (22, 24),  # 顶边
    (23, 25),  # 底边
    (22, 23),  # 左侧边
    (24, 25),  # 右侧边
]


def draw_3d():
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#f0f0f0')

    # ── 绘制球场地面 ──
    court_verts = [[
        (-3.05, -6.70, 0), (3.05, -6.70, 0),
        (3.05, 6.70, 0), (-3.05, 6.70, 0),
    ]]
    court_poly = Poly3DCollection(court_verts, alpha=0.4,
                                  facecolor='#2e7d32', edgecolor='none')
    ax.add_collection3d(court_poly)

    # ── 绘制球场线段 ──
    for (i, j) in COURT_LINE_SEGMENTS:
        x1, y1, z1 = ALL_KEYPOINTS[i]
        x2, y2, z2 = ALL_KEYPOINTS[j]
        color = '#ff9800' if (i, j) == (8, 9) else 'white'
        lw = 2.0 if (i, j) == (8, 9) else 1.2
        ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linewidth=lw)

    # ── 绘制球网（半透明填充） ──
    net_verts = [[
        (-3.05, 0, 0), (-3.05, 0, 1.55),
        (3.05, 0, 1.55), (3.05, 0, 0),
    ]]
    net_poly = Poly3DCollection(net_verts, alpha=0.3,
                                facecolor='#ff9800', edgecolor='#ff9800',
                                linewidth=1.5)
    ax.add_collection3d(net_poly)

    # ── 绘制球网线段 ──
    for (i, j) in NET_LINE_SEGMENTS:
        x1, y1, z1 = ALL_KEYPOINTS[i]
        x2, y2, z2 = ALL_KEYPOINTS[j]
        ax.plot([x1, x2], [y1, y2], [z1, z2],
                color='#ff9800', linewidth=2.0)

    # ── 绘制地面关键点 ──
    for idx, (x, y, z) in GROUND_KEYPOINTS.items():
        ax.scatter(x, y, z, color='#f44336', s=40, edgecolors='white',
                   linewidths=0.8, zorder=4, depthshade=False)

    # ── 绘制球网关键点（稍大以区分） ──
    for idx, (x, y, z) in NET_KEYPOINTS.items():
        ax.scatter(x, y, z, color='#ff9800', s=70, edgecolors='white',
                   linewidths=1.0, zorder=5, depthshade=False, marker='D')

    # ── 标注所有关键点编号 ──
    for idx, (x, y, z) in ALL_KEYPOINTS.items():
        # 标签偏移
        dx, dy, dz = 0, 0, 0
        if x < 0:
            dx = -0.35
        elif x > 0:
            dx = +0.35
        if z > 0:
            dz = +0.12
        else:
            dz = -0.10

        label = f"[{idx}]"
        fontsize = 6.5 if idx < 22 else 8
        color = '#ffeb3b' if idx < 22 else '#ffffff'
        ax.text(x + dx, y + dy, z + dz, label, fontsize=fontsize,
                color=color, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='#1b5e20',
                          edgecolor='none', alpha=0.8))

    # ── 添加图例 ──
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f44336',
               markeredgecolor='white', markersize=8,
               label='地面关键点 [0-21]'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#ff9800',
               markeredgecolor='white', markersize=8,
               label='球网关键点 [22-25]'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.8)

    # ── 视角与坐标轴 ──
    ax.view_init(elev=25, azim=-55)
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-7.5, 7.5)
    ax.set_zlim(-0.5, 3.0)
    ax.set_xlabel('X (米) — 沿球网方向', fontsize=9, labelpad=8)
    ax.set_ylabel('Y (米) — 沿球场长度方向', fontsize=9, labelpad=8)
    ax.set_zlabel('Z (米) — 竖直向上', fontsize=9, labelpad=8)
    ax.set_title('BWF 标准羽毛球场 — 26个关键点3D总览\n(22地面 + 4球网)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(labelsize=7)

    plt.tight_layout()

    # ── 保存 ──
    output_path = '/Users/duanxingxing/Desktop/badminton_code/airforce/data/court_net_3d_26keypoints.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"3D关键点总览图已保存到: {output_path}")
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    draw_3d()
