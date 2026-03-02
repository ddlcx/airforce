"""
绘制标准羽毛球场地2D平面图，标注22个地面关键点。
坐标系：原点在球场中心（球网地面线中点），X沿球网，Y沿场地长度。
"""

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── BWF 标准尺寸 (米) ──
FULL_LENGTH = 13.40
DOUBLES_WIDTH = 6.10
SINGLES_WIDTH = 5.18
HALF_LENGTH = 6.70
FRONT_SERVICE = 1.98
BACK_SERVICE_DOUBLES = 5.94  # 距网

# ── 22 个地面关键点 (X, Y) ──
KEYPOINTS = {
    0:  (-3.05, +6.70),
    1:  (+3.05, +6.70),
    2:  (-3.05, -6.70),
    3:  (+3.05, -6.70),
    4:  (-2.59, +6.70),
    5:  (+2.59, +6.70),
    6:  (-2.59, -6.70),
    7:  (+2.59, -6.70),
    8:  (-3.05,  0.00),
    9:  (+3.05,  0.00),
    10: (-3.05, +1.98),   # 改：移到双打外边线
    11: (+3.05, +1.98),   # 改：移到双打外边线
    12: (-3.05, -1.98),   # 改：移到双打外边线
    13: (+3.05, -1.98),   # 改：移到双打外边线
    14: (-3.05, +5.94),
    15: (+3.05, +5.94),
    16: (-3.05, -5.94),
    17: (+3.05, -5.94),
    18: ( 0.00, +1.98),
    19: ( 0.00, -1.98),
    20: ( 0.00, +6.70),   # 新增：远端底线中点
    21: ( 0.00, -6.70),   # 新增：近端底线中点
}

# ── 关键点中文名称 ──
KEYPOINT_NAMES = {
    0:  "远端左双打角",
    1:  "远端右双打角",
    2:  "近端左双打角",
    3:  "近端右双打角",
    4:  "远端左单打角",
    5:  "远端右单打角",
    6:  "近端左单打角",
    7:  "近端右单打角",
    8:  "网左端底部",
    9:  "网右端底部",
    10: "远端左前发球线端",
    11: "远端右前发球线端",
    12: "近端左前发球线端",
    13: "近端右前发球线端",
    14: "远端左后发球角",
    15: "远端右后发球角",
    16: "近端左后发球角",
    17: "近端右后发球角",
    18: "远端中线/前发球线交点",
    19: "近端中线/前发球线交点",
    20: "远端底线中点",
    21: "近端底线中点",
}

# ── 球场线段 (关键点索引对) ──
COURT_LINE_SEGMENTS = [
    (0, 1),    # 远端底线
    (1, 3),    # 右侧双打边线
    (3, 2),    # 近端底线
    (2, 0),    # 左侧双打边线
    (4, 6),    # 左单打边线
    (5, 7),    # 右单打边线
    (8, 9),    # 球网地面线
    (10, 11),  # 远端前发球线
    (12, 13),  # 近端前发球线
    (14, 15),  # 远端双打后发球线
    (16, 17),  # 近端双打后发球线
    (18, 20),  # 远端中线（前发球线→底线）
    (19, 21),  # 近端中线（前发球线→底线）
]


def draw_court():
    fig, ax = plt.subplots(1, 1, figsize=(10, 18))
    fig.patch.set_facecolor('#f0f0f0')
    ax.set_facecolor('#2e7d32')  # 绿色球场底色

    # ── 绘制球场线段 ──
    for (i, j) in COURT_LINE_SEGMENTS:
        x1, y1 = KEYPOINTS[i]
        x2, y2 = KEYPOINTS[j]
        lw = 2.5 if (i, j) == (8, 9) else 1.8  # 球网线加粗
        color = '#ff9800' if (i, j) == (8, 9) else 'white'
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=2)

    # ── 绘制关键点 ──
    for idx, (x, y) in KEYPOINTS.items():
        # 画点
        ax.plot(x, y, 'o', color='#f44336', markersize=8,
                markeredgecolor='white', markeredgewidth=1.2, zorder=4)

        # ── 确定标签偏移，避免重叠 ──
        dx, dy = 0, 0
        ha, va = 'center', 'center'

        # 左侧点：标签放左边
        if x < -1:
            dx = -0.25
            ha = 'right'
        # 右侧点：标签放右边
        elif x > 1:
            dx = +0.25
            ha = 'left'

        # 顶部点：标签向上
        if y > 6:
            dy = +0.30
            va = 'bottom'
        # 底部点：标签向下
        elif y < -6:
            dy = -0.30
            va = 'top'
        # 中间区域微调
        elif abs(y) < 0.5:
            dy = -0.35 if x < 0 else +0.35
            va = 'top' if x < 0 else 'bottom'
        elif abs(x) < 0.5:
            # 中心点
            dx = +0.25
            ha = 'left'

        # 特殊处理：避免标签与线段重叠
        if idx in (10, 12):  # 左前发球线端（现在在双打边线上）
            dx = -0.25
            ha = 'right'
        if idx in (11, 13):  # 右前发球线端
            dx = +0.25
            ha = 'left'
        if idx in (14, 16):  # 左后发球角
            dx = -0.25
            ha = 'right'
        if idx in (15, 17):  # 右后发球角
            dx = +0.25
            ha = 'left'
        if idx == 18:
            dx = +0.20
            ha = 'left'
            dy = +0.15
        if idx == 19:
            dx = +0.20
            ha = 'left'
            dy = -0.15
        if idx == 20:  # 远端底线中点
            dx = +0.20
            ha = 'left'
            dy = +0.30
            va = 'bottom'
        if idx == 21:  # 近端底线中点
            dx = +0.20
            ha = 'left'
            dy = -0.30
            va = 'top'

        label = f"[{idx}] {KEYPOINT_NAMES[idx]}"
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x + dx, y + dy),
            fontsize=7.5,
            fontweight='bold',
            color='#ffeb3b',
            ha=ha,
            va=va,
            zorder=5,
            fontfamily='sans-serif',
            bbox=dict(
                boxstyle='round,pad=0.15',
                facecolor='#1b5e20',
                edgecolor='none',
                alpha=0.85,
            ),
            arrowprops=dict(
                arrowstyle='-',
                color='#ffeb3b',
                lw=0.6,
            ) if (abs(dx) > 0.15 or abs(dy) > 0.15) else None,
        )

    # ── 添加球网标注文字 ──
    ax.text(0, 0.25, "← 球 网 →", fontsize=11, color='#ff9800',
            ha='center', va='bottom', fontweight='bold', zorder=3)

    # ── 添加方向标注 ──
    ax.text(0, +7.3, "远  端 (Far End)", fontsize=11, color='white',
            ha='center', va='bottom', fontweight='bold')
    ax.text(0, -7.3, "近  端 (Near End)", fontsize=11, color='white',
            ha='center', va='top', fontweight='bold')

    # ── 坐标轴设置 ──
    margin = 1.8
    ax.set_xlim(-3.05 - margin, 3.05 + margin)
    ax.set_ylim(-6.70 - margin, 6.70 + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('X (米) — 沿球网方向', fontsize=10, color='#333')
    ax.set_ylabel('Y (米) — 沿球场长度方向', fontsize=10, color='#333')
    ax.set_title('BWF 标准羽毛球场 — 22个地面关键点标注图',
                 fontsize=14, fontweight='bold', pad=15)

    # 网格
    ax.grid(True, alpha=0.15, color='white', linestyle='--')
    ax.tick_params(colors='#555', labelsize=8)

    plt.tight_layout()

    # ── 保存 ──
    output_path = '/Users/duanxingxing/Desktop/badminton_plan/data/court_diagram_22keypoints.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"球场关键点标注图已保存到: {output_path}")
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    draw_court()
