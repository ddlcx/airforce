"""
绘制球网正面视图，标注4个球网关键点。
视角：从球场一侧正面看球网（X-Z平面，Y=0）。
"""

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── 球网4个关键点 (编号22-25) ──
# 修改后：左端顶部、左端底部、右端顶部、右端底部
NET_KEYPOINTS = {
    22: (-3.05, 1.55),   # 网左端顶部 (X, Z)
    23: (-3.05, 0.00),   # 网左端底部 (X, Z)
    24: (+3.05, 1.55),   # 网右端顶部 (X, Z)
    25: (+3.05, 0.00),   # 网右端底部 (X, Z)
}

NET_KEYPOINT_NAMES = {
    22: "网左端顶部",
    23: "网左端底部",
    24: "网右端顶部",
    25: "网右端底部",
}

NET_KEYPOINT_COORDS_3D = {
    22: "(-3.05, 0.00, 1.55)",
    23: "(-3.05, 0.00, 0.00)",
    24: "(+3.05, 0.00, 1.55)",
    25: "(+3.05, 0.00, 0.00)",
}

# ── 球网线段（4条边） ──
NET_LINE_SEGMENTS = [
    (22, 24),  # 顶边
    (23, 25),  # 底边
    (22, 23),  # 左侧边
    (24, 25),  # 右侧边
]


def draw_net():
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.patch.set_facecolor('#f0f0f0')
    ax.set_facecolor('#2e7d32')

    # ── 绘制球网矩形区域（填充） ──
    net_rect = patches.Rectangle(
        (-3.05, 0.00), 6.10, 1.55,
        linewidth=0, facecolor='#1b5e20', alpha=0.5, zorder=1,
    )
    ax.add_patch(net_rect)

    # ── 绘制球网线段 ──
    for (i, j) in NET_LINE_SEGMENTS:
        x1, z1 = NET_KEYPOINTS[i]
        x2, z2 = NET_KEYPOINTS[j]
        is_top = (i in (22, 24) and j in (22, 24))
        lw = 3.0 if is_top else 2.0
        color = '#ff9800' if is_top else 'white'
        ax.plot([x1, x2], [z1, z2], color=color, linewidth=lw, zorder=2)

    # ── 绘制网格线（模拟球网纹理） ──
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        ax.plot([x, x], [0.0, 1.55], color='white', linewidth=0.3, alpha=0.3, zorder=1)
    for z in [0.3, 0.6, 0.9, 1.2]:
        ax.plot([-3.05, 3.05], [z, z], color='white', linewidth=0.3, alpha=0.3, zorder=1)

    # ── 绘制关键点 ──
    for idx, (x, z) in NET_KEYPOINTS.items():
        ax.plot(x, z, 'o', color='#f44336', markersize=12,
                markeredgecolor='white', markeredgewidth=1.5, zorder=4)

        # 确定标签偏移
        dx, dz = 0, 0
        ha, va = 'center', 'center'

        if idx == 22:  # 左上
            dx = -0.25
            dz = +0.15
            ha = 'right'
            va = 'bottom'
        elif idx == 23:  # 左下
            dx = -0.25
            dz = -0.15
            ha = 'right'
            va = 'top'
        elif idx == 24:  # 右上
            dx = +0.25
            dz = +0.15
            ha = 'left'
            va = 'bottom'
        elif idx == 25:  # 右下
            dx = +0.25
            dz = -0.15
            ha = 'left'
            va = 'top'

        label = f"[{idx}] {NET_KEYPOINT_NAMES[idx]}\n{NET_KEYPOINT_COORDS_3D[idx]}"
        ax.annotate(
            label,
            xy=(x, z),
            xytext=(x + dx, z + dz),
            fontsize=9,
            fontweight='bold',
            color='#ffeb3b',
            ha=ha,
            va=va,
            zorder=5,
            fontfamily='sans-serif',
            bbox=dict(
                boxstyle='round,pad=0.2',
                facecolor='#1b5e20',
                edgecolor='#ffeb3b',
                alpha=0.9,
                linewidth=0.8,
            ),
            arrowprops=dict(
                arrowstyle='-',
                color='#ffeb3b',
                lw=0.8,
            ),
        )

    # ── 标注尺寸 ──
    # 宽度标注
    ax.annotate('', xy=(3.05, 1.75), xytext=(-3.05, 1.75),
                arrowprops=dict(arrowstyle='<->', color='white', lw=1.2))
    ax.text(0, 1.80, '6.10m (双打宽)', fontsize=9, color='white',
            ha='center', va='bottom', fontweight='bold')

    # 高度标注（左侧）
    ax.annotate('', xy=(-3.55, 1.55), xytext=(-3.55, 0.0),
                arrowprops=dict(arrowstyle='<->', color='white', lw=1.2))
    ax.text(-3.65, 0.775, '1.55m\n(立柱处)', fontsize=8, color='white',
            ha='right', va='center', fontweight='bold')

    # ── 地面线 ──
    ax.plot([-4.0, 4.0], [0, 0], color='#8d6e63', linewidth=3, zorder=0)
    ax.text(0, -0.12, '地面 (Z=0, Y=0)', fontsize=8, color='#bcaaa4',
            ha='center', va='top')

    # ── 坐标轴设置 ──
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-0.5, 2.3)
    ax.set_aspect('equal')
    ax.set_xlabel('X (米) — 沿球网方向', fontsize=10, color='#333')
    ax.set_ylabel('Z (米) — 竖直向上', fontsize=10, color='#333')
    ax.set_title('球网正面视图 — 4个球网关键点 (编号22-25)',
                 fontsize=14, fontweight='bold', pad=15)

    ax.grid(True, alpha=0.15, color='white', linestyle='--')
    ax.tick_params(colors='#555', labelsize=8)

    plt.tight_layout()

    # ── 保存 ──
    output_path = '/Users/duanxingxing/Desktop/badminton_code/airforce/data/net_diagram_4keypoints.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"球网关键点标注图已保存到: {output_path}")
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    draw_net()
