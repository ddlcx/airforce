"""
Visualize all MHE test results on a 3D badminton court.

Generates:
  1. Overview: all trajectories on one court
  2. Grid: individual court views per scenario
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config.physics_config import MHEConfig
from module3.trajectory_estimator import TrajectoryEstimator
from tests.test_module3.synthetic_data import get_test_scenarios

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'module3_tests')


# ── Badminton court geometry (ITF standard) ──

COURT_LENGTH = 13.40   # full length (m)
COURT_WIDTH = 6.10     # doubles width (m)
SINGLES_WIDTH = 5.18
SERVICE_LINE = 1.98    # from net (short service)
BACK_LINE = 0.76       # doubles long service from back
NET_HEIGHT = 1.55      # at posts

# Origin at court center (net line)
HALF_L = COURT_LENGTH / 2   # 6.70
HALF_W = COURT_WIDTH / 2    # 3.05
HALF_SW = SINGLES_WIDTH / 2  # 2.59


def draw_court_lines(ax, alpha=0.6, linewidth=1.0):
    """Draw badminton court lines on the ground plane (z=0)."""
    c = '#FFFFFF'

    # Outer boundary (doubles)
    corners = [
        [-HALF_W, -HALF_L], [HALF_W, -HALF_L],
        [HALF_W, HALF_L], [-HALF_W, HALF_L], [-HALF_W, -HALF_L],
    ]
    xs, ys = zip(*corners)
    ax.plot(xs, ys, [0]*5, color=c, linewidth=linewidth*1.5, alpha=alpha)

    # Singles sidelines
    for x in [-HALF_SW, HALF_SW]:
        ax.plot([x, x], [-HALF_L, HALF_L], [0, 0], color=c, linewidth=linewidth, alpha=alpha*0.7)

    # Center line (net)
    ax.plot([-HALF_W, HALF_W], [0, 0], [0, 0], color=c, linewidth=linewidth, alpha=alpha)

    # Short service lines
    for y_sign in [-1, 1]:
        y = y_sign * SERVICE_LINE
        ax.plot([-HALF_W, HALF_W], [y, y], [0, 0], color=c, linewidth=linewidth, alpha=alpha*0.7)

    # Long service lines (doubles)
    for y_sign in [-1, 1]:
        y = y_sign * (HALF_L - BACK_LINE)
        ax.plot([-HALF_W, HALF_W], [y, y], [0, 0], color=c, linewidth=linewidth, alpha=alpha*0.5)

    # Center service line
    ax.plot([0, 0], [-SERVICE_LINE, SERVICE_LINE], [0, 0], color=c, linewidth=linewidth, alpha=alpha*0.5)


def draw_net(ax, alpha=0.3):
    """Draw the net as a semi-transparent surface."""
    x = np.array([-HALF_W, HALF_W])
    z = np.array([0, NET_HEIGHT])
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)

    ax.plot_surface(X, Y, Z, color='white', alpha=alpha, edgecolor='gray', linewidth=0.5)

    # Net posts
    for xp in [-HALF_W, HALF_W]:
        ax.plot([xp, xp], [0, 0], [0, NET_HEIGHT + 0.05],
                color='gray', linewidth=2, alpha=0.8)


def draw_court_floor(ax, alpha=0.15):
    """Draw court floor as a colored rectangle."""
    verts = [
        [-HALF_W, -HALF_L, 0],
        [HALF_W, -HALF_L, 0],
        [HALF_W, HALF_L, 0],
        [-HALF_W, HALF_L, 0],
    ]
    poly = Poly3DCollection([verts], alpha=alpha, facecolor='#2E7D32', edgecolor='none')
    ax.add_collection3d(poly)


def setup_court_axes(ax, elev=25, azim=-60, title=''):
    """Configure axes for court view."""
    draw_court_floor(ax)
    draw_court_lines(ax)
    draw_net(ax)

    ax.set_xlim(-HALF_W - 0.5, HALF_W + 0.5)
    ax.set_ylim(-HALF_L - 1, HALF_L + 1)
    ax.set_zlim(0, 5)

    ax.set_xlabel('X (m)', fontsize=8, labelpad=2)
    ax.set_ylabel('Y (m)', fontsize=8, labelpad=2)
    ax.set_zlabel('Z (m)', fontsize=8, labelpad=2)
    ax.tick_params(labelsize=6)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=0)

    # Dark background for contrast
    ax.set_facecolor('#1a1a2e')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#444')
    ax.yaxis.pane.set_edgecolor('#444')
    ax.zaxis.pane.set_edgecolor('#444')
    ax.grid(True, alpha=0.15)


def run_all_scenarios():
    """Run MHE estimation for all scenarios and collect results."""
    estimator = TrajectoryEstimator()
    scenarios = get_test_scenarios(fps=30.0)

    results = []
    for sc in scenarios:
        try:
            result = estimator.estimate_segment(sc['segment'], segment_index=0)
            true_world = sc['true_data']['world_pos']
            est_world = result.world_positions
            n_common = min(len(true_world), len(est_world))
            pos_errors = np.linalg.norm(true_world[:n_common] - est_world[:n_common], axis=1)

            results.append({
                'name': sc['name'],
                'description': sc['description'],
                'true_world': true_world,
                'est_world': est_world,
                'mean_err': float(np.mean(pos_errors)),
                'max_err': float(np.max(pos_errors)),
                'shot_type': sc['shot_type'],
            })
        except Exception as e:
            print(f'  {sc["name"]} failed: {e}')

    return results


# ── Shot type colors ──
SHOT_COLORS = {
    'clear': '#2196F3',      # blue
    'smash': '#F44336',      # red
    'drop': '#FF9800',       # orange
    'drive': '#4CAF50',      # green
    'cross_court': '#9C27B0', # purple
}

SCENARIO_COLORS = {
    'clear_clean': '#2196F3',
    'clear_noisy': '#42A5F5',
    'smash_noisy': '#F44336',
    'drop_noisy': '#FF9800',
    'drive_noisy': '#4CAF50',
    'cross_court_noisy': '#9C27B0',
    'clear_missing30': '#1565C0',
    'smash_high_noise': '#D32F2F',
}


def plot_overview(results, output_dir):
    """All trajectories on one court."""
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0d1117')

    ax = fig.add_subplot(111, projection='3d')
    setup_court_axes(ax, elev=28, azim=-55,
                     title='Module 3 MHE: All Test Trajectories on Badminton Court')
    ax.title.set_color('white')
    ax.title.set_fontsize(13)

    for r in results:
        color = SCENARIO_COLORS.get(r['name'], 'cyan')
        tw = r['true_world']
        ew = r['est_world']
        n = min(len(tw), len(ew))
        label_true = f'{r["name"]} true'
        label_est = f'{r["name"]} est (err={r["mean_err"]:.2f}m)'

        ax.plot(tw[:, 0], tw[:, 1], tw[:, 2],
                color=color, linewidth=2.5, alpha=0.9, label=label_true)
        ax.plot(ew[:n, 0], ew[:n, 1], ew[:n, 2],
                color=color, linewidth=1.5, linestyle='--', alpha=0.7)

        # Start and end markers
        ax.scatter(*tw[0], color=color, s=40, marker='o', zorder=5)
        ax.scatter(*tw[-1], color=color, s=40, marker='s', zorder=5)

    # Camera position
    cam_pos = np.array([3.0, -14.0, 8.0])
    ax.scatter(*cam_pos, color='yellow', s=100, marker='^', zorder=10, label='Camera')

    ax.legend(fontsize=6, loc='upper left', ncol=2,
              facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')

    path = os.path.join(output_dir, 'court_overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {path}')


def plot_individual_grid(results, output_dir):
    """2x4 grid of individual court views."""
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(24, rows * 6.5))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Module 3 MHE: True (solid) vs Estimated (dashed) Trajectories',
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    for i, r in enumerate(results):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        color = SCENARIO_COLORS.get(r['name'], 'cyan')

        title = f'{r["description"]}\n3D err = {r["mean_err"]:.2f}m (max {r["max_err"]:.2f}m)'
        setup_court_axes(ax, elev=25, azim=-55, title=title)
        ax.title.set_color('white')

        tw = r['true_world']
        ew = r['est_world']
        n_common = min(len(tw), len(ew))

        # True trajectory
        ax.plot(tw[:, 0], tw[:, 1], tw[:, 2],
                color='#00E5FF', linewidth=3, alpha=0.95, label='True')

        # Estimated trajectory
        ax.plot(ew[:n_common, 0], ew[:n_common, 1], ew[:n_common, 2],
                color='#FF6D00', linewidth=2.5, linestyle='--', alpha=0.9, label='Estimated')

        # Start/end markers
        ax.scatter(*tw[0], color='#00E5FF', s=60, marker='o', zorder=5,
                   edgecolors='white', linewidths=0.5)
        ax.scatter(*tw[-1], color='#00E5FF', s=60, marker='s', zorder=5,
                   edgecolors='white', linewidths=0.5)
        ax.scatter(*ew[0], color='#FF6D00', s=40, marker='o', zorder=5)

        # Shadow on ground
        ax.plot(tw[:, 0], tw[:, 1], np.zeros(len(tw)),
                color='#00E5FF', linewidth=1, alpha=0.25, linestyle=':')
        ax.plot(ew[:n_common, 0], ew[:n_common, 1], np.zeros(n_common),
                color='#FF6D00', linewidth=1, alpha=0.2, linestyle=':')

        ax.legend(fontsize=7, loc='upper left',
                  facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, 'court_grid.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {path}')


def plot_side_view_grid(results, output_dir):
    """2x4 grid of side views (Y-Z) with court outline."""
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 5))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Side View (Y-Z): True vs Estimated Trajectories',
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    axes = axes.flatten()
    for i, r in enumerate(results):
        ax = axes[i]
        ax.set_facecolor('#1a1a2e')

        # Court outline (side view)
        # Ground
        ax.axhline(y=0, color='#2E7D32', linewidth=2, alpha=0.5)
        ax.fill_between([-HALF_L, HALF_L], 0, -0.2, color='#2E7D32', alpha=0.15)

        # Net
        ax.plot([0, 0], [0, NET_HEIGHT], color='white', linewidth=2, alpha=0.5)
        ax.plot([-0.02, 0.02], [NET_HEIGHT, NET_HEIGHT], color='white', linewidth=3, alpha=0.5)

        # Baselines
        for y_pos in [-HALF_L, HALF_L]:
            ax.axvline(x=y_pos, color='white', linewidth=1, alpha=0.2, linestyle=':')

        tw = r['true_world']
        ew = r['est_world']
        n_common = min(len(tw), len(ew))

        # Trajectories
        ax.plot(tw[:, 1], tw[:, 2], color='#00E5FF', linewidth=2.5,
                alpha=0.9, label='True', zorder=3)
        ax.plot(ew[:n_common, 1], ew[:n_common, 2], color='#FF6D00',
                linewidth=2, linestyle='--', alpha=0.9, label='Estimated', zorder=3)

        # Start/end markers
        ax.scatter(tw[0, 1], tw[0, 2], color='#00E5FF', s=50, marker='o',
                   zorder=5, edgecolors='white', linewidths=0.5)
        ax.scatter(tw[-1, 1], tw[-1, 2], color='#00E5FF', s=50, marker='s',
                   zorder=5, edgecolors='white', linewidths=0.5)

        ax.set_xlim(-HALF_L - 0.5, HALF_L + 0.5)
        ax.set_ylim(-0.3, 5.0)
        ax.set_xlabel('Y (m)', fontsize=8, color='white')
        ax.set_ylabel('Z (m)', fontsize=8, color='white')
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#444')

        title = f'{r["description"]}\n3D err = {r["mean_err"]:.2f}m'
        ax.set_title(title, fontsize=9, fontweight='bold', color='white')
        ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
        ax.grid(True, alpha=0.1, color='white')

    # Hide unused axes
    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'court_side_views.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {path}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Running all MHE scenarios...')
    results = run_all_scenarios()
    print(f'Got {len(results)} results\n')

    for r in results:
        print(f'  {r["name"]:<25} 3D err = {r["mean_err"]:.3f}m  (max {r["max_err"]:.3f}m)')
    print()

    print('Generating court visualizations...')
    plot_overview(results, OUTPUT_DIR)
    plot_individual_grid(results, OUTPUT_DIR)
    plot_side_view_grid(results, OUTPUT_DIR)
    print('Done!')


if __name__ == '__main__':
    main()
