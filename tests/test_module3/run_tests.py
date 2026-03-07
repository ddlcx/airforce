"""
Comprehensive test runner for Module 3.

Generates synthetic trajectories, runs MHE estimation, and produces
visualization plots for review.

Usage:
    cd airforce
    python -m tests.test_module3.run_tests
"""

import os
import sys
import logging
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config.physics_config import PhysicsConfig, MHEConfig
from module3.trajectory_estimator import TrajectoryEstimator
from module3.shuttlecock_model import (
    integrate_trajectory,
    flight_plane_to_world,
    ode_rhs_numpy,
)
from module3.measurement_model import project_world_to_pixel, augmented_state_to_pixel
from tests.test_module3.synthetic_data import (
    make_camera,
    generate_true_trajectory,
    generate_observations,
    make_rally_segment,
    get_test_scenarios,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'output', 'module3_tests'
)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


# ── Unit tests ──


def test_ode_no_drag():
    """Verify ODE without drag matches analytical parabola."""
    logger.info('TEST: ODE without drag (parabolic trajectory)')
    dt = 1.0 / 30
    vs0, vz0, z0 = 20.0, 10.0, 1.0
    cd = 0.0  # no drag
    n_steps = 30

    x0 = np.array([0.0, z0, vs0, vz0])
    traj = integrate_trajectory(x0, cd, dt, n_steps, num_substeps=8)

    # Analytical solution: s(t) = vs0*t, z(t) = z0 + vz0*t - 0.5*g*t^2
    g = 9.81
    t = np.arange(n_steps + 1) * dt
    s_exact = vs0 * t
    z_exact = z0 + vz0 * t - 0.5 * g * t**2

    err_s = np.max(np.abs(traj[:, 0] - s_exact))
    err_z = np.max(np.abs(traj[:, 1] - z_exact))

    passed = err_s < 1e-4 and err_z < 1e-4
    logger.info(f'  Max error: s={err_s:.2e}, z={err_z:.2e} -> {"PASS" if passed else "FAIL"}')
    return passed


def test_ode_terminal_velocity():
    """Verify terminal velocity convergence with drag."""
    logger.info('TEST: Terminal velocity convergence')
    physics = PhysicsConfig()
    cd = physics.cd_nominal
    U_inf = physics.terminal_velocity

    # Drop from rest at height
    x0 = np.array([0.0, 100.0, 0.001, 0.0])  # tiny vs to avoid division issue
    traj = integrate_trajectory(x0, cd, 0.01, 2000, num_substeps=4)

    # Terminal speed should approach U_inf
    final_speed = np.sqrt(traj[-1, 2]**2 + traj[-1, 3]**2)
    err = abs(final_speed - U_inf) / U_inf

    passed = err < 0.05  # within 5%
    logger.info(
        f'  U_inf={U_inf:.2f}m/s, final_speed={final_speed:.2f}m/s, '
        f'err={err*100:.1f}% -> {"PASS" if passed else "FAIL"}'
    )
    return passed


def test_measurement_model():
    """Verify projection matches manual computation."""
    logger.info('TEST: Measurement model (projection)')
    P, K = make_camera()

    # Known 3D point
    world_pt = np.array([1.0, 3.0, 2.0])
    pixel = project_world_to_pixel(world_pt, P)

    # Manual: P @ [x,y,z,1]^T, then divide
    homog = P @ np.array([1.0, 3.0, 2.0, 1.0])
    pixel_manual = homog[:2] / homog[2]

    err = np.linalg.norm(pixel - pixel_manual)
    passed = err < 1e-10
    logger.info(f'  Projection error: {err:.2e} -> {"PASS" if passed else "FAIL"}')

    # Also test augmented_state_to_pixel
    psi, x0w, y0w = np.pi / 2, 0.0, 0.0
    # s = 3.0, z = 2.0 with psi=pi/2 -> X=0+3*cos(pi/2)=0, Y=0+3*sin(pi/2)=3, Z=2
    x_aug = np.array([3.0, 2.0, 0.0, 0.0, psi, 0.2, x0w, y0w])
    pixel2 = augmented_state_to_pixel(x_aug, P)
    expected_world = np.array([0.0, 3.0, 2.0])
    pixel_expected = project_world_to_pixel(expected_world, P)
    err2 = np.linalg.norm(pixel2 - pixel_expected)
    passed2 = err2 < 1e-10
    logger.info(f'  Augmented state projection error: {err2:.2e} -> {"PASS" if passed2 else "FAIL"}')

    return passed and passed2


# ── MHE integration test ──


def test_mhe_scenario(scenario: dict, estimator: TrajectoryEstimator,
                      output_dir: str) -> dict:
    """
    Run MHE on one scenario and generate plots.

    Returns:
        dict with test metrics
    """
    name = scenario['name']
    desc = scenario['description']
    true_data = scenario['true_data']
    obs_data = scenario['obs_data']
    segment = scenario['segment']
    P = scenario['P']

    logger.info(f'\nMHE TEST: {name} ({desc})')

    try:
        result = estimator.estimate_segment(segment, segment_index=0)
    except Exception as e:
        logger.error(f'  FAILED: {e}')
        return {'name': name, 'passed': False, 'error': str(e)}

    # ── Metrics ──
    true_world = true_data['world_pos']
    est_world = result.world_positions

    # Align lengths (might differ slightly)
    n_common = min(len(true_world), len(est_world))
    pos_errors = np.linalg.norm(
        true_world[:n_common] - est_world[:n_common], axis=1
    )
    mean_pos_err = float(np.mean(pos_errors))
    max_pos_err = float(np.max(pos_errors))

    cd_err = abs(result.cd - true_data['cd'])
    psi_err = abs(result.psi - true_data['psi'])
    psi_err_deg = np.degrees(psi_err)

    metrics = {
        'name': name,
        'description': desc,
        'passed': result.solve_status == 0,
        'solve_status': result.solve_status,
        'mean_reproj_error_px': result.mean_reproj_error,
        'mean_3d_error_m': mean_pos_err,
        'max_3d_error_m': max_pos_err,
        'cd_true': true_data['cd'],
        'cd_est': result.cd,
        'cd_error': cd_err,
        'psi_true_deg': np.degrees(true_data['psi']),
        'psi_est_deg': np.degrees(result.psi),
        'psi_error_deg': psi_err_deg,
        'solve_time_ms': result.solve_time_ms,
        'sqp_iterations': getattr(result, 'sqp_iterations', -1),
        'z0_true': true_data['x0_phys'][1],
        'z0_est': result.z0,
        'v0_true': np.sqrt(true_data['x0_phys'][2]**2 + true_data['x0_phys'][3]**2),
        'v0_est': result.initial_speed,
    }

    # ── Generate plots ──
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{name}: {desc}', fontsize=14, fontweight='bold')

    # Plot 1: 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(true_world[:, 0], true_world[:, 1], true_world[:, 2],
             'b-', linewidth=2, label='True')
    ax1.plot(est_world[:n_common, 0], est_world[:n_common, 1],
             est_world[:n_common, 2], 'r--', linewidth=2, label='Estimated')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # Plot 2: Side view (Y vs Z)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(true_world[:, 1], true_world[:, 2], 'b-', linewidth=2, label='True')
    ax2.plot(est_world[:n_common, 1], est_world[:n_common, 2],
             'r--', linewidth=2, label='Estimated')
    ax2.set_xlabel('Y (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Side View (Y-Z)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Plot 3: Pixel space comparison
    ax3 = fig.add_subplot(2, 3, 3)
    pixel_true = obs_data['pixel_true']
    pixel_noisy = obs_data['pixel_noisy']
    vis = obs_data['visibility']

    ax3.plot(pixel_true[:, 0], pixel_true[:, 1], 'b-', linewidth=1.5,
             label='True projection', alpha=0.7)
    vis_idx = np.where(vis)[0]
    ax3.scatter(pixel_noisy[vis_idx, 0], pixel_noisy[vis_idx, 1],
                c='green', s=15, alpha=0.6, label='Noisy detections')
    ax3.plot(result.pixel_projected[:, 0], result.pixel_projected[:, 1],
             'r--', linewidth=1.5, label='MHE reprojection')
    if (~vis).any():
        miss_idx = np.where(~vis)[0]
        ax3.scatter(pixel_true[miss_idx, 0], pixel_true[miss_idx, 1],
                    c='red', marker='x', s=30, label='Missing frames')
    ax3.set_xlabel('u (px)')
    ax3.set_ylabel('v (px)')
    ax3.set_title('Pixel Space')
    ax3.legend(fontsize=8)
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)

    # Plot 4: 3D position error over time
    ax4 = fig.add_subplot(2, 3, 4)
    t = true_data['time_stamps'][:n_common]
    ax4.plot(t, pos_errors, 'k-', linewidth=1.5)
    ax4.axhline(y=mean_pos_err, color='r', linestyle='--',
                label=f'Mean={mean_pos_err:.3f}m')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('3D Error (m)')
    ax4.set_title('Position Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Reprojection error
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, result.reprojection_errors[:n_common], 'k-', linewidth=1.5)
    ax5.axhline(y=result.mean_reproj_error, color='r', linestyle='--',
                label=f'Mean={result.mean_reproj_error:.2f}px')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Reproj Error (px)')
    ax5.set_title('Reprojection Error')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Process noise magnitude
    ax6 = fig.add_subplot(2, 3, 6)
    noise_mag = np.linalg.norm(result.process_noise, axis=1)
    t_noise = true_data['time_stamps'][:len(noise_mag)]
    ax6.plot(t_noise, noise_mag, 'k-', linewidth=1.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('|w| (noise magnitude)')
    ax6.set_title('Process Noise')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'{name}.png')
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f'  Saved plot: {fig_path}')

    return metrics


def generate_summary_report(all_metrics: list, output_dir: str):
    """Generate summary report."""
    report_path = os.path.join(output_dir, 'summary_report.txt')

    with open(report_path, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('Module 3 MHE Test Results Summary\n')
        f.write('=' * 80 + '\n\n')

        # Summary table
        f.write(f'{"Scenario":<25} {"Status":<8} {"Reproj(px)":<12} '
                f'{"3D Err(m)":<12} {"cd_err":<10} {"psi_err(°)":<12} '
                f'{"Time(ms)":<10}\n')
        f.write('-' * 89 + '\n')

        for m in all_metrics:
            if 'error' in m:
                f.write(f'{m["name"]:<25} {"FAIL":<8} {m["error"]}\n')
                continue

            status = 'PASS' if m['passed'] else 'FAIL'
            f.write(
                f'{m["name"]:<25} {status:<8} '
                f'{m["mean_reproj_error_px"]:<12.2f} '
                f'{m["mean_3d_error_m"]:<12.4f} '
                f'{m["cd_error"]:<10.4f} '
                f'{m["psi_error_deg"]:<12.2f} '
                f'{m["solve_time_ms"]:<10.1f}\n'
            )

        f.write('\n')

        # Detailed results
        for m in all_metrics:
            if 'error' in m:
                continue
            f.write(f'\n--- {m["name"]}: {m["description"]} ---\n')
            f.write(f'  Solve status:     {m["solve_status"]}\n')
            f.write(f'  Reproj error:     {m["mean_reproj_error_px"]:.2f} px\n')
            f.write(f'  3D error (mean):  {m["mean_3d_error_m"]:.4f} m\n')
            f.write(f'  3D error (max):   {m["max_3d_error_m"]:.4f} m\n')
            f.write(f'  cd: true={m["cd_true"]:.4f}, est={m["cd_est"]:.4f}, err={m["cd_error"]:.4f}\n')
            f.write(f'  psi: true={m["psi_true_deg"]:.1f}°, est={m["psi_est_deg"]:.1f}°, err={m["psi_error_deg"]:.2f}°\n')
            f.write(f'  z0: true={m["z0_true"]:.2f}, est={m["z0_est"]:.2f}\n')
            f.write(f'  v0: true={m["v0_true"]:.1f}, est={m["v0_est"]:.1f} m/s\n')
            f.write(f'  Solve time:       {m["solve_time_ms"]:.1f} ms\n')

    logger.info(f'\nSummary report saved: {report_path}')


def main():
    output_dir = ensure_output_dir()
    logger.info(f'Output directory: {output_dir}')

    # ── Unit tests ──
    logger.info('\n' + '=' * 60)
    logger.info('Running unit tests...')
    logger.info('=' * 60)

    unit_results = [
        ('ODE no drag', test_ode_no_drag()),
        ('Terminal velocity', test_ode_terminal_velocity()),
        ('Measurement model', test_measurement_model()),
    ]

    all_passed = all(r[1] for r in unit_results)
    logger.info(f'\nUnit tests: {sum(r[1] for r in unit_results)}/{len(unit_results)} passed')

    if not all_passed:
        logger.error('Unit tests failed, skipping MHE tests')
        return

    # ── MHE integration tests ──
    logger.info('\n' + '=' * 60)
    logger.info('Running MHE integration tests...')
    logger.info('=' * 60)

    estimator = TrajectoryEstimator()
    scenarios = get_test_scenarios(fps=30.0)

    all_metrics = []
    for scenario in scenarios:
        metrics = test_mhe_scenario(scenario, estimator, output_dir)
        all_metrics.append(metrics)

    # ── Summary ──
    generate_summary_report(all_metrics, output_dir)

    n_passed = sum(1 for m in all_metrics if m.get('passed', False))
    logger.info(f'\nMHE tests: {n_passed}/{len(all_metrics)} passed')
    logger.info(f'Results in: {output_dir}')


if __name__ == '__main__':
    main()
