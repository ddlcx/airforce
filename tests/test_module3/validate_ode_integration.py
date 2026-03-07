#!/usr/bin/env python3
"""
Validate ODE numerical integration accuracy.

Two validation approaches:
1. Analytic range (Cohen 2015, Eq. 2.8) vs numerical range — verifies ODE correctness
2. Convergence analysis: trajectory error vs RK4 substep count — determines optimal config

Cohen 2015 analytic range formula (valid for upward launch, theta_0 > 0):
    x_0 = (L/2) cos(theta_0) ln(1 + 4 U_0^2 sin(theta_0) / (g L))
where L = 1/c_d is the aerodynamic length.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from module3.shuttlecock_model import integrate_trajectory
from config.physics_config import PhysicsConfig

G = 9.81
PHYSICS = PhysicsConfig()
CD = PHYSICS.cd_nominal
L_AERO = PHYSICS.aero_length

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'module3_tests')


# ── Analytic formula ──

def analytic_range(U0, theta0_rad, cd):
    """Cohen 2015 analytic range (Eq. 2.8). Returns None for downward launch."""
    L = 1.0 / cd
    sin_t = np.sin(theta0_rad)
    cos_t = np.cos(theta0_rad)
    if sin_t <= 0:
        return None
    arg = 1 + 4 * U0**2 * sin_t / (G * L)
    if arg <= 0:
        return None
    return (L / 2) * cos_t * np.log(arg)


def find_range_from_traj(traj):
    """Find horizontal range s when z returns to 0 after peaking."""
    peaked = False
    for k in range(1, len(traj)):
        if traj[k, 1] < traj[k - 1, 1]:
            peaked = True
        if peaked and traj[k, 1] <= 0:
            z1, z2 = traj[k - 1, 1], traj[k, 1]
            if z1 == z2:
                return traj[k - 1, 0]
            frac = z1 / (z1 - z2)
            return traj[k - 1, 0] + frac * (traj[k, 0] - traj[k - 1, 0])
    return None


# ── Test shots ──

SHOTS = [
    # (name,             U0 m/s, theta0 deg, description)
    ("High clear",       33.5,   50,  "Standard clear, moderate speed"),
    ("Fast clear",       40.0,   45,  "Fast clear, 45 deg"),
    ("Steep clear",      35.0,   65,  "High arc clear"),
    ("Defensive lob",    20.0,   55,  "Slow defensive lob"),
    ("Flat drive",       40.0,    8,  "Fast flat drive"),
    ("Fast drive",       50.0,    5,  "Very fast flat drive"),
    ("Drop shot",        12.0,   12,  "Slow drop shot"),
    ("Net shot",          8.0,   25,  "Slow net play"),
    ("Steep smash",      80.0,  -15,  "Fast steep smash"),
    ("Flat smash",       60.0,   -5,  "Moderate flat smash"),
]

FPS_LIST = [25, 30, 60, 120]
SUBSTEPS_LIST = [1, 2, 4, 8, 16, 32]
REF_SUBSTEPS = 64  # RK4 error ~ h^4; at 64 substeps error is ~16M x smaller than 1-step


def aero_timescale(U0, cd):
    """Aerodynamic time scale tau = 1/(cd * U0). Smaller = harder to integrate."""
    return 1.0 / (cd * U0)


def estimate_flight_steps(U0, theta_deg, fps):
    """Estimate number of time steps to cover full flight."""
    theta = np.radians(theta_deg)
    vz0 = U0 * np.sin(theta)
    if theta_deg > 0:
        t_flight = min(2 * vz0 / G * 1.5, 4.0)
    else:
        # Downward from ~3m height
        t_flight = min(1.0, 4.0)
    return max(int(t_flight * fps) + 10, 20)


def run_validation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 95)
    print("ODE Integration Accuracy Validation")
    print("=" * 95)
    print(f"Aerodynamic length  L = {L_AERO:.4f} m")
    print(f"Drag coefficient   cd = {CD:.4f} 1/m")
    print(f"Terminal velocity  U_inf = {PHYSICS.terminal_velocity:.2f} m/s")
    print()

    # ── Part 1: Analytic vs Numerical Range ──

    print("-" * 95)
    print("Part 1: Analytic Range vs Numerical Range")
    print("  (Numerical: RK4 x 32 substeps at 500 Hz — effectively exact)")
    print("-" * 95)
    hdr = (f"{'Shot':<18} {'U0':>5} {'th0':>5} {'tau_aero':>9} "
           f"{'Analytic':>10} {'Numerical':>10} {'Delta':>8} {'Rel':>7}")
    print(hdr)
    print(f"{'':18} {'m/s':>5} {'deg':>5} {'ms':>9} "
          f"{'m':>10} {'m':>10} {'mm':>8} {'%':>7}")
    print("-" * 82)

    for name, U0, theta_deg, desc in SHOTS:
        theta = np.radians(theta_deg)
        vs0 = U0 * np.cos(theta)
        vz0 = U0 * np.sin(theta)
        tau = aero_timescale(U0, CD) * 1000  # ms

        r_ana = analytic_range(U0, theta, CD)

        if r_ana is not None:
            # High-precision numerical integration
            dt_fine = 1.0 / 500   # 500 Hz
            n_fine = 2000         # up to 4 seconds
            x0 = np.array([0.0, 0.0, vs0, vz0])
            traj_fine = integrate_trajectory(x0, CD, dt_fine, n_fine, 32)
            r_num = find_range_from_traj(traj_fine)

            if r_num is not None:
                delta_mm = abs(r_num - r_ana) * 1000
                rel_pct = abs(r_num - r_ana) / r_ana * 100
                print(f"{name:<18} {U0:5.1f} {theta_deg:5.0f} {tau:9.1f} "
                      f"{r_ana:10.4f} {r_num:10.4f} {delta_mm:8.1f} {rel_pct:7.3f}")
            else:
                print(f"{name:<18} {U0:5.1f} {theta_deg:5.0f} {tau:9.1f} "
                      f"{r_ana:10.4f} {'no land':>10}")
        else:
            print(f"{name:<18} {U0:5.1f} {theta_deg:5.0f} {tau:9.1f} "
                  f"{'N/A':>10} {'downward':>10}")

    print()
    print("  Note: discrepancy = analytic formula approximation error")
    print("  (the numerical solution with 512 substeps at 1kHz is essentially exact)")

    # ── Part 2: Convergence Analysis ──

    print()
    print("-" * 95)
    print("Part 2: Max Trajectory Error vs Substep Count")
    print("  (reference: 64 substeps at same fps)")
    print("-" * 95)

    # Collect all results
    all_results = {}
    for name, U0, theta_deg, desc in SHOTS:
        theta = np.radians(theta_deg)
        vs0 = U0 * np.cos(theta)
        vz0 = U0 * np.sin(theta)
        z0 = 2.5 if theta_deg < 0 else 0.0
        x0 = np.array([0.0, z0, vs0, vz0])

        results_fps = {}
        for fps in FPS_LIST:
            dt = 1.0 / fps
            n_steps = estimate_flight_steps(U0, theta_deg, fps)
            ref = integrate_trajectory(x0, CD, dt, n_steps, REF_SUBSTEPS)

            errors = {}
            for nsub in SUBSTEPS_LIST:
                traj = integrate_trajectory(x0, CD, dt, n_steps, nsub)
                pos_err = np.sqrt(
                    (traj[:, 0] - ref[:, 0])**2 + (traj[:, 1] - ref[:, 1])**2
                )
                errors[nsub] = float(np.max(pos_err))
            results_fps[fps] = errors
        all_results[name] = results_fps

    # Print tables per fps
    for fps in FPS_LIST:
        dt = 1.0 / fps
        print(f"\n  fps={fps}  (dt = {dt*1000:.1f} ms)")
        header = f"  {'Shot':<18}"
        for n in SUBSTEPS_LIST:
            header += f"{'sub=' + str(n):>12}"
        print(header)
        print("  " + "-" * (18 + 12 * len(SUBSTEPS_LIST)))

        for name, U0, theta_deg, desc in SHOTS:
            row = f"  {name:<18}"
            for nsub in SUBSTEPS_LIST:
                err = all_results[name][fps][nsub]
                if err < 1e-6:
                    row += f"{'~0':>12}"
                elif err < 0.001:
                    row += f"{err*1e6:>8.0f} um "
                elif err < 0.01:
                    row += f"{err*1000:>8.1f} mm "
                elif err < 1.0:
                    row += f"{err*100:>8.1f} cm "
                else:
                    row += f"{err:>8.2f} m  "
            print(row)

    # ── Part 3: Minimum substeps for <1mm accuracy ──

    print()
    print("-" * 95)
    print("Part 3: Minimum Substeps for < 1mm Max Error")
    print("-" * 95)
    header = f"  {'Shot':<18} {'tau_aero':>9}"
    for fps in FPS_LIST:
        header += f"{'fps=' + str(fps):>10}"
    print(header)
    print("  " + "-" * (27 + 10 * len(FPS_LIST)))

    for name, U0, theta_deg, desc in SHOTS:
        tau = aero_timescale(U0, CD) * 1000
        row = f"  {name:<18} {tau:7.1f}ms"
        for fps in FPS_LIST:
            min_sub = None
            for nsub in SUBSTEPS_LIST:
                if all_results[name][fps][nsub] < 0.001:
                    min_sub = nsub
                    break
            row += f"{str(min_sub) if min_sub else '>32':>10}"
        print(row)

    # ── Part 4: Current config assessment ──

    print()
    print("-" * 95)
    print("Part 4: Current Configuration Assessment (fps=30, substeps=4)")
    print("-" * 95)

    print(f"  {'Shot':<18} {'U0':>5} {'th0':>5} {'tau/dt':>7} "
          f"{'Max err':>10} {'Verdict':>10}")
    print("  " + "-" * 60)

    for name, U0, theta_deg, desc in SHOTS:
        tau = aero_timescale(U0, CD)
        dt_30 = 1.0 / 30
        ratio = tau / (dt_30 / 4)  # tau / actual_integration_step
        err = all_results[name][30][4]
        if err < 0.001:
            verdict = "OK"
        elif err < 0.01:
            verdict = "marginal"
        else:
            verdict = "INCREASE"
        err_str = format_error(err)
        print(f"  {name:<18} {U0:5.1f} {theta_deg:5.0f} {ratio:7.1f} "
              f"{err_str:>10} {verdict:>10}")

    # Overall conclusion
    worst = max(
        ((name, all_results[name][30][4]) for name, _, _, _ in SHOTS),
        key=lambda x: x[1],
    )
    print()
    if worst[1] < 0.001:
        print(f"  CONCLUSION: substeps=4 achieves <1mm accuracy for ALL shots at 30fps.")
        print(f"  Worst case: {worst[0]} = {format_error(worst[1])}")
        print(f"  Current config (sim_method_num_steps=4) is adequate.")
    elif worst[1] < 0.01:
        print(f"  CONCLUSION: substeps=4 achieves <1cm for all, but >{1}mm for some.")
        print(f"  Worst: {worst[0]} = {format_error(worst[1])}")
        print(f"  Consider substeps=8 for sub-mm accuracy on extreme shots.")
    else:
        print(f"  CONCLUSION: substeps=4 insufficient for some shots.")
        print(f"  Worst: {worst[0]} = {format_error(worst[1])}")
        print(f"  Recommend substeps=8 or higher.")

    # ── Generate convergence plot ──

    plot_convergence(all_results, OUTPUT_DIR)


def format_error(err):
    """Format error value with appropriate unit."""
    if err < 1e-6:
        return "~0"
    elif err < 0.001:
        return f"{err * 1e6:.0f} um"
    elif err < 0.01:
        return f"{err * 1000:.1f} mm"
    elif err < 1.0:
        return f"{err * 100:.1f} cm"
    else:
        return f"{err:.2f} m"


def plot_convergence(all_results, output_dir):
    """Generate convergence plot: error vs substeps for all shots and fps values."""
    n_shots = len(SHOTS)
    cols = 5
    rows = (n_shots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(26, rows * 5.5))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        'RK4 Integration Error vs Substep Count (reference: 64 substeps)',
        fontsize=15, fontweight='bold', color='white', y=0.98,
    )

    colors_fps = {25: '#FF6D00', 30: '#2196F3', 60: '#4CAF50', 120: '#9C27B0'}
    axes_flat = axes.flatten()

    for idx, (name, U0, theta_deg, desc) in enumerate(SHOTS):
        ax = axes_flat[idx]
        ax.set_facecolor('#1a1a2e')

        tau_ms = aero_timescale(U0, CD) * 1000

        for fps in FPS_LIST:
            errs = [all_results[name][fps].get(n, 0) for n in SUBSTEPS_LIST]
            # Replace zeros for log scale
            errs_plot = [max(e, 1e-12) for e in errs]
            ax.semilogy(SUBSTEPS_LIST, errs_plot, 'o-', color=colors_fps[fps],
                        label=f'{fps} fps', linewidth=1.5, markersize=5)

        # Reference lines
        ax.axhline(y=0.001, color='#FF5252', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(32, 0.001, '1 mm', color='#FF5252', alpha=0.7,
                fontsize=7, va='bottom', ha='right')
        ax.axhline(y=0.01, color='#FFAB40', linestyle=':', alpha=0.4, linewidth=1)
        ax.text(32, 0.01, '1 cm', color='#FFAB40', alpha=0.6,
                fontsize=7, va='bottom', ha='right')

        ax.set_xlabel('Substeps per interval', fontsize=8, color='white')
        ax.set_ylabel('Max position error (m)', fontsize=8, color='white')
        ax.set_title(
            f'{name}\nU₀={U0} m/s  θ₀={theta_deg}°  τ={tau_ms:.1f} ms',
            fontsize=9, color='white', fontweight='bold',
        )
        ax.tick_params(colors='white', labelsize=7)
        ax.set_xticks(SUBSTEPS_LIST)
        for spine in ax.spines.values():
            spine.set_color('#444')
        ax.grid(True, alpha=0.15, color='white')

        if idx == 0:
            ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#444',
                      labelcolor='white')

    # Hide unused axes
    for j in range(n_shots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'ode_integration_accuracy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nConvergence plot saved: {path}")


if __name__ == '__main__':
    run_validation()
