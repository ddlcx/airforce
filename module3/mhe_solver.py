"""
acados MHE solver for shuttlecock trajectory estimation.

Builds and solves a multiple-shooting NLP using acados OCP solver,
following the MHE convention from acados examples
(pendulum_on_cart/mhe/export_mhe_solver.py).

Key convention:
- model.u = process noise w (4-dim)
- model.p = P_flat projection matrix (12-dim)
- cost_y_expr = vertcat(h_meas, u)      at intermediate nodes
- cost_y_expr_0 = vertcat(h_meas, u, x) at node 0 (with arrival cost)
"""

import os
import time
import shutil
from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag

# Set acados environment before importing
_ACADOS_ROOT = os.path.join(
    os.path.dirname(__file__), '..', 'external', 'acados'
)
_ACADOS_ROOT = os.path.abspath(_ACADOS_ROOT)
os.environ.setdefault('ACADOS_SOURCE_DIR', _ACADOS_ROOT)

# Append acados lib to DYLD_LIBRARY_PATH
_lib_path = os.path.join(_ACADOS_ROOT, 'lib')
_existing = os.environ.get('DYLD_LIBRARY_PATH', '')
if _lib_path not in _existing:
    os.environ['DYLD_LIBRARY_PATH'] = _lib_path + ':' + _existing

from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import vertcat

from module3.shuttlecock_model import export_shuttlecock_mhe_model
from module3.measurement_model import create_measurement_expr

NX = 8  # augmented state dimension
NW = 4  # process noise dimension
NY_MEAS = 2  # measurement dimension (u_px, v_px)


@dataclass
class MHESolveResult:
    """Raw solver output."""

    states: np.ndarray       # (N+1, 8) estimated augmented states
    noises: np.ndarray       # (N, 4) estimated process noises
    cost: float
    status: int              # 0 = success
    solve_time_ms: float
    sqp_iterations: int


def build_mhe_solver(
    N: int,
    dt: float,
    R: np.ndarray,
    Q: np.ndarray,
    Q0: np.ndarray,
    cd_bounds: tuple = (0.05, 0.50),
    z_bounds: tuple = (0.0, 15.0),
    psi_bounds: tuple = (-np.pi, np.pi),
    x0w_bounds: tuple = (-10.0, 10.0),
    y0w_bounds: tuple = (-15.0, 15.0),
    integrator_type: str = 'ERK',
    sim_method_num_stages: int = 4,
    sim_method_num_steps: int = 4,
    nlp_solver_max_iter: int = 200,
    code_export_dir: str = None,
) -> AcadosOcpSolver:
    """
    Build acados MHE solver.

    Following the plan document S4 and acados MHE convention.

    Args:
        N: number of shooting intervals
        dt: time step (1/fps)
        R: (2, 2) measurement weight matrix
        Q: (4, 4) process noise penalty matrix
        Q0: (8, 8) arrival cost matrix
        Other args: bounds, solver options

    Returns:
        AcadosOcpSolver ready to be configured and solved
    """
    model = export_shuttlecock_mhe_model()
    h_meas = create_measurement_expr(model)

    ocp = AcadosOcp()
    ocp.model = model

    nx = NX
    nw = NW
    ny_meas = NY_MEAS

    ny_0 = ny_meas + nw + nx   # h(x), w, x  -> 2 + 4 + 8 = 14
    ny = ny_meas + nw          # h(x), w     -> 2 + 4 = 6

    ocp.solver_options.N_horizon = N

    # ── Node 0: measurement + noise + arrival cost ──
    ocp.cost.cost_type_0 = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_0 = vertcat(h_meas, model.u, model.x)
    ocp.cost.W_0 = block_diag(R, Q, Q0)
    ocp.cost.yref_0 = np.zeros(ny_0)

    # ── Intermediate nodes: measurement + noise ──
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = vertcat(h_meas, model.u)
    ocp.cost.W = block_diag(R, Q)
    ocp.cost.yref = np.zeros(ny)

    # ── Terminal node: measurement only (no noise at terminal) ──
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = h_meas
    ocp.cost.W_e = R.copy()
    ocp.cost.yref_e = np.zeros(ny_meas)

    # ── Parameters ──
    ocp.parameter_values = np.zeros(12)  # P_flat, set at runtime

    # ── State bounds ──
    # Index: [s, z, vs, vz, psi, cd, x0w, y0w]
    ocp.constraints.lbx = np.array([
        -50.0,             # s
        z_bounds[0],       # z >= 0
        -150.0,            # vs
        -150.0,            # vz
        psi_bounds[0],     # psi
        cd_bounds[0],      # cd
        x0w_bounds[0],     # x0w
        y0w_bounds[0],     # y0w
    ])
    ocp.constraints.ubx = np.array([
        50.0,
        z_bounds[1],
        150.0,
        150.0,
        psi_bounds[1],
        cd_bounds[1],
        x0w_bounds[1],
        y0w_bounds[1],
    ])
    ocp.constraints.idxbx = np.arange(nx)

    # Also apply bounds at initial node
    ocp.constraints.lbx_0 = ocp.constraints.lbx.copy()
    ocp.constraints.ubx_0 = ocp.constraints.ubx.copy()
    ocp.constraints.idxbx_0 = np.arange(nx)

    # And terminal node
    ocp.constraints.lbx_e = ocp.constraints.lbx.copy()
    ocp.constraints.ubx_e = ocp.constraints.ubx.copy()
    ocp.constraints.idxbx_e = np.arange(nx)

    # ── Solver options ──
    ocp.solver_options.tf = N * dt
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = nlp_solver_max_iter
    ocp.solver_options.levenberg_marquardt = 1e-2
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.alpha_min = 1e-4
    ocp.solver_options.nlp_solver_tol_stat = 1e-4
    ocp.solver_options.nlp_solver_tol_eq = 1e-4
    ocp.solver_options.nlp_solver_tol_ineq = 1e-4
    ocp.solver_options.nlp_solver_tol_comp = 1e-4
    ocp.solver_options.cost_scaling = np.ones(N + 1)  # no dt scaling

    # ── Integrator ──
    ocp.solver_options.integrator_type = integrator_type
    ocp.solver_options.sim_method_num_stages = sim_method_num_stages
    ocp.solver_options.sim_method_num_steps = sim_method_num_steps

    # ── Code export directory (unique per N to avoid conflicts) ──
    if code_export_dir is None:
        code_export_dir = os.path.join(
            os.path.dirname(__file__), '..', 'c_generated_code', f'mhe_N{N}'
        )
    code_export_dir = os.path.abspath(code_export_dir)
    ocp.code_export_directory = code_export_dir

    json_file = os.path.join(code_export_dir, 'acados_ocp_mhe.json')
    os.makedirs(code_export_dir, exist_ok=True)

    solver = AcadosOcpSolver(ocp, json_file=json_file)
    return solver


def solve_mhe(
    solver: AcadosOcpSolver,
    N: int,
    P_flat: np.ndarray,
    observations: np.ndarray,
    visibility: np.ndarray,
    x0_bar: np.ndarray,
    x_init: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    Q0: np.ndarray,
    R_zero: np.ndarray = None,
) -> MHESolveResult:
    """
    Configure and solve the MHE problem for one flight segment.

    Args:
        solver: pre-built AcadosOcpSolver
        N: number of shooting intervals
        P_flat: (12,) flattened projection matrix
        observations: (N+1, 2) pixel observations [u, v]
        visibility: (N+1,) boolean visibility flags
        x0_bar: (8,) prior state for arrival cost
        x_init: (N+1, 8) initial state guess
        R: (2, 2) measurement weight
        Q: (4, 4) noise penalty
        Q0: (8, 8) arrival cost weight
        R_zero: (2, 2) near-zero weight for missing detections

    Returns:
        MHESolveResult
    """
    if R_zero is None:
        R_zero = np.diag([1e-10, 1e-10])

    ny_0 = NY_MEAS + NW + NX  # 14
    ny = NY_MEAS + NW          # 6

    W_normal = block_diag(R, Q)
    W_missing = block_diag(R_zero, Q)

    # ── Set parameters, references, and initial guesses ──
    for k in range(N + 1):
        solver.set(k, 'p', P_flat)
        solver.set(k, 'x', x_init[k])

        if k == 0:
            # Node 0: yref = [obs, zeros_w, x0_bar]
            yref_0 = np.zeros(ny_0)
            yref_0[:NY_MEAS] = observations[0]
            yref_0[NY_MEAS + NW:] = x0_bar
            solver.set(0, 'yref', yref_0)

            if visibility[0]:
                solver.cost_set(0, 'W', block_diag(R, Q, Q0))
            else:
                solver.cost_set(0, 'W', block_diag(R_zero, Q, Q0))

        elif k < N:
            yref_k = np.zeros(ny)
            yref_k[:NY_MEAS] = observations[k]
            solver.set(k, 'yref', yref_k)

            if visibility[k]:
                solver.cost_set(k, 'W', W_normal)
            else:
                solver.cost_set(k, 'W', W_missing)

        else:
            # Terminal node (k == N): measurement only
            yref_e = observations[N].copy()
            solver.set(N, 'yref', yref_e)

            if visibility[N]:
                solver.cost_set(N, 'W', R.copy())
            else:
                solver.cost_set(N, 'W', R_zero.copy())

    # Set initial noise guess to zero
    for k in range(N):
        solver.set(k, 'u', np.zeros(NW))

    # ── Solve ──
    t_start = time.time()
    status = solver.solve()
    solve_time = (time.time() - t_start) * 1000

    # ── Extract results ──
    states = np.zeros((N + 1, NX))
    noises = np.zeros((N, NW))

    for k in range(N + 1):
        states[k] = solver.get(k, 'x')
    for k in range(N):
        noises[k] = solver.get(k, 'u')

    cost = float(solver.get_cost())
    sqp_raw = solver.get_stats('sqp_iter')
    sqp_iter = int(sqp_raw[0]) if hasattr(sqp_raw, '__len__') else int(sqp_raw)

    return MHESolveResult(
        states=states,
        noises=noises,
        cost=cost,
        status=status,
        solve_time_ms=solve_time,
        sqp_iterations=sqp_iter,
    )
