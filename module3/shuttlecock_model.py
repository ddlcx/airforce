"""
Shuttlecock aerodynamic ODE model.

Provides:
- Numpy functions for simulation and initialization
- AcadosModel export for acados MHE solver

State: x_aug = [s, z, v_s, v_z, psi, c_d, x0w, y0w]  (8-dim)
Noise: w = [w_s, w_z, w_vs, w_vz]                      (4-dim, as model.u)
Params: p = P_flat                                       (12-dim, projection matrix)

References:
    [1] Cohen et al., 2015: ds/dt = vs, dz/dt = vz,
        dv_s/dt = -c_d |v| v_s, dv_z/dt = -g - c_d |v| v_z
"""

import numpy as np
from casadi import SX, vertcat, sqrt, cos, sin

G = 9.81
_SPEED_EPS = 1e-10  # numerical guard for sqrt


# ── Numpy implementation (for simulation and initialization) ──


def ode_rhs_numpy(x_phys: np.ndarray, cd: float,
                  w: np.ndarray = None) -> np.ndarray:
    """ODE right-hand side for physical state [s, z, vs, vz]."""
    if w is None:
        w = np.zeros(4)
    s, z, vs, vz = x_phys
    speed = np.sqrt(vs**2 + vz**2 + _SPEED_EPS)
    return np.array([
        vs + w[0],
        vz + w[1],
        -cd * speed * vs + w[2],
        -G - cd * speed * vz + w[3],
    ])


def rk4_step(x_phys: np.ndarray, cd: float, dt: float,
             w: np.ndarray = None) -> np.ndarray:
    """Single RK4 step for physical state."""
    if w is None:
        w = np.zeros(4)
    k1 = ode_rhs_numpy(x_phys, cd, w)
    k2 = ode_rhs_numpy(x_phys + dt / 2 * k1, cd, w)
    k3 = ode_rhs_numpy(x_phys + dt / 2 * k2, cd, w)
    k4 = ode_rhs_numpy(x_phys + dt * k3, cd, w)
    return x_phys + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_trajectory(x0_phys: np.ndarray, cd: float, dt: float,
                         n_steps: int, num_substeps: int = 4) -> np.ndarray:
    """
    Integrate trajectory using RK4 with substeps.

    Returns:
        (n_steps+1, 4) trajectory in flight-plane coordinates [s, z, vs, vz]
    """
    dt_sub = dt / num_substeps
    traj = np.zeros((n_steps + 1, 4))
    traj[0] = x0_phys
    x = x0_phys.copy()
    for k in range(n_steps):
        for _ in range(num_substeps):
            x = rk4_step(x, cd, dt_sub)
        traj[k + 1] = x
    return traj


def flight_plane_to_world(s, z, psi: float, x0w: float,
                          y0w: float) -> np.ndarray:
    """Convert flight-plane (s, z) to world (X, Y, Z)."""
    s = np.asarray(s)
    z = np.asarray(z)
    Xw = x0w + s * np.cos(psi)
    Yw = y0w + s * np.sin(psi)
    Zw = z
    if s.ndim == 0:
        return np.array([float(Xw), float(Yw), float(Zw)])
    return np.column_stack([Xw, Yw, Zw])


def world_velocities_from_flight_plane(vs, vz, psi: float) -> np.ndarray:
    """Convert flight-plane velocities to world velocities."""
    vs = np.asarray(vs)
    vz = np.asarray(vz)
    vx_w = vs * np.cos(psi)
    vy_w = vs * np.sin(psi)
    vz_w = vz
    if vs.ndim == 0:
        return np.array([float(vx_w), float(vy_w), float(vz_w)])
    return np.column_stack([vx_w, vy_w, vz_w])


# ── AcadosModel export (for acados MHE solver) ──


def export_shuttlecock_mhe_model():
    """
    Export CasADi model for acados MHE.

    Following acados MHE convention (pendulum_on_cart/mhe example):
    - model.x = augmented state (8-dim)
    - model.u = process noise w (4-dim)
    - model.p = P_flat projection matrix (12-dim)

    Returns:
        AcadosModel
    """
    from acados_template import AcadosModel

    model = AcadosModel()

    # Augmented state (8-dim)
    s = SX.sym('s')
    z = SX.sym('z')
    vs = SX.sym('vs')
    vz = SX.sym('vz')
    psi = SX.sym('psi')
    cd = SX.sym('cd')
    x0w = SX.sym('x0w')
    y0w = SX.sym('y0w')
    x = vertcat(s, z, vs, vz, psi, cd, x0w, y0w)

    # Process noise = MHE "control input" (4-dim)
    w_s = SX.sym('w_s')
    w_z = SX.sym('w_z')
    w_vs = SX.sym('w_vs')
    w_vz = SX.sym('w_vz')
    w = vertcat(w_s, w_z, w_vs, w_vz)

    # xdot symbols (for implicit form)
    s_dot = SX.sym('s_dot')
    z_dot = SX.sym('z_dot')
    vs_dot = SX.sym('vs_dot')
    vz_dot = SX.sym('vz_dot')
    psi_dot = SX.sym('psi_dot')
    cd_dot = SX.sym('cd_dot')
    x0w_dot = SX.sym('x0w_dot')
    y0w_dot = SX.sym('y0w_dot')
    xdot = vertcat(s_dot, z_dot, vs_dot, vz_dot,
                   psi_dot, cd_dot, x0w_dot, y0w_dot)

    # Known parameters: projection matrix flattened (12-dim)
    p_proj = SX.sym('p_proj', 12)

    # Dynamics
    speed = sqrt(vs**2 + vz**2 + _SPEED_EPS)
    f_expl = vertcat(
        vs + w_s,                        # ds/dt
        vz + w_z,                        # dz/dt
        -cd * speed * vs + w_vs,         # dv_s/dt
        -G - cd * speed * vz + w_vz,     # dv_z/dt
        0, 0, 0, 0,                      # parameter zero dynamics
    )
    f_impl = xdot - f_expl

    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.x = x
    model.xdot = xdot
    model.u = w          # MHE convention: process noise in u slot
    model.p = p_proj     # projection matrix as runtime parameter
    model.name = 'shuttlecock_mhe'

    return model
