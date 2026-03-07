"""
Measurement model: augmented state -> 2D pixel coordinates.

h(x_aug, P): flight-plane state -> world -> pixel via projection matrix P.

Provides both numpy (for evaluation) and CasADi symbolic (for acados cost).
"""

import numpy as np
from casadi import SX, vertcat, cos, sin


def project_world_to_pixel(world_pts: np.ndarray,
                           P: np.ndarray) -> np.ndarray:
    """
    Project 3D world points to 2D pixel coordinates.

    Args:
        world_pts: (N, 3) or (3,) world coordinates
        P: (3, 4) projection matrix

    Returns:
        (N, 2) or (2,) pixel coordinates [u, v]
    """
    pts = np.asarray(world_pts, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 3)

    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])  # (N, 4)
    projected = (P @ pts_h.T).T     # (N, 3)
    uv = projected[:, :2] / projected[:, 2:3]

    if single:
        return uv[0]
    return uv


def augmented_state_to_pixel(x_aug: np.ndarray,
                             P: np.ndarray) -> np.ndarray:
    """
    Map augmented state to pixel coordinates (numpy).

    Args:
        x_aug: (8,) [s, z, vs, vz, psi, cd, x0w, y0w]
        P: (3, 4) projection matrix

    Returns:
        (2,) pixel coordinates [u, v]
    """
    s, z, vs, vz, psi, cd, x0w, y0w = x_aug
    Xw = x0w + s * np.cos(psi)
    Yw = y0w + s * np.sin(psi)
    Zw = z
    return project_world_to_pixel(np.array([Xw, Yw, Zw]), P)


def create_measurement_expr(model) -> SX:
    """
    Build CasADi symbolic measurement expression h(x, p) -> [u_px, v_px].

    Uses model.x (augmented state) and model.p (P_flat) to construct
    the nonlinear projection expression for acados cost_y_expr.

    Args:
        model: AcadosModel with x (8-dim) and p (12-dim)

    Returns:
        CasADi SX expression (2,1)
    """
    from casadi import vertsplit, horzcat

    s, z, vs, vz, psi, cd, x0w, y0w = vertsplit(model.x)
    P_flat = model.p

    # Reconstruct 3x4 matrix from row-major flat vector.
    # CasADi reshape is column-major, so we index explicitly instead.
    P = vertcat(
        horzcat(P_flat[0], P_flat[1], P_flat[2], P_flat[3]),
        horzcat(P_flat[4], P_flat[5], P_flat[6], P_flat[7]),
        horzcat(P_flat[8], P_flat[9], P_flat[10], P_flat[11]),
    )

    # Flight plane -> world
    Xw = x0w + s * cos(psi)
    Yw = y0w + s * sin(psi)
    Zw = z

    # World -> homogeneous pixel
    world_h = vertcat(Xw, Yw, Zw, 1)
    projected = P @ world_h

    # Perspective division
    u_px = projected[0] / projected[2]
    v_px = projected[1] / projected[2]

    return vertcat(u_px, v_px)
