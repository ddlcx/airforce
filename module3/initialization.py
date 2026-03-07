"""
Initial guess generation for MHE.

Strategy:
1. Player position -> x0w, y0w
2. Hit direction -> psi
3. Back-project 2D detections to flight plane -> (s_k, z_k)
4. Fit ODE parameters (z0, vs0, vz0, c_d) via least_squares
5. Optional: refine psi via grid search
"""

import numpy as np
from scipy.optimize import least_squares

from module3.shuttlecock_model import integrate_trajectory


def compute_azimuth(hit_start_pos: np.ndarray,
                    hit_end_pos: np.ndarray) -> float:
    """
    Compute azimuth from hitter to receiver.

    Args:
        hit_start_pos: (3,) hitter world position
        hit_end_pos: (3,) receiver world position (or court center if unknown)

    Returns:
        azimuth in radians
    """
    dx = hit_end_pos[0] - hit_start_pos[0]
    dy = hit_end_pos[1] - hit_start_pos[1]
    return np.arctan2(dy, dx)


def back_project_to_flight_plane(
    pixel_coords: np.ndarray,
    P: np.ndarray,
    psi: float,
    x0w: float,
    y0w: float,
) -> np.ndarray:
    """
    Back-project 2D pixel coordinates to flight-plane (s, z).

    Given psi, x0w, y0w, each pixel (u, v) gives a 2x2 linear system for (s, z).

    World coords: X = x0w + s*cos(psi), Y = y0w + s*sin(psi), Z = z
    Projection: P @ [X, Y, Z, 1]^T = lambda * [u, v, 1]^T

    Eliminating lambda yields 2 equations in 2 unknowns (s, z).

    Args:
        pixel_coords: (N, 2) pixel coordinates [u, v]
        P: (3, 4) projection matrix
        psi: azimuth angle
        x0w, y0w: flight plane origin

    Returns:
        (N, 2) flight-plane coordinates [s, z]
    """
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    # P @ [x0w + s*cos, y0w + s*sin, z, 1]^T = lambda * [u, v, 1]^T
    # Let d = P @ [x0w, y0w, 0, 1]^T  (constant part)
    # Let a = P @ [cos, sin, 0, 0]^T   (s coefficient)
    # Let b = P @ [0, 0, 1, 0]^T       (z coefficient)
    # Then: d + s*a + z*b = lambda * [u, v, 1]^T

    d = P @ np.array([x0w, y0w, 0, 1])        # (3,)
    a = P @ np.array([cos_psi, sin_psi, 0, 0]) # (3,)
    b = P @ np.array([0, 0, 1, 0])             # (3,)

    N = pixel_coords.shape[0]
    result = np.zeros((N, 2))

    for i in range(N):
        u, v = pixel_coords[i]

        # Eliminate lambda using rows 0,2 and rows 1,2:
        # (d[0] + s*a[0] + z*b[0]) / (d[2] + s*a[2] + z*b[2]) = u
        # (d[1] + s*a[1] + z*b[1]) / (d[2] + s*a[2] + z*b[2]) = v
        #
        # Cross-multiplying:
        # d[0] + s*a[0] + z*b[0] = u*(d[2] + s*a[2] + z*b[2])
        # d[1] + s*a[1] + z*b[1] = v*(d[2] + s*a[2] + z*b[2])
        #
        # Rearranging:
        # s*(a[0] - u*a[2]) + z*(b[0] - u*b[2]) = u*d[2] - d[0]
        # s*(a[1] - v*a[2]) + z*(b[1] - v*b[2]) = v*d[2] - d[1]

        A_mat = np.array([
            [a[0] - u * a[2], b[0] - u * b[2]],
            [a[1] - v * a[2], b[1] - v * b[2]],
        ])
        rhs = np.array([
            u * d[2] - d[0],
            v * d[2] - d[1],
        ])

        cond = np.linalg.cond(A_mat)
        if cond > 1e10:
            # Ill-conditioned: use least-squares
            sz, _, _, _ = np.linalg.lstsq(A_mat, rhs, rcond=None)
        else:
            try:
                sz = np.linalg.solve(A_mat, rhs)
            except np.linalg.LinAlgError:
                sz, _, _, _ = np.linalg.lstsq(A_mat, rhs, rcond=None)

        # Clamp to reasonable physical range
        sz[0] = np.clip(sz[0], -50, 50)   # s
        sz[1] = np.clip(sz[1], 0.01, 15)  # z
        result[i] = sz

    return result


def fit_ode_parameters(
    s_obs: np.ndarray,
    z_obs: np.ndarray,
    dt: float,
    cd_nominal: float = 0.217,
    cd_reg_weight: float = 20.0,
) -> dict:
    """
    Fit initial velocity and c_d by matching ODE to back-projected trajectory.

    Fits: [vs0, vz0, z0, cd] to minimize |ode(t_k) - (s_k, z_k)|^2
    with a regularization term penalizing deviation of cd from nominal.

    Multi-start: tries several cd initial values to avoid local minima.

    Args:
        s_obs: (N+1,) observed s coordinates
        z_obs: (N+1,) observed z coordinates
        dt: time step
        cd_nominal: nominal c_d value (prior center)
        cd_reg_weight: regularization weight for cd deviation

    Returns:
        dict with keys: vs0, vz0, z0, cd, residual
    """
    n_steps = len(s_obs) - 1
    if n_steps < 3:
        vs0 = (s_obs[-1] - s_obs[0]) / (n_steps * dt) if n_steps > 0 else 10.0
        vz0 = (z_obs[-1] - z_obs[0]) / (n_steps * dt) if n_steps > 0 else 5.0
        return {
            'vs0': vs0, 'vz0': vz0,
            'z0': z_obs[0], 'cd': cd_nominal, 'residual': 1e6,
        }

    def residual_fn(params):
        vs0, vz0, z0, cd = params
        cd = max(0.05, min(0.50, cd))
        x0 = np.array([0.0, z0, vs0, vz0])
        traj = integrate_trajectory(x0, cd, dt, n_steps, num_substeps=4)
        res_s = traj[:, 0] - s_obs
        res_z = traj[:, 1] - z_obs
        # Regularization: penalize cd deviation from nominal
        res_cd = np.array([cd_reg_weight * (cd - cd_nominal)])
        return np.concatenate([res_s, res_z, res_cd])

    vs0_init = (s_obs[min(3, n_steps)] - s_obs[0]) / (min(3, n_steps) * dt)
    vz0_init = (z_obs[min(3, n_steps)] - z_obs[0]) / (min(3, n_steps) * dt)
    z0_init = z_obs[0]

    bounds_low = np.array([0.0, -100.0, 0.0, 0.05])
    bounds_high = np.array([150.0, 100.0, 5.0, 0.50])

    # Multi-start: try several cd initial values
    cd_starts = [0.15, cd_nominal, 0.30]
    best_result = None
    best_cost = np.inf

    for cd_start in cd_starts:
        p0 = np.clip(
            [abs(vs0_init), vz0_init, z0_init, cd_start],
            bounds_low + 1e-8,
            bounds_high - 1e-8,
        )

        try:
            result = least_squares(
                residual_fn, p0,
                bounds=(bounds_low, bounds_high),
                method='trf', max_nfev=200,
            )
            cost = np.mean(result.fun**2)
            if cost < best_cost:
                best_cost = cost
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return {
            'vs0': float(abs(vs0_init)), 'vz0': float(vz0_init),
            'z0': float(z0_init), 'cd': cd_nominal, 'residual': 1e6,
        }

    return {
        'vs0': best_result.x[0],
        'vz0': best_result.x[1],
        'z0': best_result.x[2],
        'cd': best_result.x[3],
        'residual': best_cost,
    }


def generate_initial_guess(
    segment,
    psi_search_range: float = np.radians(8),
    psi_search_steps: int = 13,
) -> tuple:
    """
    Generate initial guess for MHE from a RallySegment.

    Args:
        segment: RallySegment
        psi_search_range: half-range for azimuth grid search (rad)
        psi_search_steps: number of azimuth candidates

    Returns:
        (x0_bar, x_init, best_psi) where:
            x0_bar: (8,) prior for arrival cost
            x_init: (N+1, 8) initial state guess for each node
            best_psi: best azimuth found
    """
    from module3.result_types import RallySegment

    P = segment.P
    fps = segment.fps
    dt = 1.0 / fps
    detections = segment.detections
    N = len(detections) - 1

    # Player positions
    x0w = segment.hit_start.player_position_world[0]
    y0w = segment.hit_start.player_position_world[1]

    # Azimuth from hitter to receiver (or court center)
    if segment.hit_end is not None:
        psi_center = compute_azimuth(
            segment.hit_start.player_position_world,
            segment.hit_end.player_position_world,
        )
    else:
        # Default: toward opposite baseline
        if segment.hit_start.player_id == 'near':
            psi_center = np.pi / 2   # toward far end
        else:
            psi_center = -np.pi / 2  # toward near end

    # Visible detections
    vis_mask = np.array([d.visible for d in detections])
    pixel_obs = np.array([[d.pixel_x, d.pixel_y] for d in detections])

    # Grid search over psi with prior penalty toward psi_center.
    # The prior prevents selecting wrong psi when the ODE residual
    # landscape is flat (short/slow trajectories like drop shots).
    psi_candidates = np.linspace(
        psi_center - psi_search_range,
        psi_center + psi_search_range,
        psi_search_steps,
    )

    vis_indices = np.where(vis_mask)[0]
    best_psi = psi_center
    best_score = np.inf
    best_params = None
    best_sz = None

    # Collect ODE residuals for normalization
    trial_results = []

    for psi_trial in psi_candidates:
        if len(vis_indices) < 4:
            continue

        vis_pixels = pixel_obs[vis_indices]
        sz_vis = back_project_to_flight_plane(vis_pixels, P, psi_trial, x0w, y0w)

        s_all = np.interp(np.arange(N + 1), vis_indices, sz_vis[:, 0])
        z_all = np.interp(np.arange(N + 1), vis_indices, sz_vis[:, 1])
        z_all = np.maximum(z_all, 0.01)

        params = fit_ode_parameters(s_all, z_all, dt)
        trial_results.append((psi_trial, params, s_all, z_all))

    if trial_results:
        # Normalize ODE residuals so psi prior has consistent influence
        residuals = np.array([t[1]['residual'] for t in trial_results])
        res_range = max(residuals.max() - residuals.min(), 1e-10)

        # psi prior weight: penalize deviation from psi_center
        # Scale: 1 radian deviation adds ~1.0 to normalized score
        psi_prior_weight = 1.0

        for psi_trial, params, s_all, z_all in trial_results:
            norm_residual = (params['residual'] - residuals.min()) / res_range
            psi_dev = (psi_trial - psi_center) ** 2
            score = norm_residual + psi_prior_weight * psi_dev

            if score < best_score:
                best_score = score
                best_psi = psi_trial
                best_params = params
                best_sz = (s_all, z_all)

    if best_params is None:
        # Fallback
        best_params = {'vs0': 20.0, 'vz0': 5.0, 'z0': 1.5, 'cd': 0.217}
        s_all = np.linspace(0, 5, N + 1)
        z_all = np.full(N + 1, 1.5)
        best_sz = (s_all, z_all)

    # Build x0_bar and x_init
    cd = best_params['cd']
    vs0 = best_params['vs0']
    vz0 = best_params['vz0']
    z0 = best_params['z0']

    x0_bar = np.array([0.0, z0, vs0, vz0, best_psi, cd, x0w, y0w])

    # Generate trajectory from fitted parameters
    x0_phys = np.array([0.0, z0, vs0, vz0])
    traj = integrate_trajectory(x0_phys, cd, dt, N, num_substeps=4)

    x_init = np.zeros((N + 1, 8))
    x_init[:, 0] = np.clip(traj[:, 0], -50, 50)     # s
    x_init[:, 1] = np.clip(traj[:, 1], 0.0, 15.0)   # z
    x_init[:, 2] = np.clip(traj[:, 2], -150, 150)   # vs
    x_init[:, 3] = np.clip(traj[:, 3], -150, 150)   # vz
    x_init[:, 4] = best_psi     # psi (constant)
    x_init[:, 5] = cd           # cd (constant)
    x_init[:, 6] = x0w          # x0w (constant)
    x_init[:, 7] = y0w          # y0w (constant)

    return x0_bar, x_init, best_psi
