"""
Main trajectory estimator: high-level interface for Module 3.

Orchestrates: segment_builder -> initialization -> mhe_solver -> results.
"""

import logging
from typing import List, Optional

import numpy as np

from config.physics_config import PhysicsConfig, MHEConfig
from module3.result_types import (
    RallySegment,
    TrajectoryResult3D,
)
from module3.shuttlecock_model import (
    flight_plane_to_world,
    world_velocities_from_flight_plane,
)
from module3.measurement_model import augmented_state_to_pixel
from module3.initialization import generate_initial_guess
from module3.mhe_solver import (
    build_mhe_solver,
    solve_mhe,
    NX,
    NW,
)

logger = logging.getLogger(__name__)


class TrajectoryEstimator:
    """
    Estimates 3D shuttlecock trajectories from 2D detections using MHE.

    Usage:
        estimator = TrajectoryEstimator()
        results = estimator.estimate_segments(segments)
    """

    def __init__(
        self,
        physics: PhysicsConfig = None,
        mhe_config: MHEConfig = None,
    ):
        self.physics = physics or PhysicsConfig()
        self.cfg = mhe_config or MHEConfig()
        self._solver_cache = {}  # cache solvers by N

    def _get_solver(self, N: int, dt: float):
        """Get or build acados solver for given horizon length."""
        cache_key = (N, round(dt, 6))
        if cache_key not in self._solver_cache:
            logger.info(f'Building acados solver for N={N}, dt={dt:.4f}')
            solver = build_mhe_solver(
                N=N,
                dt=dt,
                R=self.cfg.R,
                Q=self.cfg.Q,
                Q0=self.cfg.Q0,
                cd_bounds=(self.cfg.cd_min, self.cfg.cd_max),
                z_bounds=(self.cfg.z_min, self.cfg.z_max),
                integrator_type=self.cfg.integrator_type,
                sim_method_num_stages=self.cfg.sim_method_num_stages,
                sim_method_num_steps=self.cfg.sim_method_num_steps,
                nlp_solver_max_iter=self.cfg.nlp_solver_max_iter,
            )
            self._solver_cache[cache_key] = solver
        return self._solver_cache[cache_key]

    def estimate_segment(
        self,
        segment: RallySegment,
        segment_index: int = -1,
    ) -> TrajectoryResult3D:
        """
        Estimate 3D trajectory for one flight segment.

        Args:
            segment: RallySegment with detections and camera params
            segment_index: index for logging

        Returns:
            TrajectoryResult3D
        """
        detections = segment.detections
        N = len(detections) - 1
        dt = 1.0 / segment.fps
        P = segment.P
        P_flat = P.flatten()

        # ── Initialization ──
        x0_bar, x_init, best_psi = generate_initial_guess(segment)
        logger.info(
            f'Segment {segment_index}: N={N}, psi={np.degrees(best_psi):.1f}deg, '
            f'cd_init={x0_bar[5]:.3f}, v0={np.sqrt(x0_bar[2]**2+x0_bar[3]**2):.1f}m/s'
        )

        observations = np.array([
            [d.pixel_x, d.pixel_y] for d in detections
        ])
        visibility = np.array([d.visible for d in detections])

        solver = self._get_solver(N, dt)

        # ── Solve MHE ──
        result = solve_mhe(
            solver=solver,
            N=N,
            P_flat=P_flat,
            observations=observations,
            visibility=visibility,
            x0_bar=x0_bar,
            x_init=x_init,
            R=self.cfg.R,
            Q=self.cfg.Q,
            Q0=self.cfg.Q0,
        )

        # ── Extract 3D trajectory ──
        states = result.states
        psi_est = float(np.mean(states[:, 4]))
        cd_est = float(np.mean(states[:, 5]))
        x0w_est = float(np.mean(states[:, 6]))
        y0w_est = float(np.mean(states[:, 7]))

        world_pos = flight_plane_to_world(
            states[:, 0], states[:, 1], psi_est, x0w_est, y0w_est
        )
        world_vel = world_velocities_from_flight_plane(
            states[:, 2], states[:, 3], psi_est
        )

        # ── Reprojection errors ──
        pixel_proj = np.zeros((N + 1, 2))
        reproj_errors = np.zeros(N + 1)
        for k in range(N + 1):
            pixel_proj[k] = augmented_state_to_pixel(states[k], P)
            if visibility[k]:
                reproj_errors[k] = np.linalg.norm(
                    pixel_proj[k] - observations[k]
                )

        vis_mask = visibility.astype(bool)
        mean_reproj = float(np.mean(reproj_errors[vis_mask])) if vis_mask.any() else 0.0

        time_stamps = np.arange(N + 1) * dt

        logger.info(
            f'Segment {segment_index}: status={result.status}, '
            f'reproj={mean_reproj:.2f}px, cd={cd_est:.3f}, '
            f'solve={result.solve_time_ms:.1f}ms, iter={result.sqp_iterations}'
        )

        return TrajectoryResult3D(
            time_stamps=time_stamps,
            world_positions=world_pos,
            world_velocities=world_vel,
            pixel_projected=pixel_proj,
            psi=psi_est,
            cd=cd_est,
            x0w=x0w_est,
            y0w=y0w_est,
            z0=float(states[0, 1]),
            initial_speed=float(np.sqrt(states[0, 2]**2 + states[0, 3]**2)),
            process_noise=result.noises,
            reprojection_errors=reproj_errors,
            mean_reproj_error=mean_reproj,
            solve_status=result.status,
            solve_time_ms=result.solve_time_ms,
            segment_index=segment_index,
        )

    def estimate_segments(
        self,
        segments: List[RallySegment],
    ) -> List[TrajectoryResult3D]:
        """Estimate 3D trajectories for all segments."""
        results = []
        for i, seg in enumerate(segments):
            try:
                result = self.estimate_segment(seg, segment_index=i)
                results.append(result)
            except Exception as e:
                logger.error(f'Segment {i} failed: {e}')
        return results
