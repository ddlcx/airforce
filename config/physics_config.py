"""
Physics constants and MHE hyperparameters for Module 3.

References:
    [1] Cohen et al., "The physics of badminton", New J. Phys. 17, 2015
    [2] Liu & Wang, "MonoTrack", arXiv:2204.01899v2, 2022
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PhysicsConfig:
    """Shuttlecock physical parameters (Cohen 2015 [1])."""

    M: float = 5.0e-3       # mass (kg)
    rho: float = 1.2         # air density (kg/m^3) at 20 C
    S: float = 28.0e-4       # cross-section area (m^2), D=6cm
    C_D: float = 0.65        # drag coefficient (Cohen [1], feathered)
    g: float = 9.81          # gravitational acceleration (m/s^2)

    @property
    def aero_length(self) -> float:
        """Aerodynamic length L (m). Cohen [1, S2.1]."""
        return 2 * self.M / (self.rho * self.S * self.C_D)

    @property
    def cd_nominal(self) -> float:
        """Lumped drag parameter c_d = 1/L (1/m)."""
        return 1.0 / self.aero_length

    @property
    def terminal_velocity(self) -> float:
        """Terminal velocity U_inf (m/s). Cohen [1, S2.1]."""
        return (self.g * self.aero_length) ** 0.5


@dataclass
class MHEConfig:
    """MHE solver hyperparameters."""

    # Measurement noise
    sigma_pixel: float = 5.0          # pixel detection std (px)

    # Process noise penalty
    q_pos: float = 100.0              # position noise weight
    q_vel: float = 10.0               # velocity noise weight

    # Arrival cost weights
    q0_s: float = 0.1
    q0_z: float = 1.0
    q0_vs: float = 0.01
    q0_vz: float = 0.01
    q0_psi: float = 10.0
    q0_cd: float = 100.0
    q0_x0w: float = 10.0
    q0_y0w: float = 10.0

    # State bounds
    cd_min: float = 0.05
    cd_max: float = 0.50
    z_min: float = 0.0
    z_max: float = 15.0
    z0_max: float = 3.0

    # Integrator
    integrator_type: str = 'ERK'
    sim_method_num_stages: int = 4    # RK4
    sim_method_num_steps: int = 4     # substeps per interval

    # Solver
    nlp_solver_max_iter: int = 500
    nlp_tol: float = 1e-6

    @property
    def R(self) -> np.ndarray:
        """Measurement weight matrix (2x2)."""
        w = 1.0 / (self.sigma_pixel ** 2)
        return np.diag([w, w])

    @property
    def Q(self) -> np.ndarray:
        """Process noise penalty matrix (4x4)."""
        return np.diag([self.q_pos, self.q_pos, self.q_vel, self.q_vel])

    @property
    def Q0(self) -> np.ndarray:
        """Arrival cost matrix (8x8)."""
        return np.diag([
            self.q0_s, self.q0_z, self.q0_vs, self.q0_vz,
            self.q0_psi, self.q0_cd, self.q0_x0w, self.q0_y0w,
        ])
