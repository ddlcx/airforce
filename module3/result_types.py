"""
Input/output data structures for Module 3.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ── Inputs (provided by external modules) ──


@dataclass
class ShuttlecockDetection:
    """Single-frame shuttlecock detection from external module (e.g. TrackNet)."""

    frame_idx: int
    pixel_x: float
    pixel_y: float
    visible: bool
    confidence: float  # [0, 1]


@dataclass
class HitEvent:
    """Hit event from external module."""

    frame_idx: int
    player_id: str  # "near" / "far"
    player_position_world: np.ndarray  # (3,) world coordinates at hit time


@dataclass
class RallySegment:
    """One flight segment between two consecutive hits."""

    detections: List[ShuttlecockDetection]
    all_frame_indices: List[int]
    hit_start: HitEvent
    hit_end: Optional[HitEvent]
    fps: float
    P: np.ndarray  # (3, 4) projection matrix
    K: np.ndarray  # (3, 3) intrinsic matrix


# ── Output ──


@dataclass
class TrajectoryResult3D:
    """3D trajectory reconstruction result for one flight segment."""

    time_stamps: np.ndarray          # (N+1,) seconds
    world_positions: np.ndarray      # (N+1, 3) [X_w, Y_w, Z_w]
    world_velocities: np.ndarray     # (N+1, 3) world velocities
    pixel_projected: np.ndarray      # (N+1, 2) reprojected pixel coords
    psi: float                       # estimated azimuth (rad)
    cd: float                        # estimated lumped drag (1/m)
    x0w: float                       # estimated initial world X
    y0w: float                       # estimated initial world Y
    z0: float                        # estimated initial height
    initial_speed: float             # |v0| (m/s)
    process_noise: np.ndarray        # (N, 4) estimated process noise
    reprojection_errors: np.ndarray  # (N+1,) per-node reproj error (px)
    mean_reproj_error: float
    solve_status: int                # 0 = success
    solve_time_ms: float
    segment_index: int = -1
