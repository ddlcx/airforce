"""
Synthetic data generator for Module 3 testing.

Creates realistic:
- Camera projection matrix P (broadcast-style view)
- True 3D trajectories for various shot types
- 2D pixel observations with noise and missing detections
- RallySegment objects ready for the estimator
"""

import numpy as np
from module3.shuttlecock_model import integrate_trajectory, flight_plane_to_world
from module3.measurement_model import project_world_to_pixel
from module3.result_types import ShuttlecockDetection, HitEvent, RallySegment


def make_camera(
    cam_pos: np.ndarray = None,
    look_at: np.ndarray = None,
    focal_length: float = 1000.0,
    image_size: tuple = (1920, 1080),
) -> tuple:
    """
    Create a synthetic camera.

    Args:
        cam_pos: (3,) camera position in world frame
        look_at: (3,) point the camera looks at
        focal_length: focal length in pixels
        image_size: (width, height) in pixels

    Returns:
        (P, K) - (3,4) projection matrix, (3,3) intrinsic matrix
    """
    if cam_pos is None:
        # Broadcast-style: behind near baseline, elevated, slightly off-center
        # Off-center position provides lateral parallax for depth recovery
        cam_pos = np.array([3.0, -14.0, 8.0])
    if look_at is None:
        look_at = np.array([0.0, 1.0, 1.5])  # slightly in front of net

    w, h = image_size
    K = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1],
    ], dtype=np.float64)

    # Camera rotation: look-at construction
    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)

    up_hint = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # R: world -> camera rotation
    # Camera axes: x=right, y=-up (image y down), z=forward
    R = np.array([right, -up, forward])

    t = -R @ cam_pos  # translation

    Rt = np.hstack([R, t.reshape(3, 1)])
    P = K @ Rt

    return P, K


def generate_true_trajectory(
    shot_type: str,
    fps: float = 30.0,
    psi: float = None,
    x0w: float = 0.0,
    y0w: float = None,
    cd: float = 0.217,
) -> dict:
    """
    Generate a true trajectory for a given shot type.

    Shot types:
        'clear': high arc, moderate speed
        'smash': fast and downward
        'drop': slow and descending
        'drive': flat and fast
        'cross_court': diagonal trajectory

    Returns:
        dict with keys: x0_phys, psi, x0w, y0w, cd, fps, dt, n_steps,
                        traj_phys, world_pos, time_stamps
    """
    dt = 1.0 / fps

    shot_params = {
        'clear': {
            'vs0': 30.0, 'vz0': 15.0, 'z0': 2.0,
            'psi': np.pi / 2, 'y0w': -5.0, 'duration': 1.2,
        },
        'smash': {
            'vs0': 60.0, 'vz0': -5.0, 'z0': 2.8,
            'psi': np.pi / 2, 'y0w': -3.0, 'duration': 0.5,
        },
        'drop': {
            'vs0': 12.0, 'vz0': 2.0, 'z0': 2.5,
            'psi': np.pi / 2, 'y0w': -2.0, 'duration': 0.8,
        },
        'drive': {
            'vs0': 40.0, 'vz0': 3.0, 'z0': 1.5,
            'psi': np.pi / 2, 'y0w': -4.0, 'duration': 0.7,
        },
        'cross_court': {
            'vs0': 35.0, 'vz0': 10.0, 'z0': 2.0,
            'psi': np.pi / 2 + np.pi / 6, 'y0w': -4.0, 'duration': 1.0,
        },
    }

    params = shot_params[shot_type]
    if psi is None:
        psi = params['psi']
    if y0w is None:
        y0w = params['y0w']

    vs0 = params['vs0']
    vz0 = params['vz0']
    z0 = params['z0']
    duration = params['duration']
    n_steps = int(duration * fps)

    x0_phys = np.array([0.0, z0, vs0, vz0])
    traj_phys = integrate_trajectory(x0_phys, cd, dt, n_steps, num_substeps=8)

    # Clip at ground (z >= 0)
    ground_hit = np.where(traj_phys[:, 1] < 0)[0]
    if len(ground_hit) > 0:
        n_steps = ground_hit[0]
        traj_phys = traj_phys[:n_steps + 1]

    world_pos = flight_plane_to_world(
        traj_phys[:, 0], traj_phys[:, 1], psi, x0w, y0w
    )

    time_stamps = np.arange(len(traj_phys)) * dt

    return {
        'x0_phys': x0_phys,
        'psi': psi,
        'x0w': x0w,
        'y0w': y0w,
        'cd': cd,
        'fps': fps,
        'dt': dt,
        'n_steps': len(traj_phys) - 1,
        'traj_phys': traj_phys,
        'world_pos': world_pos,
        'time_stamps': time_stamps,
        'shot_type': shot_type,
    }


def generate_observations(
    true_data: dict,
    P: np.ndarray,
    noise_std: float = 3.0,
    missing_rate: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Generate noisy 2D pixel observations from true trajectory.

    Args:
        true_data: output from generate_true_trajectory
        P: (3, 4) projection matrix
        noise_std: pixel noise standard deviation
        missing_rate: fraction of frames with missing detections
        seed: random seed

    Returns:
        dict with keys: pixel_true, pixel_noisy, visibility, detections
    """
    rng = np.random.RandomState(seed)

    world_pos = true_data['world_pos']
    n_points = len(world_pos)

    pixel_true = project_world_to_pixel(world_pos, P)
    pixel_noisy = pixel_true + rng.randn(n_points, 2) * noise_std

    # Generate visibility mask
    visibility = np.ones(n_points, dtype=bool)
    if missing_rate > 0:
        n_missing = int(n_points * missing_rate)
        # Don't remove first and last frames
        candidates = np.arange(1, n_points - 1)
        if n_missing > 0 and len(candidates) > 0:
            missing_idx = rng.choice(
                candidates, min(n_missing, len(candidates)), replace=False
            )
            visibility[missing_idx] = False

    # Create detections
    detections = []
    for i in range(n_points):
        if visibility[i]:
            detections.append(ShuttlecockDetection(
                frame_idx=i,
                pixel_x=float(pixel_noisy[i, 0]),
                pixel_y=float(pixel_noisy[i, 1]),
                visible=True,
                confidence=0.9,
            ))
        else:
            detections.append(ShuttlecockDetection(
                frame_idx=i,
                pixel_x=0.0,
                pixel_y=0.0,
                visible=False,
                confidence=0.0,
            ))

    return {
        'pixel_true': pixel_true,
        'pixel_noisy': pixel_noisy,
        'visibility': visibility,
        'detections': detections,
    }


def make_rally_segment(
    true_data: dict,
    obs_data: dict,
    P: np.ndarray,
    K: np.ndarray,
) -> RallySegment:
    """Package true data + observations into a RallySegment."""
    n_points = len(obs_data['detections'])

    hit_start = HitEvent(
        frame_idx=0,
        player_id='near',
        player_position_world=np.array([
            true_data['x0w'],
            true_data['y0w'],
            0.0,
        ]),
    )

    hit_end = HitEvent(
        frame_idx=n_points - 1,
        player_id='far',
        player_position_world=np.array([
            true_data['world_pos'][-1, 0],
            true_data['world_pos'][-1, 1],
            0.0,
        ]),
    )

    return RallySegment(
        detections=obs_data['detections'],
        all_frame_indices=list(range(n_points)),
        hit_start=hit_start,
        hit_end=hit_end,
        fps=true_data['fps'],
        P=P.copy(),
        K=K.copy(),
    )


# ── Pre-built test scenarios ──


def get_test_scenarios(fps: float = 30.0) -> list:
    """
    Generate a set of test scenarios covering various conditions.

    Returns:
        list of dicts, each with keys:
            name, shot_type, noise_std, missing_rate, true_data, obs_data,
            segment, P, K, description
    """
    P, K = make_camera()
    scenarios = []

    configs = [
        ('clear_clean', 'clear', 0.0, 0.0,
         'High clear, zero noise'),
        ('clear_noisy', 'clear', 3.0, 0.0,
         'High clear, 3px noise'),
        ('smash_noisy', 'smash', 3.0, 0.0,
         'Smash, 3px noise'),
        ('drop_noisy', 'drop', 3.0, 0.0,
         'Drop shot, 3px noise'),
        ('drive_noisy', 'drive', 3.0, 0.0,
         'Drive, 3px noise'),
        ('cross_court_noisy', 'cross_court', 3.0, 0.0,
         'Cross court, 3px noise'),
        ('clear_missing30', 'clear', 3.0, 0.3,
         'High clear, 3px noise, 30% missing'),
        ('smash_high_noise', 'smash', 8.0, 0.0,
         'Smash, 8px high noise'),
    ]

    for name, shot_type, noise_std, missing_rate, desc in configs:
        true_data = generate_true_trajectory(shot_type, fps=fps)
        obs_data = generate_observations(
            true_data, P,
            noise_std=noise_std,
            missing_rate=missing_rate,
            seed=hash(name) % (2**31),
        )
        segment = make_rally_segment(true_data, obs_data, P, K)

        scenarios.append({
            'name': name,
            'shot_type': shot_type,
            'noise_std': noise_std,
            'missing_rate': missing_rate,
            'true_data': true_data,
            'obs_data': obs_data,
            'segment': segment,
            'P': P,
            'K': K,
            'description': desc,
        })

    return scenarios
