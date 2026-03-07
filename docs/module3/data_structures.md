# Module 3: 数据结构

数据结构定义位于 `module3/result_types.py`。

## 1. 输入

```python
@dataclass
class ShuttlecockDetection:
    """由外部模块提供的单帧羽毛球检测。"""
    frame_idx: int
    pixel_x: float
    pixel_y: float
    visible: bool
    confidence: float               # [0, 1]

@dataclass
class HitEvent:
    """由外部模块提供的击球事件。"""
    frame_idx: int
    player_id: str                  # "near" / "far"
    player_position_world: np.ndarray   # (3,) 击球时球员世界坐标

@dataclass
class RallySegment:
    """一次飞行段（两次击球之间）。"""
    detections: List[ShuttlecockDetection]
    all_frame_indices: List[int]
    hit_start: HitEvent
    hit_end: Optional[HitEvent]
    fps: float
    P: np.ndarray                   # (3, 4) 投影矩阵
    K: np.ndarray                   # (3, 3) 内参矩阵
```

## 2. 输出

```python
@dataclass
class TrajectoryResult3D:
    """单个飞行段的 3D 轨迹重建结果。"""
    time_stamps: np.ndarray             # (N+1,) 秒
    world_positions: np.ndarray         # (N+1, 3) 世界坐标 [X_w, Y_w, Z_w]
    world_velocities: np.ndarray        # (N+1, 3) 世界速度
    pixel_projected: np.ndarray         # (N+1, 2) 重投影像素坐标
    psi: float                          # 估计方位角 (rad)
    cd: float                           # 估计集总阻力参数 (1/m)
    x0w: float                          # 估计初始世界 X
    y0w: float                          # 估计初始世界 Y
    z0: float                           # 估计初始高度
    initial_speed: float                # 估计初始速度 |v₀| (m/s)
    process_noise: np.ndarray           # (N, 4) 估计的过程噪声 w
    reprojection_errors: np.ndarray     # (N+1,) 每节点重投影误差 (px)
    mean_reproj_error: float
    solve_status: int                   # acados 状态 (0=成功)
    solve_time_ms: float
    num_sqp_iterations: int
```
