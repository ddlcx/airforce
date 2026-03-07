# Module 3: acados MHE 实现

```mermaid
flowchart LR
    A[CasADi 符号模型] --> B[AcadosOcp 配置]
    B --> C[C 代码生成]
    C --> D[编译共享库]
    D --> E[AcadosOcpSolver]
    E --> F[设置参数/观测]
    F --> G[solver.solve]
    G --> H[提取结果]
```

## 4.1 acados Python 接口

acados 通过 `acados_template` Python 包提供完整的 MHE 接口。Python 层定义 CasADi 符号模型和 OCP 配置后，acados 自动生成优化的 C 代码并编译为共享库，通过 CFFI 绑定回 Python 调用。

**安装方式**: 详见 [installation.md](installation.md)。

## 4.2 CasADi 模型定义 (`shuttlecock_model.py`)

遵循 acados MHE 范式（参考 `examples/acados_python/pendulum_on_cart/mhe/export_mhe_ode_model.py`）：

```python
def export_shuttlecock_mhe_model() -> AcadosModel:
    model = AcadosModel()

    # 增广状态 (8维)
    s    = SX.sym('s')
    z    = SX.sym('z')
    vs   = SX.sym('vs')
    vz   = SX.sym('vz')
    psi  = SX.sym('psi')
    cd   = SX.sym('cd')
    x0w  = SX.sym('x0w')
    y0w  = SX.sym('y0w')
    x = vertcat(s, z, vs, vz, psi, cd, x0w, y0w)

    # 过程噪声 = MHE 的 "控制输入" (4维)
    w_s  = SX.sym('w_s')
    w_z  = SX.sym('w_z')
    w_vs = SX.sym('w_vs')
    w_vz = SX.sym('w_vz')
    w = vertcat(w_s, w_z, w_vs, w_vz)

    # 已知参数: 投影矩阵展平 (12维)
    p_proj = SX.sym('p_proj', 12)

    # 动力学
    speed = sqrt(vs**2 + vz**2)
    f_expl = vertcat(
        vs + w_s,                               # ds/dt
        vz + w_z,                               # dz/dt
        -cd * speed * vs + w_vs,                # dv_s/dt
        -G - cd * speed * vz + w_vz,            # dv_z/dt
        0, 0, 0, 0                              # 参数零动力学
    )

    model.x = x
    model.u = w             # MHE 范式: 过程噪声占据 u 位置
    model.p = p_proj        # 投影矩阵作为运行时参数
    model.f_expl_expr = f_expl
    model.name = 'shuttlecock_mhe'
    return model
```

## 4.3 测量模型定义 (`measurement_model.py`)

```python
def create_measurement_expr(model) -> SX:
    """构建 h(x_aug, p) → [u_px, v_px] 的 CasADi 符号表达式。"""
    s, z, vs, vz, psi, cd, x0w, y0w = vertsplit(model.x)
    P_flat = model.p

    # 从行优先 (numpy flatten) 的 P_flat 重建 3×4 矩阵。
    # 注意: CasADi 的 reshape 为列优先，不能直接使用 reshape(P_flat, 3, 4)。
    P = vertcat(
        horzcat(P_flat[0], P_flat[1], P_flat[2], P_flat[3]),
        horzcat(P_flat[4], P_flat[5], P_flat[6], P_flat[7]),
        horzcat(P_flat[8], P_flat[9], P_flat[10], P_flat[11]),
    )

    # 飞行平面 → 世界坐标
    Xw = x0w + s * cos(psi)
    Yw = y0w + s * sin(psi)
    Zw = z

    # 投影
    homog = P @ vertcat(Xw, Yw, Zw, 1)
    u_px = homog[0] / homog[2]
    v_px = homog[1] / homog[2]

    return vertcat(u_px, v_px)
```

## 4.4 求解器配置 (`mhe_setup.py`)

遵循 acados MHE 范式（参考 `examples/acados_python/pendulum_on_cart/mhe/export_mhe_solver.py`）：

```python
def export_shuttlecock_mhe_solver(model, N, dt, R, Q, Q0) -> AcadosOcpSolver:
    ocp = AcadosOcp()
    ocp.model = model

    h_meas = create_measurement_expr(model)
    nx = 8   # 增广状态维度
    nw = 4   # 过程噪声维度
    ny = 2   # 测量维度

    # ── 节点 0: 测量 + 噪声 + 到达代价 ──
    ocp.cost.cost_type_0 = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_0 = vertcat(h_meas, model.u, model.x)
    ocp.cost.W_0 = block_diag(R, Q, Q0)
    ocp.cost.yref_0 = np.zeros(ny + nw + nx)

    # ── 中间节点: 测量 + 噪声 ──
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = vertcat(h_meas, model.u)
    ocp.cost.W = block_diag(R, Q)
    ocp.cost.yref = np.zeros(ny + nw)

    # ── 终端节点: 测量代价（无噪声项） ──
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = h_meas
    ocp.cost.W_e = R.copy()
    ocp.cost.yref_e = np.zeros(ny)

    # ── 求解器选项 ──
    ocp.solver_options.cost_scaling = np.ones(N + 1)
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.nlp_solver_max_iter = 500
    ocp.solver_options.levenberg_marquardt = 1e-2
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.alpha_min = 1e-4
    ocp.solver_options.tf = N * dt
    ocp.solver_options.N_horizon = N

    # ── 积分器配置 (二选一，详见 §4.6.4) ──
    # 方案 A: ERK + RK4
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4    # RK4
    ocp.solver_options.sim_method_num_steps = 4     # 子步数
    # 方案 B: IRK + GL4
    # ocp.solver_options.integrator_type = 'IRK'
    # ocp.solver_options.sim_method_num_stages = 2  # GL4
    # ocp.solver_options.sim_method_num_steps = 1
    # ocp.solver_options.sim_method_newton_iter = 5

    # ── 状态约束 ──
    # z >= 0, c_d bounds, psi bounds, etc.
    # (通过 ocp.constraints.lbx / ubx 设置)

    return AcadosOcpSolver(ocp)
```

## 4.5 运行时参数设置 (`trajectory_estimator.py`)

```python
# 逐节点设置观测值和参数
for k in range(N + 1):
    solver.set(k, "p", P_flat)                      # 投影矩阵

    if k == 0:
        yref_0 = np.concatenate([obs_k, zeros_4, x0_bar])
        solver.set(0, "yref", yref_0)
    elif k < N:
        yref_k = np.concatenate([obs_k, zeros_4])
        solver.set(k, "yref", yref_k)

# 缺失检测: 将测量权重设为近零
if not detection.visible:
    W_missing = block_diag(R_zero, Q)               # R_zero = diag(1e-10, 1e-10)
    solver.cost_set(k, "W", W_missing)

# 设置初始猜测
for k in range(N + 1):
    solver.set(k, "x", x_init[k])
for k in range(N):
    solver.set(k, "u", np.zeros(4))                 # 初始噪声猜测为零

# 求解
status = solver.solve()

# 提取结果
states = np.array([solver.get(k, "x") for k in range(N + 1)])
noises = np.array([solver.get(k, "u") for k in range(N)])
```

## 4.6 离散化方法与时间对齐

### 4.6.1 Multiple Shooting 离散化

MHE 采用 multiple shooting 离散化：每个节点 k 的状态 x_k 是独立的优化变量，相邻节点通过 ODE 积分产生的连续性约束（shooting gap）耦合。acados 在 SQP 迭代中自动处理这些约束。

```
x₀ ──[积分器]──→ x̂₁    x₁ ──[积分器]──→ x̂₂    ...
                  ‖                      ‖
              gap₁ = x̂₁ − x₁ = 0   gap₂ = x̂₂ − x₂ = 0
```

### 4.6.2 时间步长 dt

**设计决策: dt = 1/fps，每个 MHE 节点对应一个视频帧。**

```
视频帧:    f₀       f₁       f₂       f₃       ...      fₙ
时间:     0      1/fps    2/fps    3/fps    ...    N/fps
           │        │        │        │                │
MHE节点:  k=0      k=1      k=2      k=3     ...    k=N
观测:     obs₀     obs₁     obs₂     obs₃          obsₙ
```

理由:
- 外部模块（TrackNet 等）在每个视频帧产出一个 2D 检测
- 一一对应关系消除了时间插值的需要
- 节点 k 的估计状态 x_k 精确对应帧 f_k，不存在时间对齐误差
- 典型问题规模：30 fps 下 1 秒飞行段 → N = 30，极其轻量

| 视频帧率 | dt | 1 秒飞行段节点数 |
|----------|------|----------------|
| 25 fps | 0.040 s | 25 |
| 30 fps | 0.033 s | 30 |
| 60 fps | 0.017 s | 60 |

### 4.6.3 时间对齐异常处理

| 异常情况 | 原因 | 处理策略 |
|----------|------|----------|
| 缺失检测 | TrackNet 某帧未检测到球 | 节点保留，dt 不变，R 权重设为近零（参见 [mhe_formulation.md](mhe_formulation.md) 缺失检测处理）|
| 视频丢帧 | 帧索引不连续（如 f₅ → f₇） | 按帧索引差值计算各区间实际 dt_k，通过 `ocp.solver_options.time_steps` 设置非均匀时间步长数组 |
| 帧率微抖 | 实际帧间隔 ≠ 精确 1/fps | 广播视频可忽略；若需精确处理，从视频元数据读取帧时戳 |

非均匀时间步长示例（丢帧场景）:
```python
# 帧索引: [10, 11, 12, 14, 15]  (帧13丢失)
frame_indices = segment.all_frame_indices
dt_base = 1.0 / segment.fps
time_steps = np.array([
    (frame_indices[k+1] - frame_indices[k]) * dt_base
    for k in range(len(frame_indices) - 1)
])
# time_steps = [0.033, 0.033, 0.067, 0.033]  (第3个区间跨2帧)
ocp.solver_options.time_steps = time_steps
```

### 4.6.4 积分器配置

提供两种积分器方案，实现时可选择其一。

**方案 A: ERK + RK4（显式 Runge-Kutta 4阶）**

```python
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.sim_method_num_stages = 4      # RK4
ocp.solver_options.sim_method_num_steps = 4        # 每个 shooting 区间细分为 4 个 RK4 子步
```

特点:
- 显式求值，无需 Newton 迭代，每步计算量低
- 条件稳定：需通过 `sim_method_num_steps` 保证高速段精度
- `sim_method_num_steps = 4` 使实际积分步长 = dt/4 ≈ 0.008 s (30fps)，对所有速度段精度充裕

数值精度验证（dt = 0.033 s, `sim_method_num_steps = 4`, 实际步长 ≈ 0.008 s）:

| 初速度 U | 气动时间尺度 τ = 1/(c_d·U) | τ / 实际步长 | 评估 |
|---------|--------------------------|-------------|------|
| 100 m/s (杀球) | 0.046 s | 5.7 | 安全 |
| 50 m/s (平高球) | 0.092 s | 11.5 | 充裕 |
| 20 m/s (吊球) | 0.230 s | 28.8 | 充裕 |

**方案 B: IRK + GL4（隐式 Gauss-Legendre 4阶）**

```python
ocp.solver_options.integrator_type = 'IRK'
ocp.solver_options.sim_method_num_stages = 2      # 2个 Gauss-Legendre 节点 → 4阶精度
ocp.solver_options.sim_method_num_steps = 1        # 无需子步，A-稳定性保证鲁棒
ocp.solver_options.sim_method_newton_iter = 5      # Newton 迭代次数
```

特点:
- A-稳定（无条件稳定），对大步长天然鲁棒，无需手动调节子步数
- 需要 Newton 迭代求解隐式方程，每步计算量较高
- 8 维状态系统的 Newton 迭代开销可忽略（微秒级）

**两种方案对比**:

| | 方案 A: ERK + RK4 | 方案 B: IRK + GL4 |
|---|---|---|
| 精度阶数 | 4 阶 | 4 阶 |
| 稳定性 | 条件稳定（需 substeps） | A-稳定（无条件） |
| 每步计算量 | 低 | 较高（Newton 迭代） |
| 参数调节 | 需设置 `num_steps` | 无需额外调节 |
| 适用场景 | 非刚性 ODE（本问题适用） | 刚性或需要鲁棒性保证 |
| 推荐场景 | 追求求解效率 | 追求配置简洁和鲁棒性 |

两种方案在本问题的 4 阶精度下，轨迹差异远小于测量噪声（像素级），最终估计结果无实质区别。
