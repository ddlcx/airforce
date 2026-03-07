# Module 3: MHE 数学建模

## 1. 增广状态空间模型

### 1.1 增广状态（8 维）

待估参数以零动力学形式并入状态向量：

```
x_aug = [s, z, v_s, v_z,  ψ, c_d, x0w, y0w]
         ├── 物理状态 ──┤  ├── 待估参数 ──┤
              (4维)             (4维)
```

飞行平面 ODE 及符号约定详见 [physics_model.md](physics_model.md)。

### 1.2 过程噪声（4 维）

```
w = [w_s, w_z, w_vs, w_vz]
```

过程噪声以加性形式添加到物理状态动力学方程中。在 acados MHE 框架中，过程噪声分配到 `model.u`（即 OCP 的"控制输入"位置），作为优化变量由求解器确定其最优值。

通过高权重 Q 矩阵惩罚（Q >> R），过程噪声在正常情况下接近零，仅在刚性模型无法完全匹配观测时被激活，用于吸收：
- 投影矩阵 P 的标定误差（Module 2 不完美）
- 飞行平面假设的偏差（实际存在微小侧向运动）
- 简化阻力模型未覆盖的效应（旋转、横截面变化等）

### 1.3 完整动力学方程

```
f_expl(x_aug, w) = [
    v_s + w_s,                                       # ds/dt
    v_z + w_z,                                       # dz/dt
    −c_d · √(v_s² + v_z²) · v_s + w_vs,             # dv_s/dt
    −g − c_d · √(v_s² + v_z²) · v_z + w_vz,         # dv_z/dt
    0,                                                # dψ/dt   (零动力学)
    0,                                                # dc_d/dt (零动力学)
    0,                                                # dx0w/dt (零动力学)
    0                                                 # dy0w/dt (零动力学)
]
```

### 1.4 已知参数（12 维）

```
model.p = P_flat    (投影矩阵 P 展平为 12 个元素)
```

P 在飞行段内为常数，通过 `solver.set(k, "p", P_flat)` 在运行时逐节点设置。

## 2. 测量模型

将增广状态映射到 2D 像素坐标：

```
h(x_aug, P) → [u_px, v_px]                                            (5)

步骤 1: 飞行平面 → 世界坐标
    X_w = x0w + s · cos(ψ)
    Y_w = y0w + s · sin(ψ)
    Z_w = z

步骤 2: 世界坐标 → 齐次像素坐标
    [u_h, v_h, w_h]ᵀ = P · [X_w, Y_w, Z_w, 1]ᵀ

步骤 3: 透视除法
    u_px = u_h / w_h
    v_px = v_h / w_h
```

CasADi 自动微分处理透视除法的 Jacobian。acados 使用 Gauss-Newton 近似处理非线性最小二乘的 Hessian。

## 3. 代价函数

采用 acados `NONLINEAR_LS` 代价类型。

### 3.1 节点 0（含到达代价）

```
cost_y_expr_0 = vertcat(h(x_aug, P),  w,  x_aug)
                        ├── 2维 ──┤  4维   8维      合计 14 维

W_0     = block_diag(R, Q, Q0)                      14 × 14
yref_0  = [obs_u, obs_v,  0,0,0,0,  x̄₀(8维)]       14 维
```

### 3.2 中间节点 k = 1, ..., N-1

```
cost_y_expr = vertcat(h(x_aug, P),  w)
                      ├── 2维 ──┤  4维              合计 6 维

W     = block_diag(R, Q)                             6 × 6
yref  = [obs_u_k, obs_v_k,  0,0,0,0]                6 维
```

### 3.3 终端节点 N

终端节点包含测量代价（观测 obs_N），但无过程噪声项（终端节点无控制输入）。

```
cost_type_e = 'NONLINEAR_LS'
cost_y_expr_e = h(x_aug, P)                    2 维（仅测量）
W_e   = R                                       2 × 2
yref_e = [obs_u_N, obs_v_N]                     2 维
```

### 3.4 代价缩放

```
cost_scaling = np.ones(N + 1)     # MHE 不按时间步长缩放代价
```

acados 默认按时间步长 dt 缩放各节点代价，对 MHE 需覆盖为均匀权重。

## 4. 权重设计策略

权重设计遵循层级原则：**信任物理模型 >> 信任像素检测 >> 容忍过程噪声**。

### R（2x2）: 测量权重

基于 TrackNet 等检测器的像素噪声水平：

```
R = diag(1/σ²_pixel, 1/σ²_pixel)
σ_pixel ≈ 3~5 px → R ≈ diag(0.04 ~ 0.11,  0.04 ~ 0.11)
```

### Q（4x4）: 过程噪声惩罚

```
Q = diag(q_s, q_z, q_vs, q_vz)

位置噪声:  q_s = q_z = 100      (允许 ~0.1 m/step 偏差)
速度噪声:  q_vs = q_vz = 10     (允许 ~0.3 m/s/step 偏差)

Q/R ≈ 100 ~ 1000 → 物理模型可信但允许适度柔性
```

Q/R 比值不宜过大（如 1e4~1e5），否则问题变得极度刚性，SQP 从较差初始猜测出发时难以收敛。实测表明 Q/R ≈ 100~1000 在收敛性和轨迹质量之间取得良好平衡。

### Q0（8x8）: 到达代价（初始猜测置信度）

```
Q0 = diag(q0_s, q0_z, q0_vs, q0_vz, q0_ψ, q0_cd, q0_x0w, q0_y0w)

位置:    q0_s = 0.1,  q0_z = 1.0        (初始 s=0，z 有先验)
速度:    q0_vs = 0.01, q0_vz = 0.01     (速度初始化不确定性大)
方位角:  q0_ψ = 10.0                     (球员位置给出较好先验)
阻力:    q0_cd = 100.0                   (标称值 ~0.217，允许适度调整)
位置:    q0_x0w = 10.0, q0_y0w = 10.0   (球员位置先验)
```

### 与 MonoTrack 损失函数的对应关系

MonoTrack [2, Eq.4] 的损失函数：

```
L_3d = σ · L_r + ‖x(0) − x_H‖² + ‖x(t_R) − x_R‖² + d_O²

其中 σ = ‖P‖₂⁻² 用于平衡像素空间与世界空间的量纲
```

MHE 框架中的对应关系：

| MonoTrack 损失项 | MHE 对应 |
|------------------|----------|
| σ · L_r（重投影误差） | Σ_k ‖h(x_k) − obs_k‖²_R（测量代价分散到各节点） |
| ‖x(0) − x_H‖²（起点先验） | ‖x_aug(0) − x̄₀‖²_Q0（到达代价） |
| ‖x(t_R) − x_R‖²（终点先验） | 可通过终端软约束或额外代价项实现 |
| d_O²（出界惩罚） | 状态约束（球场边界 box constraint） |

## 5. 约束

```
状态约束 (box constraints, 通过 lbx / ubx 设置):

    z ≥ 0                             # 高度非负
    0 ≤ z(0) ≤ 3.0                   # MonoTrack [2]: 击球高度约束
    0.05 ≤ c_d ≤ 0.50               # 物理范围
                                      #   标称 c_d ≈ 0.217
                                      #   对应 C_D ∈ [0.15, 1.49]
                                      #   覆盖 Cohen [1] 全部实验范围
    −π ≤ ψ ≤ π                       # 方位角
    x0w ∈ 击球方半场范围              # MonoTrack [2]: 初始位置约束
    y0w ∈ 击球方半场范围              #   近端球员: 0 ≤ y0w ≤ 6.7
                                      #   远端球员: 6.7 ≤ y0w ≤ 13.4
```

速度约束 ‖v₀‖₂ ≤ 120 m/s（MonoTrack [2]）通过初始化和到达代价 Q0 隐式处理，而非显式非线性约束，以避免增加 NLP 复杂度。

## 6. 缺失检测处理

对未检测到羽毛球的帧，将该节点的测量权重 R 设为近零矩阵：

```
solver.cost_set(k, "W", block_diag(R_zero, Q))

R_zero = diag(ε, ε),   ε ≈ 1e-10
```

ODE 动力学仍在缺失帧间传播状态，但不产生测量代价。这使得动力学模型"桥接"缺失检测区间。

acados 求解器配置的具体实现详见 [acados_implementation.md](acados_implementation.md)。

## 参考文献

1. Cohen C, Darbois Texier B, Quéré D, Clanet C. "The physics of badminton." *New Journal of Physics*, 17(6):063001, 2015.
2. Liu P, Wang J-H. "MonoTrack: Shuttle trajectory reconstruction from monocular badminton video." *arXiv:2204.01899v2*, 2022.
