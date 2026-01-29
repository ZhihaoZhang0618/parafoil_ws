# Parafoil Advanced Path Planning Module — Claude Input README

> **目标**：基于已有 parafoil 6-DOF 仿真器，新建一个**高级轨迹规划模块**，采用业界前沿方法：
>
> - **Gaussian 伪谱法（GPM）**：将轨迹优化转化为非线性规划问题
> - **6-DOF 动力学模型**：使用仿真器真值 + 噪声作为状态估计
> - **分阶段制导**：巡航（Cruise）→ 进场（Approach）→ 拉飘（Flare）
> - **预计算轨迹库**：离线计算多种着陆轨迹，在线匹配
>
> 你可以把本文件全文直接喂给 Claude 让它生成新模块。

---

## Claude Prompt（请严格照做）

你是资深飞行器轨迹优化与最优控制工程师，熟悉 Gaussian 伪谱法、6-DOF 动力学建模、MPC 以及精确空投系统（如 JPADS）的制导算法。

请基于一个已有的 parafoil 6-DOF 仿真器（ROS2 + Python），新增一个**高级轨迹规划模块**，实现从任意初始状态到目标着陆点的最优轨迹规划与跟踪。

---

## 1. 核心技术要求

### 1.1 Gaussian 伪谱法（GPM）轨迹优化

采用 Gaussian 伪谱法将连续时间最优控制问题离散化为非线性规划（NLP）问题：

**连续时间问题**：

```
min J = Φ(x(tf), tf) + ∫[t0,tf] L(x(t), u(t), t) dt

s.t.  ẋ = f(x, u, t)           # 6-DOF 动力学
      g(x, u) ≤ 0              # 路径约束
      ψ(x(t0), x(tf)) = 0      # 边界条件
```

**GPM 离散化**：

- 使用 Legendre-Gauss (LG) 或 Legendre-Gauss-Lobatto (LGL) 配点
- 状态和控制在配点处参数化
- 动力学约束转化为代数约束（配点处导数匹配）
- 积分使用 Gauss 求积公式

**实现要求**：

1. 支持可配置的配点数量（N = 10~50）
2. 使用 `scipy.optimize` 或 `casadi` 求解 NLP
3. 支持热启动（warm start）加速求解
4. 提供求解状态监控（收敛性、约束违反量）

### 1.2 6-DOF 动力学模型

规划器必须使用完整的 6-DOF 翼伞动力学模型：

**状态向量**（13维）：

```
x = [p_N, p_E, p_D,           # 位置 (NED)
     v_N, v_E, v_D,           # 速度 (NED)
     q_w, q_x, q_y, q_z,      # 姿态四元数
     ω_x, ω_y, ω_z]           # 角速度 (body)
```

**控制输入**（2维）：

```
u = [δ_L, δ_R]                # 左右刹车 [0, 1]
```

**动力学方程**：

```
ṗ = v
v̇ = (1/m)(F_aero + F_gravity + F_payload)
q̇ = 0.5 * q ⊗ [0, ω]
ω̇ = J⁻¹(τ_aero - ω × Jω)
```

**状态获取**：

- 从仿真器 `/parafoil/odom` 和 `/parafoil/imu` 获取真值
- 添加可配置的高斯噪声模拟传感器误差：
  ```yaml
  noise:
    position_sigma: [0.5, 0.5, 0.3]    # m
    velocity_sigma: [0.1, 0.1, 0.05]   # m/s
    attitude_sigma: [0.01, 0.01, 0.02] # rad
    angular_rate_sigma: [0.005, 0.005, 0.01] # rad/s
  ```

### 1.3 分阶段制导架构

实现三阶段制导系统：

```
┌─────────────────────────────────────────────────────────────────┐
│                     CRUISE PHASE（巡航阶段）                     │
│  目标：消耗多余高度，调整位置至进场点                             │
│  策略：                                                          │
│    - 若高度过剩：执行 racetrack/S-turn 盘旋                      │
│    - 使用 GPM 优化能量效率                                       │
│    - 轨迹库匹配选择最优模式                                      │
│  退出条件：                                                      │
│    - 到达 Entry Point (距目标 D_entry = 150~200m)                │
│    - 高度预算满足 Approach 需求                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APPROACH PHASE（进场阶段）                    │
│  目标：精确对准目标，建立稳定下滑道                               │
│  策略：                                                          │
│    - 迎风直线进场（Final Leg）                                   │
│    - GPM 优化下滑剖面（glide slope）                             │
│    - 精确控制航迹角和下降率                                      │
│  退出条件：                                                      │
│    - 高度 ≤ H_flare (通常 5~15m AGL)                             │
│    - 距目标 ≤ D_flare (通常 20~40m)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FLARE PHASE（拉飘阶段）                      │
│  目标：最大化减速，实现软着陆                                     │
│  策略：                                                          │
│    - 全刹车（δ_L = δ_R = 1.0）                                  │
│    - 利用气动阻力快速减小垂直速度                                │
│    - 持续时间 2~5 秒                                             │
│  退出条件：                                                      │
│    - 触地（h ≤ 0）                                               │
│    - 或 timeout                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 预计算轨迹库

**离线阶段**：预计算覆盖典型场景的轨迹库

```python
TrajectoryLibrary:
  scenarios:
    - wind_speed: [0, 2, 4, 6, 8] m/s
    - wind_direction: [0, 45, 90, 135, 180, 225, 270, 315] deg
    - initial_altitude: [15, 30, 50, 80, 120] m
    - target_distance: [50, 100, 150, 200, 250] m
    - target_bearing: [0, 90, 180, 270] deg

  trajectory_types:
    - DIRECT: 直飞进场
    - S_TURN: S弯消高
    - RACETRACK: 跑道盘旋
    - SPIRAL: 螺旋下降

  storage:
    format: HDF5 or pickle
    content_per_trajectory:
      - waypoints: [(t, x, y, z, vx, vy, vz, q, ω, δL, δR), ...]
      - metadata: {cost, duration, altitude_loss, max_bank, ...}
      - boundary_conditions: {x0, xf, wind, ...}
```

**在线阶段**：快速匹配与插值

```python
def online_trajectory_selection(current_state, target, wind):
    # 1. 计算场景特征向量
    features = compute_scenario_features(current_state, target, wind)

    # 2. K近邻搜索（KD-Tree）
    k_nearest = library.query_knn(features, k=5)

    # 3. 可行性检查
    feasible = filter_by_constraints(k_nearest, current_state)

    # 4. 代价评估与选择
    best = min(feasible, key=lambda t: evaluate_cost(t, current_state))

    # 5. 插值调整（可选：使用 GPM 微调）
    adapted = adapt_trajectory(best, current_state, target, wind)

    return adapted
```

---

## 2. 模块交付形式

创建新的 ROS2 package：**`parafoil_planner_v3`**

### 2.1 目录结构

```
parafoil_planner_v3/
├── package.xml
├── setup.py
├── setup.cfg
├── parafoil_planner_v3/
│   ├── __init__.py
│   │
│   ├── # === 数据结构 ===
│   ├── types.py                    # State, Control, Trajectory, Phase, etc.
│   │
│   ├── # === 动力学模型 ===
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── parafoil_6dof.py        # 完整 6-DOF 动力学
│   │   ├── aerodynamics.py         # 气动力/力矩模型
│   │   └── simplified_model.py     # 用于快速规划的简化模型
│   │
│   ├── # === GPM 优化器 ===
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── gpm_collocation.py      # Gaussian 伪谱配点
│   │   ├── nlp_formulation.py      # NLP 问题构建
│   │   ├── cost_functions.py       # 代价函数库
│   │   ├── constraints.py          # 约束函数库
│   │   └── solver_interface.py     # scipy/casadi 接口
│   │
│   ├── # === 轨迹库 ===
│   ├── trajectory_library/
│   │   ├── __init__.py
│   │   ├── library_generator.py    # 离线生成器
│   │   ├── library_manager.py      # 库管理与查询
│   │   ├── trajectory_adapter.py   # 在线适配器
│   │   └── scenario_features.py    # 特征提取
│   │
│   ├── # === 分阶段制导 ===
│   ├── guidance/
│   │   ├── __init__.py
│   │   ├── phase_manager.py        # 阶段状态机
│   │   ├── cruise_guidance.py      # 巡航制导
│   │   ├── approach_guidance.py    # 进场制导
│   │   ├── flare_guidance.py       # 拉飘制导
│   │   └── phase_transitions.py    # 阶段切换逻辑
│   │
│   ├── # === 核心规划器 ===
│   ├── planner_core.py             # 主规划算法（集成上述模块）
│   │
│   ├── # === ROS2 节点 ===
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── planner_node.py         # 在线规划节点
│   │   ├── library_server_node.py  # 轨迹库服务节点
│   │   └── guidance_node.py        # 制导执行节点
│   │
│   └── # === 工具 ===
│   ├── utils/
│       ├── __init__.py
│       ├── coordinate_transforms.py
│       ├── quaternion_utils.py
│       └── interpolation.py
│
├── config/
│   ├── planner_params.yaml         # 规划器参数
│   ├── dynamics_params.yaml        # 动力学参数
│   ├── gpm_params.yaml             # GPM 优化参数
│   └── library_params.yaml         # 轨迹库参数
│
├── launch/
│   ├── planner.launch.py           # 单独规划器
│   ├── full_system.launch.py       # 完整系统
│   └── library_generation.launch.py # 离线库生成
│
├── library/                        # 预计算轨迹库存储
│   └── .gitkeep
│
├── scripts/
│   ├── generate_library.py         # 离线生成脚本
│   ├── validate_library.py         # 库验证脚本
│   └── benchmark_solver.py         # 求解器性能测试
│
└── tests/
    ├── test_gpm_collocation.py
    ├── test_dynamics.py
    ├── test_phase_transitions.py
    ├── test_trajectory_library.py
    └── test_integration.py
```

---

## 3. GPM 轨迹优化详细规格

### 3.1 问题形式化

**最优控制问题**：

```
最小化：
  J = w_f * ||p(tf) - p_target||²
    + w_v * ||v(tf)||²
    + ∫[t0,tf] (w_u * ||u||² + w_ω * ||ω̇||²) dt

约束：
  动力学：ẋ = f_6dof(x, u, wind)

  控制限制：
    0 ≤ δ_L ≤ 1
    0 ≤ δ_R ≤ 1
    |δ̇_L|, |δ̇_R| ≤ δ_rate_max

  状态限制：
    V_min ≤ ||v_horizontal|| ≤ V_max
    |φ| ≤ φ_max (滚转角限制)
    |ω_z| ≤ ω_max (转弯率限制)

  路径约束：
    h(t) ≥ h_terrain(x, y) + h_clearance
    (x, y) ∈ allowed_region

  边界条件：
    x(t0) = x_current (初始状态)
    p(tf) = p_target (终端位置)
    迎风进场（可选）：ψ(tf) = ψ_headwind ± ε
```

### 3.2 GPM 配点实现

```python
class GPMCollocation:
    """Gaussian Pseudospectral Method 配点类"""

    def __init__(self, N: int, scheme: str = "LG"):
        """
        Args:
            N: 配点数量
            scheme: "LG" (Legendre-Gauss) 或 "LGL" (Legendre-Gauss-Lobatto)
        """
        self.N = N
        self.tau, self.weights = self._compute_nodes_and_weights(scheme)
        self.D = self._compute_differentiation_matrix()

    def _compute_nodes_and_weights(self, scheme):
        """计算配点位置 τ ∈ [-1, 1] 和积分权重"""
        if scheme == "LG":
            # Legendre-Gauss nodes (不包含端点)
            tau, weights = np.polynomial.legendre.leggauss(self.N)
        elif scheme == "LGL":
            # Legendre-Gauss-Lobatto nodes (包含端点)
            tau = self._lgl_nodes(self.N)
            weights = self._lgl_weights(tau)
        return tau, weights

    def _compute_differentiation_matrix(self):
        """计算 Lagrange 多项式微分矩阵 D"""
        # D[i,j] = dL_j(τ_i) / dτ
        # 用于将配点处函数值转换为导数值
        ...

    def discretize_dynamics(self, f, x_nodes, u_nodes, t0, tf):
        """
        将动力学约束离散化为代数约束

        ẋ(τ_k) ≈ (2/(tf-t0)) * Σ_j D[k,j] * x_j = f(x_k, u_k, t_k)
        """
        dt = (tf - t0) / 2
        defects = []
        for k in range(self.N):
            x_dot_approx = (1/dt) * self.D[k, :] @ x_nodes
            x_dot_exact = f(x_nodes[k], u_nodes[k], self._tau_to_t(self.tau[k], t0, tf))
            defects.append(x_dot_approx - x_dot_exact)
        return np.concatenate(defects)

    def integrate_cost(self, L, x_nodes, u_nodes, t0, tf):
        """使用 Gauss 求积计算积分代价"""
        dt = (tf - t0) / 2
        running_cost = 0
        for k in range(self.N):
            t_k = self._tau_to_t(self.tau[k], t0, tf)
            running_cost += self.weights[k] * L(x_nodes[k], u_nodes[k], t_k)
        return dt * running_cost
```

### 3.3 NLP 求解器接口

```python
class GPMSolver:
    """GPM 优化求解器"""

    def __init__(self, dynamics, cost, constraints, gpm: GPMCollocation):
        self.dynamics = dynamics
        self.cost = cost
        self.constraints = constraints
        self.gpm = gpm

    def solve(self, x0, x_target, tf_guess, warm_start=None):
        """
        求解最优控制问题

        Returns:
            solution: OptimalTrajectory
            info: SolverInfo (iterations, cost, constraint_violation, etc.)
        """
        # 决策变量：[x_nodes (N×n_x), u_nodes (N×n_u), tf]
        n_vars = self.gpm.N * (self.n_x + self.n_u) + 1

        # 初始猜测
        if warm_start is not None:
            z0 = warm_start
        else:
            z0 = self._generate_initial_guess(x0, x_target, tf_guess)

        # 约束
        constraints = [
            {'type': 'eq', 'fun': self._dynamics_constraint},
            {'type': 'eq', 'fun': self._boundary_constraint, 'args': (x0, x_target)},
            {'type': 'ineq', 'fun': self._path_constraint},
        ]

        # 变量边界
        bounds = self._compute_bounds()

        # 求解
        result = scipy.optimize.minimize(
            self._objective,
            z0,
            method='SLSQP',  # 或 'trust-constr'
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6}
        )

        return self._extract_trajectory(result.x), self._build_info(result)
```

---

## 4. 分阶段制导详细规格

### 4.1 阶段状态机

```python
class GuidancePhase(Enum):
    CRUISE = "cruise"           # 巡航（高度管理）
    APPROACH = "approach"       # 进场（精确导引）
    FLARE = "flare"            # 拉飘（减速着陆）
    LANDED = "landed"          # 已着陆
    ABORT = "abort"            # 中止

class PhaseManager:
    """分阶段制导状态机"""

    def __init__(self, config: PhaseConfig):
        self.config = config
        self.current_phase = GuidancePhase.CRUISE
        self.phase_start_time = None
        self.phase_start_state = None

    def update(self, state: State, target: Target, wind: Wind) -> PhaseTransition:
        """每个控制周期调用，检查是否需要切换阶段"""

        transition = PhaseTransition(
            from_phase=self.current_phase,
            to_phase=self.current_phase,
            triggered=False
        )

        if self.current_phase == GuidancePhase.CRUISE:
            if self._should_enter_approach(state, target, wind):
                transition = self._transition_to(GuidancePhase.APPROACH, state)

        elif self.current_phase == GuidancePhase.APPROACH:
            if self._should_enter_flare(state, target):
                transition = self._transition_to(GuidancePhase.FLARE, state)
            elif self._should_abort(state, target, wind):
                transition = self._transition_to(GuidancePhase.ABORT, state)

        elif self.current_phase == GuidancePhase.FLARE:
            if self._has_landed(state):
                transition = self._transition_to(GuidancePhase.LANDED, state)

        return transition

    def _should_enter_approach(self, state, target, wind) -> bool:
        """进场条件判断"""
        distance_to_target = np.linalg.norm(state.position_xy - target.position_xy)
        altitude_agl = state.altitude - target.altitude

        # 条件1：距离小于进场距离
        distance_ok = distance_to_target <= self.config.approach_entry_distance

        # 条件2：高度预算匹配
        required_altitude = self._estimate_approach_altitude(distance_to_target, wind)
        altitude_ok = altitude_agl <= required_altitude * self.config.altitude_margin

        return distance_ok and altitude_ok

    def _should_enter_flare(self, state, target) -> bool:
        """拉飘条件判断"""
        altitude_agl = state.altitude - target.altitude
        distance = np.linalg.norm(state.position_xy - target.position_xy)

        # 高度触发 OR 距离触发
        altitude_trigger = altitude_agl <= self.config.flare_altitude
        distance_trigger = distance <= self.config.flare_distance

        return altitude_trigger or distance_trigger
```

### 4.2 各阶段制导器

#### 巡航制导器（Cruise Guidance）

```python
class CruiseGuidance:
    """
    巡航阶段制导
    目标：管理高度，调整位置至进场点
    """

    def __init__(self, config, trajectory_library):
        self.config = config
        self.library = trajectory_library
        self.gpm_solver = GPMSolver(...)  # 用于在线微调

    def compute_control(self, state, target, wind, dt) -> ControlCommand:
        # 1. 评估当前状态与目标的关系
        altitude_excess = self._compute_altitude_excess(state, target, wind)

        if altitude_excess > self.config.hold_threshold:
            # 高度过剩：执行盘旋消高
            return self._hold_pattern_control(state, target, wind)
        else:
            # 高度合适：向进场点飞行
            return self._direct_to_entry_control(state, target, wind)

    def _hold_pattern_control(self, state, target, wind):
        """生成盘旋模式控制"""
        # 从轨迹库选择最优盘旋模式
        pattern = self.library.select_hold_pattern(state, target, wind)

        # 跟踪盘旋轨迹
        track_point = pattern.get_tracking_point(state)
        return self._track_waypoint(state, track_point, wind)

    def _direct_to_entry_control(self, state, target, wind):
        """直飞进场点控制"""
        entry_point = self._compute_entry_point(target, wind)
        return self._track_waypoint(state, entry_point, wind)
```

#### 进场制导器（Approach Guidance）

```python
class ApproachGuidance:
    """
    进场阶段制导
    目标：精确对准目标，建立稳定下滑道
    """

    def __init__(self, config):
        self.config = config
        self.glide_slope_controller = GlideSlopeController(config)
        self.lateral_controller = LateralController(config)

    def compute_control(self, state, target, wind, dt) -> ControlCommand:
        # 1. 计算期望下滑道
        desired_glide_path = self._compute_glide_path(state, target, wind)

        # 2. 纵向控制：跟踪下滑道
        vertical_control = self.glide_slope_controller.compute(
            state, desired_glide_path, wind
        )

        # 3. 横向控制：航向对准目标（迎风）
        headwind_heading = self._compute_headwind_heading(wind)
        lateral_control = self.lateral_controller.compute(
            state, target, headwind_heading
        )

        # 4. 混合控制
        return self._blend_controls(vertical_control, lateral_control)

    def _compute_glide_path(self, state, target, wind):
        """计算理想下滑道"""
        distance = np.linalg.norm(state.position_xy - target.position_xy)
        altitude_to_go = state.altitude - target.altitude - self.config.flare_altitude

        # 期望下滑角
        desired_glide_angle = np.arctan2(altitude_to_go, distance)

        # 考虑风修正
        wind_correction = self._compute_wind_correction(wind, desired_glide_angle)

        return GlidePath(
            angle=desired_glide_angle + wind_correction,
            intercept_altitude=state.altitude
        )
```

#### 拉飘制导器（Flare Guidance）

```python
class FlareGuidance:
    """
    拉飘阶段制导
    目标：最大化减速，实现软着陆

    关键策略：
    - 在着陆前 2-5 秒执行全刹车
    - 利用气动阻力最大化减小垂直速度
    - 着陆时垂直速度应 < 2 m/s
    """

    def __init__(self, config):
        self.config = config
        self.flare_start_time = None
        self.flare_start_altitude = None

    def initialize(self, state):
        """拉飘开始时调用"""
        self.flare_start_time = state.timestamp
        self.flare_start_altitude = state.altitude

    def compute_control(self, state, target, dt) -> ControlCommand:
        # 拉飘策略：渐进式全刹车
        time_since_flare = state.timestamp - self.flare_start_time

        # 阶段1：快速增加刹车（0.5秒内）
        if time_since_flare < self.config.flare_ramp_time:
            brake_command = self._ramp_up_brake(time_since_flare)
        else:
            # 阶段2：保持全刹车
            brake_command = self.config.flare_max_brake  # 通常 = 1.0

        # 保持方向稳定（小差动）
        heading_correction = self._compute_heading_correction(state, target)

        delta_L = brake_command + heading_correction
        delta_R = brake_command - heading_correction

        return ControlCommand(
            delta_L=np.clip(delta_L, 0, 1),
            delta_R=np.clip(delta_R, 0, 1)
        )

    def _ramp_up_brake(self, t):
        """平滑增加刹车量"""
        # 使用 S 曲线避免突变
        progress = t / self.config.flare_ramp_time
        # Smoothstep: 3t² - 2t³
        smooth = 3 * progress**2 - 2 * progress**3
        return self.config.flare_initial_brake + \
               (self.config.flare_max_brake - self.config.flare_initial_brake) * smooth
```

### 4.3 拉飘效果分析

```python
class FlareAnalysis:
    """拉飘机动效果分析"""

    @staticmethod
    def compute_flare_benefit(polar_table, initial_state, flare_duration):
        """
        计算拉飘机动的减速效果

        典型数值（基于 parafoil 极曲线）：
        - 无拉飘：着陆垂直速度 ≈ 1.0-1.3 m/s
        - 有拉飘：着陆垂直速度 ≈ 0.3-0.7 m/s（减少 50-70%）

        原理：
        - 全刹车大幅增加迎角和阻力
        - 虽然下沉率瞬时增加，但动能快速转化为高度
        - "zoom" 效应：短暂爬升或减速，然后缓降
        """
        # 获取全刹车气动参数
        full_brake = 1.0
        airspeed, sink_rate = polar_table.interpolate(full_brake)

        # 简化分析：能量守恒
        # 初始动能 + 初始势能 = 最终动能 + 最终势能 + 阻力功

        initial_ke = 0.5 * initial_state.velocity_magnitude**2
        initial_pe = 9.81 * initial_state.altitude

        # 拉飘期间的减速
        # 使用 6-DOF 仿真进行精确计算
        final_velocity = simulate_flare(initial_state, flare_duration)

        return FlareResult(
            initial_sink_rate=initial_state.velocity_d,
            final_sink_rate=final_velocity.z,
            velocity_reduction_percent=(1 - final_velocity.z / initial_state.velocity_d) * 100
        )
```

---

## 5. 预计算轨迹库详细规格

### 5.1 离线生成器

```python
class TrajectoryLibraryGenerator:
    """离线轨迹库生成器"""

    def __init__(self, dynamics, gpm_config, scenario_config):
        self.dynamics = dynamics
        self.gpm = GPMCollocation(**gpm_config)
        self.scenario_config = scenario_config

    def generate_library(self, output_path: str):
        """生成完整轨迹库"""
        scenarios = self._enumerate_scenarios()
        trajectories = []

        for scenario in tqdm(scenarios, desc="Generating trajectories"):
            try:
                # 对每种轨迹类型求解
                for traj_type in TrajectoryType:
                    traj = self._solve_scenario(scenario, traj_type)
                    if traj.is_feasible:
                        trajectories.append(traj)
            except SolverException as e:
                logger.warning(f"Failed to solve scenario {scenario}: {e}")

        # 构建索引
        library = TrajectoryLibrary(trajectories)
        library.build_index()

        # 保存
        library.save(output_path)

        return library

    def _enumerate_scenarios(self) -> List[Scenario]:
        """枚举所有场景组合"""
        scenarios = []

        for wind_speed in self.scenario_config.wind_speeds:
            for wind_dir in self.scenario_config.wind_directions:
                for altitude in self.scenario_config.altitudes:
                    for distance in self.scenario_config.distances:
                        for bearing in self.scenario_config.bearings:
                            scenarios.append(Scenario(
                                wind_speed=wind_speed,
                                wind_direction=wind_dir,
                                initial_altitude=altitude,
                                target_distance=distance,
                                target_bearing=bearing
                            ))

        return scenarios

    def _solve_scenario(self, scenario: Scenario, traj_type: TrajectoryType):
        """为特定场景求解最优轨迹"""
        # 构建初始状态
        x0 = self._scenario_to_initial_state(scenario)
        target = self._scenario_to_target(scenario)
        wind = self._scenario_to_wind(scenario)

        # 根据轨迹类型设置约束
        constraints = self._get_trajectory_constraints(traj_type)

        # GPM 求解
        tf_guess = self._estimate_flight_time(scenario)
        trajectory, info = self.gpm_solver.solve(x0, target, tf_guess)

        return LibraryTrajectory(
            scenario=scenario,
            trajectory_type=traj_type,
            waypoints=trajectory.waypoints,
            controls=trajectory.controls,
            cost=info.cost,
            metadata=self._compute_metadata(trajectory)
        )
```

### 5.2 在线匹配器

```python
class TrajectoryMatcher:
    """在线轨迹匹配与适配"""

    def __init__(self, library: TrajectoryLibrary, config):
        self.library = library
        self.config = config
        self.kdtree = library.build_kdtree()

    def select_trajectory(self, state, target, wind) -> AdaptedTrajectory:
        """选择并适配最优轨迹"""

        # 1. 特征提取
        features = self._extract_features(state, target, wind)

        # 2. KNN 搜索
        distances, indices = self.kdtree.query(features, k=self.config.k_neighbors)
        candidates = [self.library[i] for i in indices]

        # 3. 可行性过滤
        feasible = []
        for traj in candidates:
            if self._check_feasibility(traj, state, target, wind):
                feasible.append(traj)

        if not feasible:
            # 无可行轨迹：需要在线求解
            return self._solve_online(state, target, wind)

        # 4. 代价评估
        best = min(feasible, key=lambda t: self._evaluate_cost(t, state, target, wind))

        # 5. 轨迹适配
        adapted = self._adapt_trajectory(best, state, target, wind)

        return adapted

    def _extract_features(self, state, target, wind):
        """提取场景特征向量用于 KNN 搜索"""
        # 相对位置
        rel_pos = target.position - state.position[:2]
        distance = np.linalg.norm(rel_pos)
        bearing = np.arctan2(rel_pos[1], rel_pos[0])

        # 风相对于目标方向
        wind_angle = np.arctan2(wind.direction[1], wind.direction[0])
        relative_wind_angle = wind_angle - bearing

        return np.array([
            state.altitude,
            distance,
            bearing,
            wind.speed,
            relative_wind_angle
        ])

    def _adapt_trajectory(self, library_traj, state, target, wind):
        """
        将库中轨迹适配到当前状态

        方法：
        1. 时间/空间缩放
        2. 旋转对齐
        3. 可选：GPM 微调（如果偏差较大）
        """
        # 计算变换参数
        scale = self._compute_scale(library_traj.scenario, state, target)
        rotation = self._compute_rotation(library_traj.scenario, state, target)
        translation = state.position[:2]

        # 应用变换
        adapted_waypoints = []
        for wp in library_traj.waypoints:
            pos_transformed = rotation @ (scale * wp.position[:2]) + translation
            adapted_waypoints.append(Waypoint(
                position=np.array([pos_transformed[0], pos_transformed[1], wp.position[2] * scale[2]]),
                velocity=rotation @ (scale * wp.velocity[:2]),
                time=wp.time * scale_time
            ))

        # 验证适配后的可行性
        if not self._verify_adapted_trajectory(adapted_waypoints, state, target, wind):
            # 需要 GPM 微调
            return self._fine_tune_with_gpm(adapted_waypoints, state, target, wind)

        return AdaptedTrajectory(
            waypoints=adapted_waypoints,
            controls=library_traj.controls,  # 控制也需要适配
            source_trajectory=library_traj
        )
```

### 5.3 轨迹库参数配置

```yaml
# library_params.yaml

generation:
  # 场景采样
  wind_speeds: [0.0, 2.0, 4.0, 6.0, 8.0]  # m/s
  wind_directions: [0, 45, 90, 135, 180, 225, 270, 315]  # deg
  initial_altitudes: [100, 150, 200, 250, 300]  # m
  target_distances: [50, 100, 150, 200, 250]  # m
  target_bearings: [0, 90, 180, 270]  # deg

  # 轨迹类型
  trajectory_types:
    - DIRECT
    - S_TURN
    - RACETRACK
    - SPIRAL

  # GPM 配置
  gpm:
    num_nodes: 30
    scheme: "LGL"
    max_iterations: 500
    tolerance: 1e-6

  # 并行计算
  num_workers: 8

matching:
  k_neighbors: 5
  feasibility_margin: 0.1

  # 特征权重（用于 KNN 距离）
  feature_weights:
    altitude: 0.2
    distance: 0.25
    bearing: 0.15
    wind_speed: 0.2
    wind_angle: 0.2

adaptation:
  max_scale_deviation: 0.3      # 最大缩放偏差
  max_rotation_deviation: 30.0  # deg
  enable_gpm_fine_tuning: true
  fine_tuning_tolerance: 5.0    # m，触发微调的偏差阈值
```

---

## 6. ROS2 接口

### 6.1 话题订阅


| 话题             | 类型                           | 说明                           |
| ---------------- | ------------------------------ | ------------------------------ |
| `/parafoil/odom` | `nav_msgs/Odometry`            | 6-DOF 状态（位置、速度、姿态） |
| `/parafoil/imu`  | `sensor_msgs/Imu`              | 角速度、加速度                 |
| `/wind_estimate` | `geometry_msgs/Vector3Stamped` | 风估计（NED）                  |
| `/target`        | `geometry_msgs/PoseStamped`    | 目标着陆点                     |

### 6.2 话题发布


| 话题                  | 类型                             | 说明                   |
| --------------------- | -------------------------------- | ---------------------- |
| `/planned_trajectory` | `nav_msgs/Path`                  | 完整轨迹（3D）         |
| `/control_command`    | `geometry_msgs/Vector3Stamped`   | 刹车指令 [δL, δR, 0] |
| `/guidance_phase`     | `std_msgs/String`                | 当前制导阶段           |
| `/planner_status`     | `std_msgs/String`                | 规划器状态与调试信息   |
| `/trajectory_preview` | `visualization_msgs/MarkerArray` | RViz 可视化            |

### 6.3 服务


| 服务             | 类型                         | 说明       |
| ---------------- | ---------------------------- | ---------- |
| `/replan`        | `std_srvs/Trigger`           | 强制重规划 |
| `/set_target`    | `parafoil_msgs/SetTarget`    | 设置新目标 |
| `/query_library` | `parafoil_msgs/QueryLibrary` | 查询轨迹库 |

### 6.4 参数

所有参数集中在 `config/*.yaml`，通过 launch 文件加载。

---

## 7. 测试要求

### 7.1 单元测试（pytest）

```python
# test_gpm_collocation.py
def test_lg_nodes_weights():
    """验证 Legendre-Gauss 节点和权重正确性"""

def test_differentiation_matrix():
    """验证微分矩阵精度"""

def test_dynamics_discretization():
    """验证动力学离散化精度（与 RK4 对比）"""

# test_dynamics.py
def test_6dof_trim_state():
    """验证配平状态下的动力学"""

def test_turn_response():
    """验证转弯响应"""

def test_flare_dynamics():
    """验证拉飘期间的动力学行为"""

# test_phase_transitions.py
def test_cruise_to_approach_transition():
    """验证巡航到进场的切换条件"""

def test_approach_to_flare_transition():
    """验证进场到拉飘的切换条件"""

def test_flare_to_landed_transition():
    """验证拉飘到着陆的切换条件"""

# test_trajectory_library.py
def test_library_generation():
    """验证轨迹库生成"""

def test_trajectory_matching():
    """验证轨迹匹配精度"""

def test_trajectory_adaptation():
    """验证轨迹适配正确性"""

# test_integration.py
def test_full_mission_no_wind():
    """无风条件下完整任务测试"""

def test_full_mission_with_wind():
    """有风条件下完整任务测试"""

def test_flare_effectiveness():
    """验证拉飘减速效果"""
```

### 7.2 集成测试

```bash
# 1. 启动仿真器
ros2 launch parafoil_simulator_ros sim.launch.py

# 2. 启动规划器（使用预计算库）
ros2 launch parafoil_planner_v3 full_system.launch.py use_library:=true

# 3. 设置目标并观察
ros2 topic pub /target geometry_msgs/PoseStamped "{pose: {position: {x: 150, y: 50, z: 0}}}"

# 4. 验证着陆精度
# - 落点误差 < 10m（CEP）
# - 着陆垂直速度 < 2 m/s（有拉飘）
```

---

## 8. 输出要求

- 输出完整文件树 + 每个文件完整代码（不要省略）
- 代码清晰、可运行、无多余功能
- 必须提供：
  - `colcon build --symlink-install` 构建命令
  - 离线库生成脚本
  - 完整系统启动命令
  - 单元测试命令

---

## 9. 性能指标

### 9.1 规划性能


| 指标                 | 目标    |
| -------------------- | ------- |
| GPM 求解时间（N=30） | < 1.0 s |
| 轨迹库匹配时间       | < 10 ms |
| 轨迹适配时间         | < 50 ms |
| 重规划频率           | 1-5 Hz  |

### 9.2 着陆精度


| 指标                   | 目标      |
| ---------------------- | --------- |
| 落点 CEP（无风）       | < 5 m     |
| 落点 CEP（4m/s 风）    | < 10 m    |
| 着陆垂直速度（有拉飘） | < 2 m/s   |
| 着陆垂直速度（无拉飘） | < 1.3 m/s |

### 9.3 拉飘效果


| 指标             | 预期   |
| ---------------- | ------ |
| 拉飘持续时间     | 2-5 s  |
| 垂直速度降低比例 | 30-70% |
| 动能降低比例     | 50-80% |

---

## 10. 验证与测试框架

采用**分层验证架构**，独立评估 Planner 和 Controller 的性能，便于问题定位和调试。

> **⚠️ 重要：执行模式选择**
>
> 验证可在两种模式下运行（详见 11.0 节）：
>
> - **OFFLINE 模式**：纯 Python 仿真，速度快，适合 Level 1-3 的开发迭代和 Level 4 的批量统计
> - **ROS2 模式**：完整系统，速度慢，适合 Level 3 的最终集成验证和部署前测试
>
> 推荐工作流：OFFLINE 开发调试 → OFFLINE 蒙特卡洛验证 → ROS2 最终确认

### 10.1 验证层次

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Level 1: 单元测试                                              [OFFLINE]  │
│    - GPM 配点精度、动力学模型、阶段切换逻辑                                  │
│    - pytest 运行，CI/CD 自动化                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Level 2: 模块测试                                              [OFFLINE]  │
│    - Planner 开环验证：假设完美跟踪，评估规划质量                            │
│    - Controller 跟踪测试：给定标准轨迹，评估跟踪误差                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Level 3: 集成测试（闭环仿真）                        [OFFLINE 或 ROS2]     │
│    - Planner + Controller 联合运行，端到端验证                              │
│    - 开发阶段用 OFFLINE，最终验证用 ROS2                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Level 4: 鲁棒性测试（蒙特卡洛）                      [OFFLINE（推荐）]      │
│    - 随机场景批量测试，统计性能指标                                          │
│    - N > 100 时必须用 OFFLINE（可并行，速度快 20x）                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Level 5: 部署前验证                                              [ROS2]   │
│    - 完整系统测试，验证节点通信和时序                                        │
│    - HIL/SIL 测试（可选）                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Planner 验证（开环）

**目的**：评估规划质量，假设 Controller 能完美跟踪。

```python
# verification/verify_planner.py

class PlannerVerifier:
    """规划器开环验证"""

    def __init__(self, dynamics_model):
        self.dynamics = dynamics_model

    def verify_trajectory(self, trajectory, initial_state, target, wind):
        """
        验证规划轨迹质量

        Returns:
            results: 包含可行性、最优性、终端精度等指标
        """
        results = {
            'feasibility': {},
            'optimality': {},
            'terminal_accuracy': {}
        }

        # 1. 动力学可行性：规划轨迹是否满足动力学约束
        results['feasibility']['dynamics_violation'] = \
            self._check_dynamics_feasibility(trajectory)

        # 2. 控制可行性：控制量是否在限制内
        results['feasibility']['control_bounds'] = \
            self._check_control_bounds(trajectory)

        # 3. 状态约束：是否满足状态约束（速度、姿态限制）
        results['feasibility']['state_constraints'] = \
            self._check_state_constraints(trajectory)

        # 4. 终端精度：轨迹终点与目标的偏差
        final_state = trajectory.waypoints[-1]
        results['terminal_accuracy']['position_error'] = \
            np.linalg.norm(final_state.position[:2] - target.position[:2])
        results['terminal_accuracy']['altitude_error'] = \
            abs(final_state.position[2] - target.altitude)

        # 5. 高度利用率：是否有效消耗高度
        results['optimality']['altitude_efficiency'] = \
            self._compute_altitude_efficiency(trajectory, initial_state, target)

        # 6. 控制能量：控制量的积分
        results['optimality']['control_effort'] = \
            self._compute_control_effort(trajectory)

        return results

    def simulate_open_loop(self, trajectory, initial_state, wind):
        """
        开环仿真：按规划控制执行动力学

        用于验证规划轨迹与动力学模型的一致性
        """
        actual_states = [initial_state]
        state = initial_state

        for i, wp in enumerate(trajectory.waypoints[:-1]):
            control = trajectory.controls[i]
            dt = trajectory.waypoints[i+1].time - wp.time
            state = self.dynamics.step(state, control, wind, dt)
            actual_states.append(state)

        return actual_states

    def compare_planned_vs_simulated(self, trajectory, actual_states):
        """对比规划轨迹与开环仿真结果"""
        errors = {'position': [], 'velocity': []}

        for planned, actual in zip(trajectory.waypoints, actual_states):
            errors['position'].append(
                np.linalg.norm(planned.position - actual.position)
            )
            errors['velocity'].append(
                np.linalg.norm(planned.velocity - actual.velocity)
            )

        return {
            'position_rmse': np.sqrt(np.mean(np.array(errors['position'])**2)),
            'velocity_rmse': np.sqrt(np.mean(np.array(errors['velocity'])**2)),
            'max_position_error': np.max(errors['position'])
        }
```

**Planner 评估指标**：


| 指标         | 计算方法                   | 目标      |
| ------------ | -------------------------- | --------- |
| 动力学一致性 | 规划轨迹 vs 开环仿真 RMSE  | < 1 m     |
| 终端位置误差 | `|p_final - p_target|`     | < 5 m     |
| 高度利用率   | `H_consumed / H_available` | 0.9 ~ 1.0 |
| 控制平滑度   | `mean(|u̇|)`              | 越小越好  |
| GPM 求解时间 | 单次求解耗时               | < 1.0 s   |

### 10.3 Controller 验证（跟踪测试）

**目的**：评估 Controller 的轨迹跟踪能力，使用标准测试轨迹。

```python
# verification/verify_controller.py

class ControllerVerifier:
    """控制器跟踪验证"""

    def __init__(self, simulator, controller):
        self.sim = simulator
        self.controller = controller

    def verify_tracking(self, reference_trajectory, initial_state, wind):
        """
        闭环跟踪测试

        给定参考轨迹，使用 Controller 跟踪，评估跟踪误差
        """
        self.sim.reset(initial_state)

        actual_states = []
        tracking_errors = []
        control_commands = []

        for t in np.arange(0, reference_trajectory.duration, self.controller.dt):
            state = self.sim.get_state()
            actual_states.append(state)

            # 计算跟踪误差
            cross_track_error = self._compute_cross_track_error(
                state, reference_trajectory
            )
            along_track_error = self._compute_along_track_error(
                state, reference_trajectory, t
            )
            tracking_errors.append({
                'cross_track': cross_track_error,
                'along_track': along_track_error,
                'altitude': state.altitude - reference_trajectory.get_altitude_at(t)
            })

            # Controller 计算控制量
            control = self.controller.compute(state, reference_trajectory, wind)
            control_commands.append(control)

            # 执行仿真
            self.sim.step(control)

        return {
            'actual_trajectory': actual_states,
            'tracking_errors': tracking_errors,
            'controls': control_commands,
            'metrics': self._compute_metrics(tracking_errors)
        }

    def _compute_metrics(self, tracking_errors):
        """计算跟踪性能指标"""
        cross_track = [e['cross_track'] for e in tracking_errors]
        altitude = [e['altitude'] for e in tracking_errors]

        return {
            'cross_track_rmse': np.sqrt(np.mean(np.array(cross_track)**2)),
            'cross_track_max': np.max(np.abs(cross_track)),
            'altitude_rmse': np.sqrt(np.mean(np.array(altitude)**2)),
            'altitude_max': np.max(np.abs(altitude))
        }


class StandardTestTrajectories:
    """标准测试轨迹生成器"""

    @staticmethod
    def straight_line(start, end, duration):
        """直线轨迹：测试直线跟踪能力"""
        ...

    @staticmethod
    def constant_turn(center, radius, duration, direction='CW'):
        """恒定半径转弯：测试横向跟踪能力"""
        ...

    @staticmethod
    def s_turn(waypoints, turn_radius):
        """S弯轨迹：测试过渡跟踪能力"""
        ...

    @staticmethod
    def glide_slope(start_alt, end_alt, distance, glide_angle):
        """下滑道轨迹：测试纵向跟踪能力"""
        ...

    @staticmethod
    def flare_maneuver(entry_state, flare_altitude, duration):
        """拉飘轨迹：测试拉飘控制能力"""
        ...
```

**Controller 评估指标**：


| 指标         | 计算方法         | 目标     |
| ------------ | ---------------- | -------- |
| 横向跟踪误差 | Cross-track RMSE | < 3 m    |
| 纵向跟踪误差 | Along-track RMSE | < 5 m    |
| 高度跟踪误差 | Altitude RMSE    | < 2 m    |
| 航向跟踪误差 | Heading RMSE     | < 10°   |
| 控制抖动     | `std(u̇)`       | 越小越好 |
| 响应延迟     | 阶跃响应上升时间 | < 2 s    |

### 10.4 端到端验证（Planner + Controller）

```python
# verification/verify_e2e.py

class EndToEndVerifier:
    """端到端验证"""

    def __init__(self, simulator, planner, controller):
        self.sim = simulator
        self.planner = planner
        self.controller = controller

    def run_mission(self, initial_state, target, wind_config):
        """
        完整任务测试
        """
        self.sim.reset(initial_state)
        wind_model = WindModel(wind_config)

        mission_log = {
            'states': [],
            'plans': [],
            'controls': [],
            'phases': [],
            'events': []
        }

        trajectory = None

        while not self._mission_complete():
            state = self.sim.get_state()
            wind = wind_model.get_wind(state.position)

            # Planner 重规划
            if self._should_replan(state, trajectory):
                trajectory = self.planner.plan(state, target, wind)
                mission_log['plans'].append({
                    'time': state.timestamp,
                    'trajectory': trajectory
                })

            # Controller 计算控制
            control = self.controller.compute(state, trajectory, wind)

            # 记录
            mission_log['states'].append(state)
            mission_log['controls'].append(control)
            mission_log['phases'].append(self.controller.current_phase)

            # 执行
            self.sim.step(control)

        return self._analyze_mission(mission_log, target)

    def _analyze_mission(self, log, target):
        """分析任务结果"""
        final_state = log['states'][-1]
        initial_state = log['states'][0]

        return {
            # 着陆精度
            'landing_error': np.linalg.norm(
                final_state.position[:2] - target.position[:2]
            ),
            'landing_vertical_velocity': abs(final_state.velocity[2]),

            # 任务效率
            'mission_duration': final_state.timestamp - initial_state.timestamp,
            'altitude_used': initial_state.altitude - final_state.altitude,
            'replan_count': len(log['plans']),

            # 阶段分析
            'phase_durations': self._compute_phase_durations(log),

            # 控制分析
            'control_effort': np.mean([np.sum(c**2) for c in log['controls']]),
        }
```

### 10.5 蒙特卡洛测试

```python
# verification/monte_carlo.py

def run_monte_carlo_tests(verifier, n_runs=100, seed=42):
    """蒙特卡洛测试：评估统计性能"""
    np.random.seed(seed)
    results = []

    for i in range(n_runs):
        # 随机生成场景
        scenario = generate_random_scenario(
            altitude_range=(50, 200),
            distance_range=(50, 250),
            wind_speed_range=(0, 6),
            wind_direction_range=(0, 360)
        )

        # 运行任务
        result = verifier.run_mission(
            scenario.initial_state,
            scenario.target,
            scenario.wind
        )
        results.append(result)

    # 统计分析
    landing_errors = [r['landing_error'] for r in results]
    vertical_velocities = [r['landing_vertical_velocity'] for r in results]

    return {
        # 着陆精度统计
        'CEP': np.percentile(landing_errors, 50),       # 50% 圆概率误差
        'CEP95': np.percentile(landing_errors, 95),     # 95% 圆概率误差
        'mean_error': np.mean(landing_errors),
        'std_error': np.std(landing_errors),

        # 成功率
        'landing_success_rate': sum(1 for e in landing_errors if e < 10) / n_runs,
        'flare_success_rate': sum(1 for v in vertical_velocities if v < 2.0) / n_runs,

        # 详细结果
        'all_results': results
    }
```

### 10.6 验证脚本与命令

```bash
# ==============================================================================
# OFFLINE 模式命令（纯 Python，快速迭代，推荐用于开发和批量测试）
# ==============================================================================

# === Level 1: 单元测试 [OFFLINE] ===
pytest tests/test_gpm_collocation.py -v
pytest tests/test_dynamics.py -v
pytest tests/test_phase_transitions.py -v
pytest tests/test_trajectory_library.py -v

# === Level 2: 模块测试 [OFFLINE] ===
# Planner 开环验证
python scripts/verify_planner.py \
    --mode offline \
    --scenarios 50 \
    --output reports/planner_verification.html

# Controller 跟踪测试
python scripts/verify_controller.py \
    --mode offline \
    --test-trajectories standard \
    --output reports/controller_verification.html

# === Level 3: 端到端测试 [OFFLINE - 开发阶段] ===
python scripts/verify_e2e.py \
    --mode offline \
    --scenarios 20 \
    --visualize \
    --output reports/e2e_verification.html

# === Level 4: 蒙特卡洛测试 [OFFLINE - 推荐并行] ===
python scripts/monte_carlo.py \
    --mode offline \
    --runs 1000 \
    --parallel 8 \
    --seed 42 \
    --output reports/monte_carlo_results.json

# ==============================================================================
# ROS2 模式命令（完整系统，最终验证，推荐用于部署前确认）
# ==============================================================================

# === Level 3: 端到端测试 [ROS2 - 最终验证] ===
ros2 launch parafoil_planner_v3 e2e_verification.launch.py \
    scenario:=standard \
    n_runs:=10 \
    record_bag:=true \
    visualize:=true

# === Level 5: 部署前验证 [ROS2] ===
ros2 launch parafoil_planner_v3 full_system_test.launch.py \
    scenario:=comprehensive \
    rviz:=true \
    record_bag:=true \
    bag_output:=final_validation_bags/

# ==============================================================================
# 混合工作流示例
# ==============================================================================

# Step 1: 开发阶段 - OFFLINE 快速迭代
python scripts/verify_e2e.py --mode offline --scenarios 50

# Step 2: 调参阶段 - OFFLINE 蒙特卡洛
python scripts/monte_carlo.py --mode offline --runs 500 --parallel 8

# Step 3: 最终验证 - ROS2 完整系统
ros2 launch parafoil_planner_v3 e2e_verification.launch.py n_runs:=10

# Step 4: 部署确认 - ROS2 + rosbag 录制
ros2 launch parafoil_planner_v3 full_system_test.launch.py record_bag:=true
```

### 10.7 可视化验证工具

```python
# verification/visualize.py

def plot_planner_verification(trajectory, actual_states, target):
    """规划器验证可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. XY 平面轨迹对比
    ax = axes[0, 0]
    ax.plot([w.position[0] for w in trajectory.waypoints],
            [w.position[1] for w in trajectory.waypoints],
            'b-', label='Planned', linewidth=2)
    ax.plot([s.position[0] for s in actual_states],
            [s.position[1] for s in actual_states],
            'r--', label='Open-loop Sim', linewidth=1.5)
    ax.scatter(*target.position[:2], c='g', s=100, marker='*', label='Target')
    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.legend()
    ax.set_title('XY Trajectory Comparison')
    ax.axis('equal')
    ax.grid(True)

    # 2. 高度剖面
    ax = axes[0, 1]
    # ... 距离 vs 高度

    # 3. 控制量时间历程
    ax = axes[1, 0]
    # ... 时间 vs δL, δR

    # 4. 误差分析
    ax = axes[1, 1]
    # ... 时间 vs 位置误差

    plt.tight_layout()
    return fig


def plot_controller_verification(reference, actual, errors):
    """控制器验证可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 轨迹对比（3D）
    ax = axes[0, 0]
    # ...

    # 2. 跟踪误差
    ax = axes[0, 1]
    # ...

    # 3. 控制量
    ax = axes[1, 0]
    # ...

    # 4. 阶段时间线
    ax = axes[1, 1]
    # ...

    plt.tight_layout()
    return fig


def plot_monte_carlo_results(results):
    """蒙特卡洛结果可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 落点分布散点图
    ax = axes[0, 0]
    # ...

    # 2. 落点误差直方图
    ax = axes[0, 1]
    # ...

    # 3. 着陆速度直方图
    ax = axes[1, 0]
    # ...

    # 4. 成功率统计
    ax = axes[1, 1]
    # ...

    plt.tight_layout()
    return fig
```

### 10.8 验证报告模板

每次验证运行后生成 HTML 报告，包含：

```
验证报告 - parafoil_planner_v3
================================

1. 测试概要
   - 测试时间: 2024-01-15 14:30:00
   - 测试类型: End-to-End
   - 场景数量: 100

2. Planner 性能
   - GPM 求解时间: 0.85s (avg), 1.2s (max)
   - 终端位置误差: 3.2m (avg), 8.1m (max)
   - 动力学一致性: 0.8m RMSE

3. Controller 性能
   - 横向跟踪误差: 2.1m RMSE
   - 高度跟踪误差: 1.5m RMSE
   - 控制平滑度: 良好

4. 着陆精度
   - CEP (50%): 4.8m
   - CEP (95%): 12.3m
   - 成功率 (<10m): 92%

5. 拉飘效果
   - 平均垂直速度: 1.1 m/s
   - 拉飘成功率 (<2m/s): 95%

6. 失败案例分析
   - 案例 #23: 强顺风导致高度不足
   - 案例 #67: 侧风超限导致不可达

7. 可视化图表
   [落点分布图] [误差直方图] [典型轨迹图]
```

---

## 11. 自动分析与优化框架

实现 AI 驱动的日志分析、问题诊断和参数优化，形成闭环自改进系统。

### 11.0 执行模式选择（重要）

验证和优化可以在两种模式下运行，**必须根据任务目的选择正确的模式**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           执行模式决策流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐                                                   │
│  │ 任务目标是什么？      │                                                   │
│  └──────────┬───────────┘                                                   │
│             │                                                               │
│    ┌────────┴────────┐                                                      │
│    ▼                 ▼                                                      │
│ ┌──────────────┐ ┌──────────────┐                                           │
│ │ 快速迭代调参  │ │ 系统集成验证  │                                           │
│ │ 算法原型开发  │ │ 部署前测试    │                                           │
│ │ 批量蒙特卡洛  │ │ 时序验证      │                                           │
│ └──────┬───────┘ └──────┬───────┘                                           │
│        │                │                                                   │
│        ▼                ▼                                                   │
│  ┌───────────┐   ┌───────────┐                                              │
│  │  OFFLINE  │   │   ROS2    │                                              │
│  │   模式    │   │   模式    │                                              │
│  └───────────┘   └───────────┘                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 模式 A: Offline 模式（纯 Python 仿真）

**适用场景**：

- 参数调优迭代（需要运行数百次）
- 算法开发和调试
- 蒙特卡洛统计分析
- AI 自动优化循环
- 单元测试和模块测试

**特点**：

- ✅ 运行速度快（无 ROS2 消息传递开销）
- ✅ 可精确控制仿真步长和时间
- ✅ 易于并行化（多进程）
- ✅ 易于复现（确定性随机种子）
- ❌ 不测试实际 ROS2 节点通信
- ❌ 不测试实际时序和延迟

**运行方式**：

```bash
# 单次 Offline 测试
python scripts/offline_simulate.py \
    --config config/test_scenario.yaml \
    --output results/

# 批量蒙特卡洛（并行）
python scripts/offline_monte_carlo.py \
    --n-runs 1000 \
    --parallel 8 \
    --output monte_carlo_results/

# AI 自动优化（推荐 Offline 模式，更快）
python scripts/auto_optimize.py \
    --mode offline \
    --iterations 20 \
    --runs-per-iter 50
```

**核心代码结构**：

```python
# scripts/offline_simulate.py

from parafoil_planner_v3.dynamics import SixDOFDynamics
from parafoil_planner_v3.planner import GPMPlanner
from parafoil_planner_v3.guidance import GuidanceController

class OfflineSimulator:
    """纯 Python 离线仿真器（无 ROS2 依赖）"""

    def __init__(self, config):
        # 初始化动力学模型（与模拟器相同的方程）
        self.dynamics = SixDOFDynamics(config.dynamics)

        # 初始化规划器和控制器（与 ROS2 节点相同的代码）
        self.planner = GPMPlanner(config.planner)
        self.controller = GuidanceController(config.controller)

        self.dt = config.get('dt', 0.02)  # 50 Hz

    def run(self, initial_state, target, wind, max_time=300.0):
        """运行仿真直到着陆或超时"""
        state = initial_state.copy()
        trajectory = None
        t = 0.0

        while state.altitude > 0 and t < max_time:
            # 1. 规划（低频，或条件触发）
            if self._should_replan(t, state, trajectory):
                trajectory = self.planner.plan(state, target, wind)

            # 2. 制导（高频）
            control = self.controller.compute(state, trajectory, target)

            # 3. 动力学积分（RK4）
            state = self.dynamics.step(state, control, wind, self.dt)
            t += self.dt

            # 4. 记录日志
            self.logger.log_step(t, state, trajectory, control)

        return self.logger.get_results()
```

#### 模式 B: ROS2 模式（完整系统集成）

**适用场景**：

- 最终部署前验证
- 测试节点间通信和时序
- 验证 QoS 设置和消息同步
- 与实际传感器/执行器接口测试
- 硬件在环（HIL）测试

**特点**：

- ✅ 测试真实的节点通信
- ✅ 验证消息延迟和丢包处理
- ✅ 与 RViz 可视化集成
- ✅ 可连接实际硬件
- ❌ 运行速度较慢
- ❌ 并行化困难
- ❌ 需要 ROS2 环境

**运行方式**：

```bash
# 启动完整 ROS2 仿真
ros2 launch parafoil_planner_v3 full_simulation.launch.py \
    scenario:=test_scenario.yaml

# ROS2 集成测试
ros2 launch parafoil_planner_v3 integration_test.launch.py

# 录制 rosbag 用于回放分析
ros2 launch parafoil_planner_v3 full_simulation.launch.py \
    record_bag:=true \
    bag_output:=test_bags/
```

**Launch 文件结构**：

```python
# launch/full_simulation.launch.py

def generate_launch_description():
    return LaunchDescription([
        # 1. 仿真器节点（发布真实状态）
        Node(
            package='parafoil_simulator_ros',
            executable='sim_node',
            parameters=[config_file],
        ),

        # 2. Planner 节点
        Node(
            package='parafoil_planner_v3',
            executable='planner_node',
            parameters=[planner_config],
        ),

        # 3. Guidance 节点
        Node(
            package='parafoil_planner_v3',
            executable='guidance_node',
            parameters=[guidance_config],
        ),

        # 4. (可选) RViz 可视化
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            condition=IfCondition(LaunchConfiguration('rviz'))
        ),
    ])
```

#### 模式选择总结


| 任务类型                         | 推荐模式    | 原因                     |
| -------------------------------- | ----------- | ------------------------ |
| 参数调优（贝叶斯优化、网格搜索） | **Offline** | 需要数百次运行，速度优先 |
| AI 自动优化循环                  | **Offline** | 迭代次数多，需要快速反馈 |
| 蒙特卡洛分析（N > 100）          | **Offline** | 并行化，统计显著性       |
| 算法开发调试                     | **Offline** | 可断点调试，易复现       |
| 单元测试                         | **Offline** | CI/CD 流水线，快速执行   |
| 部署前最终验证                   | **ROS2**    | 必须验证完整系统         |
| 节点通信测试                     | **ROS2**    | 测试真实消息传递         |
| 时序/延迟验证                    | **ROS2**    | Offline 无法模拟         |
| RViz 可视化调试                  | **ROS2**    | 需要 ROS2 话题           |
| HIL/SIL 测试                     | **ROS2**    | 与硬件接口               |

#### 混合工作流（推荐）

```
┌────────────────────────────────────────────────────────────────────┐
│                     推荐的开发验证流程                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Stage 1: 算法开发                    [OFFLINE]                    │
│  ├── 单元测试 (pytest)                                             │
│  ├── 模块测试 (Planner/Controller 单独)                            │
│  └── 快速迭代调试                                                  │
│                                                                    │
│  Stage 2: 参数调优                    [OFFLINE]                    │
│  ├── AI 自动优化循环 (N=10-20 iterations)                          │
│  ├── 每轮 50-100 次蒙特卡洛                                        │
│  └── 输出最优参数                                                  │
│                                                                    │
│  Stage 3: 批量验证                    [OFFLINE]                    │
│  ├── 1000 次蒙特卡洛（覆盖各种场景）                                │
│  ├── 计算 CEP、成功率等统计指标                                     │
│  └── 确认性能达标                                                  │
│                                                                    │
│  Stage 4: 集成验证                    [ROS2]                       │
│  ├── 完整系统测试（10-20 次）                                      │
│  ├── 验证节点通信和时序                                            │
│  ├── RViz 可视化检查                                               │
│  └── 录制 rosbag 存档                                              │
│                                                                    │
│  Stage 5: 部署前确认                  [ROS2 + 硬件]                 │
│  ├── HIL/SIL 测试                                                  │
│  └── 最终确认                                                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

#### 代码共享原则

为确保 Offline 和 ROS2 模式结果一致，**核心算法代码必须共享**：

```
parafoil_planner_v3/
├── core/                      # 核心算法（Offline 和 ROS2 共用）
│   ├── dynamics.py            # 6-DOF 动力学
│   ├── planner_core.py        # GPM 规划器核心
│   ├── guidance_core.py       # 制导算法核心
│   └── trajectory_lib.py      # 轨迹库
│
├── ros2/                      # ROS2 封装层
│   ├── planner_node.py        # 调用 core/planner_core.py
│   ├── guidance_node.py       # 调用 core/guidance_core.py
│   └── interfaces.py          # ROS2 消息转换
│
├── offline/                   # Offline 仿真封装
│   ├── simulator.py           # 调用 core/dynamics.py
│   └── runner.py              # 批量运行管理
│
└── tests/
    ├── test_offline.py        # Offline 模式测试
    └── test_ros2_integration.py  # ROS2 集成测试
```

**关键点**：`core/` 目录下的代码是纯 Python，无 ROS2 依赖，被 `ros2/` 和 `offline/` 共同调用。

---

### 11.1 日志记录规范

所有运行必须记录详细日志，用于后续 AI 分析：

```python
# logging/mission_logger.py

class MissionLogger:
    """任务日志记录器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.log_data = {
            'metadata': {},
            'config': {},
            'timeline': [],
            'planner_logs': [],
            'controller_logs': [],
            'state_history': [],
            'events': [],
            'metrics': {}
        }

    def log_config(self, planner_config, controller_config, scenario):
        """记录配置参数"""
        self.log_data['config'] = {
            'planner': planner_config,
            'controller': controller_config,
            'scenario': scenario
        }

    def log_planner_step(self, timestamp, state, trajectory, solver_info):
        """记录规划器每步"""
        self.log_data['planner_logs'].append({
            'timestamp': timestamp,
            'state': state.to_dict(),
            'trajectory_length': len(trajectory.waypoints),
            'solver_time': solver_info.solve_time,
            'solver_iterations': solver_info.iterations,
            'cost': solver_info.cost,
            'constraint_violation': solver_info.max_violation,
            'trajectory_type': trajectory.type.name,
            'terminal_error': solver_info.terminal_error
        })

    def log_controller_step(self, timestamp, state, reference, control, errors):
        """记录控制器每步"""
        self.log_data['controller_logs'].append({
            'timestamp': timestamp,
            'phase': self.current_phase.name,
            'state': state.to_dict(),
            'reference': reference.to_dict(),
            'control': {'delta_L': control.delta_L, 'delta_R': control.delta_R},
            'errors': {
                'cross_track': errors.cross_track,
                'along_track': errors.along_track,
                'altitude': errors.altitude,
                'heading': errors.heading
            }
        })

    def log_event(self, event_type, details):
        """记录关键事件"""
        self.log_data['events'].append({
            'timestamp': self.current_time,
            'type': event_type,
            'details': details
        })

    def compute_summary_metrics(self):
        """计算汇总指标"""
        controller_logs = self.log_data['controller_logs']

        self.log_data['metrics'] = {
            # Planner 指标
            'planner': {
                'avg_solve_time': np.mean([l['solver_time'] for l in self.log_data['planner_logs']]),
                'max_solve_time': np.max([l['solver_time'] for l in self.log_data['planner_logs']]),
                'replan_count': len(self.log_data['planner_logs']),
                'avg_constraint_violation': np.mean([l['constraint_violation'] for l in self.log_data['planner_logs']])
            },

            # Controller 指标
            'controller': {
                'cross_track_rmse': np.sqrt(np.mean([l['errors']['cross_track']**2 for l in controller_logs])),
                'cross_track_max': np.max([abs(l['errors']['cross_track']) for l in controller_logs]),
                'altitude_rmse': np.sqrt(np.mean([l['errors']['altitude']**2 for l in controller_logs])),
                'control_effort': np.mean([l['control']['delta_L']**2 + l['control']['delta_R']**2 for l in controller_logs]),
                'control_rate': np.mean([self._compute_control_rate(i, controller_logs) for i in range(1, len(controller_logs))])
            },

            # 着陆指标
            'landing': {
                'position_error': self._compute_landing_error(),
                'vertical_velocity': abs(controller_logs[-1]['state']['velocity'][2]),
                'horizontal_velocity': np.linalg.norm(controller_logs[-1]['state']['velocity'][:2])
            },

            # 阶段时间
            'phase_durations': self._compute_phase_durations()
        }

    def save(self, filename: str):
        """保存日志为 JSON"""
        with open(self.output_dir / filename, 'w') as f:
            json.dump(self.log_data, f, indent=2, cls=NumpyEncoder)
```

### 11.2 AI 分析 Prompt 模板

#### Prompt 1: 日志分析与问题诊断

```markdown
# 任务日志分析 Prompt

你是翼伞自主着陆系统的性能分析专家。请分析以下任务日志，识别问题并给出诊断。

## 日志数据

```json
{log_data_json}
```

## 分析要求

### 1. 整体性能评估

- 着陆精度是否达标（CEP < 10m）？
- 着陆速度是否安全（垂直速度 < 2 m/s）？
- 任务是否成功完成？

### 2. Planner 诊断

请检查以下问题：

- [ ]  GPM 求解时间是否过长（> 1s）？
- [ ]  约束违反量是否过大（> 0.1）？
- [ ]  规划轨迹是否频繁变化（抖动）？
- [ ]  终端误差是否过大？
- [ ]  是否有不必要的重规划？

如发现问题，请说明：

- 问题描述
- 可能原因
- 建议的参数调整或算法修改

### 3. Controller 诊断

请检查以下问题：

- [ ]  横向跟踪误差是否过大（RMSE > 3m）？
- [ ]  高度跟踪误差是否过大（RMSE > 2m）？
- [ ]  控制量是否抖动（rate > 0.5/s）？
- [ ]  是否有饱和（持续触及限幅）？
- [ ]  阶段切换是否正确？
- [ ]  拉飘效果是否达标？

如发现问题，请说明：

- 问题描述
- 可能原因（增益过大/过小、前视距离不当、阶段阈值不当等）
- 建议的参数调整

### 4. 阶段分析

分析每个阶段的表现：

- CRUISE: 高度消耗效率、盘旋稳定性
- APPROACH: 下滑道跟踪精度、航向对准
- FLARE: 减速效果、着陆姿态

### 5. 输出格式

请以 JSON 格式输出分析结果：

```json
{
  "overall_assessment": "success|partial_success|failure",
  "landing_accuracy": {"error_m": X, "status": "ok|warning|critical"},
  "landing_velocity": {"velocity_mps": X, "status": "ok|warning|critical"},

  "planner_diagnosis": {
    "status": "healthy|degraded|failed",
    "issues": [
      {
        "type": "issue_type",
        "severity": "low|medium|high",
        "description": "...",
        "root_cause": "...",
        "recommendation": {
          "action": "adjust_parameter|modify_algorithm|investigate",
          "parameter": "param_name",
          "current_value": X,
          "suggested_value": Y,
          "rationale": "..."
        }
      }
    ]
  },

  "controller_diagnosis": {
    "status": "healthy|degraded|failed",
    "issues": [...]
  },

  "phase_analysis": {
    "cruise": {"status": "...", "notes": "..."},
    "approach": {"status": "...", "notes": "..."},
    "flare": {"status": "...", "notes": "..."}
  },

  "priority_actions": [
    "最重要的改进建议 1",
    "最重要的改进建议 2",
    "..."
  ]
}
```

```

#### Prompt 2: 批量日志统计分析

```markdown
# 批量任务统计分析 Prompt

你是翼伞系统性能优化专家。请分析以下 N 次任务的统计数据。

## 汇总统计

```json
{
  "n_runs": 100,
  "success_rate": 0.85,
  "landing_errors": {
    "mean": 7.2, "std": 4.1, "min": 1.2, "max": 28.5,
    "percentiles": {"50": 6.1, "90": 12.3, "95": 15.8}
  },
  "vertical_velocities": {
    "mean": 1.4, "std": 0.6, "min": 0.3, "max": 3.2
  },
  "failure_cases": [
    {"run_id": 23, "error": 28.5, "cause": "strong_headwind"},
    {"run_id": 67, "error": 22.1, "cause": "crosswind_limit"}
  ],
  "scenario_breakdown": {
    "wind_0_2": {"n": 30, "success_rate": 0.97, "mean_error": 4.2},
    "wind_2_4": {"n": 40, "success_rate": 0.90, "mean_error": 6.8},
    "wind_4_6": {"n": 30, "success_rate": 0.67, "mean_error": 11.3}
  }
}
```

## 分析任务

1. **性能瓶颈识别**

   - 哪些场景下性能最差？
   - 失败案例的共同模式是什么？
2. **参数敏感性分析**

   - 哪些参数对性能影响最大？
   - 当前参数是否处于最优区间？
3. **改进优先级**

   - 应该优先改进 Planner 还是 Controller？
   - 最有效的改进方向是什么？
4. **输出建议的参数调整**

```json
{
  "parameter_recommendations": [
    {
      "parameter": "L1_distance",
      "current": 13.67,
      "suggested": 15.0,
      "expected_improvement": "减少高风速下的横向跟踪误差",
      "confidence": "medium"
    }
  ],
  "algorithm_recommendations": [
    {
      "component": "flare_guidance",
      "issue": "高风速下拉飘效果不足",
      "suggestion": "增加风速自适应的拉飘高度触发"
    }
  ]
}
```

```

#### Prompt 3: 自动参数优化

```markdown
# 参数优化 Prompt

你是控制系统参数调优专家。基于历史数据，请优化以下参数。

## 当前参数

```yaml
# Planner 参数
planner:
  gpm_nodes: 30
  approach_entry_distance: 150.0
  flare_altitude: 10.0
  flare_distance: 30.0
  hold_threshold: 20.0

# Controller 参数
controller:
  L1_distance: 13.67
  K_yaw_rate: 1.13
  max_delta_a: 0.39
  terminal_radius: 50.4
  terminal_brake: 0.49
  distance_blend: 0.73
```

## 历史性能数据

```json
{performance_history}
```

## 优化目标

1. **主目标**：最小化 CEP（圆概率误差）
2. **约束**：
   - 着陆垂直速度 < 2 m/s
   - GPM 求解时间 < 1.0 s
   - 控制量变化率 < 0.5/s

## 优化方法

请使用以下方法之一：

1. **网格搜索建议**：给出参数搜索范围
2. **梯度估计**：基于历史数据估计参数梯度
3. **贝叶斯优化**：给出下一个探索点

## 输出

```json
{
  "optimization_method": "bayesian",
  "suggested_parameters": {
    "L1_distance": 15.2,
    "K_yaw_rate": 1.05,
    "flare_altitude": 12.0
  },
  "expected_improvement": {
    "CEP": {"from": 7.2, "to": 5.8},
    "landing_velocity": {"from": 1.4, "to": 1.2}
  },
  "confidence": 0.75,
  "next_experiments": [
    {"L1_distance": 14.0, "rationale": "验证 L1 敏感性"},
    {"flare_altitude": 15.0, "rationale": "探索更早拉飘"}
  ]
}
```

```

### 11.3 自动优化流程

```python
# optimization/auto_optimizer.py

from enum import Enum

class ExecutionMode(Enum):
    OFFLINE = "offline"    # 纯 Python 仿真，快速迭代
    ROS2 = "ros2"          # 完整 ROS2 系统，最终验证


class AutoOptimizer:
    """AI 驱动的自动优化器，支持 Offline 和 ROS2 两种执行模式"""

    def __init__(self, ai_client, config, mode: ExecutionMode = ExecutionMode.OFFLINE):
        """
        Args:
            ai_client: Claude API client
            config: 优化配置
            mode: 执行模式
                - OFFLINE: 使用纯 Python 仿真器，速度快，适合参数调优
                - ROS2: 使用完整 ROS2 系统，速度慢，适合最终验证
        """
        self.ai = ai_client
        self.config = config
        self.mode = mode
        self.history = OptimizationHistory()

        # 根据模式初始化不同的验证器
        if mode == ExecutionMode.OFFLINE:
            from parafoil_planner_v3.offline import OfflineVerifier
            self.verifier = OfflineVerifier(config)
            print("🚀 Optimizer initialized in OFFLINE mode (fast, parallel)")
        else:
            from parafoil_planner_v3.ros2 import ROS2Verifier
            self.verifier = ROS2Verifier(config)
            print("🤖 Optimizer initialized in ROS2 mode (full system)")

    def run_optimization_loop(self, n_iterations=10, runs_per_iter=20):
        """
        运行优化循环

        推荐配置:
            - OFFLINE 模式: n_iterations=20, runs_per_iter=50
            - ROS2 模式: n_iterations=5, runs_per_iter=10
        """
        # 检查配置合理性
        if self.mode == ExecutionMode.ROS2 and runs_per_iter > 20:
            print(f"⚠️  Warning: ROS2 mode with {runs_per_iter} runs/iter may be slow")
            print("   Consider using OFFLINE mode for parameter tuning")

        for iteration in range(n_iterations):
            print(f"\n=== Optimization Iteration {iteration + 1}/{n_iterations} ===")
            print(f"    Mode: {self.mode.value}, Runs: {runs_per_iter}")

            # 1. 运行当前参数的测试
            results = self.verifier.run_monte_carlo(
                n_runs=runs_per_iter,
                parallel=(self.mode == ExecutionMode.OFFLINE)  # 仅 Offline 支持并行
            )
            self.history.add_result(self.current_params, results)

            # 2. AI 分析日志
            diagnosis = self._ai_analyze_logs(results)

            # 3. AI 建议参数调整
            suggestions = self._ai_suggest_parameters(diagnosis)

            # 4. 应用建议（带安全检查）
            if self._validate_suggestions(suggestions):
                self.current_params = self._apply_suggestions(suggestions)
                print(f"Applied parameter changes: {suggestions}")
            else:
                print("Suggestions rejected by safety check")

            # 5. 检查收敛
            if self._check_convergence():
                print("Optimization converged!")
                break

        return self.history.get_best_params()

    def _ai_analyze_logs(self, results):
        """使用 AI 分析日志"""
        prompt = self._build_analysis_prompt(results)

        response = self.ai.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.content[0].text)

    def _ai_suggest_parameters(self, diagnosis):
        """使用 AI 建议参数"""
        prompt = self._build_optimization_prompt(diagnosis, self.history)

        response = self.ai.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.content[0].text)

    def _validate_suggestions(self, suggestions):
        """验证建议的安全性"""
        for param, value in suggestions.items():
            bounds = self.config.param_bounds.get(param)
            if bounds and not (bounds[0] <= value <= bounds[1]):
                print(f"Warning: {param}={value} out of bounds {bounds}")
                return False
        return True

    def _check_convergence(self):
        """检查是否收敛"""
        if len(self.history) < 3:
            return False

        recent = self.history.get_recent(3)
        cep_values = [r['metrics']['CEP'] for r in recent]

        # 如果最近 3 次 CEP 变化 < 5%，认为收敛
        improvement = (cep_values[0] - cep_values[-1]) / cep_values[0]
        return abs(improvement) < 0.05
```

### 11.4 增量改进建议生成

```python
# optimization/improvement_advisor.py

class ImprovementAdvisor:
    """生成代码改进建议"""

    def __init__(self, ai_client):
        self.ai = ai_client

    def analyze_and_suggest_code_changes(self, diagnosis, source_files):
        """分析问题并建议代码修改"""

        prompt = f"""
# 代码改进建议 Prompt

基于以下诊断结果，请建议具体的代码修改。

## 诊断结果

```json
{json.dumps(diagnosis, indent=2)}
```

## 当前代码

### planner_core.py

```python
{source_files['planner_core.py']}
```

### guidance/flare_guidance.py

```python
{source_files['flare_guidance.py']}
```

## 任务

1. 识别导致问题的具体代码位置
2. 提出修改建议（diff 格式）
3. 解释修改的预期效果

## 输出格式

---
（Prompt 结束）
---

```json
{{
  "code_changes": [
    {{
      "file": "guidance/flare_guidance.py",
      "issue": "拉飘触发时机不适应风速",
      "location": "line 45-50",
      "current_code": "...",
      "suggested_code": "...",
      "explanation": "增加风速自适应逻辑..."
    }}
  ],
  "new_features": [
    {{
      "description": "增加风速自适应拉飘高度",
      "implementation_sketch": "..."
    }}
  ]
}}
```

"""

    response = self.ai.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.content[0].text)
```

### 11.5 自动化优化脚本

```bash
# scripts/auto_optimize.py

#!/usr/bin/env python3
"""
自动优化脚本

===========================================================================
用法 (推荐工作流):
===========================================================================

1. 参数调优阶段 - 使用 OFFLINE 模式（快速，可并行）:

   python scripts/auto_optimize.py \
       --mode offline \
       --iterations 20 \
       --runs-per-iter 50 \
       --parallel 8 \
       --output optimization_results/

2. 最终验证阶段 - 使用 ROS2 模式（完整系统）:

   python scripts/auto_optimize.py \
       --mode ros2 \
       --iterations 3 \
       --runs-per-iter 10 \
       --output final_validation/

===========================================================================
模式说明:
===========================================================================

OFFLINE 模式 (--mode offline):
  - 使用纯 Python 仿真器，无 ROS2 依赖
  - 支持多进程并行 (--parallel N)
  - 速度快: ~0.5s/run vs ROS2 的 ~10s/run
  - 推荐用于: 参数调优、算法开发、蒙特卡洛统计

ROS2 模式 (--mode ros2):
  - 使用完整 ROS2 系统
  - 测试真实节点通信和时序
  - 速度慢，不支持并行
  - 推荐用于: 最终验证、集成测试、部署前确认

===========================================================================
"""

import argparse
from pathlib import Path
import anthropic

from parafoil_planner_v3.optimization import AutoOptimizer, ExecutionMode


def main():
    parser = argparse.ArgumentParser(
        description='AI-driven parameter optimization for parafoil planner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 执行模式
    parser.add_argument(
        '--mode', type=str, choices=['offline', 'ros2'], default='offline',
        help='执行模式: offline(快速,并行) 或 ros2(完整系统). 默认: offline'
    )

    # 优化参数
    parser.add_argument('--iterations', type=int, default=10,
                        help='优化迭代次数. OFFLINE 推荐 10-20, ROS2 推荐 3-5')
    parser.add_argument('--runs-per-iter', type=int, default=20,
                        help='每轮蒙特卡洛次数. OFFLINE 推荐 50-100, ROS2 推荐 10-20')
    parser.add_argument('--parallel', type=int, default=1,
                        help='并行进程数 (仅 OFFLINE 模式有效)')

    # 输出
    parser.add_argument('--output', type=str, default='optimization_results/')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Anthropic API key (或设置 ANTHROPIC_API_KEY 环境变量)')

    args = parser.parse_args()

    # 模式转换
    mode = ExecutionMode.OFFLINE if args.mode == 'offline' else ExecutionMode.ROS2

    # 参数合理性检查
    if mode == ExecutionMode.ROS2:
        if args.runs_per_iter > 20:
            print(f"⚠️  Warning: ROS2 mode with {args.runs_per_iter} runs/iter is very slow!")
            print(f"   Consider: --mode offline --runs-per-iter {args.runs_per_iter}")
        if args.parallel > 1:
            print(f"⚠️  Warning: --parallel ignored in ROS2 mode")
            args.parallel = 1

    print(f"\n{'='*60}")
    print(f"Auto Optimization - {mode.value.upper()} Mode")
    print(f"{'='*60}")
    print(f"Iterations: {args.iterations}")
    print(f"Runs per iteration: {args.runs_per_iter}")
    if mode == ExecutionMode.OFFLINE:
        print(f"Parallel processes: {args.parallel}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    # 初始化 AI client
    ai_client = anthropic.Anthropic(api_key=args.api_key)

    # 初始化优化器（根据模式自动选择验证器）
    optimizer = AutoOptimizer(
        ai_client=ai_client,
        config=load_config('config/optimization.yaml'),
        mode=mode
    )

    # 设置并行数（仅 OFFLINE 有效）
    if mode == ExecutionMode.OFFLINE:
        optimizer.verifier.set_parallel(args.parallel)

    # 运行优化
    best_params = optimizer.run_optimization_loop(
        n_iterations=args.iterations,
        runs_per_iter=args.runs_per_iter
    )

    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'best_params.yaml', 'w') as f:
        yaml.dump(best_params, f)

    with open(output_dir / 'optimization_history.json', 'w') as f:
        json.dump(optimizer.history.to_dict(), f, indent=2)

    # 记录使用的模式
    with open(output_dir / 'run_info.yaml', 'w') as f:
        yaml.dump({
            'mode': mode.value,
            'iterations': args.iterations,
            'runs_per_iter': args.runs_per_iter,
            'parallel': args.parallel if mode == ExecutionMode.OFFLINE else 1
        }, f)

    print(f"\n{'='*60}")
    print(f"Optimization complete!")
    print(f"{'='*60}")
    print(f"Mode: {mode.value}")
    print(f"Best CEP: {optimizer.history.get_best()['CEP']:.2f} m")
    print(f"Results saved to: {output_dir}")

    # 如果是 OFFLINE 模式，建议 ROS2 验证
    if mode == ExecutionMode.OFFLINE:
        print(f"\n💡 Tip: Run final validation with ROS2:")
        print(f"   python scripts/auto_optimize.py --mode ros2 --iterations 1 --runs-per-iter 10")


if __name__ == '__main__':
    main()
```
### 11.6 优化参数边界配置

```yaml
# config/optimization.yaml

optimization:
  # 优化目标
  objectives:
    primary: CEP  # 圆概率误差
    constraints:
      - name: landing_velocity
        max: 2.0  # m/s
      - name: solve_time
        max: 1.0  # s

  # 参数边界（安全范围）
  param_bounds:
    # Planner
    gpm_nodes: [15, 50]
    approach_entry_distance: [100.0, 250.0]
    flare_altitude: [5.0, 20.0]
    flare_distance: [15.0, 50.0]
    hold_threshold: [10.0, 40.0]

    # Controller
    L1_distance: [8.0, 25.0]
    K_yaw_rate: [0.5, 2.0]
    max_delta_a: [0.2, 0.5]
    terminal_radius: [30.0, 80.0]
    terminal_brake: [0.3, 0.6]
    distance_blend: [0.5, 0.9]
    flare_ramp_time: [0.3, 1.0]

  # 优化配置
  settings:
    max_iterations: 20
    runs_per_iteration: 30
    convergence_threshold: 0.05  # 5% improvement
    exploration_rate: 0.3  # 探索 vs 利用

  # AI 配置
  ai:
    model: "claude-sonnet-4-20250514"
    temperature: 0.3
    max_retries: 3
```
### 11.7 优化报告模板

```markdown
# 自动优化报告

## 优化概要

- 开始时间: 2024-01-15 10:00:00
- 结束时间: 2024-01-15 14:30:00
- 迭代次数: 10
- 总测试次数: 200

## 性能改进

| 指标 | 初始值 | 最终值 | 改进 |
|------|--------|--------|------|
| CEP | 12.3 m | 5.8 m | -53% |
| CEP95 | 24.1 m | 11.2 m | -54% |
| 着陆速度 | 1.8 m/s | 1.2 m/s | -33% |
| 成功率 | 78% | 94% | +16% |

## 参数变化

| 参数 | 初始值 | 最终值 | 变化 |
|------|--------|--------|------|
| L1_distance | 13.67 | 16.2 | +18% |
| flare_altitude | 10.0 | 14.5 | +45% |
| K_yaw_rate | 1.13 | 0.95 | -16% |

## AI 分析洞察

1. **主要问题**: 高风速下拉飘触发过晚，导致减速不足
2. **根因分析**: 固定拉飘高度未考虑风速影响
3. **解决方案**: 增加风速自适应拉飘触发

## 建议的代码改进

```python
# flare_guidance.py - 建议修改

# 原代码
def _should_enter_flare(self, state, target):
    return state.altitude <= self.config.flare_altitude

# 建议修改
def _should_enter_flare(self, state, target, wind):
    # 风速自适应拉飘高度
    wind_factor = 1.0 + 0.5 * (wind.speed / 4.0)  # 4m/s 时增加 50%
    adaptive_flare_altitude = self.config.flare_altitude * wind_factor
    return state.altitude <= adaptive_flare_altitude
```
## 下一步建议

1. 实施风速自适应拉飘改进
2. 增加侧风场景的测试覆盖
3. 考虑加入 MPC 预测控制

```

---

## 附录 A：6-DOF 动力学方程

### A.1 运动方程

```
位置导数（NED）：
ṗ = v

速度导数（NED）：
v̇ = R_nb * (F_aero / m) + g_ned

其中：

- R_nb: body 到 NED 的旋转矩阵（由四元数计算）
- F_aero: 气动力（body 系）
- m: 总质量
- g_ned = [0, 0, 9.81]ᵀ

姿态导数（四元数）：
q̇ = 0.5 * q ⊗ [0, ω]ᵀ

角速度导数（body 系）：
ω̇ = J⁻¹ * (τ_aero - ω × Jω)

其中：

- J: 惯性张量
- τ_aero: 气动力矩（body 系）

```

### A.2 气动力模型

```
气动力（body 系）：
F_aero = [F_x, F_y, F_z]ᵀ

其中：
F_x = -D * cos(α) + L * sin(α)  # 阻力和升力的 x 分量
F_y = Y                          # 侧力
F_z = -D * sin(α) - L * cos(α)  # 阻力和升力的 z 分量

升力：L = 0.5 * ρ * V² * S * C_L(α, δ)
阻力：D = 0.5 * ρ * V² * S * C_D(α, δ)
侧力：Y = 0.5 * ρ * V² * S * C_Y(β, δ)

刹车效应：

- δ_sym = (δ_L + δ_R) / 2  # 对称刹车 → 增加阻力和下沉率
- δ_diff = δ_L - δ_R       # 差动刹车 → 产生偏航力矩

```

### A.3 气动力矩模型

```
气动力矩（body 系）：
τ_aero = [τ_roll, τ_pitch, τ_yaw]ᵀ

滚转力矩：τ_roll = 0.5 * ρ * V² * S * b * C_l(β, p, δ)
俯仰力矩：τ_pitch = 0.5 * ρ * V² * S * c * C_m(α, q, δ)
偏航力矩：τ_yaw = 0.5 * ρ * V² * S * b * C_n(β, r, δ)

差动刹车偏航力矩：
τ_yaw_δ = k_yaw * δ_diff * V²

```

---

## 附录 B：GPM 数学背景

### B.1 Legendre 多项式

```
P_0(τ) = 1
P_1(τ) = τ
P_n(τ) = ((2n-1)τP_{n-1}(τ) - (n-1)P_{n-2}(τ)) / n

Legendre-Gauss 节点：P_N(τ) = 0 的根
Legendre-Gauss-Lobatto 节点：τ = ±1 和 P'_{N-1}(τ) = 0 的根

```

### B.2 配点法原理

```
状态近似：
x(τ) ≈ Σ_{i=0}^{N} x_i * L_i(τ)

其中 L_i(τ) 是 Lagrange 插值基函数：
L_i(τ) = Π_{j≠i} (τ - τ_j) / (τ_i - τ_j)

导数近似：
ẋ(τ_k) ≈ Σ_{j=0}^{N} D_{kj} * x_j

其中 D 是微分矩阵：
D_{kj} = dL_j(τ_k) / dτ

```

### B.3 NLP 问题规模

```
决策变量数：n_vars = N * (n_x + n_u) + 1
等式约束数：n_eq = N * n_x + n_bc
不等式约束数：n_ineq = N * (n_path + n_control)

典型值（N=30, n_x=13, n_u=2）：

- 决策变量：451
- 等式约束：~400
- 不等式约束：~150

```

---

## 附录 C：轨迹库存储格式

### C.1 HDF5 结构

```
library.h5
├── metadata/
│   ├── version: "1.0"
│   ├── creation_date: "2024-01-15"
│   ├── num_trajectories: 5000
│   └── scenario_config: {...}
├── index/
│   ├── features: [N_traj, N_features]  # 用于 KNN
│   └── kdtree_params: {...}
└── trajectories/
├── traj_0000/
│   ├── scenario: {wind_speed, wind_dir, altitude, ...}
│   ├── trajectory_type: "DIRECT"
│   ├── waypoints: [N_points, 13]  # [t, x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
│   ├── controls: [N_points, 2]    # [δL, δR]
│   ├── cost: 123.45
│   └── metadata: {duration, altitude_loss, max_bank, ...}
├── traj_0001/
│   └── ...
└── ...

```

---

（Prompt 结束）
```
