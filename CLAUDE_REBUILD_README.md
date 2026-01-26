# Parafoil Simulator Rebuild (Flightmare-style) — Claude Input README

> 目的：把这份文件**整段交给 Claude**，让它按要求从零重建一个新项目（ROS 2 + Python），动力学组织方式对齐 Flightmare：
> - 明确连续动力学：\(\dot x = f(x,u)\)
> - 数值积分器与动力学彻底解耦（Euler / Semi-implicit / RK4 可切换）
> - 支持控制周期内子步长 `dt_max`（Flightmare 风格稳定性关键）
> - 同时满足：1) 下滑比/落点 2) 转弯半径/航向响应 3) 风扰鲁棒性

---

## Claude Prompt（请 Claude 严格照做）

你是资深机器人仿真工程师。请从零重建一个 ROS 2（Python）parafoil（滑翔伞翼）6DoF 仿真项目，目标是同时做好：

1) **下滑比/落点**（沉降率与水平航迹积分要可信）
2) **转弯半径/航向响应**（差动刹车到偏航/滚转响应与阻尼要合理）
3) **风扰鲁棒性**（相对来流一致 + 风模型可控可复现 + domain randomization）

必须仿照 Flightmare 的“动力学组织方式”：把动力学写成显式微分方程 \(\dot x=f(x,u)\)，并将数值积分器与动力学完全解耦，支持 **控制周期内子步长 dt_max** 与 **RK4**（可切换 Euler / Semi-implicit / RK4）。项目必须可 `colcon build`，可 `ros2 run` 启动，ROS 节点与核心仿真库解耦（核心可脱离 ROS 单独运行 + pytest 单元测试）。

### 非目标（不要做）
- 不做 Unity/渲染/视觉传感器
- 不做 Gazebo 插件
- 不做复杂 UI、RViz、bag 工具

---

## 1. 工程结构（必须按这个生成）

生成一个 workspace（相当于 src 内容），至少 2 个 ROS2 package：

### 1) `parafoil_dynamics`（纯 Python 库）
必须包含：状态、参数、动力学 \(f(x,u)\)、积分器、风模型、传感器模拟、pytest。

文件结构必须包含（并给出每个文件完整代码）：
- `parafoil_dynamics/__init__.py`
- `parafoil_dynamics/state.py`：`State` / `ControlCmd` dataclass
- `parafoil_dynamics/params.py`：`Params` dataclass + 从 YAML/ROS 参数加载
- `parafoil_dynamics/math3d.py`：四元数/旋转矩阵/叉乘工具（自己实现，不依赖私有库）
- `parafoil_dynamics/dynamics.py`：核心 `dynamics(state, cmd, params) -> state_dot`
- `parafoil_dynamics/integrators.py`：Euler / Semi-implicit / RK4 + `dt_max` 子步
- `parafoil_dynamics/wind.py`：steady + gust + colored wind（默认可关闭，支持 seed）
- `parafoil_dynamics/sensors.py`：从 state 生成 position/acc/gyro 测量 + 噪声
- `parafoil_dynamics/tests/`：pytest 单元测试（见第 7 节）

### 2) `parafoil_simulator_ros`（ROS2 节点 package）
- `parafoil_simulator_ros/__init__.py`
- `parafoil_simulator_ros/sim_node.py`：仿真节点（timer loop）
- `parafoil_simulator_ros/config/params.yaml`：全部参数集中配置
- 可选：`parafoil_simulator_ros/launch/sim.launch.py`

要求：项目必须能 `colcon build --symlink-install` 通过。

---

## 2. 坐标系、状态、输入（必须严格定义并说明）

- 坐标系采用 **NED**：x=North, y=East, z=Down
- 重力为 \(+9.81\) 沿 z 正方向（z-down），高度 \(h=-z\)
- 落地判定：`z > 0`（进入地面后停止仿真或停止 timer）

### 状态 `State`（dataclass，至少包含）
- `p_I: np.ndarray (3,)` 位置（I=NED）
- `v_I: np.ndarray (3,)` 速度（I=NED）
- `q_IB: np.ndarray (4,)` 四元数，表示 **B->I**，格式 `[w,x,y,z]`
- `w_I: np.ndarray (3,)` 角速度（I 表达）
- `delta: np.ndarray (2,)` 执行机构内部状态 `[delta_l, delta_r]`（实际生效刹车）
- `t: float`

### 控制输入 `ControlCmd`
- `delta_cmd: np.ndarray (2,)` 命令左右刹车 `[delta_l_cmd, delta_r_cmd]`，范围 [0,1]
- 在动力学中必须定义：
  - `delta_s = (delta_l + delta_r)/2`（对称刹车：影响升阻 -> 下滑比/沉降/落点）
  - `delta_a = (delta_r - delta_l)`（差动刹车：影响滚转/偏航 -> 航向响应/转弯半径）

### 变换约定（必须实现）
- `C_IB = quat_to_rotmat(q_IB)` 为 B->I
- `v_B = C_IB.T @ v_I`，`w_B = C_IB.T @ w_I`

---

## 3. 动力学 \(\dot x=f(x,u)\)（必须包含的物理项：服务 1/2/3）

模型类型：**准静态气动 + 刚体 6DoF + payload 阻力 + 执行机构一阶滞后 + 风扰相对来流**。

### 3.1 执行机构一阶滞后（必须）
左右刹车不是瞬时生效：

\[
\dot\delta = \frac{\mathrm{clamp}(\delta_{cmd},0,1)-\delta}{\tau_{act}}
\]

- `tau_act` 可配置
- `delta` 在用于气动之前必须 clamp 到 [0,1]

### 3.2 风与相对来流（必须）
所有气动力必须基于相对来流：
- `wind_I` 来自 `wind.py`（steady/gust/colored），默认可关闭
- `v_rel_I = v_I - wind_I`
- 转到 B、再到伞气动坐标系计算 \(V,\alpha,\beta\)
- 必须对小 \(V\) 做 epsilon 保护，避免除零/NaN

### 3.3 气动力/力矩：最小可调系数集合（必须）
> 关键要求：不要求复杂伞型几何；但必须把“几何效果”折算进少数可辨识/可调的系数/导数里。

#### 升力/阻力（决定下滑比与落点）
\[
L = \tfrac12\rho V^2 S\,C_L(\alpha,\delta_s),\quad
D = \tfrac12\rho V^2 S\,C_D(\alpha,\delta_s)
\]

建议最小形式（可配置参数）：
- \(C_L = c_{L0}+c_{L\alpha}\alpha+c_{L\delta_s}\delta_s\)
- \(C_D = c_{D0}+c_{D\alpha2}\alpha^2+c_{D\delta_s}\delta_s\)

#### 侧力（风侧吹时航迹合理性）
- \(C_Y \approx c_{Y\beta}\beta\)
- \(Y = \tfrac12\rho V^2 S C_Y\)

#### 力矩（决定航向响应/转弯半径/阻尼）
必须至少包含这些可调参数：
- 滚转阻尼：`c_l_p`
- 差动刹车滚转：`c_l_da`
- 俯仰：`c_m0`, `c_m_a`, `c_m_q`
- 偏航阻尼：`c_n_r`
- 差动刹车偏航：`c_n_da`

并使用无量纲角速度（推荐，避免速度变化导致响应怪异）：
- \(\hat p = p b/(2V)\), \(\hat q = q c/(2V)\), \(\hat r = r b/(2V)\)

所有涉及 \(1/V\) 的地方必须 epsilon 保护。

### 3.4 payload 阻力（必须）
\[
\mathbf{F}_{pd}=-\tfrac12\rho C_{D,pd} S_{pd}\,\|\mathbf{v}_{rel}\|\,\mathbf{v}_{rel}
\]
并通过力臂 `r x F` 贡献力矩。

### 3.5 刚体 6DoF（必须）
- \(\dot p = v\)
- \(\dot v = \frac{1}{m}F_I\)
- 四元数：\(\dot q = \tfrac12\, q \otimes [0,\omega_B]\)（或等价形式，需在同一坐标系一致）
- 角速度：\(\dot\omega = I^{-1}(\tau - \omega \times I\omega)\)
- 每个子步后必须四元数归一化

---

## 4. 数值积分器（必须：解耦 + dt_max 子步 + 可切换）

在 `integrators.py` 实现：
- `euler_step(dynamics, x, u, dt)`
- `semi_implicit_step(...)`（说明清楚更新顺序）
- `rk4_step(...)`
- `integrate_with_substeps(dynamics, x, u, ctl_dt, dt_max, method)`：将一个控制周期拆 N 个子步执行

硬性要求：
- 每个子步：归一化四元数 + 检查 finite，否则抛异常（便于定位数值问题）

---

## 5. 风模型（必须支持三层，默认可关）

在 `wind.py`：
1) steady wind：常值向量
2) gust：分段阶跃阵风（每 T 秒抽样一次）
3) colored wind：一阶滤波白噪声（连续扰动谱）

要求：全部支持 `seed`，保证可复现；并能在每个 episode reset 时重新采样。

---

## 6. ROS2 节点接口（必须与现有工程兼容）

`parafoil_simulator_ros/sim_node.py`：
- 订阅：`/rockpara_actuators_node/auto_commands` (`geometry_msgs/msg/Vector3Stamped`)
  - `x=delta_left_cmd`，`y=delta_right_cmd`，clamp [0,1]
- 发布：
  - `/position`：`geometry_msgs/msg/Vector3Stamped`（NED，米）
  - `/body_acc`：`geometry_msgs/msg/Vector3Stamped`（B 系加速度或比力，说明清楚）
  - `/body_ang_vel`：`geometry_msgs/msg/Vector3Stamped`（B 系角速度 rad/s）
- timer：以 `ctl_dt` 触发；每次 step 内部用 `dt_max` 子步积分
- 参数化：`integrator_type`, `ctl_dt`, `dt_max`, `actuator_tau`, 风开关/强度、噪声 sigma、气动系数等全部来自 `params.yaml`
- 落地时：停止 timer 或退出，并打印落点/时间等关键信息

---

## 7. 最小测试与验证（必须提供 pytest）

至少包含 3 类测试：
1) 四元数归一：连续积分 N 步后 `|q|≈1` 不发散
2) 子步收敛：同一 `ctl_dt`，`dt_max` 越小末状态变化越小（相对收敛）
3) 有限性：随机合理初值下 `dynamics` 与积分结果不产生 NaN/Inf（含小 V 防护）

---

## 8. 最小运行说明（必须给出命令）

- `colcon build --symlink-install`
- `ros2 run parafoil_simulator_ros sim_node --ros-args --params-file <path>`
- 给一个发布控制命令的例子：
  - `ros2 topic pub -r 20 /rockpara_actuators_node/auto_commands geometry_msgs/msg/Vector3Stamped "{vector: {x: 0.2, y: 0.6, z: 0.0}}"`

---

## 9. 参数命名（贴近现有工程，方便迁移）

请尽量兼容下面这些参数名（来自现有 parafoil 仿真代码习惯），以便我后续把真实参数直接替换进去：

- 升力：`c_L0, c_La, c_Lds`
- 阻力：`c_D0, c_Da2, c_Dds`
- 侧力：`c_Yb`
- 力矩/阻尼：`c_lp, c_lda, c_m0, c_ma, c_mq, c_nr, c_nda`

并保持控制分解：
- `delta_s = (delta_l + delta_r)/2`
- `delta_a = (delta_r - delta_l)`

---

## 10. 输出要求

- 输出完整文件树 + 每个文件的完整代码（不要省略）
- 代码简洁、可运行、无多余功能
- 所有系数/参数给可运行的默认占位值（后续我会用真实参数替换）
- 不引用外部私有仓库；四元数/旋转矩阵等数学函数请直接实现
- 对所有可能除零位置加 epsilon，保证数值稳定性

（Prompt 结束）
