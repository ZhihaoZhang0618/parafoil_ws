# Parafoil 6DoF Simulator

基于 ROS2 的滑翔伞（Parafoil）六自由度动力学仿真器。

## 目录

- [项目结构](#项目结构)
- [坐标系定义](#坐标系定义)
- [状态与控制](#状态与控制)
- [动力学模型](#动力学模型)
- [当前翼伞性能（极曲线/转弯率）](#当前翼伞性能极曲线转弯率)
- [参数说明](#参数说明)
- [风模型](#风模型)
- [安装与运行](#安装与运行)
- [ROS2 接口](#ros2-接口)

---

## 项目结构

```
parafoil_ws/
├── src/
│   ├── parafoil_dynamics/              # 纯 Python 动力学库（不依赖 ROS）
│   │   └── parafoil_dynamics/
│   │       ├── state.py                # 状态定义
│   │       ├── params.py               # 参数定义
│   │       ├── dynamics.py             # 核心动力学方程
│   │       ├── math3d.py               # 四元数/旋转工具
│   │       ├── integrators.py          # 数值积分器
│   │       ├── wind.py                 # 风模型
│   │       └── sensors.py              # 传感器模拟
│   │
│   └── parafoil_simulator_ros/         # ROS2 仿真节点
│       ├── parafoil_simulator_ros/
│       │   ├── sim_node.py             # 仿真主节点
│       │   └── keyboard_teleop.py      # 键盘遥控
│       ├── config/
│       │   ├── params.yaml             # 参数配置文件
│       │   └── parafoil_rviz.rviz      # RViz2 配置
│       └── launch/
│           └── sim.launch.py           # 启动文件
│
└── scripts/                            # 测试脚本
    ├── test_weathercock.py
    ├── test_wind_yaw_coupling.py
    └── ...
```

---

## 坐标系定义

### 1. 惯性坐标系 (Inertial Frame, I)

采用 **NED (North-East-Down)** 惯例：

```
      North (x+)
         ↑
         |
   West ←+→ East (y+)
         |
         ↓
      Down (z+)
```


| 轴 | 方向  | 说明         |
| -- | ----- | ------------ |
| x  | North | 地理北向为正 |
| y  | East  | 地理东向为正 |
| z  | Down  | 垂直向下为正 |

- **高度**：`h = -z`（z 为负表示在地面上方）
- **重力**：`g_I = [0, 0, +g]`，其中 `g = 9.81` m/s²
- **落地判定**：`z >= 0`

### 2. 机体坐标系 (Body Frame, B)

原点位于系统质心，轴定义：

```
        Forward (x+)
            ↑
            |
    Left ←--+--→ Right (y+)
            |
            ↓
         Down (z+)
```


| 轴 | 方向    | 说明         |
| -- | ------- | ------------ |
| x  | Forward | 前进方向为正 |
| y  | Right   | 右翼方向为正 |
| z  | Down    | 垂直向下为正 |

**角速度分量**：

- `p` = 滚转角速度 (roll rate)，绕 x 轴
- `q` = 俯仰角速度 (pitch rate)，绕 y 轴
- `r` = 偏航角速度 (yaw rate)，绕 z 轴

### 3. 风轴坐标系 / 稳定轴坐标系 (Stability Frame)

用于气动力计算，以相对气流速度定义：

- **x 轴**：沿相对速度方向
- **z 轴**：在机体 x-z 平面内，垂直于 x 轴向下
- **y 轴**：右手系确定

**气动角定义**：

```
迎角 (Angle of Attack):   α = arctan(w / u)
侧滑角 (Sideslip Angle):  β = arcsin(v / V)
```

其中 `[u, v, w]` 为机体坐标系下的相对气流速度，`V` 为空速。

### 4. 坐标变换

**四元数约定**：`q = [w, x, y, z]`（标量在前）

**旋转矩阵**：`C_IB` 将向量从机体系变换到惯性系

```
v_I = C_IB @ v_B
v_B = C_IB.T @ v_I = C_BI @ v_I
```

**欧拉角 (ZYX 顺序)**：

- Roll (φ)：绕 x 轴
- Pitch (θ)：绕 y 轴
- Yaw (ψ)：绕 z 轴

```python
# 欧拉角 → 四元数
q = quat_from_euler(roll, pitch, yaw)

# 四元数 → 欧拉角
roll, pitch, yaw = quat_to_euler(q)
```

---

## 状态与控制

### 状态向量


| 变量       | 符号    | 维度 | 单位  | 描述                    |
| ---------- | ------- | ---- | ----- | ----------------------- |
| 位置       | `p_I`   | 3    | m     | 惯性系位置 [N, E, D]    |
| 速度       | `v_I`   | 3    | m/s   | 惯性系速度 [vN, vE, vD] |
| 姿态       | `q_IB`  | 4    | -     | 四元数 [w, x, y, z]     |
| 角速度     | `w_B`   | 3    | rad/s | 机体系角速度 [p, q, r]  |
| 执行器状态 | `delta` | 2    | [0,1] | 刹车位置 [δL, δR]     |
| 时间       | `t`     | 1    | s     | 仿真时间                |

**总状态维度**：3 + 3 + 4 + 3 + 2 = **15**

### 控制输入


| 变量       | 符号          | 范围   | 描述           |
| ---------- | ------------- | ------ | -------------- |
| 左刹车命令 | `delta_l_cmd` | [0, 1] | 0=释放, 1=全拉 |
| 右刹车命令 | `delta_r_cmd` | [0, 1] | 0=释放, 1=全拉 |

**控制分解**：

```
对称刹车: δs = (δL + δR) / 2     → 影响升力、阻力、失速
差动刹车: δa = δL - δR           → 影响转弯
```

**控制效果**：

- 拉左刹车（δL↑）→ 左转
- 拉右刹车（δR↑）→ 右转
- 双侧刹车（δs↑）→ 减速、增加下沉率、可能失速

---

## 动力学模型

核心方程形式：`x_dot = f(x, u, params, wind)`

### 1. 运动学方程

**位置导数**：

```
p_I_dot = v_I
```

**四元数导数**：

```
q_IB_dot = 0.5 * q_IB ⊗ [0, w_B]
```

### 2. 气动力计算

#### 2.1 相对速度与气动角

```python
# 相对速度（惯性系）
v_rel_I = v_I - wind_I

# 转换到机体系
v_rel_B = C_BI @ v_rel_I

# 气动角
V = ||v_rel_B||                      # 空速
α = arctan2(w, u)                    # 迎角
β = arcsin(v / V)                    # 侧滑角
```

#### 2.2 气动力系数

**升力系数**（含失速模型）：

```
C_L = C_L_linear * stall_factor

C_L_linear = c_L0 + c_La * α + c_Lds * δs
```

失速模型：

```
α_stall = α_stall_0 - α_stall_brake * δs

当 α > α_stall 时：
stall_ratio = (α - α_stall) / α_stall_width
stall_factor = 0.3 + 0.7 * exp(-stall_ratio²)
```

**阻力系数**：

```
C_D = c_D0 + c_Da2 * α² + c_Dds * δs + C_D_stall

失速时：C_D_stall = c_D_stall * (1 - stall_factor)
```

**侧力系数**：

```
C_Y = c_Yb * β
```

#### 2.3 气动力（机体系）

```python
q_bar = 0.5 * ρ * V²                    # 动压
L = q_bar * S * C_L                      # 升力
D = q_bar * S * C_D                      # 阻力
Y = q_bar * S * C_Y                      # 侧力

# 机体系气动力
F_aero_B = [
    -D * cos(α) + L * sin(α),            # 前向力
    Y,                                     # 侧向力
    -D * sin(α) - L * cos(α)             # 垂向力
]
```

### 3. 气动力矩计算

**无量纲角速度**：

```
p_hat = p * b / (2V)
q_hat = q * c / (2V)
r_hat = r * b / (2V)
```

**滚转力矩系数**：

```
C_l = c_lp * p_hat + c_lda * δa + c_lb * β
```

**俯仰力矩系数**：

```
C_m = c_m0 + c_ma * α + c_mq * q_hat
```

**偏航力矩系数**：

```
C_n = c_nr * r_hat + c_nda * δa + c_nb * β + c_n_weath * (wind_y_B / V)
```

其中最后一项为风向舵效应（weathercock effect）。

**力矩向量（机体系）**：

```
M_aero_B = q_bar * S * [C_l * b, C_m * c, C_n * b]
```

### 4. 其他力与力矩

**载荷阻力**：

```
F_pd_B = -0.5 * ρ * c_D_pd * S_pd * V_rel * v_rel_B
```

**摆锤恢复力矩**（滚转/俯仰稳定性）：

```
k_pendulum = m_payload * g * L
M_pendulum_B = -k_pendulum * [sin(roll), sin(pitch), 0]
```

**重力**：

```
F_gravity_I = [0, 0, m * g]
```

### 5. 动力学方程

**平动方程**：

```
v_I_dot = (1/m) * (C_IB @ (F_aero_B + F_pd_B) + F_gravity_I)
```

**转动方程**：

```
w_B_dot = I_B_inv @ (M_total_B - w_B × (I_B @ w_B))
```

**执行器动力学**（一阶滞后）：

```
delta_dot = (delta_cmd - delta) / tau_act
```

---

## 当前翼伞性能（极曲线/转弯率）

以下表格用于描述**当前模拟器参数集**下的翼伞性能（稳态数值仿真得到），便于 planner/controller 选参与效果评估：

- 参数来源：`src/parafoil_simulator_ros/config/params.yaml`
- 仿真假设：无风、对称刹车保持常值、取稳态时间窗统计

### 气动极曲线（Polar Table）

| 刹车量 | 空速 (m/s) | 下沉率 (m/s) | 滑翔比 |
|--------|-----------|-------------|--------|
| 0.0    | 4.44      | 0.90        | 4.9    |
| 0.1    | 4.19      | 1.03        | 4.1    |
| 0.2    | 3.97      | 1.13        | 3.5    |
| 0.3    | 3.78      | 1.20        | 3.1    |
| 0.4    | 3.61      | 1.26        | 2.9    |
| 0.5    | 3.47      | 1.30        | 2.7    |
| 0.6    | 3.33      | 1.33        | 2.5    |
| 0.7    | 3.22      | 1.36        | 2.4    |
| 0.8    | 3.11      | 1.39        | 2.2    |
| 0.9    | 3.01      | 1.40        | 2.1    |
| 1.0    | 2.92      | 1.42        | 2.1    |

### 典型转弯率（yaw_rate vs Δa）

说明：
- `delta_a (L-R) = left_brake - right_brake`
- 下表是在 `b_sym≈0.25`、无风、稳态下统计的 yaw rate（来自动力学仿真）
- 可近似为：`yaw_rate ≈ -1.7 * delta_a`（单位 rad/s）

| delta_a (L-R) | yaw_rate (rad/s) | yaw_rate (deg/s) |
|--------------:|-----------------:|-----------------:|
| 0.10          | -0.173           | -9.9             |
| 0.20          | -0.345           | -19.8            |
| 0.30          | -0.515           | -29.5            |
| 0.40          | -0.683           | -39.2            |
| 0.50          | -0.848           | -48.6            |

## 参数说明

参数文件位置：`src/parafoil_simulator_ros/config/params.yaml`

### 仿真时序参数


| 参数              | 默认值 | 单位 | 描述                                |
| ----------------- | ------ | ---- | ----------------------------------- |
| `ctl_dt`          | 0.02   | s    | 控制周期 (50 Hz)                    |
| `dt_max`          | 0.005  | s    | 最大积分子步长                      |
| `integrator_type` | "rk4"  | -    | 积分器类型: euler/semi_implicit/rk4 |

### 初始条件


| 参数               | 默认值       | 单位 | 描述                                     |
| ------------------ | ------------ | ---- | ---------------------------------------- |
| `initial_position` | [0, 0, -100] | m    | 初始位置 [N, E, D]，z=-100 表示 100m 高度 |
| `initial_velocity` | [4.5, 0, 0.9] | m/s  | 初始速度 [vN, vE, vD]（NED，vD>0 表示下沉） |
| `initial_euler`    | [0, 0, 0]    | rad  | 初始姿态 [roll, pitch, yaw]              |
| `initial_altitude` | 50.0 (launch) | m   | 可选：若设置则覆盖 `initial_position[2]`（`sim.launch.py` 默认 50m） |

### 物理常数


| 参数  | 默认值 | 单位   | 描述       |
| ----- | ------ | ------ | ---------- |
| `rho` | 1.29   | kg/m³ | 空气密度   |
| `g`   | 9.81   | m/s²  | 重力加速度 |

### 质量特性


| 参数          | 默认值            | 单位    | 描述                           |
| ------------- | ----------------- | ------- | ------------------------------ |
| `m`           | 2.45              | kg      | 总质量（伞衣+载荷，已按实飞极曲线标定） |
| `I_B_diag`    | [0.8, 0.15, 0.85] | kg·m² | 惯性张量对角线 [Ixx, Iyy, Izz] |

### 几何参数


| 参数     | 默认值 | 单位 | 描述         |
| -------- | ------ | ---- | ------------ |
| `S`      | 1.5    | m²  | 伞翼参考面积 |
| `b`      | 1.88   | m    | 翼展         |
| `c`      | 0.80   | m    | 弦长         |
| `S_pd`   | 0.1    | m²  | 载荷阻力面积 |
| `c_D_pd` | 1.0    | -    | 载荷阻力系数 |
| `r_canopy_B` | [0, 0, -0.3] | m | 伞翼气动中心相对质心位置（机体系） |
| `r_pd_B` | [0, 0, 0.2] | m | 载荷相对质心位置（机体系） |
| `pendulum_arm` | 0.3 | m | 预留：当前摆锤恢复力矩仍使用 `line_length`（parafoil_dynamics 默认 0.5m） |

### 升力系数


| 参数    | 默认值 | 描述                 |
| ------- | ------ | -------------------- |
| `c_L0`  | 0.55   | 零迎角升力系数       |
| `c_La`  | 3.80   | 升力曲线斜率 [1/rad] |
| `c_Lds` | 0.30   | 对称刹车对升力的影响（已标定） |

**升力模型**：`C_L = c_L0 + c_La * α + c_Lds * δs`

### 阻力系数


| 参数    | 默认值 | 描述                 |
| ------- | ------ | -------------------- |
| `c_D0`  | 0.16   | 零升阻力系数         |
| `c_Da2` | 0.50   | 诱导阻力因子         |
| `c_Dds` | 0.75   | 对称刹车对阻力的影响（已标定） |

**阻力模型**：`C_D = c_D0 + c_Da2 * α² + c_Dds * δs`

### 失速模型参数


| 参数                | 默认值 | 单位 | 描述                 |
| ------------------- | ------ | ---- | -------------------- |
| `alpha_stall`       | 0.35   | rad  | 失速迎角（更温和，避免中等刹车出现“断崖”） |
| `alpha_stall_brake` | 0.02   | rad  | 刹车对失速角的减小量（更温和） |
| `alpha_stall_width` | 0.15   | rad  | 失速过渡宽度（更平滑） |
| `c_D_stall`         | 0.15   | -    | 失速附加阻力（更温和） |

### 侧力系数


| 参数   | 默认值 | 描述                                   |
| ------ | ------ | -------------------------------------- |
| `c_Yb` | -6.8   | 侧滑产生的侧力 [1/rad]，负值表示恢复力 |

**关键作用**：提供协调转弯能力，使飞行方向跟随偏航方向。

### 滚转力矩系数


| 参数    | 默认值 | 描述                       |
| ------- | ------ | -------------------------- |
| `c_lp`  | -0.84  | 滚转阻尼导数（负值=稳定）  |
| `c_lda` | -0.005 | 差动刹车产生的滚转         |

### 俯仰力矩系数


| 参数   | 默认值 | 描述                                |
| ------ | ------ | ----------------------------------- |
| `c_m0` | 0.1    | 零迎角俯仰力矩（抬头趋势）          |
| `c_ma` | -0.72  | 俯仰力矩斜率 [1/rad]（负值=静稳定） |
| `c_mq` | -1.49  | 俯仰阻尼导数                        |

**配平迎角**：`α_trim = -c_m0/c_ma ≈ 8°`

### 偏航力矩系数


| 参数        | 默认值 | 描述                                     |
| ----------- | ------ | ---------------------------------------- |
| `c_nr`      | -0.27  | 偏航阻尼导数（负值=稳定）                |
| `c_nda`     | -0.133 | 差动刹车产生的偏航（负值：左刹车→左转） |
| `c_nb`      | 0.15   | 侧滑产生的偏航（风-偏航耦合）            |
| `c_n_weath` | 0.02   | 风向舵效应（正值：逆风→顺风不稳定）     |

**偏航力矩模型**：

```
C_n = c_nr * r_hat + c_nda * δa + c_nb * β + c_n_weath * (wind_y_B / V)
```

**风向舵效应 (Weathercock)**：

- `c_n_weath > 0`：逆风平衡点不稳定，翼伞会自动转向顺风
- `c_n_weath = 0`：无风向舵效应，偏航稳定
- 物理解释：侧风作用于伞翼产生不对称阻力，引起偏航力矩

### 执行器参数


| 参数      | 默认值 | 单位 | 描述           |
| --------- | ------ | ---- | -------------- |
| `tau_act` | 0.2    | s    | 执行器时间常数 |

**执行器模型**：一阶滞后 `delta_dot = (delta_cmd - delta) / tau_act`

### 数值参数


| 参数    | 默认值  | 描述             |
| ------- | ------- | ---------------- |
| `eps`   | 1e-6    | 除法保护小量     |
| `V_min` | 1.0 m/s | 气动计算最小速度 |

### 传感器与 IMU

> 说明：`sensor.*` 的噪声标准差按“每个采样点”的高斯噪声配置；`/parafoil/imu` 默认按 `imu.publish_rate` 低频发布（~1Hz），更贴近实飞日志的采样频率。

| 参数 | 默认值 | 单位 | 描述 |
| ---- | ------ | ---- | ---- |
| `sensor.position_noise_std` | [0, 0, 0] | m | 位置噪声标准差 |
| `sensor.accel_noise_std` | [6.74, 7.30, 8.72] | m/s² | 加速度计噪声标准差（机体系比力） |
| `sensor.gyro_noise_std` | [0.520, 0.567, 0.769] | rad/s | 陀螺仪噪声标准差（机体系角速度） |
| `sensor.seed` | -1 | - | 随机种子（-1=随机） |
| `imu.publish_rate` | 1.0 | Hz | `/parafoil/imu` 发布频率（<=0 关闭） |
| `imu.frame_id` | `parafoil_body` | - | `/parafoil/imu` 的 frame_id |

---

## 风模型

风模型支持三种叠加成分：

### 1. 稳态风 (Steady Wind)

```yaml
wind:
  enable_steady: true
  steady_wind: [0.0, 2.0, 0.0]  # [N, E, D] m/s
```

**风向量约定**：`wind_I` 表示风的速度向量（风往哪吹）

- `[2, 0, 0]`：风向北吹（从南来的风）
- `[0, 2, 0]`：风向东吹（从西来的风）

### 2. 阵风 (Gust)

离散阵风，随机出现：

```yaml
wind:
  enable_gust: true
  gust_interval: 10.0    # 平均间隔 [s]
  gust_duration: 2.0     # 持续时间 [s]
  gust_magnitude: 3.0    # 最大风速 [m/s]
```

### 3. 有色噪声风 (Colored Noise)

一阶滤波白噪声，模拟持续湍流：

```yaml
wind:
  enable_colored: true
  colored_tau: 2.0       # 滤波时间常数 [s]
  colored_sigma: 1.0     # 标准差 [m/s]
```

### 风对动力学的影响

1. **相对速度**：`v_rel = v_I - wind_I`
2. **侧滑角**：侧风产生 `β ≠ 0`
3. **风-偏航耦合**：`c_nb * β` 产生偏航力矩
4. **风向舵效应**：`c_n_weath * wind_y_B/V` 产生持续偏航力矩

---

## 安装与运行

### 依赖

- ROS2 (Humble/Iron/Jazzy)
- Python >= 3.8
- numpy, pyyaml

### 编译

```bash
cd ~/parafoil_ws
source /opt/ros/humble/setup.zsh
colcon build --symlink-install
source install/setup.zsh
```

### 运行仿真

```bash
# 方式1：使用参数文件
ros2 run parafoil_simulator_ros sim_node --ros-args \
    --params-file src/parafoil_simulator_ros/config/params.yaml

# 方式2：使用 launch 文件
ros2 launch parafoil_simulator_ros sim.launch.py

# 方式3：使用默认参数
ros2 run parafoil_simulator_ros sim_node
```

### 键盘控制

```bash
ros2 run parafoil_simulator_ros keyboard_teleop
```


| 按键       | 功能                 |
| ---------- | -------------------- |
| `W` / `↑` | 双侧刹车（减速）     |
| `S` / `↓` | 释放双侧刹车（加速） |
| `A` / `←` | 拉左刹车（左转）     |
| `D` / `→` | 拉右刹车（右转）     |
| `Q`        | 释放左刹车           |
| `E`        | 释放右刹车           |
| `Space`    | 全部释放             |
| `R`        | 全刹车               |
| `ESC`      | 退出                 |

### RViz2 可视化

```bash
rviz2 -d ~/parafoil_ws/src/parafoil_simulator_ros/config/parafoil_rviz.rviz
```

### 发送控制命令

```bash
# 直飞（无刹车）
ros2 topic pub -r 20 /rockpara_actuators_node/auto_commands \
    geometry_msgs/msg/Vector3Stamped \
    "{vector: {x: 0.0, y: 0.0, z: 0.0}}"

# 左转（拉左刹车）
ros2 topic pub -r 20 /rockpara_actuators_node/auto_commands \
    geometry_msgs/msg/Vector3Stamped \
    "{vector: {x: 0.5, y: 0.0, z: 0.0}}"

# 右转（拉右刹车）
ros2 topic pub -r 20 /rockpara_actuators_node/auto_commands \
    geometry_msgs/msg/Vector3Stamped \
    "{vector: {x: 0.0, y: 0.5, z: 0.0}}"
```

---

## ROS2 接口

### 订阅话题


| 话题                                     | 类型                           | 描述                   |
| ---------------------------------------- | ------------------------------ | ---------------------- |
| `/rockpara_actuators_node/auto_commands` | `geometry_msgs/Vector3Stamped` | 控制命令：x=δL, y=δR |

### 发布话题 - 传感器数据


| 话题            | 类型                           | 坐标系 | 描述           |
| --------------- | ------------------------------ | ------ | -------------- |
| `/position`     | `geometry_msgs/Vector3Stamped` | NED    | 位置 [m]       |
| `/body_acc`     | `geometry_msgs/Vector3Stamped` | Body   | 比力 [m/s²]    |
| `/body_ang_vel` | `geometry_msgs/Vector3Stamped` | Body   | 角速度 [rad/s] |
| `/parafoil/imu` | `sensor_msgs/Imu`              | Body   | 低频 IMU（默认 1Hz），含 accel/gyro 噪声 |

### 发布话题 - RViz2 可视化


| 话题               | 类型                        | 坐标系 | 描述                   |
| ------------------ | --------------------------- | ------ | ---------------------- |
| `/parafoil/odom`   | `nav_msgs/Odometry`         | ENU    | 里程计                 |
| `/parafoil/path`   | `nav_msgs/Path`             | ENU    | 飞行轨迹               |
| `/parafoil/pose`   | `geometry_msgs/PoseStamped` | ENU    | 当前位姿               |
| `/parafoil/marker` | `visualization_msgs/Marker` | ENU    | 简易模型（canopy + payload + 连线） |
| `/tf`              | `tf2_msgs/TFMessage`        | -      | world → parafoil_body |

**注意**：RViz2 话题使用 ENU 坐标系（East-North-Up），与内部 NED 坐标系不同。

---

## 单元测试

```bash
cd ~/parafoil_ws/src/parafoil_dynamics
python3 -m pytest parafoil_dynamics/tests/ -v
```

---

## 典型飞行性能（默认参数）


| 指标       | 数值        | 条件         |
| ---------- | ----------- | ------------ |
| 空速       | ~4.44 m/s   | 零刹车       |
| 下沉率     | ~0.90 m/s   | 零刹车       |
| 滑翔比     | ~4.9        | 零刹车       |
| 典型转弯率 | ~49°/s      | `delta_a=0.5`, `b_sym≈0.25` |
| 失速迎角   | ~20°        | `alpha_stall≈0.35 rad` |
| 执行器响应 | ~0.6s (95%) | `tau_act=0.2s` |

更完整的极曲线/转弯率见：[当前翼伞性能（极曲线/转弯率）](#当前翼伞性能极曲线转弯率)。
