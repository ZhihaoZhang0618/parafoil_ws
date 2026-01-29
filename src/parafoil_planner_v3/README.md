# parafoil_planner_v3

翼伞自主迫降规划与制导系统：**安全优先落点选择 + 轨迹规划 + 三阶段制导**。

## 这是什么

`parafoil_planner_v3` 是一个 ROS2 规划/制导包，面向翼伞失效迫降场景。它接收里程计、风估计和目标点，先做**安全落点选择**（可选），再进行**轨迹规划**，最后由**制导器**输出刹车指令。

## 关键特性

- **规划三层回退**：轨迹库匹配 → GPM 在线优化 → 直线回退（风修正）。
- **安全优先落点选择**：风险栅格 + 可达域 + 禁飞区约束 + 目标更新策略（滞后/锁定/紧急重选）。
- **三阶段制导**：CRUISE → APPROACH → FLARE → LANDED（含 ABORT）。
- **强风策略**：入场点切换、地速投影/下风偏置、最低地速保护。
- **ROS2 一键联动**：`planner_node`/`guidance_node`/`library_server_node`/`safety_viz_node`。

## 整体逻辑（建议先读这一节）

### 数据流

```
/parafoil/odom  /parafoil/imu  /wind_estimate  /target
            │         │            │             │
            └─────────┴────────────┴─────────────┘
                              ▼
                 PlannerNode (1 Hz)
     1) 状态估计 + 坐标转换(ENU→NED) + 噪声
     2) 目标点选择（target.auto_mode）
     3) 安全落点选择（可选）+ 目标更新策略
     4) PlannerCore 规划（库/GPM/回退）
                              ▼
                    /planned_trajectory (ENU)
                              │
                              ▼
                 GuidanceNode (20 Hz)
     1) PhaseManager 状态机
     2) 有轨迹则 L1 跟踪，无轨迹则直追目标
     3) CRUISE/APPROACH/FLARE 控制
                              ▼
         /control_command (δL, δR)
```

### PlannerNode 逻辑（`parafoil_planner_v3/nodes/planner_node.py`）

- **目标点来源**：
  - `target.auto_mode=manual`：使用 `/target` 或 `target.position_ned`。
  - `target.auto_mode=current`：落点取当前位置。
  - `target.auto_mode=reach_center`：落点取风漂移中心。
- **安全落点选择（可选）**：
  - `LandingSiteSelector` 从可达域中采样候选点，融合风险栅格与禁飞区惩罚，得到最优落点。
  - `TargetUpdatePolicy` 负责滞后/锁定/紧急重选，避免频繁跳变。
- **规划流程（PlannerCore）**：
  1) 轨迹库匹配（KNN）→ 适配 → 约束检查 → 成本排序
  2) 可选 GPM 精修（偏差过大时）
  3) 失败时 GPM 在线求解
  4) 仍失败则生成直线路径并做风漂移修正
- **输出**：
  - `/planned_trajectory` (`nav_msgs/Path`, ENU)
  - `/planner_status` (JSON 字符串)
  - RViz 预览 Marker（目标点红色，安全落点绿色）

### GuidanceNode 逻辑（`parafoil_planner_v3/nodes/guidance_node.py`）

- **相位切换**（`phase_manager.py`）：
  - CRUISE → APPROACH：距离接近且高度满足滑翔需求
  - APPROACH → FLARE：高度或距离触发
  - ABORT：低空且远离目标/滑翔比过大/阶段超时
- **跟踪策略**：
  - 有规划路径：用 L1 lookahead 跟踪（`track_point_control`）。
  - 无路径：直接跟踪目标点。
- **控制输出**：
  - CRUISE：自动消高（直飞/盘旋/S-turn/跑道型）
  - APPROACH：风修正 + 下滑道跟踪
  - FLARE：刹车斜坡（两种模式：spec_full_brake / touchdown_brake）

### 轨迹库服务器（`parafoil_planner_v3/nodes/library_server_node.py`）

- **用途**：独立加载轨迹库并提供 KNN 查询服务，便于外部调试/分析；不参与 Planner 的内部规划流程。
- **参数**：
  - `library_path`：轨迹库文件路径；为空则不加载。
- **服务**：
  - `/query_library`（`parafoil_msgs/QueryLibrary`）：输入 5D 特征 `[altitude_m, distance_m, bearing_rad, wind_speed_mps, wind_angle_rad]`，返回匹配索引、距离与 `trajectory_types`；`k<=0` 时默认取 5。
  - `/reload_library`（`std_srvs/Trigger`）：按当前 `library_path` 重新加载。
- **说明（业内常见做法）**：翼伞轨迹库检索一般采用“低维物理特征”索引（高度/距离/方位/风），以便在样本有限时保持可检索性与可解释性。该 5D 组合并非行业统一标准，但属于工程上常见的方案：用最小信息刻画**可达性**与**进场几何**，并与库生成阶段使用的特征保持一致。
- **启动逻辑**：`full_system.launch.py` / `planner.launch.py` / `e2e_verification.launch.py` 中通过 `start_library_server` 条件启动，默认 `true`，且与 `library_path` 复用同一路径。
- **与 Planner 的关系**：Planner 在 `use_library=true` 时自行从 `library_path` 加载库；关闭服务器不会影响规划，仅影响 `/query_library` 服务是否可用。

## 坐标系

- **内部统一使用 NED**（北-东-下）。
- ROS2 中：
  - `/target` 为 ENU，节点内部会转换为 NED。
  - `/planned_trajectory` 发布为 ENU（RViz 兼容）。
  - `/wind_estimate` 默认假定为 **NED + wind-to**（风向量表示空气团在惯性系中的速度，单位 m/s）。若你的风估计输出为 ENU 或 wind-from（气象约定），请在 `planner_params.yaml` / `guidance_params.yaml` 中配置 `wind.input_frame` 与 `wind.convention` 做转换。

## 风（Wind）配置

这一节专门说明 **风输入的约定/配置**，避免“坐标系/气象约定”混用导致规划与制导方向反了。

### 1) 内部约定：NED + wind-to（推荐）

- **内部统一**：`wind_ned_to = [north, east, down]`（m/s），表示“空气团在惯性系中的速度”（wind-to）。
- 关系式：`v_ground = v_air + wind_to`（地速 = 空速 + 风）。
- 若上游给的是 **wind-from**（气象约定，“风从哪里来”），内部会做：`wind_to = -wind_from`。

### 2) `/wind_estimate` 消息约定（Vector3Stamped）

`/wind_estimate` 使用 `geometry_msgs/Vector3Stamped`，我们约定向量分量含义如下：

- 当输入是 **NED**：`vector = [north, east, down]`，推荐 `header.frame_id="ned"`。
- 当输入是 **ENU**：`vector = [east, north, up]`，推荐 `header.frame_id="enu"`/`"world"`/`"map"`。

> 说明：`wind.input_frame=auto` 时会尝试用 `header.frame_id` 推断 ENU/NED（推断失败默认按 NED）。

### 3) 风输入源：topic 与默认值回退

Planner 与 Guidance 都有同一套风参数（分别在 `config/planner_params.yaml`、`config/guidance_params.yaml`）：

- `wind.use_topic=true`：订阅 `wind.topic`（默认 `/wind_estimate`）。
  - 若 **没有收到消息** 或 **超过 `wind.timeout_s` 未更新**，自动回退到 `wind.default_ned`。
- `wind.use_topic=false`：始终使用 `wind.default_ned`（离线验证/没接估计器时有用）。

注意：`wind.default_ned` 只是 **参数兜底**，不会“自动从仿真器读取”。如果你在跑仿真，建议启用 `/wind_estimate`（见下文的 `wind_estimator_node`）。

### 4) 上游格式转换（只描述“输入是什么”）

内部标准始终是 **NED + wind-to**；以下参数仅用于把“上游输入”转换成内部标准：

```yaml
wind:
  input_frame: "ned"   # ned|enu|auto
  convention: "to"     # to|from (from 会取反转换成 wind-to)
```

典型场景：
- **推荐/默认（本仓库仿真 + fake wind estimator）**：`input_frame=ned`，`convention=to`（不需要额外转换）。
- 若你的估计器输出是 **ENU + wind-from**：设置 `input_frame=enu`，`convention=from`。

### 5) 限幅、滤波与强风保护

- `wind.max_speed_mps`：对水平风（N/E）限幅；0 表示关闭（D 分量不处理）。
- Guidance 可启用风低通/抗突变：`wind.filter.*`（见 `config/guidance_params.yaml`）。
- 轨迹库在强风下更容易“选到勉强可行但误差大”的邻居，因此 PlannerCore 提供保护：
  - `library.skip_if_unreachable_wind=true`：当“沿目标方向的地速”过小/不可达时跳过轨迹库，直接走 GPM/回退链路。
  - `library.min_track_ground_speed_mps`：上述判定的最低地速阈值。
  - 相关诊断可在 `/planner_status` 里看到：`wind_src` / `wind_age` / `ratio` / `solver_info.message`。

### 6) Launch 中与风相关的参数（仿真）

在 `full_system.launch.py` / `e2e_verification.launch.py` 里：

- `start_wind_estimator:=true`（默认）：启动 `wind_estimator_node` 发布 `wind_topic`（默认 `/wind_estimate`）。
- `wind_enable_steady|wind_enable_gust|wind_enable_colored`、`wind_steady_n/e/d`、`wind_gust_*`、`wind_colored_*`、`wind_seed`：
  - 传给模拟器用于生成风场；
  - 同时也传给 `wind_estimator_node`，保证“仿真真值风”和“planner/guidance 使用的风话题”一致。

## 快速开始

### 构建

```bash
cd /home/aims/parafoil_ws
colcon build --packages-select parafoil_msgs parafoil_dynamics parafoil_planner_v3 --symlink-install
source install/setup.bash
```

### 启动全系统（含仿真）

```bash
ros2 launch parafoil_planner_v3 full_system.launch.py \
  run_sim:=true use_library:=false
```

### 设置目标点

```bash
# 方式 A：Topic（ENU）
ros2 topic pub -1 /target geometry_msgs/PoseStamped \
  "{pose: {position: {x: 150.0, y: 50.0, z: 0.0}}}"

# 方式 B：Service（NED）
ros2 service call /set_target parafoil_msgs/srv/SetTarget \
  "{north_m: 150.0, east_m: 50.0, down_m: 0.0}"
```

### 生成轨迹库（可选）

```bash
# coarse library (simplified, faster, wider coverage)
python3 $(ros2 pkg prefix parafoil_planner_v3)/share/parafoil_planner_v3/scripts/generate_library.py \
  --config $(ros2 pkg prefix parafoil_planner_v3)/share/parafoil_planner_v3/config/library_params.yaml \
  --output /tmp/parafoil_library_coarse.pkl

# fine library (high-risk focus, non-uniform grid, mixed 6DOF sampling)
python3 $(ros2 pkg prefix parafoil_planner_v3)/share/parafoil_planner_v3/scripts/generate_library.py \
  --config $(ros2 pkg prefix parafoil_planner_v3)/share/parafoil_planner_v3/config/library_params_full.yaml \
  --output /tmp/parafoil_library_fine.pkl

# optional 6-DOF validation library (small grid, extreme conditions)
python3 $(ros2 pkg prefix parafoil_planner_v3)/share/parafoil_planner_v3/scripts/generate_library.py \
  --config $(ros2 pkg prefix parafoil_planner_v3)/share/parafoil_planner_v3/config/library_params_6dof_validation.yaml \
  --output /tmp/parafoil_library_6dof_validation.pkl
```

为避免多进程 + BLAS 线程过度抢占导致变慢，建议在生成时限制数值库线程：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 $(ros2 pkg prefix parafoil_planner_v3)/share/parafoil_planner_v3/scripts/generate_library.py \
  --config $(ros2 pkg prefix parafoil_planner_v3)/share/parafoil_planner_v3/config/library_params_full.yaml \
  --output /tmp/parafoil_library_fine.pkl
```

估算生成耗时（抽样测速 + 外推）：

```bash
python3 $(ros2 pkg prefix parafoil_planner_v3)/share/parafoil_planner_v3/scripts/generate_library.py \
  --config $(ros2 pkg prefix parafoil_planner_v3)/share/parafoil_planner_v3/config/library_params_full.yaml \
  --output /tmp/parafoil_library_fine.pkl \
  --estimate --sample-fraction 0.01
```

## 配置文件一览

完整索引见：`config/README.md`

- `config/planner_params.yaml`：Planner 主参数（含安全落点选择 & 目标更新策略）。
- `config/planner_params_safety_demo.yaml`：安全 Demo 默认配置。
- `config/planner_params_strongwind.yaml`：强风场景参数。
- `config/gpm_params.yaml`：GPM 求解器权重与约束。
- `config/optimization.yaml`：优化策略参数（辅助）。
- `config/dynamics_params.yaml`：动力学/气动参数。
- `config/guidance_params.yaml`：CRUISE/APPROACH/FLARE 参数。
- `config/library_params.yaml`：轨迹库生成网格（coarse）。
- `config/library_params_full.yaml`：高风险 fine 网格（非均匀 + mixed 6DOF 抽检）。
- `config/library_params_6dof_validation.yaml`：小规模 6DOF 验证网格。

常用开关：

```yaml
# planner_params.yaml
use_library: true
library_path: "/tmp/parafoil_library.pkl"
library:
  coarse_path: "/tmp/parafoil_library_coarse.pkl"
  fine_path: "/tmp/parafoil_library_fine.pkl"
  require_coarse_match: true
  fallback_to_coarse: true

# 仅 coarse 库也可运行（fine 为空时自动跳过 fine 阶段）
use_library: true
library:
  coarse_path: "/tmp/parafoil_library_coarse.pkl"
  fine_path: ""
  require_coarse_match: true
  fallback_to_coarse: true

safety:
  enable: true
  risk:
    grid_file: "config/demo_risk_grid.npz"
```

## 安全优先落点选择（Safety First）

- **RiskGrid**：支持 `.npz/.yaml/.json` 风险栅格。
- **Reachability**：基于刹车档位估计可达圈 + 风裕度。
- **No-fly**：支持 Circle / Polygon / GeoJSON。
- **TargetUpdatePolicy**：相位锁定 + 滞后 + 紧急重选。

### RiskGrid 示例（`config/demo_risk_grid.npz`）

- **用途**：用于 Safety-first 落点选择的“风险代价图”（软约束）。当 `safety.enable=true` 且 `safety.risk.grid_file` 指向该文件时，Selector 会倾向选择低风险区域。
- **坐标与原点**：`origin_n/origin_e` 是风险栅格左下角（最小 N/E）的坐标，不是飞机/目标原点。默认示例覆盖约 `N/E ∈ [-200, 200)`，把 `(0,0)` 放在栅格中间便于 Demo 造型。
- **越界行为**：栅格范围外按 `safety.risk.oob_value` 处理（默认 1.0，等价“很危险”），真实任务建议按任务区域重建/平移栅格。
- **与 No-fly 的关系**：`constraints.no_fly_circles/no_fly_polygons` 是硬约束（落点/轨迹进入即判违规），RiskGrid 是软代价（同范围不冲突，硬约束优先）。`safety_demo.launch.py` 里的 `no_fly_*` 参数主要用于 `safety_viz_node` 可视化。
- **更大面积**：若要生成更大覆盖范围（例如约 `800m×800m`，仍把 `(0,0)` 放中间），可用 `generate_shenzhen_like_scene.py` 重新生成并指定：`--cells 400 --resolution 2.0 --origin-n -400 --origin-e -400`。

### 城市场景 Demo（Shenzhen-like，2D）

> 注意：这是**合成的示例场景**（“像深圳中心公园附近的复杂城市环境”风格），不是任何真实城市地图/数据。

包含的元素（以正方形/长方形为主）：
- **办公楼/居民楼**：作为 **no-fly 多边形（硬约束）**，并在 RiskGrid 中叠加“靠近更危险”的缓冲区。
- **高压线走廊**：作为贯穿区域的 **no-fly 多边形（硬约束）**，RiskGrid 中也赋高风险。
- **推荐落点空地**：公园/荒地作为 **低风险区域（软偏好）**；另外在公园内放置“水体”高风险块，避免落水。

相关文件（已生成）：
- 风险栅格：`src/parafoil_planner_v3/config/shenzhen_like_risk_grid_400x400.npz`（400×400，1m 分辨率，覆盖 `N/E ∈ [-200, 200)`）
- 禁飞多边形：`src/parafoil_planner_v3/config/shenzhen_like_no_fly_polygons.json`（带 `clearance_m`，坐标为 NED 平面 `(north,east)` 米）
- 对应参数：`src/parafoil_planner_v3/config/planner_params_safety_shenzhen_like.yaml`

重新生成（默认 400×400、1m）：

```bash
python3 /home/aims/parafoil_ws/src/parafoil_planner_v3/scripts/generate_shenzhen_like_scene.py \
  --output-npz /home/aims/parafoil_ws/src/parafoil_planner_v3/config/shenzhen_like_risk_grid_400x400.npz \
  --output-nofly /home/aims/parafoil_ws/src/parafoil_planner_v3/config/shenzhen_like_no_fly_polygons.json
```

生成更大覆盖（约 800m×800m，仍把 `(0,0)` 放中间）。**大范围会自动增加楼群/道路/电力走廊等元素**：

```bash
python3 /home/aims/parafoil_ws/src/parafoil_planner_v3/scripts/generate_shenzhen_like_scene.py \
  --output-npz /home/aims/parafoil_ws/src/parafoil_planner_v3/config/shenzhen_like_risk_grid_800x800.npz \
  --output-nofly /home/aims/parafoil_ws/src/parafoil_planner_v3/config/shenzhen_like_no_fly_polygons_800x800.json \
  --cells 400 --resolution 2.0 --origin-n -400 --origin-e -400
```

运行（带 RViz + SafetyViz 可视化风险/禁飞）：

```bash
cd /home/aims/parafoil_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch parafoil_planner_v3 safety_demo.launch.py \
  planner_params:=/home/aims/parafoil_ws/src/parafoil_planner_v3/config/planner_params_safety_shenzhen_like.yaml \
  risk_grid_file:=/home/aims/parafoil_ws/src/parafoil_planner_v3/config/shenzhen_like_risk_grid_800x800.npz \
  no_fly_polygons_file:=/home/aims/parafoil_ws/src/parafoil_planner_v3/config/shenzhen_like_no_fly_polygons_800x800.json
```

### 一键安全 Demo

```bash
ros2 launch parafoil_planner_v3 safety_demo.launch.py
```

该 launch 会额外启动 `safety_viz_node`，在 RViz 中可视化：风险热力图、可达域、禁飞区等。

## ROS2 接口速查

**订阅**
- `/parafoil/odom` (nav_msgs/Odometry)
- `/parafoil/imu` (sensor_msgs/Imu)
- `/wind_estimate` (geometry_msgs/Vector3Stamped)
- `/target` (geometry_msgs/PoseStamped)

**发布**
- `/planned_trajectory` (nav_msgs/Path)
- `/planned_path` (nav_msgs/Path, v2 兼容)
- `/control_command` (geometry_msgs/Vector3Stamped)
- `/guidance_phase` (std_msgs/String)
- `/planner_status` (std_msgs/String)
- `/trajectory_preview` (visualization_msgs/MarkerArray)

**服务**
- `/replan` (std_srvs/Trigger)
- `/set_target` (parafoil_msgs/SetTarget)
- `/query_library` (parafoil_msgs/QueryLibrary)
- `/reload_library` (std_srvs/Trigger)

## 仿真用风估计（可选）

`parafoil_planner_v3` 本身不做风估计，但提供一个仿真用的发布节点 `wind_estimator_node`（移植自 v2），用于在不接入真实估计器时发布 `/wind_estimate`（NED + wind-to）。

在 `e2e_verification.launch.py` / `full_system.launch.py` 中默认会启动该节点（`start_wind_estimator:=true`，默认 `wind_topic:=/wind_estimate`），并复用仿真的风参数（steady/gust/colored）。

## 验证与工具

```bash
# 单元测试
pytest src/parafoil_planner_v3/tests/ -v

# 端到端离线验证
python3 $(ros2 pkg prefix parafoil_planner_v3)/share/parafoil_planner_v3/scripts/verify_e2e.py --runs 20

# 安全功能验证
python3 $(ros2 pkg prefix parafoil_planner_v3)/share/parafoil_planner_v3/scripts/verify_safety.py --runs 50
```

更多脚本见 `src/parafoil_planner_v3/scripts/`。

## 目录速览

```
parafoil_planner_v3/
├── parafoil_planner_v3/        # 核心代码（planner/guidance/optimization/...）
├── config/                     # YAML 配置
├── launch/                     # ROS2 启动文件
├── scripts/                    # 验证与生成脚本
└── tests/                      # pytest
```

## 常见问题

- **规划结果为空或不稳定**：检查 `gpm_params.yaml` 约束是否过严，或风速是否大于空速。
- **安全落点频繁跳变**：增大 `target.update_policy.*` 的滞后阈值。
- **轨迹库加载失败**：确认 `library_path` 与库版本匹配，必要时重新生成。
- **风估计导致落点明显偏移**：优先确认 `/wind_estimate` 的坐标系与约定是否正确（默认 NED + wind-to）。可查看 `/planner_status` 中的 `wind_src/wind_age/ratio`，必要时配置 `wind.input_frame`（enu/ned/auto）、`wind.convention`（to/from）以及 `wind.timeout_s`。

## 相关阅读

- `README_SAFETY_FIRST.md`
- `SAFETY_FIRST_DESIGN.md`
- `../CLAUDE_PATH_PLANNING_README.md`
