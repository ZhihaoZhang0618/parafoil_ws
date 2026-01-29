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

## 坐标系

- **内部统一使用 NED**（北-东-下）。
- ROS2 中：
  - `/target` 为 ENU，节点内部会转换为 NED。
  - `/planned_trajectory` 发布为 ENU（RViz 兼容）。

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

## 相关阅读

- `README_SAFETY_FIRST.md`
- `SAFETY_FIRST_DESIGN.md`
- `../CLAUDE_PATH_PLANNING_README.md`
