# Planner v3 Strong-Wind Trajectory Library Prompt (Headwind / Backward-Drift)

> 目标：生成一套**能在强风（含 `|wind| >= |V_air|`）下仍可用**的离线轨迹库，使规划层在“安全落点优先”的框架里能够选出/匹配到**顶风减速（地速变小、甚至出现“机头顶风但地面仍下风漂移”的倒退现象）**的轨迹；并与 Planner v3 的现有上下游接口保持一致（特征索引、库格式、在线适配、可达性判定与日志字段）。

---

## 0) 术语与约定（必须统一）

1. **坐标系**：规划/库内部统一使用 **NED**：
   - `x=N (north)`, `y=E (east)`, `z=D (down)`；高度 `altitude = -p_D`
2. **风向约定**：库与规划内部统一使用 **wind-to**（空气团在惯性系中的速度）：
   - `wind.v_I = [w_N, w_E, w_D]`（通常 `w_D=0`）
   - 与 ROS 外部消息的 `to/from`、`ENU/NED` 转换必须在节点入口处理（本 prompt 不改 ROS 消息侧，只要求库内部一致）
3. **速度分解**：
   - 对空速度（空速矢量）`v_air`：由极曲线/动力学给出（与风无关）
   - 对地速度（地速矢量）`v_g = v_air_vec + wind_xy`
   - “倒退/倒飞”指：**机头（或对空速度方向）与对地速度方向相反/夹角大**（典型条件：`|wind| > |v_air|` 且对空速度朝上风）
4. **滑翔率**：
   - 对空滑翔率 `L/D_air = |v_air_xy| / sink`
   - 对地滑翔率 `L/D_ground = |v_g_xy| / sink`
5. **库检索特征（默认保持 5D，不轻易改动）**：
   - `[altitude, distance_to_target, bearing_to_target, wind_speed, relative_wind_angle]`
   - 其中 `relative_wind_angle = wind_angle - bearing_to_target`（wrap 到 `[-pi, pi]`）

---

## 1) 为什么现有库在强风下可能不够

在强风（尤其 `wind_ratio = |wind|/|V_air|` 接近或超过 1）时：
- 仅用 “DIRECT/S_TURN/RACETRACK/SPIRAL + 形状约束” 的库，可能**把需要大幅改航向/长时间顶风**的解过滤掉（例如 DIRECT 的 `net_turn` 上限很小）。
- warm-start（初始猜测）常假设“初始速度朝目标”，这在“需要顶风减速才能到达下风近距离落点”的场景里可能很差，导致 GPM 离线求解失败或解质量差。
- 在线适配 `adapt_trajectory()` 当前几乎不处理风差异；因此离线库需要覆盖更密的强风网格，或显式提供更稳健的风敏感轨迹族。

---

## 2) 需求（离线库必须覆盖的强风行为）

### 2.1 必须支持的轨迹行为（至少一种能稳定生成）
1. **顶风减速下风漂移（Backward-drift behavior）**
   - 在 `wind_ratio >= 1` 且目标在下风方向（或更一般地：到目标的“可达方向”受风强烈约束）时，
   - 轨迹应允许长时间保持对空速度指向上风（或接近上风），从而使对地沿风分量接近最小值 `|wind|-|V_air|`（仍为下风但更慢）。
2. **强风不可达过滤（与规划层一致）**
   - 对于“上风不可达”的场景（规划层会标记 `reason=unreachable_wind` 并换安全落点），库生成阶段应：
     - 要么不生成该场景（节省离线时间）
     - 要么生成但明确打标签 `scenario_unreachable_wind=true`，并在加载/索引阶段排除

### 2.2 输出元数据（强烈建议，便于上游调参/排障）
对每条库轨迹存储以下元数据（写入 `LibraryTrajectory.metadata` 或 `Trajectory.metadata` 均可）：
- `wind_ratio`（用库生成时的 `V_air(brake_sym)` 或 `V_air_max` 明确说明）
- `touchdown_tgo_est_s = altitude / sink`（库坐标里 target 高度为 0）
- `v_g_along_to_target_min/max`（统计轨迹上沿目标方向的地速分量范围）
- `v_g_along_wind_min/max`（统计沿风向的地速分量范围）
- `backward_drift_fraction`：统计 `dot(v_g_xy, v_air_xy_hat) < 0` 的时间占比（若能从状态估计 `v_air_xy_hat`）
- `shape_ok/shape_reason`（已有就沿用）
- `generation_mode`（gpm/template、simplified/6dof/mixed、solve_mode 等）

---

## 3) 实现策略（推荐方案）

### 3.1 场景网格与过滤（上游：`library_generator.py`）
在 `ScenarioConfig` 的枚举阶段加入**强风专用网格**（推荐非均匀）：
- wind_speeds：覆盖到 `>= V_air_max`（例如 `[0, 2, 4, 6, 8, 10, 12]`）
- wind_directions：至少 8 向（0/45/.../315）
- target_bearings：至少 8 向（同上）
- distances：同时覆盖“近距离下风”（会触发顶风减速需求）与“远距离大下风漂移”
- altitudes：覆盖低空（更容易不可达/需要快速策略）与中高空

对每个场景，在离线求解前做一次快速可行性判定（与规划层同构，但可更保守/更乐观要写清楚）：
- `tgo_est = altitude / sink(brake_sym)`（或用 `brake=0` 给乐观上界）
- `v_req_ground = (p_target - p0) / tgo_est`（库中 p_target=0）
- `v_req_air = v_req_ground - wind`
- 若 `||v_req_air_xy|| > V_air_max + eps`：标记不可达（建议直接跳过）

这样保证：库只覆盖规划层会用到的“可达目标”，不可达的交给规划层安全选点逻辑。

### 3.2 新的“强风顶风轨迹族”（不一定要新增 TrajectoryType，但要能稳定生成）

给出两条可选路径（二选一或都做）：

**方案 A（推荐，最小侵入）**：沿用现有 `TrajectoryType`，但在强风子集里放宽形状约束并改 warm-start
- 对强风场景（例如 `wind_ratio >= 0.8`）：
  - 放宽 DIRECT/S_TURN 的 `net_turn`/`total_turn` 上限，允许出现“先大转向顶风、再微调对准”的解
  - warm-start 的初始航向不再固定为“朝目标”，而是用 `v_req_air` 的方向：
    - `psi0 = atan2(v_req_air_E, v_req_air_N)`
    - 初始速度 `v0_xy = V_air(brake_sym) * [cos psi0, sin psi0]`
  - 必要时对 `u_ref` 做“先转向后保持”的 pattern（例如前 20% 时间给常值 `delta_a`，后面归零），以引导出顶风段

**方案 B（更清晰，可解释性更强）**：新增一个 TrajectoryType（例如 `UPWIND_HOLD` / `WIND_HOLD`）
- 扩展 `TrajectoryType` enum，并在：
  - `TrajectoryLibraryGenerator._path_xy()`
  - `_u_ref_pattern()`
  - `_shape_constraints_ok()`
  - 配置 `library_params_*.yaml` 的 `trajectory_types`
  中加上该类型
- 该类型的模板与约束目标是：
  - 轨迹中间大部分时间对空速度方向接近 `-wind_hat`（上风）
  - 允许地面仍整体向下风漂移（尤其 `wind_ratio>1`）
  - 末端允许对准/偏置以满足落点

两种方案都必须保证：库中能出现“顶风减速”轨迹，并且不会被 shape_enforce 过滤掉。

### 3.3 与在线匹配/适配的对齐（下游：PlannerCore/TrajectoryAdapter）
保持 5D 特征索引不变时，仍需对齐以下点：
- `Scenario.wind_direction_deg` 必须是 **wind-to**；`TrajectoryLibrary._scenario_to_features()` 与 `compute_scenario_features()` 的相对风角定义必须一致
- 强风库建议更密的 `wind_speed` 网格，否则在线 `adapt_trajectory()` 不处理风差异会导致误差放大
- 若引入新 `TrajectoryType`：
  - PlannerCore 的 `_evaluate_library_cost()` 不需要改动即可工作，但建议在日志里输出 `traj_type`
  - 若希望规划在强风下偏好“顶风减速型轨迹”，可在 PlannerCore 增加可选 cost 项：
    - 例如 `library_cost_w_downwind_speed`（惩罚沿风向的平均地速过大）
    - 或 `library_cost_w_backward_drift`（奖励 backward_drift_fraction 高）

### 3.4 与“强风不可达判定/安全选点”的对齐（规划侧可能需要的小修改）
如果规划器已实现 `reason=unreachable_wind` 并切换到安全落点：
- 库生成器应确保“不可达场景”不会被离线生成（或生成但标记并过滤）
- PlannerCore 侧建议：
  - 当 `LandingSiteSelection.reason == unreachable_wind` 时，优先使用安全落点（已实现则无需改）
  - 对“强风下的 aimpoint（上风偏置点）”要谨慎：
    - 如果新库/GPM 已显式建模风，规划目标建议仍以 touchdown 为主，aimpoint 仅作为可视化/引导辅助；避免“把 aimpoint 当作落点”导致落在风险更高处

---

## 4) 验收标准（必须可量化）

### 4.1 离线库质量
- 在 `wind_ratio >= 1` 的下风目标子集中，库里至少能生成一定比例的轨迹（例如 >70% 的可达场景成功生成）
- 轨迹元数据里 `backward_drift_fraction` 在强风样例中显著 > 0（说明确实出现“顶风但仍下风漂移”的段）

### 4.2 在线集成效果（Planner v3）
- 在强风场景：
  - `unreachable_wind` 能被正确触发（上风目标）
  - 对下风安全落点，库匹配/轨迹可行性通过率提高，落点误差不显著恶化
- 日志能定位问题：
  - 失败时能区分：`unreachable_wind` / `no_reachable_candidate` / `skip_library=...` / `control_rate` / `Vh_out_of_bounds` 等

---

## 5) 交付清单（代码层面）

必须包含：
- 新/更新的库生成配置（建议新增）：`src/parafoil_planner_v3/config/library_params_strongwind_headwind.yaml`
- `TrajectoryLibraryGenerator` / `GPMTrajectoryLibraryGenerator` 中强风 warm-start/过滤/轨迹族策略
- 至少 2 个单元测试：
  - “特征一致性”：库的 scenario->features 与在线 compute_scenario_features 一致（尤其 relative_wind）
  - “强风行为存在”：对一个构造场景生成轨迹后，元数据里 backward_drift_fraction > 0（或其它可证明“顶风减速”的指标）

---

# Prompt（可直接交给实现同学/另一个 AI）

你是 parafoil_planner_v3 的离线轨迹库（trajectory_library）生成器开发助手。目标是在**强风/逆风（含 wind_ratio>=1）**下，生成能支撑“安全降落优先”的轨迹库，使系统能在强风时产生“顶风减速、机头顶风但地面仍下风漂移”的轨迹，并与现有 Planner v3 上下游接口对齐。

请你在不破坏现有库格式与在线匹配流程的前提下完成：

1) 统一约定：库内部一律使用 NED + wind-to；明确 wind_direction_deg 的含义与在线 compute_scenario_features 的一致性。
2) 场景网格：扩展 wind_speeds 到 >= V_air_max，并覆盖下风近距离与远距离；保持 5D 特征索引不变（除非你能给出强理由并同步修改匹配侧）。
3) 强风可行性过滤：在离线求解前用 polar + tgo_est 做快速判定，跳过与规划侧 `unreachable_wind` 同类的不可达场景（或标记并在索引中排除）。
4) 生成“顶风减速轨迹族”：
   - 方案 A：在强风子集中放宽 DIRECT/S_TURN 的形状约束，并把 warm-start 的初始航向改成 `v_req_air` 的方向；必要时引入“先转向后保持”的 u_ref pattern。
   - 或方案 B：新增 TrajectoryType（如 UPWIND_HOLD），并实现其模板、u_ref、shape 约束与配置支持。
5) 元数据：为每条轨迹记录 wind_ratio、tgo_est、v_g_along_*、v_g_along_wind_*、backward_drift_fraction 等，便于在线 debug 与调参。
6) 下游对齐：
   - 确保 TrajectoryLibrary 的 KDTree 特征与在线 compute_scenario_features 完全一致。
   - 评估 PlannerCore 是否需要避免“把 aimpoint 当落点”的风险；如需要，给出最小修改建议（例如 aimpoint 只用于可视化/引导，不替换 touchdown_target）。
7) 验证：
   - 提供最小可复现配置（新增 YAML），用 `scripts/generate_library.py` 生成一个小库。
   - 跑 `python3 -m pytest src/parafoil_planner_v3/tests -q` 并新增至少 2 个测试覆盖强风逻辑。

交付物请给出：
- 修改的文件列表与关键函数
- 新增/修改的 YAML 配置
- 如何生成库与如何验证“顶风减速轨迹确实生成”的步骤与指标

