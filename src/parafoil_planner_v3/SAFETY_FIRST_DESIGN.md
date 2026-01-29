# 安全优先落点选择与目标更新策略 — 设计说明（基于现有仓库）

> 适用范围：`parafoil_planner_v3` 安全优先规划改造（最小改动 + 可运行）。
> 数据来源：仓库内 `README.md`（极曲线/转弯率）、`README_SAFETY_FIRST.md`（安全优先架构要求）、`prompts/target_update_policy.md`（阵风鲁棒目标更新策略）。
> 日期：2026-01-28。

---

## 1) 架构总览图（文字说明即可）

```
Sensors/Odom/Wind  RiskMap(Grid/GeoJSON)  Terrain/No-fly
        │                   │                    │
        └────────┬──────────┴──────────┬─────────┘
                 ▼                     ▼
         Reachability Estimator    Risk Map Aggregator
                 │                     │
                 └──────────┬──────────┘
                            ▼
                 LandingSiteSelector (安全落点)
                            │
                            ▼
                 TargetUpdatePolicy (阵风鲁棒)
                            │
                            ▼
             PlannerCore (Library/GPM/Fallback)
                            │
                            ▼
          Guidance (CRUISE/APPROACH/FLARE)
                            │
                            ▼
                    /control_command
```

**核心变化**：目标点不再直接来自 `/target`，而是由 **Selector + Policy** 输出“安全落点”，再进入 PlannerCore；CRUISE/APPROACH/FLARE 阶段通过 Policy 控制目标更新频率与稳定性。

---

## 2) 新增/改造模块列表（路径、职责、输入/输出）

| 模块/路径 | 职责 | 主要输入 | 主要输出 |
|---|---|---|---|
| `parafoil_planner_v3/landing_site_selector.py` | 安全落点选择（可达域 + 风险融合 + 禁飞区过滤） | `State`, `Wind`, `TerrainModel`, `NoFly*`, RiskGrid | `LandingSiteSelection(target, score, margin, metadata)` |
| `parafoil_planner_v3/target_update_policy.py` | 阵风鲁棒目标更新（滞后、阶段锁定、紧急重选） | `LandingSiteSelection`, `GuidancePhase`, `current_margin` | `(Target, UpdateReason)` |
| `parafoil_planner_v3/environment.py` | 地形/禁飞区模型（Circle/Polygon/GeoJSON） | terrain/no-fly 配置 | `TerrainModel`, `NoFlyCircle/Polygon` |
| `parafoil_planner_v3/nodes/planner_node.py` | 组装 Selector + Policy + PlannerCore，发布轨迹 | odom/imu/wind/target/params | `/planned_trajectory` |
| `parafoil_planner_v3/planner_core.py` | 轨迹库 + GPM + 直线回退 | `Target`, `Wind`, `State` | 轨迹（waypoints） |
| `parafoil_planner_v3/guidance/*` | 三阶段制导 + 风修正 | `/planned_trajectory` | `/control_command` |
| `parafoil_planner_v3/reporting/report_utils.py` | 指标统计与可视化 | log/metrics | 报告/图表 |
| `scripts/generate_demo_risk_grid.py` | 示例风险栅格生成 | 输出路径 | `.npz` risk grid |

**接口示例（输入/输出数据格式）**

- 风场：`Wind(v_I=[vN, vE, vD])`，单位 m/s，NED。
- 风不确定性：`wind_uncertainty_mps`, `gust_margin_mps`（作为保守裕度）。
- 风险图（npz）：`risk` 或 `risk_map`(H×W), `origin_n`, `origin_e`, `resolution_m`。
- 极曲线（来自 `parafoil_planner_v3/dynamics/aerodynamics.py`）：

| brake | airspeed (m/s) | sink (m/s) |
|---:|---:|---:|
| 0.0 | 4.44 | 0.90 |
| 0.1 | 4.19 | 1.03 |
| 0.2 | 3.97 | 1.13 |
| 0.3 | 3.78 | 1.20 |
| 0.4 | 3.61 | 1.26 |
| 0.5 | 3.47 | 1.30 |
| 0.6 | 3.33 | 1.33 |
| 0.7 | 3.22 | 1.36 |
| 0.8 | 3.11 | 1.39 |
| 0.9 | 3.01 | 1.40 |
| 1.0 | 2.92 | 1.42 |

- 转弯率近似（用于约束/评估）：`yaw_rate ≈ -1.7 * delta_a` (rad/s)。

---

## 3) 核心算法设计

### 3.1 可达域估计（风 > 空速时处理）

1) **估计落地时间**

```
H_agl = altitude - terrain_h - clearance
sink = PolarTable.interpolate(brake).sink
t_go = H_agl / sink
```

2) **可达裕度**

```
v_req = (p_target_xy - p_xy) / t_go
margin = V_air - ||v_req - wind_xy|| - wind_uncertainty - gust_margin
ok if margin >= wind_margin_mps
```

3) **可达域圆（强风与保守处理）**

```
center = p_xy + wind_xy * t_go
radius = (V_air - wind_margin - wind_uncertainty - gust_margin) * t_go
```

- 若 `radius <= 0`：只能在下风方向、接近 `center` 的区域选择（强风极限）。
- `enforce_circle = true` 时，Selector 仅在圆内采样候选点，保证保守。

### 3.2 风险代价函数（硬/软约束）

- **硬约束**：进入禁飞区（Circle/Polygon）立即剔除。
- **软约束**：禁飞区边缘缓冲 `nofly_buffer_m`，代价线性上升。
- **风险融合**：

```
risk = Σ (weight_i * clip(risk_i, clip_min, clip_max)) + nofly_weight * nofly_penalty
```

### 3.3 安全落点搜索策略（网格/采样/优化）

- **候选生成**：在可达域圆内按 `grid_resolution_m` 网格采样；候选过多时随机采样（上限 `max_candidates`）。
- **评分函数（越小越好）**：

```
score = w_risk * risk
      + w_distance * dist_cost
      + w_reach_margin * margin_cost
      + w_energy * energy_cost
```

其中：
- `dist_cost = ||p - p_desired|| / radius`（偏离期望目标的惩罚）
- `margin_cost = 1 - clip(margin / V_air, 0..1)`（可达裕度越大越好）
- `energy_cost = clip(k_req / k_nom, 0..3)`
  - `k_req = H_agl / S_rem`（所需滑翔坡度）
  - `k_nom` 由极曲线（默认 brake=0.2）给出

### 3.4 目标点更新策略（方案 D：阵风鲁棒）

- **保守风模型**：选择候选时使用 `wind_uncertainty_mps` + `gust_margin_mps` 收缩可达域。
- **滞后机制**：新目标需显著优于当前才切换：
  - `Δscore > score_hysteresis` 或 `Δdist > dist_hysteresis_m`
- **阶段锁定**：
  - `FLARE`：完全锁定
  - `APPROACH`：仅显著改善时切换
- **紧急重选**：当前目标不可达时（`margin < emergency_margin_mps`）强制重选；带冷却时间。

---

## 4) 与现有模块的对接点

- **PlannerNode**：
  - 读取 `safety.*` + `target.update_policy.*` 参数；
  - 构建 `RiskMapAggregator` 与 `LandingSiteSelector`；
  - 在 1 Hz 规划周期中：`selection = selector.select(...)` → `policy.update(...)` → `PlannerCore.plan(...)`。
- **PlannerCore**：目标从安全落点输入（可保留 `/target` 作为 fallback）。
- **Guidance / PhaseManager**：
  - 发布当前阶段给 PlannerNode，驱动 `TargetUpdatePolicy` 锁定逻辑。
- **Environment**：禁飞区与地形在 Selector 与 PlannerCore 中同时生效（保持一致的硬约束）。
- **Logging / Reporting**：记录 `selection.reason`、`UpdateReason`、`risk`、`margin`，用于评估与回归。

---

## 5) 验证方案

### 5.1 ROS2 场景

- **Calm / Crosswind / Gusty / Strongwind** 四类风场对比；
- **城市风险场景**：多 polygon 禁飞区 + 风险栅格。

### 5.2 Offline 批量验证

- `offline/e2e.py` + `scripts/verify_*.py`：
  - 目标切换次数
  - 平均风险/最大风险
  - 可达失败率
  - 落点误差
  - 轨迹违规率（禁飞区/地形）

### 5.3 关键指标（建议阈值）

- 目标切换次数（60s 阵风场景）：≤ 5 次
- 可达失败率：≤ 2%
- 路径风险积分：< 基线 50%
- 禁飞区违规率：0%

---

## 6) 配置与参数建议（新增 YAML 字段 + 默认值）

```yaml
parafoil_planner_v3:
  ros__parameters:
    target:
      auto_mode: "safety"          # manual|current|reach_center|safety
      update_policy:
        enabled: true
        enable_hysteresis: true
        score_hysteresis: 0.5
        dist_hysteresis_m: 20.0
        cruise_allow_update: true
        approach_allow_update: "emergency_only"
        flare_lock: true
        emergency_margin_mps: -0.5
        emergency_cooldown_s: 2.0
        approach_significant_factor: 2.0

    safety:
      enable: true
      selector:
        grid_resolution_m: 20.0
        max_candidates: 800
        w_risk: 5.0
        w_distance: 1.0
        w_reach_margin: 1.0
        w_energy: 0.5
        nofly_buffer_m: 20.0
        nofly_weight: 5.0
      reachability:
        brake: 0.2
        wind_margin_mps: 0.2
        wind_uncertainty_mps: 0.5
        gust_margin_mps: 0.5
        enforce_circle: true
      risk:
        grid_file: "/home/aims/parafoil_ws/src/parafoil_planner_v3/config/demo_risk_grid.npz"
        grid_weight: 1.0
        clip_min: 0.0
        clip_max: 1.0
        oob_value: 1.0
```

**默认值建议**
- `reachability.brake=0.2`：与当前极曲线一致（速度 3.97 m/s，sink 1.13 m/s）。
- `wind_uncertainty_mps` + `gust_margin_mps`：风不确定性与阵风裕度（强风场景应增大）。
- `score_hysteresis/dist_hysteresis_m`：兼顾目标稳定与响应速度。

---

## 7) 迭代路线图

- **M1 最小可跑（当前）**
  - LandingSiteSelector + 风险栅格 + 禁飞区过滤
  - PlannerCore 接入安全落点
  - 基础 metrics 与日志

- **M2 风场鲁棒**
  - TargetUpdatePolicy 完整接入（阶段锁定 + 紧急重选）
  - 风不确定性评估（wind_uncertainty + gust_margin）
  - 批量回归测试脚本

- **M3 城市复杂环境**
  - 多源风险融合（人口/建筑/道路）
  - 动态风险层（时间相关）
  - 风场时空变化下的鲁棒可达域

---

如需进一步落代码或将 `target.update_policy` 接入 `planner_node.py`，我可以基于现有实现直接补齐。
