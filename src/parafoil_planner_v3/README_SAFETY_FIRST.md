# parafoil_planner_v3 — 安全优先规划架构总览 & 专业 Prompt

本 README 旨在：  
1) 总览当前 `parafoil_planner_v3` 架构；  
2) 给出“从命中目标 → 安全优先”的架构优化方向；  
3) 提供一份 **完整且专业的 Prompt**，用于驱动后续设计/优化。

---

## 1) 当前规划器架构（简要总览）

### 1.1 模块清单（核心逻辑）

- **Planner Core**  
  - `parafoil_planner_v3/planner_core.py`  
  - GPM 轨迹优化（`optimization/*`）  
  - 轨迹库匹配（`trajectory_library/*`）  
  - Fallback 直线轨迹（含风修正）

- **Guidance（CRUISE / APPROACH / FLARE）**  
  - `guidance/cruise_guidance.py`  
  - `guidance/approach_guidance.py`  
  - `guidance/flare_guidance.py`  
  - `guidance/phase_manager.py`  
  - 强风策略：ground-speed aimpoint / 强风 entry / 路径覆盖

- **Environment & Constraints**  
  - `environment.py`  
  - 支持地形：Flat / Plane / Grid  
  - 支持禁飞区：Circle / Polygon / GeoJSON

- **Trajectory Library**  
  - `trajectory_library/library_generator.py`（离线生成）  
  - `library_manager.py` / `trajectory_adapter.py` / `trajectory_metrics.py`

- **ROS2 Nodes**  
  - `nodes/planner_node.py`  
  - `nodes/guidance_node.py`  
  - `nodes/library_server_node.py`  
  - `nodes/mission_logger_node.py`

- **Offline 验证**  
  - `offline/e2e.py` + `scripts/verify_*.py`  

- **日志 + 报告**  
  - `logging/mission_logger.py`  
  - `reporting/report_utils.py`

### 1.2 现有数据流（简化）

```
Sensors/Odom/Wind --> PlannerCore (GPM/Library/Fallback)
                           |
                           v
                    /planned_trajectory
                           |
                           v
            Guidance (CRUISE/APPROACH/FLARE)
                           |
                           v
                 /control_command (actuators)
```

**约束**目前主要在 PlannerCore/GPM 层面生效（地形/禁飞区/终端方向），**Guidance** 更侧重跟踪与风修正。

---

## 2) 安全优先目标下的架构优化方向

### 2.1 目标变化

原目标：**“命中目标点”**  
新目标：**“在可达区域内选择最安全落点并安全着陆”**

因此系统应从 “目标点 → 轨迹规划” 改为：

> **风险地图 / 可达域 → 安全落点选择 → 规划/制导**

### 2.2 建议新增/重构的关键子模块

1) **Landing Site Selector（安全落点选择器）**  
   - 输入：可达域 + 风场 + 风险地图（人口密度/建筑高度/电力线/禁飞区）  
   - 输出：最优安全落点（含置信度/可达性证明）

2) **Reachability Estimator（可达域估计）**  
   - 基于风速/空速/地速与下降时间推估“可达扇形/半平面”  
   - 当风速 > 空速：自动切换为“下风可达域”  

3) **Risk Map Aggregator（风险地图融合）**  
   - 将建筑高度/人口密度/道路密度/禁飞区等融合为统一风险代价  
   - 支持硬约束（禁飞区/高压线/高楼）与软约束（人群密度）

4) **Planner 接口改造**  
   - Planner 不再直接接受“任意目标点”，而是使用 Selector 输出的**安全落点**  
   - 允许 Planner 在强风/不可达时触发“替代落点”或“安全退出策略”

5) **验证/评估框架扩展**  
   - 指标从 “落点误差” 转为 “安全指标 + 风险代价 + 可达率”
   - 示例指标：最小距离人群、路径穿越风险、违规率、可达失败率

---

## 3) 专业 Prompt（用于安全优先规划改造）

> **用途**：将本项目从“命中目标”重构为“安全优先落点选择 + 规避规划”。  
> **输出要求**：必须给出架构、模块、输入输出、算法、验证与参数配置方案。

```
你是翼伞自主迫降系统的规划架构专家。请基于本仓库 parafoil_planner_v3 的现有代码结构，
将规划目标从“命中指定目标点”改造为“安全优先落点选择并安全着陆”。

### 背景与约束
1) 本系统用于无人机失事后的可控降落伞。
2) 必须优先避开人员密集区、高楼大厦、高压电线、禁飞区等风险区域。
3) 风速可能接近或超过空速；此时迎风地速可能为 0 或负，应以地速可达性为准。
4) 仓库已支持：GPM 优化、轨迹库、地形/禁飞区（Circle/Polygon/GeoJSON）、CRUISE/APPROACH/FLARE 制导。
5) 当前 Planner 以目标点为输入，需要改为“安全落点选择器输出的落点”。

### 目标
设计一个“安全优先”规划架构，并给出可落地的最小修改方案：
- 安全落点选择（Landing Site Selector）
- 可达域估计（Reachability）
- 风险地图融合（Risk Map）
- 规划与制导接口改造
- 完整验证流程

### 输入信息（占位）
请假设或定义以下输入接口：
- 风场估计：wind(NED) + 风速/方向不确定性
- 空速估计：当前 polar table / 机动约束
- 地形：平面/栅格/高程图
- 风险：人口密度/建筑高度/高压线/禁飞区
- 任务限制：最低高度、安全降落半径、最大可接受风险

### 输出要求（必须完整）
请按以下结构输出：
1) **架构总览图（文字说明即可）**
2) **新增/改造模块列表**（路径、职责、输入/输出）
3) **核心算法设计**
   - 可达域估计方法（风>空速时的处理）
   - 风险代价函数设计（硬/软约束）
   - 安全落点搜索策略（网格/采样/优化）
4) **与现有模块的对接点**
   - PlannerCore / Guidance / Environment / Logging
5) **验证方案**
   - ROS2 与 OFFLINE 测试场景
   - 评估指标（风险/安全/可达率）
6) **配置与参数建议**
   - 新增 YAML 字段
   - 默认参数建议
7) **迭代路线图**
   - M1 最小可跑
   - M2 风场鲁棒
   - M3 城市复杂环境

### 约束与风格
- 不要泛泛而谈，必须贴合本仓库现有结构和代码路径。
- 设计要“最小改动 + 可运行”，不要一次性推翻全部架构。
- 给出具体接口示例（参数/数据格式）。
```

---

## 4) 下一步建议（按优先级）

1) **先新增 Landing Site Selector（最小可跑）**  
   - 在 `parafoil_planner_v3/` 下新增 `landing_site_selector.py`  
   - 输入：当前 state、wind、风险地图、可达域估计  
   - 输出：候选安全落点（NED）

2) **给 Planner 增加“安全落点输入接口”**  
   - 允许由 Selector 自动设置 target  
   - 保留原始 /target 入口作为 fallback

3) **补充验证脚本（OFFLINE）**  
   - 用随机风场 + 禁飞多边形 + 人群密度做批量评估

---

如需我继续，我可以直接落代码实现 **Landing Site Selector** + **风险代价函数** 的最小可运行版本，并更新 `README.md` 的安全优先流程。

---

## 5) 已落地的最小实现（2026-01-28）

- 新增：`parafoil_planner_v3/landing_site_selector.py`  
  - Reachability（风三角可达性）+ 风险栅格聚合 + 禁飞区软/硬约束  
- PlannerCore：支持接入 LandingSiteSelector 并用安全落点替换目标点  
- PlannerNode：新增 `safety.*` 参数，RViz 同时显示目标点（红）与安全落点（绿）  
- 默认参数：已写入 `config/planner_params*.yaml`（默认关闭，可按需启用）

## 6) 小 Demo（示例风险栅格）

已提供一个**示例风险栅格**与**生成脚本**，用于快速体验安全落点选择：

- 生成脚本：`scripts/generate_demo_risk_grid.py`
- 示例文件（已生成）：`config/demo_risk_grid.npz`

一键生成（可重复覆盖）：

```bash
python3 src/parafoil_planner_v3/scripts/generate_demo_risk_grid.py \
  --output src/parafoil_planner_v3/config/demo_risk_grid.npz
```

启用方式（planner 参数示例）：

```yaml
parafoil_planner_v3:
  ros__parameters:
    safety:
      enable: true
      risk:
        grid_file: "/home/aims/parafoil_ws/src/parafoil_planner_v3/config/demo_risk_grid.npz"
```

---

## 7) 安全优先框架补齐（风不确定性 + 可达域约束 + 能量代理）

针对“强风/阵风/风向误差”场景，框架补齐如下：

1) **保守风模型**  
   - 在可达性判断中引入 `wind_uncertainty_mps` 与 `gust_margin_mps`  
   - 解释：把风不确定性视作“额外风速消耗”，**收缩可达域**，保证保守

2) **可达域硬约束**  
   - 用“风漂移圆”表示可达域：`center = p + wind * t_go`  
   - 半径 `R = (V_air - wind_margin - wind_uncertainty - gust_margin) * t_go`  
   - Selector 仅在该圆内选择安全落点

3) **能量代理（避免高动能落地）**  
   - 使用“所需滑翔坡度” `k_req = H / S` 作为能量代理  
   - 倾向选择 `k_req` 更小（更平缓、更有时间耗能）的落点  
   - 由权重 `w_energy` 调节影响

### 关键参数（新增）

```yaml
safety:
  selector:
    w_energy: 0.5  # 能量代理权重（偏好平缓坡度）
  reachability:
    wind_uncertainty_mps: 0.0  # 风向/风速不确定性等效风速
    gust_margin_mps: 0.0       # 阵风裕度（最大风速超额）
    enforce_circle: true       # 强制落点必须位于可达圆内
target:
  auto_mode: "manual"          # manual|current|reach_center
```

### 可视化对齐

- 可达域圈与 Selector 使用 **相同风裕度与风不确定性参数**  
- 强风时圆心会显著下风漂移，半径随风裕度收缩
