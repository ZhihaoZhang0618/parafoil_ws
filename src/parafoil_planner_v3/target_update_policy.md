# Prompt: 阵风鲁棒的目标点更新策略（方案 D）

## 1. 背景与问题

### 1.1 当前架构

翼伞自主迫降系统 `parafoil_planner_v3` 的目标点选择流程：

```
风估计 → LandingSiteSelector → 安全落点 → PlannerCore → 轨迹 → Guidance
         (1 Hz 重选)
```

相关代码：
- `parafoil_planner_v3/landing_site_selector.py` — 安全落点选择器
- `parafoil_planner_v3/nodes/planner_node.py` — 规划节点（调用 Selector）
- `parafoil_planner_v3/guidance/phase_manager.py` — 阶段管理（CRUISE/APPROACH/FLARE）
- `parafoil_planner_v3/guidance/wind_filter.py` — 风滤波器

### 1.2 问题描述

**阵风导致目标点频繁跳动**：

1. 阵风使风估计突变 → 可达域形状/大小变化
2. `LandingSiteSelector` 每周期（1 Hz）重新计算 → 安全落点位置变化
3. 目标点跳动 → 轨迹不连续 → 制导指令振荡 → 飞行不稳定

**期望行为**：

- 在合理范围内保持目标点稳定
- 仅当风场显著变化或当前目标变为不可达时才切换
- 末端阶段（APPROACH/FLARE）锁定目标点以保证跟踪精度

---

## 2. 方案 D：组合策略

### 2.1 策略组成

| 子策略 | 作用 | 触发条件 |
|--------|------|----------|
| **保守风模型** | 选择时用 `wind + gust_margin` 估计可达域 | 始终生效 |
| **滞后机制** | 新目标需显著优于当前才切换 | `Δscore > ε` 或 `Δdist > d_th` |
| **阶段锁定** | APPROACH/FLARE 阶段不更新目标 | `phase ∈ {APPROACH, FLARE}` |
| **紧急重选** | 当前目标变为不可达时强制重选 | `reach_margin < margin_emergency` |

### 2.2 决策流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    目标点更新决策（每规划周期）                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │ 当前目标是否仍可达？    │
                 │ margin >= margin_min   │
                 └───────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
              │ NO                          │ YES
              ▼                             ▼
    ┌─────────────────┐          ┌─────────────────────┐
    │ 紧急重选        │          │ 检查制导阶段        │
    │ 强制调用Selector│          │                     │
    │ 忽略滞后/锁定   │          └──────────┬──────────┘
    └────────┬────────┘                     │
             │                   ┌──────────┴──────────┐
             │                   │ FLARE               │ CRUISE/APPROACH
             │                   ▼                     ▼
             │         ┌─────────────────┐   ┌─────────────────────┐
             │         │ 完全锁定        │   │ 计算新候选目标      │
             │         │ 保持当前目标    │   │ new = Selector()    │
             │         └─────────────────┘   └──────────┬──────────┘
             │                                          │
             │                               ┌──────────┴──────────┐
             │                               │ APPROACH            │ CRUISE
             │                               ▼                     ▼
             │                    ┌─────────────────┐   ┌─────────────────┐
             │                    │ 仅紧急切换      │   │ 滞后判断        │
             │                    │ score_new <<    │   │ Δscore > ε OR   │
             │                    │ score_current   │   │ Δdist > d_th?   │
             │                    └────────┬────────┘   └────────┬────────┘
             │                             │                     │
             │                    ┌────────┴────────┐   ┌────────┴────────┐
             │                    │ 满足            │   │ YES       │ NO  │
             │                    ▼                 ▼   ▼           ▼     │
             │               切换目标          保持目标  切换    保持     │
             └───────────────────┴──────────────┴───────┴─────────────────┘
```

---

## 3. 详细设计

### 3.1 新增配置参数

在 `config/planner_params.yaml` 中新增 `target.update_policy` 节：

```yaml
parafoil_planner_v3:
  ros__parameters:
    target:
      auto_mode: "safety"    # manual | current | reach_center | safety

      update_policy:
        # === 保守风模型 ===
        # 已有参数，复用 safety.reachability.*
        # gust_margin_mps: 2.0
        # wind_uncertainty_mps: 0.5

        # === 滞后机制 ===
        enable_hysteresis: true
        score_hysteresis: 0.5        # score 差阈值（无量纲）
        dist_hysteresis_m: 20.0      # 位置差阈值（米）

        # === 阶段锁定 ===
        cruise_allow_update: true
        approach_allow_update: "emergency_only"  # true | false | emergency_only
        flare_lock: true

        # === 紧急重选 ===
        emergency_margin_mps: -0.5   # 可达裕度低于此值触发强制重选
        emergency_cooldown_s: 2.0    # 紧急重选后的冷却时间

        # === 平滑（可选）===
        smooth_transition: false     # 是否平滑过渡到新目标
        smooth_rate_mps: 5.0         # 目标移动速率上限
```

### 3.2 新增/修改模块

#### 3.2.1 `TargetUpdatePolicy` 类（新增）

路径：`parafoil_planner_v3/target_update_policy.py`

```python
@dataclass
class TargetUpdatePolicyConfig:
    enable_hysteresis: bool = True
    score_hysteresis: float = 0.5
    dist_hysteresis_m: float = 20.0
    cruise_allow_update: bool = True
    approach_allow_update: str = "emergency_only"  # true/false/emergency_only
    flare_lock: bool = True
    emergency_margin_mps: float = -0.5
    emergency_cooldown_s: float = 2.0
    smooth_transition: bool = False
    smooth_rate_mps: float = 5.0


class TargetUpdatePolicy:
    """目标点更新策略管理器"""

    def __init__(self, config: TargetUpdatePolicyConfig):
        self.config = config
        self._current_target: Optional[Target] = None
        self._current_score: float = float("inf")
        self._current_margin: float = 0.0
        self._last_emergency_time: Optional[float] = None
        self._locked: bool = False

    def update(
        self,
        new_selection: LandingSiteSelection,
        phase: GuidancePhase,
        current_time: float,
        current_margin: float,  # 当前目标的实时可达裕度
    ) -> Tuple[Target, str]:
        """
        决定是否切换目标点

        Returns:
            (target, reason): 选定的目标点及决策原因
        """
        ...

    def reset(self):
        """重置状态（任务开始时调用）"""
        ...

    def force_update(self, target: Target, score: float, margin: float):
        """强制设置目标（用于手动模式或初始化）"""
        ...
```

核心逻辑 `update()` 伪代码：

```python
def update(self, new_selection, phase, current_time, current_margin):
    # 1. 紧急检查：当前目标是否仍可达
    if current_margin < self.config.emergency_margin_mps:
        if self._in_cooldown(current_time):
            return self._current_target, "emergency_cooldown"
        self._trigger_emergency(new_selection, current_time)
        return new_selection.target, "emergency_reselect"

    # 2. 阶段锁定检查
    if phase == GuidancePhase.FLARE and self.config.flare_lock:
        return self._current_target, "flare_locked"

    if phase == GuidancePhase.APPROACH:
        mode = self.config.approach_allow_update
        if mode == "false" or mode is False:
            return self._current_target, "approach_locked"
        if mode == "emergency_only":
            # 仅当新目标显著优于当前（score 差 > 2*阈值）
            if self._is_significantly_better(new_selection, factor=2.0):
                self._switch_target(new_selection)
                return new_selection.target, "approach_significant_improvement"
            return self._current_target, "approach_hysteresis"

    # 3. CRUISE 阶段：滞后判断
    if self.config.enable_hysteresis:
        if not self._should_switch(new_selection):
            return self._current_target, "cruise_hysteresis"

    # 4. 切换目标
    self._switch_target(new_selection)
    return new_selection.target, "cruise_update"
```

#### 3.2.2 修改 `PlannerNode`

在 `planner_node.py` 中集成 `TargetUpdatePolicy`：

```python
class PlannerNode(Node):
    def __init__(self):
        ...
        # 新增
        self._target_policy = TargetUpdatePolicy(self._load_policy_config())
        self._current_phase = GuidancePhase.CRUISE

    def _planning_callback(self):
        ...
        # 原逻辑：直接使用 Selector 输出
        # selection = self._selector.select(...)
        # target = selection.target

        # 新逻辑：通过 Policy 过滤
        selection = self._selector.select(...)
        current_margin = self._compute_current_target_margin()
        target, reason = self._target_policy.update(
            selection,
            self._current_phase,
            self.get_clock().now().nanoseconds * 1e-9,
            current_margin,
        )
        self._log_target_decision(target, reason)
        ...

    def _phase_callback(self, msg: String):
        """订阅 /guidance_phase 更新阶段"""
        self._current_phase = GuidancePhase[msg.data.upper()]
```

#### 3.2.3 新增 `_compute_current_target_margin()` 方法

计算当前目标点在当前风场下的实时可达裕度：

```python
def _compute_current_target_margin(self) -> float:
    """计算当前目标的实时可达裕度"""
    if self._current_target is None:
        return float("-inf")
    target_xy = self._current_target.position_xy
    state = self._current_state
    wind = self._current_wind
    tgo = self._estimate_time_to_land(target_xy)
    if tgo <= 0:
        return float("-inf")
    d = target_xy - state.position_xy
    v_req = d / max(tgo, 1e-6)
    v_air, _ = self._polar.interpolate(brake=0.2)
    margin = v_air - np.linalg.norm(v_req - wind.v_I[:2])
    return float(margin)
```

### 3.3 可视化扩展（可选）

在 `safety_viz_node.py` 中增加目标点状态可视化：

- **绿色实心球**：当前锁定目标
- **绿色空心球**：Selector 推荐的新候选（仅当与当前不同时显示）
- **黄色连线**：当前目标 → 新候选（表示潜在切换）
- **文本标注**：`target: locked/cruise/emergency`

---

## 4. 测试验证

### 4.1 单元测试

新增 `tests/test_target_update_policy.py`：

```python
class TestTargetUpdatePolicy:
    def test_hysteresis_blocks_small_change(self):
        """小幅度变化不触发切换"""
        ...

    def test_hysteresis_allows_significant_change(self):
        """显著变化触发切换"""
        ...

    def test_flare_lock(self):
        """FLARE 阶段完全锁定"""
        ...

    def test_approach_emergency_only(self):
        """APPROACH 阶段仅紧急切换"""
        ...

    def test_emergency_reselect(self):
        """当前目标不可达时强制重选"""
        ...

    def test_emergency_cooldown(self):
        """紧急重选冷却时间"""
        ...
```

### 4.2 离线验证

新增或修改 `scripts/verify_target_stability.py`：

```bash
python3 scripts/verify_target_stability.py \
  --runs 50 \
  --gust-magnitude 3.0 \
  --gust-interval 5.0 \
  --output reports/target_stability.html
```

输出指标：
- **目标点切换次数**（越少越好）
- **切换时的 score 差**（应 > 阈值）
- **紧急重选次数**
- **最终着陆误差**

### 4.3 ROS2 场景验证

```bash
# 阵风场景
ros2 launch parafoil_planner_v3 scenario_gusty.launch.py \
  target_update_policy.enable_hysteresis:=true \
  target_update_policy.score_hysteresis:=0.5

# 对比：无滞后
ros2 launch parafoil_planner_v3 scenario_gusty.launch.py \
  target_update_policy.enable_hysteresis:=false
```

---

## 5. 实现步骤

### Phase 1：核心逻辑（最小可跑）

1. 新建 `target_update_policy.py`，实现 `TargetUpdatePolicy` 类
2. 修改 `planner_node.py`，集成 Policy
3. 添加配置参数到 `planner_params.yaml`
4. 编写基础单元测试

### Phase 2：阶段联动

1. 在 `planner_node.py` 订阅 `/guidance_phase`
2. 实现阶段依赖的锁定逻辑
3. 测试 CRUISE → APPROACH → FLARE 全流程

### Phase 3：紧急重选

1. 实现 `_compute_current_target_margin()`
2. 添加紧急重选逻辑和冷却机制
3. 测试极端风场变化场景

### Phase 4：验证与调优

1. 离线批量验证（不同阵风强度）
2. ROS2 场景验证
3. 参数调优（阈值、冷却时间等）

---

## 6. 注意事项

1. **线程安全**：`TargetUpdatePolicy` 需考虑多线程访问（规划回调 vs 阶段回调）
2. **状态持久化**：任务重启时需重置 Policy 状态
3. **日志记录**：每次目标决策需记录 reason，便于调试
4. **参数边界**：`score_hysteresis` 和 `dist_hysteresis_m` 不宜过大，否则响应太慢
5. **与现有逻辑兼容**：当 `target.auto_mode = "manual"` 时，Policy 应直接透传用户设定的目标

---

## 7. 预期效果

| 指标 | 无策略 | 方案 D |
|------|--------|--------|
| 目标切换次数（60s 阵风场景） | 15-30 次 | 2-5 次 |
| 着陆误差 | 5-15 m | 5-12 m（略增） |
| 轨迹平滑度 | 差（振荡） | 好 |
| 紧急重选响应 | N/A | < 1s |

---

*此 Prompt 用于指导 `parafoil_planner_v3` 目标点更新策略的实现。*
