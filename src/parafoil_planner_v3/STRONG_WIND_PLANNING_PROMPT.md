# Planner v3 Strong-Wind / Backward-Drift Prompt

## Summary (from the conversation starting at “库几乎都命中…问题可能出在控制/强逆风后退”)
- **Library match is not the bottleneck**: coarse library matches are frequent, yet landing error stays large. The mismatch is likely in **tracking/controls** and **strong headwind reachability**.
- **Strong headwind can produce backward ground speed**: when `|wind| > |airspeed|`, ground velocity cannot point upwind. The vehicle can only **reduce downwind drift**, not move upwind.
- **Landing error spikes** are consistent with “tracking a target that is physically unreachable upwind” and with **control that does not compensate wind in heading logic**.
- **Risk map + no‑fly polygons are already present** (Shenzhen-like demo). We should leverage existing risk grid + no‑fly constraints for safety-first target selection.
- **Reachability circle is wind‑shifted** and should be kept; it’s a conservative subset of the reachable set and useful for fast pruning.

## Planner v3 code structure (relevant pieces)
- **PlannerCore** (`parafoil_planner_v3/planner_core.py`)
  - Library match → optional GPM → fallback.
  - Uses `PlannerConfig` flags such as `use_library`, `enable_gpm_fine_tuning`, `library_*` gates.
- **LandingSiteSelector** (`parafoil_planner_v3/landing_site_selector.py`)
  - Risk grid + no‑fly + reachability circle (`center = wind * tgo`).
  - Current reachability uses `v_air` and `wind` to check a **wind‑shifted circle**, but does not explicitly gate **upwind‑unreachable** cases.
- **Guidance** (`parafoil_planner_v3/guidance/*` + `nodes/guidance_node.py`)
  - `track_point_control()` uses **ground heading** from velocity; no wind‑aware heading correction.
  - `ApproachGuidance` has strong wind strategies (projection/downwind bias/ground_speed), but does **not** change reachability.
- **Risk map + no‑fly** (Shenzhen-like demo)
  - Risk grid: `config/shenzhen_like_risk_grid_*.npz`
  - No‑fly polygons: `config/shenzhen_like_no_fly_polygons*.json`
  - Launch example in README already wired into `safety_demo.launch.py`.

## Problem framing
- **Planner currently assumes target is reachable** unless risk/no‑fly say otherwise.
- In strong headwind, **groundspeed along target direction can be negative**, so the target is **physically unreachable**.
- That can yield large landing errors even when library matches are “successful”.

## Requirements for the new logic (planning first, control later)
1. **Explicit strong-wind reachability gate** before planning:
   - Define `d_hat` toward target.
   - `v_g_along_max = V_air + dot(wind, d_hat)`.
   - If `v_g_along_max <= 0`, target is **upwind-unreachable**.
2. **Safety-first target re-selection** when unreachable:
   - If target is unreachable, switch to safety selector output (risk grid + no‑fly).
   - Prefer **downwind safe zones**; treat “fallback downwind drift” as an intentional strategy.
3. **Keep reachability circle** (wind‑shifted) as fast pruning:
   - It is a conservative subset of the reachable set; do not remove it.
4. **Do not change control yet** (only planning layer change for now).

## Implementation guidance (planning layer)
- Add a function in PlannerCore or LandingSiteSelector:
  - Compute `V_air` from Polar for configured brake.
  - If `v_g_along_max <= 0`, mark target unreachable.
- When unreachable:
  - Force `TargetUpdatePolicy` to use safety selection or `reach_center/downwind` mode.
  - Ensure logs/diagnostics state the reason (e.g., `reason=unreachable_wind`).
- Keep existing `risk_grid` and `no_fly` files as is.

## Acceptance criteria
- In strong headwind scenarios, planner should **avoid attempting unreachable upwind targets**.
- The selected landing point should move toward **lower-risk zones** in the risk grid, respecting no‑fly.
- Reachability visualization remains wind‑shifted and consistent with planner.

## Tests / verification ideas
- Offline: add a small test case where `|wind| > |airspeed|` and target is upwind.
  - Expect: planner switches to safety target; reason logged.
- Safety demo: verify target jumps to downwind safe region when headwind is too strong.
- Optional: log metric “v_g_along_max” to correlate with landing error.

## Notes
- The existing reachability circle already shifts with wind. The complaint about “moving circle” is expected because **wind + time‑to‑go** changes it.
- Risk grid + no‑fly polygons in Shenzhen-like demo are sufficient for safety‑first behavior; no new map is needed.

---

# Prompt (ready to use)

你是 parafoil_planner_v3 的开发助手。请在**规划层**优先实现“强逆风/倒退可达性”逻辑，不先改控制层。已知：轨迹库匹配几乎都命中，但落点误差大，怀疑原因是**强逆风上风不可达**与**风不被规划明确处理**。现有代码框架如下：
- PlannerCore：库匹配 → GPM → fallback
- LandingSiteSelector：风险栅格 + no‑fly + 风修正可达圆
- Guidance（L1 跟踪）目前不考虑风修正
- 深圳样例风险图和 no‑fly 已齐全（无需新地图）

请完成：
1) 在规划层加入**强逆风不可达判定**：
   - 计算 `v_g_along_max = V_air + dot(wind, d_hat)`
   - 若 `v_g_along_max <= 0`，标记目标不可达（上风不可达）
2) 不可达时触发**安全落点重选**（risk grid + no‑fly），并优先**下风安全区**
3) 保留现有**风修正可达圆**作为可达域子集/快速筛选
4) 给出清晰日志/状态字段（例如 `reason=unreachable_wind`）

交付要求：
- 只改规划层，不改控制层
- 用 Shenzhen-like 风险地图与 no‑fly 文件验证
- 提供最小可复现实验（离线或 safety_demo）

请输出：
- 代码修改点（文件/函数）
- 配置参数建议
- 验证步骤与预期结果
