# 参数优化 Prompt

你是控制系统参数调优专家。基于历史数据，请优化以下参数。

## 当前参数

```yaml
L1_distance: 20.0
flare:
  mode: spec_full_brake
  touchdown_brake_altitude_m: 0.2
  flare_ramp_time: 0.5
  flare_full_brake_duration_s: 3.0
  flare_initial_brake: 0.2
  flare_max_brake: 1.0
lateral:
  K_heading: 1.0
  max_delta_a: 0.6
  turn_rate_per_delta: 1.7
  yaw_rate_max: 1.57
  max_brake: 1.0

```

## 历史性能数据

```json
{
  "entries": [
    {
      "iteration": 0,
      "params": {
        "L1_distance": 20.0,
        "flare": {
          "mode": "spec_full_brake",
          "touchdown_brake_altitude_m": 0.2,
          "flare_ramp_time": 0.5,
          "flare_full_brake_duration_s": 3.0,
          "flare_initial_brake": 0.2,
          "flare_max_brake": 1.0
        },
        "lateral": {
          "K_heading": 1.0,
          "max_delta_a": 0.6,
          "turn_rate_per_delta": 1.7,
          "yaw_rate_max": 1.57,
          "max_brake": 1.0
        }
      },
      "summary": {
        "n_runs": 1,
        "success_rate": 0.0,
        "landing_error": {
          "mean": 163.3407462102353,
          "p50": 163.3407462102353,
          "p95": 163.3407462102353,
          "max": 163.3407462102353
        },
        "touchdown_error": {
          "mean": 163.3407462102353,
          "p95": 163.3407462102353
        },
        "vertical_velocity_mps": {
          "mean": 1.1316968866644563,
          "p95": 1.1316968866644563
        },
        "touchdown_vertical_velocity_mps": {
          "mean": 1.1316968866644563,
          "p95": 1.1316968866644563
        },
        "mission_time_s": {
          "mean": 1769612165.463376,
          "p95": 1769612165.463376
        },
        "replan_count": {
          "mean": 0.0,
          "p95": 0.0
        },
        "control_effort_mean": {
          "mean": 0.32804055443381447,
          "p95": 0.32804055443381447
        },
        "phase_durations_s_mean": {}
      },
      "score": 163.3407462102353,
      "constraints_ok": false,
      "timestamp": 1769612976.922132
    }
  ]
}
```

## 参数边界

```yaml
L1_distance:
- 8.0
- 35.0
flare.flare_full_brake_duration_s:
- 2.0
- 5.0
flare.touchdown_brake_altitude_m:
- 0.05
- 1.0
flare.flare_ramp_time:
- 0.05
- 1.0
flare.flare_initial_brake:
- 0.1
- 0.4
flare.flare_max_brake:
- 0.6
- 1.0
lateral.K_heading:
- 0.5
- 2.0
lateral.max_delta_a:
- 0.2
- 0.8
lateral.turn_rate_per_delta:
- 1.0
- 3.0
lateral.yaw_rate_max:
- 0.8
- 2.5
lateral.max_brake:
- 0.6
- 1.0

```

## 优化目标

1) 最小化落点误差（或主目标指标）
2) 约束：
   - 着陆垂直速度 < 2 m/s
   - 求解时间 < 1.0 s
   - 控制量变化率 < 0.5/s

## 输出

```json
{
  "optimization_method": "grid|bayesian|heuristic",
  "suggested_parameters": {
    "param_a": 0.0,
    "param_b": 0.0
  },
  "expected_improvement": {
    "primary_metric": {"from": 0.0, "to": 0.0}
  },
  "confidence": 0.0,
  "next_experiments": [
    {"param_a": 0.0, "rationale": "..."}
  ]
}
```
