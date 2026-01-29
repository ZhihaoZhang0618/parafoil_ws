# 批量任务统计分析 Prompt

你是翼伞系统性能优化专家。请分析以下 N 次任务的统计数据。

## 汇总统计

```json
{
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
}
```

## 分析任务

1) 性能瓶颈识别
- 哪些场景下性能最差？
- 失败案例的共同模式是什么？

2) 参数敏感性分析
- 哪些参数对性能影响最大？
- 当前参数是否处于最优区间？

3) 改进优先级
- 应该优先改进 Planner 还是 Controller？
- 最有效的改进方向是什么？

4) 输出建议的参数调整

```json
{
  "parameter_recommendations": [
    {
      "parameter": "param_name",
      "current": 0.0,
      "suggested": 0.0,
      "expected_improvement": "...",
      "confidence": "low|medium|high"
    }
  ],
  "algorithm_recommendations": [
    {
      "component": "planner|controller|flare",
      "issue": "...",
      "suggestion": "..."
    }
  ]
}
```
