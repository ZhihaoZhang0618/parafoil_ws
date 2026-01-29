# 参数优化 Prompt

你是控制系统参数调优专家。基于历史数据，请优化以下参数。

## 当前参数

```yaml
{{PARAMS_YAML}}
```

## 历史性能数据

```json
{{HISTORY_JSON}}
```

## 参数边界

```yaml
{{BOUNDS_YAML}}
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
