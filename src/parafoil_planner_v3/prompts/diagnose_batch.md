# 批量任务统计分析 Prompt

你是翼伞系统性能优化专家。请分析以下 N 次任务的统计数据。

## 汇总统计

```json
{{SUMMARY_JSON}}
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
