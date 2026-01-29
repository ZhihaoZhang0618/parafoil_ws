# 任务日志分析 Prompt

你是翼伞自主着陆系统的性能分析专家。请分析以下任务日志，识别问题并给出诊断。

## 日志数据

```json
{{LOG_JSON}}
```

## 分析要求

### 1) 整体性能评估
- 着陆精度是否达标（如 CEP < 10m）？
- 着陆速度是否安全（垂直速度 < 2 m/s）？
- 任务是否成功完成？

### 2) Planner 诊断
- 求解时间是否过长（>1s）？
- 约束违反量是否过大（>0.1）？
- 终端误差是否过大？
- 是否有不必要的重规划？

### 3) Controller 诊断
- 横向跟踪误差是否过大（RMSE > 3m）？
- 高度跟踪误差是否过大（RMSE > 2m）？
- 控制量是否抖动或饱和？
- 阶段切换是否合理？
- 拉飘效果是否达标？

### 4) 阶段分析
- CRUISE / APPROACH / FLARE 阶段表现与问题点

## 输出格式

请以 JSON 形式输出：

```json
{
  "overall_assessment": "success|partial_success|failure",
  "landing_accuracy": {"error_m": 0.0, "status": "ok|warning|critical"},
  "landing_velocity": {"velocity_mps": 0.0, "status": "ok|warning|critical"},
  "planner_diagnosis": {
    "status": "healthy|degraded|failed",
    "issues": [
      {
        "type": "issue_type",
        "severity": "low|medium|high",
        "description": "...",
        "root_cause": "...",
        "recommendation": {
          "action": "adjust_parameter|modify_algorithm|investigate",
          "parameter": "param_name",
          "current_value": 0.0,
          "suggested_value": 0.0,
          "rationale": "..."
        }
      }
    ]
  },
  "controller_diagnosis": {
    "status": "healthy|degraded|failed",
    "issues": []
  },
  "phase_analysis": {
    "cruise": {"status": "...", "notes": "..."},
    "approach": {"status": "...", "notes": "..."},
    "flare": {"status": "...", "notes": "..."}
  },
  "priority_actions": [
    "最重要的改进建议 1",
    "最重要的改进建议 2"
  ]
}
```
