# 代码改进建议 Prompt

你是翼伞系统的架构与算法专家。基于诊断结果，请建议具体的代码改进。

## 诊断结果

```json
{{DIAGNOSIS_JSON}}
```

## 代码片段

```text
{{SOURCE_CONTEXT}}
```

## 任务

1) 指出可能导致问题的代码位置
2) 给出改动建议（伪 diff 或伪代码）
3) 解释预期效果与风险

## 输出格式

```json
{
  "code_changes": [
    {
      "file": "path/to/file.py",
      "issue": "...",
      "location": "line xx-yy",
      "current_code": "...",
      "suggested_code": "...",
      "explanation": "..."
    }
  ],
  "new_features": [
    {
      "description": "...",
      "implementation_sketch": "..."
    }
  ]
}
```
