#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from parafoil_planner_v3.optimization.auto_optimizer import AutoOptimizer, ExecutionMode, OptimizationConfig, PromptLibrary
from parafoil_planner_v3.optimization.llm_client import call_llm, extract_yaml_block


def _share_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _get_metric(summary: dict, key: str) -> float | None:
    cur = summary
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _write_report(
    output_dir: Path,
    history: dict,
    primary_key: str,
    start_time: float,
    end_time: float,
) -> None:
    entries = history.get("entries", [])
    if not entries:
        return
    best = min(entries, key=lambda e: float(e.get("score", float("inf"))))
    initial = entries[0]
    best_summary = best.get("summary", {})
    initial_summary = initial.get("summary", {})

    best_metric = _get_metric(best_summary, primary_key)
    init_metric = _get_metric(initial_summary, primary_key)
    success_best = best_summary.get("success_rate")
    success_init = initial_summary.get("success_rate")
    v_best = _get_metric(best_summary, "touchdown_vertical_velocity_mps.p95")
    v_init = _get_metric(initial_summary, "touchdown_vertical_velocity_mps.p95")

    lines = [
        "# 自动优化报告",
        "",
        "## 优化概要",
        f"- 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}",
        f"- 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}",
        f"- 迭代次数: {len(entries)}",
        "",
        "## 性能改进",
        "",
        "| 指标 | 初始值 | 最终值 | 改进 |",
        "|------|--------|--------|------|",
        f"| {primary_key} | {init_metric} | {best_metric} |  |",
        f"| touchdown_vertical_velocity_mps.p95 | {v_init} | {v_best} |  |",
        f"| success_rate | {success_init} | {success_best} |  |",
        "",
        "## 参数变化",
        "",
        "| 参数 | 初始值 | 最终值 |",
        "|------|--------|--------|",
    ]

    def _flatten(prefix: str, obj: dict) -> list[tuple[str, float]]:
        items = []
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                items.extend(_flatten(key, v))
            else:
                items.append((key, v))
        return items

    init_params = dict(initial.get("params", {}) or {})
    best_params = dict(best.get("params", {}) or {})
    init_flat = dict(_flatten("", init_params))
    best_flat = dict(_flatten("", best_params))
    for k in sorted(best_flat.keys()):
        lines.append(f"| {k} | {init_flat.get(k)} | {best_flat.get(k)} |")

    (output_dir / "optimization_report.md").write_text("\n".join(lines))


def _merge_dict(dst: dict, src: dict) -> dict:
    out = dict(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _apply_params_to_yaml(path: Path, params: dict) -> None:
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    if raw is None:
        raw = {}
    # unwrap ros__parameters if present
    if "parafoil_planner_v3" in raw and isinstance(raw["parafoil_planner_v3"], dict):
        node = raw["parafoil_planner_v3"]
        ros_params = node.get("ros__parameters", {})
        node["ros__parameters"] = _merge_dict(ros_params, params)
        raw["parafoil_planner_v3"] = node
    elif "parafoil_guidance_v3" in raw and isinstance(raw["parafoil_guidance_v3"], dict):
        node = raw["parafoil_guidance_v3"]
        ros_params = node.get("ros__parameters", {})
        node["ros__parameters"] = _merge_dict(ros_params, params)
        raw["parafoil_guidance_v3"] = node
    elif "ros__parameters" in raw and isinstance(raw["ros__parameters"], dict):
        raw["ros__parameters"] = _merge_dict(raw["ros__parameters"], params)
    elif isinstance(raw, dict):
        raw = _merge_dict(raw, params)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))


def main() -> None:
    share_dir = _share_dir()
    default_cfg = share_dir / "config" / "optimization.yaml"
    default_prompts = share_dir / "prompts"

    parser = argparse.ArgumentParser(description="Auto optimization for parafoil_planner_v3 (offline-first).")
    parser.add_argument("--config", type=str, default=str(default_cfg))
    parser.add_argument("--mode", type=str, choices=["offline", "ros2"], default="offline")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--runs-per-iter", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dynamics-mode", type=str, default=None)
    parser.add_argument("--record-history", action="store_true")
    parser.add_argument("--no-record-logs", action="store_true")
    parser.add_argument("--output", type=str, default="optimization_results")
    parser.add_argument("--ai-mode", type=str, choices=["none", "prompt", "api"], default="prompt")
    parser.add_argument("--prompts-dir", type=str, default="")
    parser.add_argument("--ros2-log-dir", type=str, default="")
    parser.add_argument("--llm-api-base", type=str, default="")
    parser.add_argument("--llm-api-key", type=str, default="")
    parser.add_argument("--llm-model", type=str, default="")
    parser.add_argument("--apply-llm", action="store_true", help="Apply LLM-suggested params to target YAML.")
    parser.add_argument("--apply-target", type=str, default="", help="YAML path to update when --apply-llm is set.")
    args = parser.parse_args()

    cfg = OptimizationConfig.from_yaml(args.config)

    if args.iterations is not None:
        cfg.settings["max_iterations"] = int(args.iterations)
    if args.runs_per_iter is not None:
        cfg.settings["runs_per_iteration"] = int(args.runs_per_iter)
    if args.parallel is not None:
        cfg.settings["parallel"] = int(args.parallel)
    if args.seed is not None:
        cfg.settings["seed"] = int(args.seed)
    if args.dynamics_mode is not None:
        cfg.simulation["dynamics_mode"] = str(args.dynamics_mode)
    if args.ros2_log_dir:
        cfg.settings["ros2_log_dir"] = str(args.ros2_log_dir)

    mode = ExecutionMode.OFFLINE if args.mode == "offline" else ExecutionMode.ROS2
    iterations = int(cfg.settings.get("max_iterations", 10))
    runs_per_iter = int(cfg.settings.get("runs_per_iteration", 20))
    parallel = int(cfg.settings.get("parallel", 1))
    record_history = bool(args.record_history or cfg.settings.get("record_history", False))
    record_logs = not bool(args.no_record_logs)

    output_dir = Path(args.output)
    start_time = time.time()
    optimizer = AutoOptimizer(cfg, mode=mode)
    history = optimizer.run(
        iterations=iterations,
        runs_per_iter=runs_per_iter,
        parallel=parallel,
        record_history=record_history,
        output_dir=output_dir,
        record_logs=record_logs,
    )
    end_time = time.time()

    hist_dict = history.to_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "optimization_history.json").write_text(json.dumps(hist_dict, indent=2, ensure_ascii=False))

    best = history.best()
    if best is not None:
        (output_dir / "best_params.yaml").write_text(yaml.safe_dump(best.get("params", {}), sort_keys=False))

    (output_dir / "run_info.json").write_text(
        json.dumps(
            {
                "mode": mode.value,
                "iterations": iterations,
                "runs_per_iteration": runs_per_iter,
                "parallel": parallel,
                "record_history": record_history,
                "record_logs": record_logs,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    primary = str(cfg.objectives.get("primary", "landing_error.p95"))
    _write_report(output_dir, hist_dict, primary, start_time, end_time)

    if args.ai_mode == "prompt" and best is not None:
        prompts_dir = Path(args.prompts_dir) if args.prompts_dir else default_prompts
        prompt_lib = PromptLibrary(prompts_dir)
        prompt_out = output_dir / "prompts"
        prompt_out.mkdir(parents=True, exist_ok=True)

        summary_json = json.dumps(best.get("summary", {}), indent=2, ensure_ascii=False)
        history_json = json.dumps(hist_dict, indent=2, ensure_ascii=False)
        params_yaml = yaml.safe_dump(best.get("params", {}), sort_keys=False)
        bounds_yaml = yaml.safe_dump(cfg.param_bounds, sort_keys=False)

        analysis_prompt = prompt_lib.render(
            "diagnose_batch.md",
            {"SUMMARY_JSON": summary_json},
        )
        (prompt_out / "analysis_prompt.md").write_text(analysis_prompt)

        optimize_prompt = prompt_lib.render(
            "optimize_parameters.md",
            {
                "PARAMS_YAML": params_yaml,
                "HISTORY_JSON": history_json,
                "BOUNDS_YAML": bounds_yaml,
            },
        )
        (prompt_out / "optimization_prompt.md").write_text(optimize_prompt)

        if record_logs:
            log_files = list((output_dir / "logs").rglob("*.json"))
            if log_files:
                log_json = log_files[0].read_text()
                single_prompt = prompt_lib.render("diagnose_single.md", {"LOG_JSON": log_json})
                (prompt_out / "single_run_prompt.md").write_text(single_prompt)

    if args.ai_mode == "api" and best is not None:
        prompts_dir = Path(args.prompts_dir) if args.prompts_dir else default_prompts
        prompt_lib = PromptLibrary(prompts_dir)
        prompt_out = output_dir / "prompts"
        prompt_out.mkdir(parents=True, exist_ok=True)

        summary_json = json.dumps(best.get("summary", {}), indent=2, ensure_ascii=False)
        history_json = json.dumps(hist_dict, indent=2, ensure_ascii=False)
        params_yaml = yaml.safe_dump(best.get("params", {}), sort_keys=False)
        bounds_yaml = yaml.safe_dump(cfg.param_bounds, sort_keys=False)

        optimize_prompt = prompt_lib.render(
            "optimize_parameters.md",
            {
                "PARAMS_YAML": params_yaml,
                "HISTORY_JSON": history_json,
                "BOUNDS_YAML": bounds_yaml,
            },
        )
        response = call_llm(
            optimize_prompt,
            model=str(args.llm_model) if args.llm_model else None,
            api_base=str(args.llm_api_base) if args.llm_api_base else None,
            api_key=str(args.llm_api_key) if args.llm_api_key else None,
        )
        (prompt_out / "llm_response.md").write_text(response)
        suggested = extract_yaml_block(response)
        if isinstance(suggested, dict):
            (output_dir / "llm_suggested_params.yaml").write_text(yaml.safe_dump(suggested, sort_keys=False))
            if args.apply_llm and args.apply_target:
                _apply_params_to_yaml(Path(args.apply_target), suggested)


if __name__ == "__main__":
    main()
