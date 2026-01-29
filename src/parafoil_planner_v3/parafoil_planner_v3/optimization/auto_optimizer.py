from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from parafoil_planner_v3.guidance.control_laws import LateralControlConfig
from parafoil_planner_v3.guidance.flare_guidance import FlareGuidanceConfig
from parafoil_planner_v3.logging.mission_logger import MissionLogger
from parafoil_planner_v3.offline.e2e import Scenario, simulate_one
from parafoil_planner_v3.planner_core import PlannerConfig


class ExecutionMode(Enum):
    OFFLINE = "offline"
    ROS2 = "ros2"


@dataclass
class OptimizationConfig:
    objectives: dict[str, Any]
    param_bounds: dict[str, list[float]]
    settings: dict[str, Any]
    scenario: dict[str, Any]
    base_params: dict[str, Any]
    simulation: dict[str, Any]
    ai: dict[str, Any]

    @staticmethod
    def _unwrap(data: dict[str, Any]) -> dict[str, Any]:
        if "parafoil_planner_v3" in data:
            data = data["parafoil_planner_v3"]
        if "ros__parameters" in data:
            data = data["ros__parameters"]
        return data

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OptimizationConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        raw = cls._unwrap(raw)
        if "optimization" in raw:
            raw = raw["optimization"]
        return cls(
            objectives=raw.get("objectives", {}),
            param_bounds=raw.get("param_bounds", {}),
            settings=raw.get("settings", {}),
            scenario=raw.get("scenario", {}),
            base_params=raw.get("base_params", {}),
            simulation=raw.get("simulation", {}),
            ai=raw.get("ai", {}),
        )


@dataclass
class OptimizationHistory:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)

    def best(self) -> dict[str, Any] | None:
        if not self.entries:
            return None
        return min(self.entries, key=lambda e: float(e.get("score", float("inf"))))

    def to_dict(self) -> dict[str, Any]:
        return {"entries": self.entries}


def _get_nested(d: dict[str, Any], key: str) -> Any:
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_nested(d: dict[str, Any], key: str, value: Any) -> None:
    cur = d
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _maybe_cast(value: float, ref: Any) -> Any:
    if isinstance(ref, int):
        return int(round(value))
    return float(value)


def _random_params(
    rng: np.random.Generator,
    base_params: dict[str, Any],
    bounds: dict[str, list[float]],
) -> dict[str, Any]:
    params = json.loads(json.dumps(base_params))
    for key, bound in bounds.items():
        if len(bound) < 2:
            continue
        lo, hi = float(bound[0]), float(bound[1])
        ref = _get_nested(params, key)
        val = rng.uniform(lo, hi)
        _set_nested(params, key, _maybe_cast(val, ref))
    return params


def _perturb_params(
    rng: np.random.Generator,
    params: dict[str, Any],
    bounds: dict[str, list[float]],
    scale: float,
) -> dict[str, Any]:
    out = json.loads(json.dumps(params))
    for key, bound in bounds.items():
        if len(bound) < 2:
            continue
        lo, hi = float(bound[0]), float(bound[1])
        span = max(hi - lo, 1e-9)
        ref = _get_nested(out, key)
        if ref is None:
            ref = (lo + hi) / 2.0
        val = float(ref) + float(rng.normal(0.0, scale * span))
        val = float(np.clip(val, lo, hi))
        _set_nested(out, key, _maybe_cast(val, ref))
    return out


def _build_lateral_config(params: dict[str, Any]) -> LateralControlConfig:
    base = LateralControlConfig()
    lateral = params.get("lateral", {}) if isinstance(params, dict) else {}
    return LateralControlConfig(
        turn_rate_per_delta=float(lateral.get("turn_rate_per_delta", base.turn_rate_per_delta)),
        K_heading=float(lateral.get("K_heading", base.K_heading)),
        yaw_rate_max=float(lateral.get("yaw_rate_max", base.yaw_rate_max)),
        max_delta_a=float(lateral.get("max_delta_a", base.max_delta_a)),
        max_brake=float(lateral.get("max_brake", base.max_brake)),
    )


def _build_flare_config(params: dict[str, Any], lateral: LateralControlConfig) -> FlareGuidanceConfig:
    base = FlareGuidanceConfig(lateral=lateral)
    flare = params.get("flare", {}) if isinstance(params, dict) else {}
    return FlareGuidanceConfig(
        mode=str(flare.get("mode", base.mode)),
        flare_initial_brake=float(flare.get("flare_initial_brake", base.flare_initial_brake)),
        flare_max_brake=float(flare.get("flare_max_brake", base.flare_max_brake)),
        flare_ramp_time=float(flare.get("flare_ramp_time", base.flare_ramp_time)),
        flare_full_brake_duration_s=float(flare.get("flare_full_brake_duration_s", base.flare_full_brake_duration_s)),
        touchdown_brake_altitude_m=float(flare.get("touchdown_brake_altitude_m", base.touchdown_brake_altitude_m)),
        lateral=lateral,
    )


def _build_planner_config(params: dict[str, Any]) -> PlannerConfig | None:
    planner = params.get("planner") if isinstance(params, dict) else None
    if not isinstance(planner, dict):
        return None
    base = PlannerConfig()
    merged = dict(base.__dict__)
    merged.update(planner)
    return PlannerConfig(**merged)


def _run_offline_task(task: tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]) -> dict[str, Any]:
    scenario_dict, params, sim_cfg, record_history = task
    scenario = Scenario(**scenario_dict)
    lateral = _build_lateral_config(params)
    flare = _build_flare_config(params, lateral)
    planner = _build_planner_config(params)
    return simulate_one(
        scenario=scenario,
        planner_rate_hz=float(sim_cfg.get("planner_rate_hz", 1.0)),
        control_rate_hz=float(sim_cfg.get("control_rate_hz", 20.0)),
        max_time_s=float(sim_cfg.get("max_time_s", 200.0)),
        L1_distance=float(params.get("L1_distance", 20.0)),
        use_gpm=bool(sim_cfg.get("use_gpm", False)),
        dynamics_mode=str(sim_cfg.get("dynamics_mode", "simplified")),
        record_history=bool(record_history),
        planner_config=planner,
        lateral_config=lateral,
        flare_config=flare,
    )


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    landing_errors = []
    touchdown_errors = []
    vertical_v = []
    touchdown_v = []
    times = []
    replans = []
    effort = []
    phase_sum: dict[str, float] = {}
    success = []

    for r in results:
        metrics = r.get("metrics", {}) or {}
        landing_errors.append(float(metrics.get("landing_error_m", 0.0)))
        touchdown_errors.append(float(metrics.get("touchdown_landing_error_m", metrics.get("landing_error_m", 0.0))))
        vertical_v.append(float(metrics.get("vertical_velocity_mps", 0.0)))
        touchdown_v.append(float(metrics.get("touchdown_vertical_velocity_mps", metrics.get("vertical_velocity_mps", 0.0))))
        times.append(float(metrics.get("time_s", 0.0)))
        replans.append(float(metrics.get("replan_count", 0.0)))
        effort.append(float(metrics.get("control_effort_mean", 0.0)))
        success.append(1.0 if bool(metrics.get("success", False)) else 0.0)
        phase = metrics.get("phase_durations_s", {}) or {}
        for k, v in phase.items():
            phase_sum[k] = phase_sum.get(k, 0.0) + float(v)

    n = max(len(results), 1)
    summary = {
        "n_runs": len(results),
        "success_rate": float(np.mean(success)) if success else 0.0,
        "landing_error": {
            "mean": float(np.mean(landing_errors)) if landing_errors else 0.0,
            "p50": float(np.percentile(landing_errors, 50)) if landing_errors else 0.0,
            "p95": float(np.percentile(landing_errors, 95)) if landing_errors else 0.0,
            "max": float(np.max(landing_errors)) if landing_errors else 0.0,
        },
        "touchdown_error": {
            "mean": float(np.mean(touchdown_errors)) if touchdown_errors else 0.0,
            "p95": float(np.percentile(touchdown_errors, 95)) if touchdown_errors else 0.0,
        },
        "vertical_velocity_mps": {
            "mean": float(np.mean(vertical_v)) if vertical_v else 0.0,
            "p95": float(np.percentile(vertical_v, 95)) if vertical_v else 0.0,
        },
        "touchdown_vertical_velocity_mps": {
            "mean": float(np.mean(touchdown_v)) if touchdown_v else 0.0,
            "p95": float(np.percentile(touchdown_v, 95)) if touchdown_v else 0.0,
        },
        "mission_time_s": {
            "mean": float(np.mean(times)) if times else 0.0,
            "p95": float(np.percentile(times, 95)) if times else 0.0,
        },
        "replan_count": {
            "mean": float(np.mean(replans)) if replans else 0.0,
            "p95": float(np.percentile(replans, 95)) if replans else 0.0,
        },
        "control_effort_mean": {
            "mean": float(np.mean(effort)) if effort else 0.0,
            "p95": float(np.percentile(effort, 95)) if effort else 0.0,
        },
        "phase_durations_s_mean": {k: float(v / n) for k, v in phase_sum.items()},
    }
    return summary


class OfflineVerifier:
    def __init__(self, scenario_cfg: dict[str, Any], sim_cfg: dict[str, Any]) -> None:
        self.scenario_cfg = scenario_cfg
        self.sim_cfg = sim_cfg

    def _sample_scenarios(self, n_runs: int, seed: int) -> list[dict[str, Any]]:
        rng = np.random.default_rng(int(seed))
        alt = self.scenario_cfg.get("altitude_m", [50.0, 200.0])
        dist = self.scenario_cfg.get("distance_m", [50.0, 250.0])
        bearing = self.scenario_cfg.get("bearing_deg", [-180.0, 180.0])
        wind_speed = self.scenario_cfg.get("wind_speed_mps", [0.0, 6.0])
        wind_dir = self.scenario_cfg.get("wind_direction_deg", [0.0, 360.0])
        return [
            {
                "altitude_m": float(rng.uniform(*alt)),
                "distance_m": float(rng.uniform(*dist)),
                "bearing_deg": float(rng.uniform(*bearing)),
                "wind_speed_mps": float(rng.uniform(*wind_speed)),
                "wind_direction_deg": float(rng.uniform(*wind_dir)),
            }
            for _ in range(int(n_runs))
        ]

    def run_monte_carlo(
        self,
        params: dict[str, Any],
        n_runs: int,
        parallel: int,
        seed: int,
        record_history: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        scenarios = self._sample_scenarios(n_runs, seed)
        tasks = [(s, params, self.sim_cfg, record_history) for s in scenarios]
        results: list[dict[str, Any]] = []
        if int(parallel) <= 1:
            for task in tasks:
                results.append(_run_offline_task(task))
        else:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=int(parallel)) as pool:
                for item in pool.imap_unordered(_run_offline_task, tasks, chunksize=1):
                    results.append(item)
        summary = summarize_results(results)
        return summary, results


class ROS2Verifier:
    def __init__(self, log_dir: str | Path | None = None) -> None:
        self.log_dir = Path(log_dir) if log_dir else None
        self.last_error = "ROS2 mode requires log_dir with mission logs."

    def _load_logs(self, n_runs: int | None = None) -> list[dict[str, Any]]:
        if self.log_dir is None or not self.log_dir.exists():
            raise RuntimeError(self.last_error)
        files = sorted(self.log_dir.rglob("*.json"))
        if n_runs is not None and n_runs > 0:
            files = files[: int(n_runs)]
        logs = []
        for f in files:
            try:
                logs.append(json.loads(f.read_text()))
            except Exception:
                continue
        if not logs:
            raise RuntimeError("No mission logs found in log_dir")
        return logs

    @staticmethod
    def _metrics_from_log(log: dict[str, Any]) -> dict[str, Any]:
        cfg = log.get("config", {}) if isinstance(log, dict) else {}
        scenario = cfg.get("scenario", {}) if isinstance(cfg, dict) else {}
        target = scenario.get("target_ned") or scenario.get("target") or scenario.get("target_position_ned")
        if target is None:
            target = [0.0, 0.0, 0.0]
        target = np.asarray(target, dtype=float).reshape(3)

        state_hist = log.get("state_history", []) or []
        last_state = None
        if state_hist:
            last_state = state_hist[-1].get("state") if isinstance(state_hist[-1], dict) else None
            if last_state is None and isinstance(state_hist[-1], dict):
                last_state = state_hist[-1]

        landing_error = 0.0
        vertical_v = 0.0
        time_s = 0.0
        if isinstance(last_state, dict):
            pos = np.asarray(last_state.get("position", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
            vel = np.asarray(last_state.get("velocity", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
            landing_error = float(np.linalg.norm(pos[:2] - target[:2]))
            vertical_v = float(abs(vel[2]))
            time_s = float(last_state.get("t", 0.0))

        control_hist = log.get("control_history", []) or []
        effort = []
        for item in control_hist:
            ctrl = item.get("control") if isinstance(item, dict) else None
            if isinstance(ctrl, dict):
                dl = float(ctrl.get("delta_L", 0.0))
                dr = float(ctrl.get("delta_R", 0.0))
                effort.append(dl * dl + dr * dr)
        control_effort_mean = float(np.mean(effort)) if effort else 0.0

        # Phase duration from events
        phase_durations: dict[str, float] = {}
        events = log.get("events", []) or []
        last_phase = None
        last_t = None
        for ev in events:
            if not isinstance(ev, dict):
                continue
            if ev.get("event") == "phase_transition":
                t = float(ev.get("t", 0.0))
                to_phase = str(ev.get("to", ""))
                if last_phase is not None and last_t is not None:
                    phase_durations[last_phase] = phase_durations.get(last_phase, 0.0) + float(t - last_t)
                last_phase = to_phase
                last_t = t

        metrics = {
            "landing_error_m": float(landing_error),
            "vertical_velocity_mps": float(vertical_v),
            "touchdown_vertical_velocity_mps": float(vertical_v),
            "time_s": float(time_s),
            "control_effort_mean": float(control_effort_mean),
            "phase_durations_s": phase_durations,
            "success": bool(landing_error < 10.0),
        }
        return metrics

    def run_monte_carlo(self, params, n_runs: int, parallel: int, seed: int, record_history: bool):  # noqa: D102
        logs = self._load_logs(n_runs=n_runs if n_runs > 0 else None)
        results: list[dict[str, Any]] = []
        for log in logs:
            metrics = self._metrics_from_log(log)
            results.append({"scenario": log.get("config", {}).get("scenario", {}), "metrics": metrics})
        summary = summarize_results(results)
        return summary, results


class AutoOptimizer:
    def __init__(self, config: OptimizationConfig, mode: ExecutionMode = ExecutionMode.OFFLINE) -> None:
        self.config = config
        self.mode = mode
        self.history = OptimizationHistory()
        self.rng = np.random.default_rng(int(self.config.settings.get("seed", 42)))
        if self.mode == ExecutionMode.OFFLINE:
            self.verifier = OfflineVerifier(self.config.scenario, self.config.simulation)
        else:
            log_dir = self.config.settings.get("ros2_log_dir") if isinstance(self.config.settings, dict) else None
            self.verifier = ROS2Verifier(log_dir=log_dir)

    def _objective_score(self, summary: dict[str, Any]) -> float:
        primary = str(self.config.objectives.get("primary", "landing_error.p95"))
        direction = str(self.config.objectives.get("direction", "minimize")).lower()
        value = _get_nested(summary, primary)
        if value is None:
            return float("inf")
        score = float(value)
        return -score if direction == "maximize" else score

    def _constraints_ok(self, summary: dict[str, Any]) -> bool:
        constraints = self.config.objectives.get("constraints", []) or []
        for item in constraints:
            name = str(item.get("name", ""))
            if not name:
                continue
            value = _get_nested(summary, name)
            if value is None:
                continue
            if "max" in item and float(value) > float(item["max"]):
                return False
            if "min" in item and float(value) < float(item["min"]):
                return False
        return True

    def _propose_next(self, best_params: dict[str, Any]) -> dict[str, Any]:
        explore = float(self.config.settings.get("exploration_rate", 0.3))
        scale = float(self.config.settings.get("step_scale", 0.15))
        if self.rng.random() < explore or not best_params:
            return _random_params(self.rng, self.config.base_params, self.config.param_bounds)
        return _perturb_params(self.rng, best_params, self.config.param_bounds, scale)

    def run(
        self,
        iterations: int,
        runs_per_iter: int,
        parallel: int,
        record_history: bool,
        output_dir: Path,
        record_logs: bool = True,
    ) -> OptimizationHistory:
        output_dir.mkdir(parents=True, exist_ok=True)
        params = json.loads(json.dumps(self.config.base_params))
        best_score = float("inf")
        no_improve = 0
        tol = float(self.config.settings.get("convergence_tol", 0.0))
        patience = int(self.config.settings.get("convergence_patience", 0))
        for i in range(int(iterations)):
            seed = int(self.config.settings.get("seed", 42)) + i
            summary, results = self.verifier.run_monte_carlo(
                params=params,
                n_runs=int(runs_per_iter),
                parallel=int(parallel),
                seed=seed,
                record_history=bool(record_history),
            )
            score = self._objective_score(summary)
            constraints_ok = self._constraints_ok(summary)
            entry = {
                "iteration": i,
                "params": params,
                "summary": summary,
                "score": float(score),
                "constraints_ok": bool(constraints_ok),
                "timestamp": time.time(),
            }
            self.history.add(entry)

            if record_logs:
                log_dir = output_dir / "logs" / f"iter_{i:02d}"
                log_dir.mkdir(parents=True, exist_ok=True)
                for k, res in enumerate(results):
                    logger = MissionLogger.from_offline_result(
                        result=res,
                        output_dir=log_dir,
                        run_id=f"iter{i:02d}_run{k:03d}",
                        params=params,
                    )
                    logger.save()

            best = self.history.best()
            best_params = best["params"] if best else params
            params = self._propose_next(best_params)

            if tol > 0.0 and patience > 0:
                if score + tol < best_score:
                    best_score = float(score)
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break

        return self.history


class PromptLibrary:
    def __init__(self, prompt_dir: Path) -> None:
        self.prompt_dir = prompt_dir

    def render(self, name: str, mapping: dict[str, str]) -> str:
        text = (self.prompt_dir / name).read_text()
        for key, value in mapping.items():
            text = text.replace(f"{{{{{key}}}}}", value)
        return text
