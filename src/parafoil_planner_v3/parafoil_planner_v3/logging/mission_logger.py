from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):  # noqa: D102
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return super().default(obj)


@dataclass
class MissionLogger:
    output_dir: Path
    run_id: str = ""
    mode: str = "offline"
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.run_id:
            self.run_id = uuid.uuid4().hex[:8]
        self.log_data: dict[str, Any] = {
            "metadata": {
                "run_id": self.run_id,
                "mode": self.mode,
                "timestamp": _utc_now(),
                "tags": list(self.tags),
                "schema_version": "v1",
            },
            "config": {},
            "timeline": [],
            "planner_logs": [],
            "planner_status_history": [],
            "controller_logs": [],
            "state_history": [],
            "control_history": [],
            "events": [],
            "tracking_history": [],
            "metrics": {},
        }

    def log_config(
        self,
        planner_config: dict | None = None,
        controller_config: dict | None = None,
        scenario: dict | None = None,
        extra: dict | None = None,
    ) -> None:
        payload: dict[str, Any] = {}
        if planner_config is not None:
            payload["planner"] = planner_config
        if controller_config is not None:
            payload["controller"] = controller_config
        if scenario is not None:
            payload["scenario"] = scenario
        if extra is not None:
            payload["extra"] = extra
        self.log_data["config"] = payload

    def log_state(self, timestamp: float, state: dict) -> None:
        self.log_data["state_history"].append({"timestamp": float(timestamp), "state": state})

    def log_control(self, timestamp: float, control: dict) -> None:
        self.log_data["control_history"].append({"timestamp": float(timestamp), "control": control})

    def log_planner_step(self, timestamp: float, state: dict, trajectory: dict | None, solver_info: dict | None) -> None:
        self.log_data["planner_logs"].append(
            {
                "timestamp": float(timestamp),
                "state": state,
                "trajectory": trajectory,
                "solver": solver_info,
            }
        )

    def log_planner_status(self, timestamp: float, status: dict | str | None) -> None:
        if status is None:
            return
        self.log_data["planner_status_history"].append({"timestamp": float(timestamp), "status": status})

    def log_tracking_mode(self, timestamp: float, mode: str | None) -> None:
        if not mode:
            return
        self.log_data["tracking_history"].append({"timestamp": float(timestamp), "mode": str(mode)})

    def log_controller_step(
        self,
        timestamp: float,
        phase: str,
        state: dict,
        reference: dict | None,
        control: dict,
        errors: dict | None,
    ) -> None:
        self.log_data["controller_logs"].append(
            {
                "timestamp": float(timestamp),
                "phase": str(phase),
                "state": state,
                "reference": reference,
                "control": control,
                "errors": errors or {},
            }
        )

    def log_event(self, timestamp: float, event_type: str, details: dict | None = None) -> None:
        self.log_data["events"].append({"timestamp": float(timestamp), "type": str(event_type), "details": details or {}})

    def set_metrics(self, metrics: dict) -> None:
        self.log_data["metrics"] = metrics

    def compute_summary_metrics(self) -> None:
        metrics = dict(self.log_data.get("metrics") or {})

        planner_logs = self.log_data.get("planner_logs") or []
        if planner_logs and "planner" not in metrics:
            solver_times = []
            violations = []
            iterations = []
            for item in planner_logs:
                solver = item.get("solver") if isinstance(item, dict) else None
                if isinstance(solver, dict):
                    if "solve_time" in solver:
                        solver_times.append(float(solver["solve_time"]))
                    if "max_violation" in solver:
                        violations.append(float(solver["max_violation"]))
                    if "iterations" in solver:
                        iterations.append(float(solver["iterations"]))
            if solver_times or violations or iterations:
                metrics["planner"] = {
                    "avg_solve_time": float(np.mean(solver_times)) if solver_times else None,
                    "max_solve_time": float(np.max(solver_times)) if solver_times else None,
                    "replan_count": int(len(planner_logs)),
                    "avg_constraint_violation": float(np.mean(violations)) if violations else None,
                    "avg_iterations": float(np.mean(iterations)) if iterations else None,
                }

        controller_logs = self.log_data.get("controller_logs") or []
        if controller_logs and "controller" not in metrics:
            cross_track = []
            altitude = []
            for item in controller_logs:
                errors = item.get("errors") if isinstance(item, dict) else None
                if isinstance(errors, dict):
                    if "cross_track" in errors:
                        cross_track.append(float(errors["cross_track"]))
                    if "altitude" in errors:
                        altitude.append(float(errors["altitude"]))
            metrics["controller"] = {
                "cross_track_rmse": float(np.sqrt(np.mean(np.square(cross_track)))) if cross_track else None,
                "altitude_rmse": float(np.sqrt(np.mean(np.square(altitude)))) if altitude else None,
            }

        controls = self.log_data.get("control_history") or []
        if controls and "control_effort_mean" not in metrics:
            effort = []
            for item in controls:
                ctrl = item.get("control") if isinstance(item, dict) else None
                if isinstance(ctrl, dict):
                    dl = float(ctrl.get("delta_L", 0.0))
                    dr = float(ctrl.get("delta_R", 0.0))
                    effort.append(dl * dl + dr * dr)
            if effort:
                metrics["control_effort_mean"] = float(np.mean(effort))

        self.log_data["metrics"] = metrics

    def save(self, filename: str | None = None) -> Path:
        self.compute_summary_metrics()
        name = filename or f"{self.run_id}.json"
        path = self.output_dir / name
        path.write_text(json.dumps(self.log_data, indent=2, ensure_ascii=False, cls=NumpyEncoder))
        return path

    @staticmethod
    def from_offline_result(
        result: dict[str, Any],
        output_dir: str | Path,
        run_id: str | None = None,
        mode: str = "offline",
        params: dict | None = None,
        tags: list[str] | None = None,
    ) -> "MissionLogger":
        logger = MissionLogger(output_dir=Path(output_dir), run_id=run_id or "", mode=mode, tags=tags or [])
        logger.log_config(scenario=result.get("scenario"), extra={"params": params or {}})
        logger.log_data["planner_logs"] = result.get("planner_logs", []) or []
        logger.log_data["planner_status_history"] = result.get("planner_status_history", []) or []
        logger.log_data["state_history"] = result.get("state_history", []) or []
        logger.log_data["control_history"] = result.get("control_history", []) or []
        logger.log_data["events"] = result.get("events", []) or []
        logger.log_data["tracking_history"] = result.get("tracking_history", []) or []
        logger.log_data["metrics"] = result.get("metrics", {}) or {}
        return logger
