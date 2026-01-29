#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from parafoil_planner_v3.environment import FlatTerrain, NoFlyCircle, NoFlyPolygon, load_no_fly_polygons
from parafoil_planner_v3.landing_site_selector import (
    LandingSiteSelector,
    LandingSiteSelectorConfig,
    ReachabilityConfig,
    RiskGrid,
    RiskLayer,
    RiskMapAggregator,
)
from parafoil_planner_v3.offline.e2e import Scenario, make_initial_state


@dataclass
class SafetyResult:
    scenario: dict
    reachable: bool
    reason: str
    risk: float
    desired_risk: float
    risk_reduction: float
    distance_to_desired_m: float
    reach_margin_mps: float
    time_to_land_s: float


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _parse_no_fly_circles(payload: str) -> list[NoFlyCircle]:
    payload = payload.strip()
    if not payload:
        return []
    parsed = json.loads(payload)
    circles: list[NoFlyCircle] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, list) and len(item) >= 3:
                cn = float(item[0])
                ce = float(item[1])
                r = float(item[2])
                c = float(item[3]) if len(item) >= 4 else 0.0
                circles.append(NoFlyCircle(center_n=cn, center_e=ce, radius_m=r, clearance_m=c))
    return circles


def _load_risk_map(path: str, weight: float, clip_min: float, clip_max: float, oob_value: float) -> RiskMapAggregator | None:
    if not path:
        return None
    grid = RiskGrid.from_file(path, oob_value=oob_value)
    layer = RiskLayer(name=Path(path).stem or "risk", weight=float(weight), grid=grid)
    return RiskMapAggregator(layers=[layer], clip_min=float(clip_min), clip_max=float(clip_max))


def _generate_scenarios(n: int, seed: int) -> list[Scenario]:
    rng = np.random.default_rng(int(seed))
    scenarios: list[Scenario] = []
    for _ in range(n):
        altitude = float(rng.uniform(50.0, 120.0))
        distance = float(rng.uniform(80.0, 220.0))
        bearing = float(rng.uniform(-180.0, 180.0))
        wind_speed = float(rng.uniform(0.0, 6.0))
        wind_dir = float(rng.uniform(0.0, 360.0))
        scenarios.append(
            Scenario(
                altitude_m=altitude,
                distance_m=distance,
                bearing_deg=bearing,
                wind_speed_mps=wind_speed,
                wind_direction_deg=wind_dir,
            )
        )
    return scenarios


def _evaluate_one(
    scenario: Scenario,
    selector: LandingSiteSelector,
    terrain: FlatTerrain,
    no_fly_circles: list[NoFlyCircle],
    no_fly_polygons: list[NoFlyPolygon],
    risk_map: RiskMapAggregator | None,
) -> SafetyResult:
    state, target, wind = make_initial_state(scenario)
    selection = selector.select(
        state=state,
        desired_target=target,
        wind=wind,
        terrain=terrain,
        no_fly_circles=no_fly_circles,
        no_fly_polygons=no_fly_polygons,
    )

    desired_risk = 0.0
    if risk_map is not None:
        desired_risk, _ = risk_map.risk(float(target.position_xy[0]), float(target.position_xy[1]))

    reachable = selection.reason == "ok"
    risk = float(selection.risk) if np.isfinite(selection.risk) else float("inf")
    risk_reduction = float(desired_risk - risk) if np.isfinite(risk) else 0.0

    return SafetyResult(
        scenario=asdict(scenario),
        reachable=reachable,
        reason=str(selection.reason),
        risk=risk,
        desired_risk=float(desired_risk),
        risk_reduction=float(risk_reduction),
        distance_to_desired_m=float(selection.distance_to_desired_m),
        reach_margin_mps=float(selection.reach_margin_mps),
        time_to_land_s=float(selection.time_to_land_s),
    )


def _write_reports(results: list[SafetyResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]

    reachable = [r for r in results if r.reachable]
    reach_rate = float(len(reachable) / max(len(results), 1))
    risk_vals = [r.risk for r in reachable if np.isfinite(r.risk)]
    desired_risks = [r.desired_risk for r in reachable if np.isfinite(r.desired_risk)]
    reductions = [r.risk_reduction for r in reachable if np.isfinite(r.risk_reduction)]
    dist_vals = [r.distance_to_desired_m for r in reachable]
    margin_vals = [r.reach_margin_mps for r in reachable]

    summary = {
        "n_runs": len(results),
        "reachable_rate": reach_rate,
        "risk": _stats(risk_vals),
        "desired_risk": _stats(desired_risks),
        "risk_reduction": _stats(reductions),
        "distance_to_desired_m": _stats(dist_vals),
        "reach_margin_mps": _stats(margin_vals),
    }

    json_path = output_dir / "safety_summary.json"
    json_path.write_text(json.dumps({"summary": summary, "results": payload}, indent=2))

    md_path = output_dir / "safety_summary.md"
    lines = [
        "# Safety Verification Summary",
        f"- Total runs: {summary['n_runs']}",
        f"- Reachable rate: {summary['reachable_rate']:.3f}",
        "",
        "## Metrics (reachable only)",
        "| Metric | Mean | P50 | P95 |",
        "|--------|------|-----|-----|",
        f"| Risk | {summary['risk']['mean']:.3f} | {summary['risk']['p50']:.3f} | {summary['risk']['p95']:.3f} |",
        f"| Desired Risk | {summary['desired_risk']['mean']:.3f} | {summary['desired_risk']['p50']:.3f} | {summary['desired_risk']['p95']:.3f} |",
        f"| Risk Reduction | {summary['risk_reduction']['mean']:.3f} | {summary['risk_reduction']['p50']:.3f} | {summary['risk_reduction']['p95']:.3f} |",
        f"| Distance to Desired (m) | {summary['distance_to_desired_m']['mean']:.2f} | {summary['distance_to_desired_m']['p50']:.2f} | {summary['distance_to_desired_m']['p95']:.2f} |",
        f"| Reach Margin (m/s) | {summary['reach_margin_mps']['mean']:.2f} | {summary['reach_margin_mps']['p50']:.2f} | {summary['reach_margin_mps']['p95']:.2f} |",
        "",
        "## Notes",
        "- Risk metrics are computed from the risk grid at the selected landing point.",
        "- Reachable rate uses selector reason == 'ok'.",
        "",
    ]
    md_path.write_text("\n".join(lines))

    print(f"Summary written to {json_path}")
    print(f"Report written to {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline safety verification for landing site selection.")
    parser.add_argument("--runs", type=int, default=30, help="Number of random scenarios")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--output", type=str, default="reports/safety_offline", help="Output directory")

    parser.add_argument("--risk-grid", type=str, default="", help="Risk grid (.npz/.yaml/.json)")
    parser.add_argument("--risk-weight", type=float, default=1.0, help="Risk grid weight")
    parser.add_argument("--risk-clip-min", type=float, default=0.0)
    parser.add_argument("--risk-clip-max", type=float, default=1.0)
    parser.add_argument("--risk-oob", type=float, default=1.0)

    parser.add_argument("--grid-resolution", type=float, default=20.0, help="Selector grid resolution")
    parser.add_argument("--search-radius", type=float, default=0.0, help="Search radius (0=auto)")
    parser.add_argument("--max-candidates", type=int, default=800)
    parser.add_argument("--w-risk", type=float, default=5.0)
    parser.add_argument("--w-distance", type=float, default=1.0)
    parser.add_argument("--w-reach-margin", type=float, default=1.0)
    parser.add_argument("--w-energy", type=float, default=0.5)
    parser.add_argument("--nofly-buffer", type=float, default=20.0)
    parser.add_argument("--nofly-weight", type=float, default=5.0)

    parser.add_argument("--brake", type=float, default=0.2)
    parser.add_argument("--min-time", type=float, default=2.0)
    parser.add_argument("--max-time", type=float, default=200.0)
    parser.add_argument("--wind-margin", type=float, default=0.2)
    parser.add_argument("--wind-uncertainty", type=float, default=0.0)
    parser.add_argument("--gust-margin", type=float, default=0.0)
    parser.add_argument("--min-altitude", type=float, default=5.0)

    parser.add_argument("--terrain-height0", type=float, default=0.0)
    parser.add_argument("--no-fly-circles", type=str, default="", help="JSON list [[n,e,r,clearance], ...]")
    parser.add_argument("--no-fly-polygons", type=str, default="", help="YAML/JSON/GeoJSON file")

    args = parser.parse_args()

    risk_map = _load_risk_map(
        path=str(args.risk_grid),
        weight=float(args.risk_weight),
        clip_min=float(args.risk_clip_min),
        clip_max=float(args.risk_clip_max),
        oob_value=float(args.risk_oob),
    )

    selector_cfg = LandingSiteSelectorConfig(
        enabled=True,
        grid_resolution_m=float(args.grid_resolution),
        search_radius_m=float(args.search_radius),
        max_candidates=int(args.max_candidates),
        random_seed=int(args.seed),
        w_risk=float(args.w_risk),
        w_distance=float(args.w_distance),
        w_reach_margin=float(args.w_reach_margin),
        w_energy=float(args.w_energy),
        nofly_buffer_m=float(args.nofly_buffer),
        nofly_weight=float(args.nofly_weight),
        snap_to_terrain=True,
        reachability=ReachabilityConfig(
            brake=float(args.brake),
            min_time_s=float(args.min_time),
            max_time_s=float(args.max_time),
            wind_margin_mps=float(args.wind_margin),
            wind_uncertainty_mps=float(args.wind_uncertainty),
            gust_margin_mps=float(args.gust_margin),
            min_altitude_m=float(args.min_altitude),
            terrain_clearance_m=0.0,
        ),
    )

    selector = LandingSiteSelector(config=selector_cfg, risk_map=risk_map)

    terrain = FlatTerrain(height0_m=float(args.terrain_height0))
    no_fly_circles = _parse_no_fly_circles(str(args.no_fly_circles))
    no_fly_polygons = load_no_fly_polygons(str(args.no_fly_polygons)) if str(args.no_fly_polygons).strip() else []

    scenarios = _generate_scenarios(int(args.runs), int(args.seed))
    results = [
        _evaluate_one(sc, selector, terrain, no_fly_circles, no_fly_polygons, risk_map)
        for sc in scenarios
    ]

    _write_reports(results, Path(args.output))


if __name__ == "__main__":
    main()
