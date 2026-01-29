from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from parafoil_planner_v3.dynamics.aerodynamics import PolarTable
from parafoil_planner_v3.environment import NoFlyCircle, NoFlyPolygon, TerrainModel
from parafoil_planner_v3.types import State, Target, Wind


@dataclass(frozen=True)
class ReachabilityConfig:
    brake: float = 0.2
    min_time_s: float = 2.0
    max_time_s: float = 200.0
    wind_margin_mps: float = 0.2
    wind_uncertainty_mps: float = 0.0
    gust_margin_mps: float = 0.0
    min_altitude_m: float = 5.0
    terrain_clearance_m: float = 0.0
    enforce_circle: bool = True


@dataclass(frozen=True)
class RiskGrid:
    origin_n: float
    origin_e: float
    resolution_m: float
    risk_map: np.ndarray
    oob_value: float = 1.0

    def __post_init__(self) -> None:
        data = np.asarray(self.risk_map, dtype=float)
        if data.ndim != 2:
            raise ValueError("RiskGrid risk_map must be 2D")
        object.__setattr__(self, "risk_map", data)

    def risk_at(self, north_m: float, east_m: float) -> float:
        h = self.risk_map
        res = float(self.resolution_m)
        if res <= 1e-9:
            return float(h[0, 0])
        u = (float(north_m) - float(self.origin_n)) / res
        v = (float(east_m) - float(self.origin_e)) / res
        if u < 0.0 or v < 0.0 or u > (h.shape[0] - 1) or v > (h.shape[1] - 1):
            return float(self.oob_value)
        i0 = int(np.floor(u))
        j0 = int(np.floor(v))
        i1 = int(np.clip(i0 + 1, 0, h.shape[0] - 1))
        j1 = int(np.clip(j0 + 1, 0, h.shape[1] - 1))
        i0 = int(np.clip(i0, 0, h.shape[0] - 1))
        j0 = int(np.clip(j0, 0, h.shape[1] - 1))
        fu = float(np.clip(u - i0, 0.0, 1.0))
        fv = float(np.clip(v - j0, 0.0, 1.0))
        h00 = float(h[i0, j0])
        h10 = float(h[i1, j0])
        h01 = float(h[i0, j1])
        h11 = float(h[i1, j1])
        h0 = h00 * (1.0 - fu) + h10 * fu
        h1 = h01 * (1.0 - fu) + h11 * fu
        return float(h0 * (1.0 - fv) + h1 * fv)

    @staticmethod
    def from_file(path: str | Path, oob_value: float = 1.0) -> "RiskGrid":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        ext = path.suffix.lower()
        if ext == ".npz":
            data = np.load(path)
            risk = None
            for key in ("risk", "risk_map", "risk_grid"):
                if key in data:
                    risk = np.asarray(data[key], dtype=float)
                    break
            if risk is None:
                raise ValueError("RiskGrid npz missing risk array")
            origin_n = float(data["origin_n"]) if "origin_n" in data else 0.0
            origin_e = float(data["origin_e"]) if "origin_e" in data else 0.0
            resolution_m = float(data["resolution_m"]) if "resolution_m" in data else 1.0
            return RiskGrid(
                origin_n=origin_n,
                origin_e=origin_e,
                resolution_m=resolution_m,
                risk_map=risk,
                oob_value=oob_value,
            )
        if ext in {".yaml", ".yml", ".json"}:
            if ext in {".yaml", ".yml"}:
                import yaml

                payload = yaml.safe_load(path.read_text()) or {}
            else:
                import json

                payload = json.loads(path.read_text())
            risk = None
            for key in ("risk", "risk_map", "risk_grid"):
                if key in payload:
                    risk = np.asarray(payload.get(key), dtype=float)
                    break
            if risk is None:
                raise ValueError("RiskGrid file missing risk array")
            origin_n = float(payload.get("origin_n", 0.0))
            origin_e = float(payload.get("origin_e", 0.0))
            resolution_m = float(payload.get("resolution_m", 1.0))
            return RiskGrid(
                origin_n=origin_n,
                origin_e=origin_e,
                resolution_m=resolution_m,
                risk_map=risk,
                oob_value=oob_value,
            )
        raise ValueError(f"Unsupported risk grid file format: {ext}")


@dataclass(frozen=True)
class RiskLayer:
    name: str
    weight: float
    grid: RiskGrid


class RiskMapAggregator:
    def __init__(self, layers: Sequence[RiskLayer], clip_min: float = 0.0, clip_max: float = 1.0) -> None:
        self.layers = list(layers)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def risk(self, north_m: float, east_m: float) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        details: Dict[str, float] = {}
        for layer in self.layers:
            val = layer.grid.risk_at(north_m, east_m)
            val = float(np.clip(val, self.clip_min, self.clip_max))
            details[layer.name] = val
            total += float(layer.weight) * val
        return float(total), details


@dataclass(frozen=True)
class LandingSiteSelectorConfig:
    enabled: bool = True
    grid_resolution_m: float = 25.0
    search_radius_m: float = 0.0  # 0 => auto from reachability
    max_candidates: int = 800
    random_seed: int = 0
    w_risk: float = 5.0
    w_distance: float = 1.0
    w_reach_margin: float = 1.0
    w_energy: float = 0.5
    nofly_buffer_m: float = 20.0
    nofly_weight: float = 5.0
    snap_to_terrain: bool = True
    min_progress_mps: float = 0.0
    reachability: ReachabilityConfig = field(default_factory=ReachabilityConfig)


@dataclass(frozen=True)
class LandingSiteSelection:
    target: Target
    score: float
    risk: float
    distance_to_desired_m: float
    reach_margin_mps: float
    time_to_land_s: float
    reason: str
    metadata: Dict[str, float] = field(default_factory=dict)


class LandingSiteSelector:
    def __init__(self, config: LandingSiteSelectorConfig, risk_map: RiskMapAggregator | None = None) -> None:
        self.config = config
        self.risk_map = risk_map
        self._rng = np.random.default_rng(int(config.random_seed))
        self._polar = PolarTable()

    def _time_to_land(self, state: State, terrain: TerrainModel | None, north_m: float, east_m: float) -> float:
        altitude = float(state.altitude)
        terrain_h = float(terrain.height_m(north_m, east_m)) if terrain is not None else 0.0
        h_agl = altitude - terrain_h - float(self.config.reachability.terrain_clearance_m)
        if h_agl < float(self.config.reachability.min_altitude_m):
            return -1.0
        _, sink = self._polar.interpolate(float(self.config.reachability.brake))
        if sink <= 1e-6:
            return -1.0
        return float(h_agl / sink)

    def _reachable(self, state: State, wind: Wind, target_xy: np.ndarray, tgo: float) -> Tuple[bool, float, float]:
        if tgo < float(self.config.reachability.min_time_s) or tgo > float(self.config.reachability.max_time_s):
            return False, -1.0, 0.0
        d = target_xy - state.position_xy
        v_req = d / max(tgo, 1e-6)
        v_air, _ = self._polar.interpolate(float(self.config.reachability.brake))
        wind_unc = float(self.config.reachability.wind_uncertainty_mps)
        gust_margin = float(self.config.reachability.gust_margin_mps)
        margin = float(v_air - np.linalg.norm(v_req - wind.v_I[:2]) - wind_unc - gust_margin)
        ok = margin >= float(self.config.reachability.wind_margin_mps)
        if ok and bool(self.config.reachability.enforce_circle):
            center = state.position_xy + wind.v_I[:2] * float(tgo)
            radius = float(
                max(
                    v_air
                    - float(self.config.reachability.wind_margin_mps)
                    - wind_unc
                    - gust_margin,
                    0.0,
                )
                * float(tgo)
            )
            if float(np.linalg.norm(target_xy - center)) > radius:
                ok = False
        return ok, margin, float(v_air)

    def _ground_speed_along_max(
        self,
        state: State,
        wind: Wind,
        target_xy: np.ndarray,
    ) -> Tuple[float, float, float, float, float, float]:
        v_air, sink = self._polar.interpolate(float(self.config.reachability.brake))
        wind_xy = np.asarray(wind.v_I[:2], dtype=float)
        rel = np.asarray(target_xy, dtype=float) - state.position_xy
        dist = float(np.linalg.norm(rel))
        if dist <= 1e-6:
            d_hat = np.array([1.0, 0.0], dtype=float)
            v_g_along_max = float(v_air)
        else:
            d_hat = rel / dist
            v_g_along_max = float(v_air) + float(np.dot(wind_xy, d_hat))
        v_ground_vec = float(v_air) * d_hat + wind_xy
        v_ground = float(np.linalg.norm(v_ground_vec))
        ld_air = float(v_air / sink) if sink > 1e-6 else float("inf")
        ld_ground = float(v_ground / sink) if sink > 1e-6 else float("inf")
        return float(v_g_along_max), float(v_air), float(sink), float(v_ground), float(ld_air), float(ld_ground)

    def _nofly_penalty(
        self,
        north_m: float,
        east_m: float,
        no_fly_circles: Iterable[NoFlyCircle],
        no_fly_polygons: Iterable[NoFlyPolygon],
    ) -> Tuple[bool, float]:
        min_dist = float("inf")
        for zone in no_fly_circles:
            d = float(zone.signed_distance_m(north_m, east_m))
            if d < 0.0:
                return True, float("inf")
            min_dist = min(min_dist, d)
        for zone in no_fly_polygons:
            d = float(zone.signed_distance_m(north_m, east_m))
            if d < 0.0:
                return True, float("inf")
            min_dist = min(min_dist, d)
        buffer_m = float(self.config.nofly_buffer_m)
        if not np.isfinite(min_dist) or buffer_m <= 1e-6:
            return False, 0.0
        if min_dist >= buffer_m:
            return False, 0.0
        penalty = (buffer_m - min_dist) / buffer_m
        return False, float(penalty)

    def _search_radius(self, state: State, wind: Wind, terrain: TerrainModel | None) -> float:
        v_air, _ = self._polar.interpolate(float(self.config.reachability.brake))
        tgo = self._time_to_land(state, terrain, float(state.p_I[0]), float(state.p_I[1]))
        if tgo <= 0.0:
            tgo = float(self.config.reachability.min_time_s)
        max_range = (float(v_air) + float(np.linalg.norm(wind.v_I[:2]))) * tgo
        radius = float(self.config.search_radius_m)
        if radius <= 1e-6:
            return float(max_range)
        return float(min(radius, max_range))

    def _generate_candidates(self, center_xy: np.ndarray, radius: float) -> np.ndarray:
        res = float(self.config.grid_resolution_m)
        if radius <= 1e-6 or res <= 1e-6:
            return np.empty((0, 2), dtype=float)
        nmin = float(center_xy[0] - radius)
        nmax = float(center_xy[0] + radius)
        emin = float(center_xy[1] - radius)
        emax = float(center_xy[1] + radius)
        n_vals = np.arange(nmin, nmax + res * 0.5, res, dtype=float)
        e_vals = np.arange(emin, emax + res * 0.5, res, dtype=float)
        total = int(n_vals.size * e_vals.size)
        max_cand = int(max(self.config.max_candidates, 1))
        if total > max_cand * 4:
            # Random sampling inside circle
            count = max_cand
            r = radius * np.sqrt(self._rng.random(count))
            theta = 2.0 * np.pi * self._rng.random(count)
            n = center_xy[0] + r * np.cos(theta)
            e = center_xy[1] + r * np.sin(theta)
            return np.stack([n, e], axis=1)
        nn, ee = np.meshgrid(n_vals, e_vals, indexing="ij")
        dn = nn - float(center_xy[0])
        de = ee - float(center_xy[1])
        mask = (dn * dn + de * de) <= float(radius * radius)
        pts = np.stack([nn[mask], ee[mask]], axis=1)
        if pts.shape[0] > max_cand:
            idx = self._rng.choice(pts.shape[0], size=max_cand, replace=False)
            pts = pts[idx]
        return pts.astype(float)

    def select(
        self,
        state: State,
        desired_target: Target,
        wind: Wind,
        terrain: TerrainModel | None,
        no_fly_circles: Iterable[NoFlyCircle],
        no_fly_polygons: Iterable[NoFlyPolygon],
    ) -> LandingSiteSelection:
        if not bool(self.config.enabled):
            return LandingSiteSelection(
                target=desired_target,
                score=float("inf"),
                risk=0.0,
                distance_to_desired_m=0.0,
                reach_margin_mps=0.0,
                time_to_land_s=0.0,
                reason="disabled",
            )

        desired_xy = desired_target.position_xy
        desired_tgo = self._time_to_land(state, terrain, float(desired_xy[0]), float(desired_xy[1]))
        desired_ok = False
        desired_margin = float("-inf")
        desired_vg_along = float("nan")
        desired_v_air = float("nan")
        desired_sink = float("nan")
        desired_v_ground = float("nan")
        desired_ld_air = float("nan")
        desired_ld_ground = float("nan")
        if desired_tgo > 0.0:
            desired_ok, desired_margin, _ = self._reachable(state, wind, desired_xy, desired_tgo)
            (
                desired_vg_along,
                desired_v_air,
                desired_sink,
                desired_v_ground,
                desired_ld_air,
                desired_ld_ground,
            ) = self._ground_speed_along_max(state, wind, desired_xy)
        desired_unreachable = bool(
            desired_tgo > 0.0 and desired_vg_along <= float(self.config.min_progress_mps)
        )
        radius = self._search_radius(state, wind, terrain)
        center_xy = state.position_xy
        candidates = self._generate_candidates(center_xy, radius)
        # Always include desired target location
        if candidates.size == 0:
            candidates = desired_xy.reshape(1, 2)
        else:
            candidates = np.vstack([candidates, desired_xy.reshape(1, 2)])

        best: Optional[LandingSiteSelection] = None
        for cand in candidates:
            n = float(cand[0])
            e = float(cand[1])
            hard_violation, nofly_penalty = self._nofly_penalty(n, e, no_fly_circles, no_fly_polygons)
            if hard_violation:
                continue
            tgo = self._time_to_land(state, terrain, n, e)
            if tgo <= 0.0:
                continue
            ok, margin, v_air = self._reachable(state, wind, cand, tgo)
            if not ok:
                continue
            (
                touchdown_vg_along,
                touchdown_v_air,
                touchdown_sink,
                touchdown_v_ground,
                touchdown_ld_air,
                touchdown_ld_ground,
            ) = self._ground_speed_along_max(state, wind, cand)

            risk_val = 0.0
            risk_details: Dict[str, float] = {}
            if self.risk_map is not None and self.risk_map.layers:
                risk_val, risk_details = self.risk_map.risk(n, e)
            risk_val = float(risk_val + float(self.config.nofly_weight) * float(nofly_penalty))

            dist = float(np.linalg.norm(cand - desired_xy))
            dist_cost = dist / max(radius, 1.0)
            margin_cost = 1.0 - float(np.clip(margin / max(v_air, 1e-6), 0.0, 1.0))

            # Energy proxy: prefer shallow required glide slope (more distance to bleed energy)
            terrain_h = float(terrain.height_m(n, e)) if terrain is not None else 0.0
            h_agl = float(state.altitude - terrain_h - float(self.config.reachability.terrain_clearance_m))
            s_rem = float(max(np.linalg.norm(cand - state.position_xy), 1e-3))
            k_req = float(max(h_agl / s_rem, 0.0))
            k_nom = float(self._polar.slope(float(self.config.reachability.brake)))
            energy_cost = float(np.clip(k_req / max(k_nom, 1e-6), 0.0, 3.0))

            score = (
                float(self.config.w_risk) * risk_val
                + float(self.config.w_distance) * dist_cost
                + float(self.config.w_reach_margin) * margin_cost
                + float(self.config.w_energy) * energy_cost
            )

            metadata = dict(risk_details)
            metadata.update(
                {
                    "nofly_penalty": float(nofly_penalty),
                    "dist_to_desired_m": float(dist),
                    "reachable_margin_mps": float(margin),
                    "desired_reachable": float(1.0 if desired_ok else 0.0),
                    "desired_margin_mps": float(desired_margin),
                    "desired_tgo_s": float(desired_tgo),
                    "desired_v_g_along_max_mps": float(desired_vg_along),
                    "desired_v_air_mps": float(desired_v_air),
                    "desired_sink_mps": float(desired_sink),
                    "desired_v_ground_mps": float(desired_v_ground),
                    "desired_ld_air": float(desired_ld_air),
                    "desired_ld_ground": float(desired_ld_ground),
                    "desired_wind_speed_mps": float(np.linalg.norm(wind.v_I[:2])),
                    "touchdown_tgo_s": float(tgo),
                    "touchdown_v_g_along_max_mps": float(touchdown_vg_along),
                    "touchdown_v_air_mps": float(touchdown_v_air),
                    "touchdown_sink_mps": float(touchdown_sink),
                    "touchdown_v_ground_mps": float(touchdown_v_ground),
                    "touchdown_ld_air": float(touchdown_ld_air),
                    "touchdown_ld_ground": float(touchdown_ld_ground),
                    "touchdown_wind_speed_mps": float(np.linalg.norm(wind.v_I[:2])),
                }
            )

            if best is None or score < best.score:
                if self.config.snap_to_terrain and terrain is not None:
                    down = float(-terrain.height_m(n, e))
                else:
                    down = float(desired_target.p_I[2])
                best = LandingSiteSelection(
                    target=Target(p_I=np.array([n, e, down], dtype=float)),
                    score=float(score),
                    risk=float(risk_val),
                    distance_to_desired_m=float(dist),
                    reach_margin_mps=float(margin),
                    time_to_land_s=float(tgo),
                    reason="unreachable_wind" if desired_unreachable else "ok",
                    metadata=metadata,
                )

        if best is not None:
            return best

        # Fallback: use desired target as-is
        return LandingSiteSelection(
            target=desired_target,
            score=float("inf"),
            risk=float("inf"),
            distance_to_desired_m=0.0,
            reach_margin_mps=-1.0,
            time_to_land_s=-1.0,
            reason="no_reachable_candidate",
        )
