from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class TerrainModel(Protocol):
    def height_m(self, north_m: float, east_m: float) -> float:  # noqa: D102
        ...


@dataclass(frozen=True)
class FlatTerrain:
    height0_m: float = 0.0

    def height_m(self, north_m: float, east_m: float) -> float:  # noqa: ARG002
        return float(self.height0_m)


@dataclass(frozen=True)
class PlaneTerrain:
    """
    Simple planar terrain model: h = h0 + slope_n * N + slope_e * E.
    Heights are in meters "Up" (altitude).
    """

    height0_m: float = 0.0
    slope_n: float = 0.0
    slope_e: float = 0.0

    def height_m(self, north_m: float, east_m: float) -> float:
        return float(self.height0_m + self.slope_n * float(north_m) + self.slope_e * float(east_m))


@dataclass(frozen=True)
class NoFlyCircle:
    """
    No-fly circle in the planning NED XY plane.
    Constraint: distance_to_center >= radius + clearance.
    """

    center_n: float
    center_e: float
    radius_m: float
    clearance_m: float = 0.0

    def signed_distance_m(self, north_m: float, east_m: float) -> float:
        d = float(np.hypot(float(north_m) - self.center_n, float(east_m) - self.center_e))
        return float(d - (self.radius_m + self.clearance_m))


@dataclass(frozen=True)
class NoFlyPolygon:
    """
    No-fly polygon in the planning NED XY plane.
    Constraint: signed_distance >= 0 (positive outside), with optional clearance.
    """

    vertices: np.ndarray
    clearance_m: float = 0.0

    def __post_init__(self) -> None:
        verts = np.asarray(self.vertices, dtype=float).reshape(-1, 2)
        if verts.shape[0] < 3:
            raise ValueError("NoFlyPolygon requires at least 3 vertices")
        object.__setattr__(self, "vertices", verts)

    def _point_inside(self, p: np.ndarray) -> bool:
        x, y = float(p[0]), float(p[1])
        verts = self.vertices
        inside = False
        n = verts.shape[0]
        j = n - 1
        for i in range(n):
            xi, yi = float(verts[i, 0]), float(verts[i, 1])
            xj, yj = float(verts[j, 0]), float(verts[j, 1])
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        closest = a + t * ab
        return float(np.linalg.norm(p - closest))

    def signed_distance_m(self, north_m: float, east_m: float) -> float:
        p = np.array([float(north_m), float(east_m)], dtype=float)
        verts = self.vertices
        n = verts.shape[0]
        min_dist = float("inf")
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            d = self._segment_distance(p, a, b)
            if d < min_dist:
                min_dist = d
        inside = self._point_inside(p)
        if inside:
            return float(-(min_dist + float(self.clearance_m)))
        return float(min_dist - float(self.clearance_m))

    @staticmethod
    def from_dict(payload: dict) -> "NoFlyPolygon":
        verts = payload.get("vertices") or payload.get("points") or payload.get("polygon")
        if verts is None:
            raise ValueError("NoFlyPolygon dict missing vertices")
        clearance = float(payload.get("clearance_m", payload.get("clearance", 0.0)))
        return NoFlyPolygon(vertices=np.asarray(verts, dtype=float), clearance_m=clearance)


def load_no_fly_polygons(path: str | Path) -> list[NoFlyPolygon]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    ext = path.suffix.lower()
    if ext in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(path.read_text()) or {}
    elif ext in {".json", ".geojson"}:
        import json

        payload = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported no-fly file format: {ext}")

    def _parse_list(items) -> list[NoFlyPolygon]:
        out: list[NoFlyPolygon] = []
        for item in items:
            if isinstance(item, dict):
                out.append(NoFlyPolygon.from_dict(item))
            else:
                out.append(NoFlyPolygon(vertices=np.asarray(item, dtype=float)))
        return out

    if isinstance(payload, dict):
        if "polygons" in payload:
            return _parse_list(payload.get("polygons") or [])
        if "no_fly_polygons" in payload:
            return _parse_list(payload.get("no_fly_polygons") or [])
        # GeoJSON FeatureCollection
        if payload.get("type") == "FeatureCollection":
            feats = payload.get("features", []) or []
            polys: list[NoFlyPolygon] = []
            for feat in feats:
                geom = feat.get("geometry", {}) if isinstance(feat, dict) else {}
                if geom.get("type") in {"Polygon", "MultiPolygon"}:
                    coords = geom.get("coordinates", [])
                    if geom.get("type") == "Polygon" and coords:
                        verts = coords[0]
                        polys.append(NoFlyPolygon(vertices=np.asarray(verts, dtype=float)))
                    elif geom.get("type") == "MultiPolygon":
                        for poly in coords:
                            if poly:
                                polys.append(NoFlyPolygon(vertices=np.asarray(poly[0], dtype=float)))
            return polys
        # Single polygon dict
        if "vertices" in payload or "points" in payload or "polygon" in payload:
            return [NoFlyPolygon.from_dict(payload)]
    if isinstance(payload, list):
        return _parse_list(payload)
    raise ValueError("Invalid no-fly polygon file format")


@dataclass(frozen=True)
class GridTerrain:
    """
    Heightmap terrain loaded from file.

    Grid format:
      - height_m: 2D array (rows=N, cols=E)
      - origin_n / origin_e: world N/E of grid (0,0) corner
      - resolution_m: meters per cell
    """

    origin_n: float
    origin_e: float
    resolution_m: float
    height_map: np.ndarray

    def __post_init__(self) -> None:
        h = np.asarray(self.height_map, dtype=float)
        if h.ndim != 2:
            raise ValueError("GridTerrain height_m must be 2D")
        object.__setattr__(self, "height_map", h)

    def height_m_at(self, north_m: float, east_m: float) -> float:
        h = self.height_map
        res = float(self.resolution_m)
        if res <= 1e-9:
            return float(h[0, 0])
        u = (float(north_m) - float(self.origin_n)) / res
        v = (float(east_m) - float(self.origin_e)) / res
        i0 = int(np.floor(u))
        j0 = int(np.floor(v))
        i1 = i0 + 1
        j1 = j0 + 1
        i0 = int(np.clip(i0, 0, h.shape[0] - 1))
        i1 = int(np.clip(i1, 0, h.shape[0] - 1))
        j0 = int(np.clip(j0, 0, h.shape[1] - 1))
        j1 = int(np.clip(j1, 0, h.shape[1] - 1))
        fu = float(np.clip(u - i0, 0.0, 1.0))
        fv = float(np.clip(v - j0, 0.0, 1.0))
        h00 = float(h[i0, j0])
        h10 = float(h[i1, j0])
        h01 = float(h[i0, j1])
        h11 = float(h[i1, j1])
        h0 = h00 * (1.0 - fu) + h10 * fu
        h1 = h01 * (1.0 - fu) + h11 * fu
        return float(h0 * (1.0 - fv) + h1 * fv)

    def height_m(self, north_m: float, east_m: float) -> float:  # noqa: D102
        return self.height_m_at(north_m, east_m)

    @staticmethod
    def from_file(path: str | Path) -> "GridTerrain":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        ext = path.suffix.lower()
        if ext == ".npz":
            data = np.load(path)
            height = np.asarray(data["height_m"], dtype=float)
            origin_n = float(data["origin_n"]) if "origin_n" in data else 0.0
            origin_e = float(data["origin_e"]) if "origin_e" in data else 0.0
            resolution_m = float(data["resolution_m"]) if "resolution_m" in data else 1.0
            return GridTerrain(origin_n=origin_n, origin_e=origin_e, resolution_m=resolution_m, height_map=height)
        if ext in {".yaml", ".yml", ".json"}:
            if ext in {".yaml", ".yml"}:
                import yaml

                payload = yaml.safe_load(path.read_text()) or {}
            else:
                import json

                payload = json.loads(path.read_text())
            height = np.asarray(payload.get("height_m"), dtype=float)
            origin_n = float(payload.get("origin_n", 0.0))
            origin_e = float(payload.get("origin_e", 0.0))
            resolution_m = float(payload.get("resolution_m", 1.0))
            return GridTerrain(origin_n=origin_n, origin_e=origin_e, resolution_m=resolution_m, height_map=height)
        raise ValueError(f"Unsupported terrain file format: {ext}")
