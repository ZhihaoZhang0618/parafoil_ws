#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Rect:
    n_min: float
    e_min: float
    n_max: float
    e_max: float
    name: str
    kind: str
    clearance_m: float = 0.0

    def expanded(self, margin_m: float) -> "Rect":
        m = float(margin_m)
        return Rect(
            n_min=float(self.n_min - m),
            e_min=float(self.e_min - m),
            n_max=float(self.n_max + m),
            e_max=float(self.e_max + m),
            name=self.name,
            kind=self.kind,
            clearance_m=float(self.clearance_m),
        )

    def vertices(self) -> list[list[float]]:
        return [
            [float(self.n_min), float(self.e_min)],
            [float(self.n_max), float(self.e_min)],
            [float(self.n_max), float(self.e_max)],
            [float(self.n_min), float(self.e_max)],
        ]


def _mask_rect(nn: np.ndarray, ee: np.ndarray, rect: Rect) -> np.ndarray:
    return (nn >= float(rect.n_min)) & (nn <= float(rect.n_max)) & (ee >= float(rect.e_min)) & (ee <= float(rect.e_max))


def _apply_max(risk: np.ndarray, mask: np.ndarray, value: float) -> None:
    if not np.any(mask):
        return
    risk[mask] = np.maximum(risk[mask], float(value))


def _apply_min(risk: np.ndarray, mask: np.ndarray, value: float) -> None:
    if not np.any(mask):
        return
    risk[mask] = np.minimum(risk[mask], float(value))


def _add_cluster(
    dest: list[Rect],
    *,
    center_n: float,
    center_e: float,
    rows: int,
    cols: int,
    step_n: float,
    step_e: float,
    base_w: float,
    base_h: float,
    name_prefix: str,
    kind: str,
    clearance_m: float,
) -> None:
    idx = 0
    row_offset = (rows - 1) * 0.5
    col_offset = (cols - 1) * 0.5
    for r in range(rows):
        for c in range(cols):
            cn = float(center_n + (r - row_offset) * float(step_n))
            ce = float(center_e + (c - col_offset) * float(step_e))
            w = float(base_w + 2.0 * ((r + c) % 2))
            h = float(base_h + 2.0 * ((r * 2 + c) % 2))
            dest.append(
                Rect(
                    n_min=cn - w / 2.0,
                    e_min=ce - h / 2.0,
                    n_max=cn + w / 2.0,
                    e_max=ce + h / 2.0,
                    name=f"{name_prefix}_{idx:02d}",
                    kind=kind,
                    clearance_m=float(clearance_m),
                )
            )
            idx += 1


def build_scene_geometry(size_m: float) -> tuple[list[Rect], list[Rect], list[Rect], list[Rect]]:
    """
    Build a synthetic "Shenzhen-like" urban scene (NOT an exact map).

    Returns:
      no_fly_office, no_fly_residential, no_fly_powerline, low_risk_open_spaces
    """

    no_fly_office: list[Rect] = []
    no_fly_residential: list[Rect] = []
    no_fly_powerline: list[Rect] = []
    low_risk_open: list[Rect] = []
    half = float(size_m) * 0.5
    rich = float(size_m) >= 700.0

    # Central park / open field (recommended landing areas)
    low_risk_open.append(Rect(-70.0, -95.0, 70.0, 95.0, name="central_park_core", kind="park"))
    low_risk_open.append(Rect(-35.0, 95.0, 45.0, 140.0, name="central_park_extension_e", kind="park"))
    low_risk_open.append(Rect(-170.0, 110.0, -120.0, 180.0, name="wasteland_sw", kind="wasteland"))
    if rich:
        low_risk_open.append(Rect(120.0, -260.0, 260.0, -140.0, name="park_southeast", kind="park"))
        low_risk_open.append(Rect(-300.0, -320.0, -200.0, -220.0, name="wasteland_southwest", kind="wasteland"))
        low_risk_open.append(Rect(-260.0, 220.0, -140.0, 320.0, name="park_northwest", kind="park"))

    # Office cluster (rectangular towers) - keep outside the park area.
    # East-south quadrant
    office_centers = [
        (130.0, 120.0),
        (120.0, 150.0),
        (150.0, 145.0),
        (165.0, 110.0),
        (110.0, 105.0),
        (95.0, 135.0),
    ]
    for i, (cn, ce) in enumerate(office_centers, start=1):
        w = 18.0 + 2.0 * (i % 3)
        h = 22.0 + 2.0 * ((i + 1) % 3)
        no_fly_office.append(
            Rect(
                n_min=cn - w / 2.0,
                e_min=ce - h / 2.0,
                n_max=cn + w / 2.0,
                e_max=ce + h / 2.0,
                name=f"office_{i:02d}",
                kind="office",
                clearance_m=10.0,
            )
        )

    # Residential blocks (many small rectangles) - north-west quadrant
    base_n, base_e = 140.0, -140.0
    for r in range(3):
        for c in range(4):
            cn = base_n - r * 28.0
            ce = base_e + c * 32.0
            w = 14.0 + 2.0 * ((r + c) % 2)
            h = 18.0 + 2.0 * ((r + 2 * c) % 2)
            no_fly_residential.append(
                Rect(
                    n_min=cn - w / 2.0,
                    e_min=ce - h / 2.0,
                    n_max=cn + w / 2.0,
                    e_max=ce + h / 2.0,
                    name=f"res_{r}_{c}",
                    kind="residential",
                    clearance_m=8.0,
                )
            )

    # Add a denser residential block cluster south-west of that
    base_n, base_e = 60.0, -170.0
    for i in range(8):
        cn = base_n - (i % 4) * 22.0
        ce = base_e + (i // 4) * 40.0
        no_fly_residential.append(
            Rect(
                n_min=cn - 10.0,
                e_min=ce - 12.0,
                n_max=cn + 10.0,
                e_max=ce + 12.0,
                name=f"res_dense_{i:02d}",
                kind="residential",
                clearance_m=8.0,
            )
        )

    # High-voltage powerline corridor (long thin rectangle)
    no_fly_powerline.append(
        Rect(
            n_min=-10.0,
            e_min=-half,
            n_max=10.0,
            e_max=half,
            name="powerline_corridor",
            kind="powerline",
            clearance_m=12.0,
        )
    )
    # Add a few "pylon" squares along the corridor
    if rich:
        pylon_es = np.linspace(-0.8 * half, 0.8 * half, 7).tolist()
    else:
        pylon_es = [-160.0, -80.0, 0.0, 80.0, 160.0]
    for i, ce in enumerate(pylon_es, start=1):
        no_fly_powerline.append(
            Rect(
                n_min=-6.0,
                e_min=ce - 6.0,
                n_max=6.0,
                e_max=ce + 6.0,
                name=f"pylon_{i:02d}",
                kind="powerline_pylon",
                clearance_m=6.0,
            )
        )

    if rich:
        # Extra office clusters
        _add_cluster(
            no_fly_office,
            center_n=200.0,
            center_e=260.0,
            rows=2,
            cols=3,
            step_n=28.0,
            step_e=32.0,
            base_w=20.0,
            base_h=24.0,
            name_prefix="office_ne",
            kind="office",
            clearance_m=10.0,
        )
        _add_cluster(
            no_fly_office,
            center_n=-200.0,
            center_e=260.0,
            rows=2,
            cols=2,
            step_n=30.0,
            step_e=34.0,
            base_w=22.0,
            base_h=26.0,
            name_prefix="office_nw",
            kind="office",
            clearance_m=10.0,
        )
        # Industrial warehouses (larger rectangles)
        no_fly_office.append(Rect(220.0, -80.0, 320.0, 20.0, name="industrial_east_01", kind="industrial", clearance_m=12.0))
        no_fly_office.append(Rect(-340.0, 40.0, -250.0, 130.0, name="industrial_west_01", kind="industrial", clearance_m=12.0))

        # Extra residential clusters
        _add_cluster(
            no_fly_residential,
            center_n=-220.0,
            center_e=-120.0,
            rows=3,
            cols=3,
            step_n=26.0,
            step_e=28.0,
            base_w=14.0,
            base_h=18.0,
            name_prefix="res_sw",
            kind="residential",
            clearance_m=8.0,
        )
        _add_cluster(
            no_fly_residential,
            center_n=120.0,
            center_e=-260.0,
            rows=2,
            cols=4,
            step_n=24.0,
            step_e=26.0,
            base_w=14.0,
            base_h=18.0,
            name_prefix="res_se",
            kind="residential",
            clearance_m=8.0,
        )

        # Additional powerline corridors (one horizontal, one vertical)
        no_fly_powerline.append(
            Rect(
                n_min=210.0,
                e_min=-half,
                n_max=230.0,
                e_max=half,
                name="powerline_corridor_north",
                kind="powerline",
                clearance_m=12.0,
            )
        )
        no_fly_powerline.append(
            Rect(
                n_min=-half,
                e_min=-230.0,
                n_max=half,
                e_max=-210.0,
                name="powerline_corridor_west",
                kind="powerline",
                clearance_m=12.0,
            )
        )
        # Pylons along the vertical corridor
        pylon_ns = np.linspace(-0.8 * half, 0.8 * half, 6).tolist()
        for i, cn in enumerate(pylon_ns, start=1):
            no_fly_powerline.append(
                Rect(
                    n_min=cn - 6.0,
                    e_min=-226.0,
                    n_max=cn + 6.0,
                    e_max=-214.0,
                    name=f"pylon_w_{i:02d}",
                    kind="powerline_pylon",
                    clearance_m=6.0,
                )
            )

    return no_fly_office, no_fly_residential, no_fly_powerline, low_risk_open


def generate_risk_grid(
    *,
    n_cells: int,
    resolution_m: float,
    origin_n: float,
    origin_e: float,
    seed: int,
) -> tuple[np.ndarray, float, float, float, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    n = float(origin_n) + np.arange(int(n_cells), dtype=float) * float(resolution_m)
    e = float(origin_e) + np.arange(int(n_cells), dtype=float) * float(resolution_m)
    nn, ee = np.meshgrid(n, e, indexing="ij")

    size_m = float(n_cells) * float(resolution_m)
    half = size_m * 0.5

    # Base risk: "dense city" baseline
    risk = np.full((int(n_cells), int(n_cells)), 0.22, dtype=np.float32)

    # Light urban density gradient (more dense towards +E, +N)
    scale = max(120.0, size_m / 3.0)
    grad = 0.08 * (1.0 / (1.0 + np.exp(-(ee / scale)))) + 0.06 * (1.0 / (1.0 + np.exp(-(nn / scale))))
    risk = np.clip(risk + grad.astype(np.float32), 0.0, 1.0)

    no_fly_office, no_fly_residential, no_fly_powerline, low_risk_open = build_scene_geometry(size_m)

    # Roads (high risk stripes) - axes-aligned
    roads = [
        Rect(-half, -14.0, half, 14.0, name="main_road_ew", kind="road"),
        Rect(-14.0, -half, 14.0, half, name="main_road_ns", kind="road"),
        Rect(85.0, -half, 105.0, half, name="ring_road_n", kind="road"),
        Rect(-140.0, -half, -120.0, half, name="ring_road_s", kind="road"),
    ]
    if size_m >= 700.0:
        extra_offsets = [-200.0, 200.0]
        road_w = 16.0
        for off in extra_offsets:
            roads.append(Rect(off - road_w / 2.0, -half, off + road_w / 2.0, half, name=f"road_ns_{int(off)}", kind="road"))
            roads.append(Rect(-half, off - road_w / 2.0, half, off + road_w / 2.0, name=f"road_ew_{int(off)}", kind="road"))
    for rd in roads:
        _apply_max(risk, _mask_rect(nn, ee, rd), 0.65)

    # Water / lake inside the park: very high risk (not strictly no-fly here)
    lake = Rect(-18.0, -18.0, 18.0, 18.0, name="park_lake", kind="water")
    if size_m >= 700.0:
        lake2 = Rect(160.0, -220.0, 210.0, -170.0, name="park_lake_se", kind="water")
        _apply_max(risk, _mask_rect(nn, ee, lake2), 0.95)
    _apply_max(risk, _mask_rect(nn, ee, lake), 0.95)

    # Open spaces: force low risk
    for open_rect in low_risk_open:
        _apply_min(risk, _mask_rect(nn, ee, open_rect), 0.06 if open_rect.kind == "park" else 0.08)
    # Keep the lake risky even though it's inside the park.
    _apply_max(risk, _mask_rect(nn, ee, lake), 0.95)

    # Office buildings: no-fly + high-risk buffer
    for b in no_fly_office:
        _apply_max(risk, _mask_rect(nn, ee, b.expanded(6.0)), 0.85)
        _apply_max(risk, _mask_rect(nn, ee, b), 0.98)

    # Residential buildings: no-fly + moderate-high buffer
    for b in no_fly_residential:
        _apply_max(risk, _mask_rect(nn, ee, b.expanded(5.0)), 0.78)
        _apply_max(risk, _mask_rect(nn, ee, b), 0.96)

    # Powerlines: no-fly corridor + strong buffer
    for b in no_fly_powerline:
        _apply_max(risk, _mask_rect(nn, ee, b.expanded(6.0)), 0.86)
        _apply_max(risk, _mask_rect(nn, ee, b), 0.99)

    # Small texture noise (kept subtle to preserve "blocky" style)
    noise = rng.normal(0.0, 0.015, size=risk.shape).astype(np.float32)
    risk = np.clip(risk + noise, 0.0, 1.0)

    metadata: dict[str, Any] = {
        "note": "Synthetic shenzhen-like demo scene (NOT real map). Squares/rectangles dominate by design.",
        "n_cells": int(n_cells),
        "resolution_m": float(resolution_m),
        "origin_n": float(origin_n),
        "origin_e": float(origin_e),
        "seed": int(seed),
        "size_m": float(size_m),
        "variant": "rich" if size_m >= 700.0 else "base",
        "elements": {
            "office_buildings": len(no_fly_office),
            "residential_buildings": len(no_fly_residential),
            "powerline_polygons": len(no_fly_powerline),
            "open_spaces": len(low_risk_open),
        },
    }
    return risk, float(origin_n), float(origin_e), float(resolution_m), metadata


def write_outputs(
    *,
    output_npz: Path,
    output_nofly_json: Path,
    risk: np.ndarray,
    origin_n: float,
    origin_e: float,
    resolution_m: float,
    metadata: dict[str, Any],
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_nofly_json.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_npz,
        risk_map=np.asarray(risk, dtype=np.float32),
        origin_n=float(origin_n),
        origin_e=float(origin_e),
        resolution_m=float(resolution_m),
        metadata=json.dumps(metadata),
    )

    size_m = float(risk.shape[0]) * float(resolution_m)
    no_fly_office, no_fly_residential, no_fly_powerline, _ = build_scene_geometry(size_m)
    polygons: list[dict[str, Any]] = []
    for rect in [*no_fly_office, *no_fly_residential, *no_fly_powerline]:
        polygons.append(
            {
                "name": rect.name,
                "kind": rect.kind,
                "clearance_m": float(rect.clearance_m),
                "vertices": rect.vertices(),
            }
        )
    payload = {
        "note": "Synthetic shenzhen-like no-fly polygons (NOT real map). Coordinates are NED (north,east) meters.",
        "no_fly_polygons": polygons,
    }
    output_nofly_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic Shenzhen-like 2D safety scene (risk grid + no-fly polygons).")
    parser.add_argument("--output-npz", type=str, required=True, help="Output risk grid .npz path")
    parser.add_argument("--output-nofly", type=str, required=True, help="Output no-fly polygons .json path")
    parser.add_argument("--cells", type=int, default=400, help="Grid cells per axis (default: 400 => 400x400)")
    parser.add_argument("--resolution", type=float, default=1.0, help="Grid resolution in meters (default: 1.0m)")
    parser.add_argument("--origin-n", type=float, default=-200.0, help="Origin north (meters), lower-left corner")
    parser.add_argument("--origin-e", type=float, default=-200.0, help="Origin east (meters), lower-left corner")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for minor texture noise")

    args = parser.parse_args()
    risk, origin_n, origin_e, resolution_m, metadata = generate_risk_grid(
        n_cells=int(args.cells),
        resolution_m=float(args.resolution),
        origin_n=float(args.origin_n),
        origin_e=float(args.origin_e),
        seed=int(args.seed),
    )

    write_outputs(
        output_npz=Path(args.output_npz),
        output_nofly_json=Path(args.output_nofly),
        risk=risk,
        origin_n=origin_n,
        origin_e=origin_e,
        resolution_m=resolution_m,
        metadata=metadata,
    )

    print(f"Saved risk grid: {args.output_npz}")
    print(f"Saved no-fly polygons: {args.output_nofly}")
    elems = metadata.get('elements', {})
    print(f"Elements: {json.dumps(elems, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
