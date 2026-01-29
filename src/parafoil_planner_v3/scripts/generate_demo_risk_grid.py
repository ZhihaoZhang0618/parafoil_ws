#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _gaussian(x: np.ndarray, y: np.ndarray, cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
    dx = (x - cx) / max(sx, 1e-6)
    dy = (y - cy) / max(sy, 1e-6)
    return np.exp(-0.5 * (dx * dx + dy * dy))


def generate_demo_risk_grid(
    size_m: float,
    resolution_m: float,
    origin_n: float,
    origin_e: float,
    seed: int,
) -> tuple[np.ndarray, float, float, float]:
    rng = np.random.default_rng(int(seed))
    n_cells = max(int(np.ceil(size_m / resolution_m)), 2)
    n_vals = origin_n + np.arange(n_cells, dtype=float) * float(resolution_m)
    e_vals = origin_e + np.arange(n_cells, dtype=float) * float(resolution_m)
    nn, ee = np.meshgrid(n_vals, e_vals, indexing="ij")

    risk = np.full((n_cells, n_cells), 0.05, dtype=float)

    # A "city center" high-risk blob
    risk += 0.8 * _gaussian(nn, ee, cx=0.0, cy=0.0, sx=60.0, sy=60.0)

    # A secondary residential area
    risk += 0.5 * _gaussian(nn, ee, cx=120.0, cy=-80.0, sx=50.0, sy=40.0)

    # A powerline corridor (east-west)
    corridor = np.exp(-0.5 * (ee / 12.0) ** 2)
    risk += 0.35 * corridor

    # A park / open field (lower risk)
    risk -= 0.25 * _gaussian(nn, ee, cx=-150.0, cy=100.0, sx=70.0, sy=60.0)

    # Small random texture
    noise = rng.normal(0.0, 0.03, size=risk.shape)
    risk += noise

    risk = np.clip(risk, 0.0, 1.0)
    return risk, float(origin_n), float(origin_e), float(resolution_m)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a demo risk grid for safety-first landing selection.")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    parser.add_argument("--size", type=float, default=400.0, help="Grid size in meters (square)")
    parser.add_argument("--resolution", type=float, default=5.0, help="Grid resolution in meters")
    parser.add_argument("--origin-n", type=float, default=-200.0, help="Origin north (meters)")
    parser.add_argument("--origin-e", type=float, default=-200.0, help="Origin east (meters)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for noise")

    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    risk, origin_n, origin_e, resolution_m = generate_demo_risk_grid(
        size_m=float(args.size),
        resolution_m=float(args.resolution),
        origin_n=float(args.origin_n),
        origin_e=float(args.origin_e),
        seed=int(args.seed),
    )

    np.savez(
        output,
        risk_map=risk,
        origin_n=origin_n,
        origin_e=origin_e,
        resolution_m=resolution_m,
    )
    print(f"Saved demo risk grid to {output}")


if __name__ == "__main__":
    main()
