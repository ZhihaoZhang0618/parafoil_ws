# Config Index (parafoil_planner_v3)

This folder contains all YAML configuration files used by the planner, guidance, library generation, and demos.
The layout is flat to keep ROS2 install paths stable. Use the grouped index below to find the right file quickly.

## Core planner
- `planner_params.yaml` — baseline planner parameters (online use)
- `planner_params_safety_demo.yaml` — safety-first demo tuned for library + fast GPM fallback
- `planner_params_strongwind.yaml` — strong-wind scenario overrides

## Library generation (offline)
- `library_params.yaml` — coarse library generation grid (simplified, denser coverage)
- `library_params_full.yaml` — fine library generation (high-risk focus, non-uniform grid, mixed 6DOF sampling)
- `library_params_6dof_validation.yaml` — small 6DOF validation grid (extreme conditions)

## GPM / optimization / dynamics
- `gpm_params.yaml` — GPM cost/constraint defaults (solver weights + constraints)
- `optimization.yaml` — optimization flags/weights (legacy or auxiliary)
- `dynamics_params.yaml` — aerodynamic/vehicle parameters

## Guidance profiles
- `guidance_params.yaml` — default guidance profile
- `guidance_params_calm_low.yaml` — calm/low wind profile
- `guidance_params_crosswind.yaml` — crosswind profile
- `guidance_params_gusty.yaml` — gusty profile
- `guidance_params_strongwind.yaml` — strong-wind profile

## Scenarios / datasets
- `batch_scenarios.yaml` — batch scenario definitions
- `demo_risk_grid.npz` — demo risk grid for safety-first landing selection
