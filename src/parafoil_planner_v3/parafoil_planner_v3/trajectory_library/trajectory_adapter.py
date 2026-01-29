from __future__ import annotations

import numpy as np

from parafoil_planner_v3.types import State, Target, Trajectory, Waypoint, Wind

from .library_manager import LibraryTrajectory


def translate_trajectory(traj: Trajectory, delta_xy: np.ndarray) -> Trajectory:
    d = np.asarray(delta_xy, dtype=float).reshape(2)
    waypoints = []
    for wp in traj.waypoints:
        s = wp.state.copy()
        s.p_I[0:2] = s.p_I[0:2] + d
        waypoints.append(Waypoint(t=wp.t, state=s))
    return Trajectory(waypoints=waypoints, controls=list(traj.controls), trajectory_type=traj.trajectory_type, metadata=traj.metadata)


def adapt_trajectory(library_entry: LibraryTrajectory, state: State, target: Target, wind: Wind) -> Trajectory:  # noqa: ARG001
    """
    Online adaptation:
    - Scale distance/altitude to match current state relative to target.
    - Rotate to align target bearing.
    - Shift time origin to current time.

    Library trajectories are generated in a target-centered frame where target is at (0,0,0).
    """
    traj = library_entry.trajectory
    if not traj.waypoints:
        return traj

    # Bearing convention: from state -> target.
    rel_xy = target.position_xy - state.position_xy
    dist = float(np.linalg.norm(rel_xy))
    bearing = float(np.arctan2(rel_xy[1], rel_xy[0]))

    dist_lib = float(max(library_entry.scenario.target_distance_m, 1e-3))
    bearing_lib = float(np.deg2rad(library_entry.scenario.target_bearing_deg))

    scale_xy = dist / dist_lib
    alt_agl = float(max(state.altitude - target.altitude, 0.0))
    alt_lib = float(max(library_entry.scenario.initial_altitude_m, 1e-3))
    scale_z = alt_agl / alt_lib

    yaw = bearing - bearing_lib
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    R = np.array([[c, -s], [s, c]], dtype=float)

    # Time scaling proportional to distance scaling (simple heuristic).
    scale_t = scale_xy

    waypoints = []
    for wp in traj.waypoints:
        sp = wp.state

        # Positions are stored in NED; interpret as target-relative and transform.
        p_lib_xy = sp.p_I[:2]
        p_new_xy = target.position_xy + R @ (scale_xy * p_lib_xy)
        p_new_d = target.p_I[2] + scale_z * float(sp.p_I[2])  # target-relative down

        s_new = sp.copy()
        s_new.p_I = np.array([p_new_xy[0], p_new_xy[1], p_new_d], dtype=float)

        # Velocity scaling consistent with time scaling.
        # If scale_t == scale_xy, horizontal speed stays unchanged.
        vel_scale_xy = scale_xy / max(scale_t, 1e-6)
        vel_scale_z = scale_z / max(scale_t, 1e-6)
        v_new_xy = R @ (vel_scale_xy * sp.v_I[:2])
        v_new_d = vel_scale_z * float(sp.v_I[2])
        s_new.v_I = np.array([v_new_xy[0], v_new_xy[1], v_new_d], dtype=float)

        t_new = float(state.t + scale_t * wp.t)
        s_new.t = t_new
        waypoints.append(Waypoint(t=t_new, state=s_new))

    return Trajectory(
        waypoints=waypoints,
        controls=list(traj.controls),
        trajectory_type=traj.trajectory_type,
        metadata={
            **(traj.metadata or {}),
            "adapted": True,
            "scale_xy": scale_xy,
            "scale_z": scale_z,
            "scale_t": scale_t,
            "yaw": yaw,
        },
    )
