from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize

from parafoil_planner_v3.environment import FlatTerrain, NoFlyCircle, NoFlyPolygon, TerrainModel
from parafoil_planner_v3.types import Control, State, Trajectory, TrajectoryType, Waypoint
from parafoil_planner_v3.utils.quaternion_utils import quat_to_rpy

from .gpm_collocation import GPMCollocation


@dataclass(frozen=True)
class SolverInfo:
    success: bool
    status: int
    message: str
    iterations: int
    cost: float
    solve_time: float
    max_violation: float
    terminal_error_m: float


@dataclass(frozen=True)
class SolverConfig:
    # Objective weights
    w_terminal_pos: float = 50.0
    w_terminal_vel: float = 1.0
    w_running_u: float = 0.05
    w_running_du: float = 0.01
    w_running_w: float = 0.0
    w_running_wdot: float = 0.0
    w_running_energy: float = 0.0
    w_u_ref: float = 0.0

    # Bounds / constraints
    tf_min: float = 1.0
    tf_max: float = 120.0
    delta_rate_max: float = 3.0  # 1/s
    Vh_min: float = 0.5  # m/s
    Vh_max: float = 8.0  # m/s
    roll_max_rad: float = np.deg2rad(60.0)
    yaw_rate_max: float = np.deg2rad(120.0)
    terminal_heading_tol_rad: float = np.deg2rad(5.0)

    # Solver options
    method: str = "SLSQP"
    maxiter: int = 300
    ftol: float = 1e-6
    # Wall-time budget for a single solve (<=0 disables timeout).
    max_solve_time_s: float = 0.0


class _SolveTimeout(RuntimeError):
    """Internal exception to abort a solve when time budget is exceeded."""



class GPMSolver:
    """
    SciPy-based NLP solver for GPM collocation.

    Decision variable layout:
      z = [X(0..N-1) flattened, U(0..N-1) flattened, tf]
    """

    def __init__(
        self,
        f: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        gpm: GPMCollocation,
        n_x: int = 13,
        n_u: int = 2,
        config: Optional[SolverConfig] = None,
    ) -> None:
        self.f = f
        self.gpm = gpm
        self.n_x = int(n_x)
        self.n_u = int(n_u)
        self.config = config or SolverConfig()
        self.last_solution_z: Optional[np.ndarray] = None
        self._u_ref: Optional[np.ndarray] = None
        self._terrain: TerrainModel | None = None
        self._terrain_clearance_m: float = 0.0
        self._no_fly_circles: tuple[NoFlyCircle, ...] = ()
        self._no_fly_polygons: tuple[NoFlyPolygon, ...] = ()
        self._terminal_heading_hat_xy: Optional[np.ndarray] = None

    def _unpack(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        z = np.asarray(z, dtype=float).reshape(-1)
        N = self.gpm.N
        nx = self.n_x
        nu = self.n_u
        nX = N * nx
        nU = N * nu
        if z.size != nX + nU + 1:
            raise ValueError(f"Bad decision vector size {z.size}, expected {nX+nU+1}")
        X = z[:nX].reshape(N, nx)
        U = z[nX : nX + nU].reshape(N, nu)
        tf = float(z[-1])
        return X, U, tf

    def _pack(self, X: np.ndarray, U: np.ndarray, tf: float) -> np.ndarray:
        return np.concatenate([X.reshape(-1), U.reshape(-1), np.array([tf], dtype=float)])

    def _initial_guess(self, x0: np.ndarray, p_target: np.ndarray, tf_guess: float) -> np.ndarray:
        x0 = np.asarray(x0, dtype=float).reshape(self.n_x)
        p_target = np.asarray(p_target, dtype=float).reshape(3)
        N = self.gpm.N

        X = np.zeros((N, self.n_x), dtype=float)
        # Position linear interpolation
        p0 = x0[0:3]
        for k in range(N):
            a = 0.0 if N == 1 else k / (N - 1)
            X[k, 0:3] = (1.0 - a) * p0 + a * p_target
        # Velocity: start with current
        X[:, 3:6] = x0[3:6]
        # Attitude/omega: keep at initial
        X[:, 6:10] = x0[6:10]
        X[:, 10:13] = x0[10:13]

        U = np.full((N, self.n_u), 0.2, dtype=float)
        tf = float(np.clip(tf_guess, self.config.tf_min, self.config.tf_max))
        return self._pack(X, U, tf)

    def _objective(self, z: np.ndarray, x0: np.ndarray, p_target: np.ndarray) -> float:
        X, U, tf = self._unpack(z)
        if not (self.config.tf_min <= tf <= self.config.tf_max):
            return 1e9 + 1e6 * abs(tf)

        # Terminal costs
        p_final = X[-1, 0:3]
        v_final = X[-1, 3:6]
        pos_err = p_final - np.asarray(p_target, dtype=float).reshape(3)
        J = 0.0
        J += self.config.w_terminal_pos * float(np.dot(pos_err, pos_err))
        J += self.config.w_terminal_vel * float(np.dot(v_final, v_final))

        # Running costs
        w_u = self.config.w_running_u
        w_du = self.config.w_running_du
        w_w = self.config.w_running_w
        w_wdot = self.config.w_running_wdot
        w_energy = self.config.w_running_energy

        # Use piecewise approx of integral with collocation weights
        def L(x_k: np.ndarray, u_k: np.ndarray, _: float) -> float:
            v = x_k[3:6]
            w = x_k[10:13]
            # Specific mechanical energy proxy: E = g*h + 0.5*||v||^2, with h = -p_D.
            h = float(-x_k[2])
            E = 9.81 * h + 0.5 * float(np.dot(v, v))
            return float(
                w_u * float(np.dot(u_k, u_k))
                + w_w * float(np.dot(w, w))
                + w_energy * float(E * E)
            )

        J += self.gpm.integrate_cost(L, X, U, 0.0, tf)

        # Optional reference control tracking (trajectory-type shaping)
        if self._u_ref is not None and self.config.w_u_ref > 0.0:
            U_ref = np.asarray(self._u_ref, dtype=float)
            if U_ref.shape != U.shape:
                return 1e9
            tau, w = self.gpm.nodes_and_weights()
            dt_half = 0.5 * tf
            diff = U - U_ref
            J += float(dt_half * np.sum(w * (self.config.w_u_ref * np.sum(diff * diff, axis=1))))

        # Smoothness: finite differences on u
        if w_du > 0.0 and U.shape[0] >= 2:
            tau = self.gpm.tau
            dt_half = 0.5 * tf
            for k in range(U.shape[0] - 1):
                dt = float(dt_half * (tau[k + 1] - tau[k]))
                if dt <= 1e-9:
                    continue
                du = (U[k + 1] - U[k]) / dt
                J += w_du * float(np.dot(du, du)) * dt

        # Angular-rate smoothness: finite differences on w_B
        if w_wdot > 0.0 and X.shape[0] >= 2:
            tau = self.gpm.tau
            dt_half = 0.5 * tf
            for k in range(X.shape[0] - 1):
                dt = float(dt_half * (tau[k + 1] - tau[k]))
                if dt <= 1e-9:
                    continue
                wdot = (X[k + 1, 10:13] - X[k, 10:13]) / dt
                J += w_wdot * float(np.dot(wdot, wdot)) * dt

        return float(J)

    def _constraint_dynamics(self, z: np.ndarray) -> np.ndarray:
        X, U, tf = self._unpack(z)
        return self.gpm.discretize_dynamics(self.f, X, U, 0.0, tf)

    def _constraint_boundary(self, z: np.ndarray, x0: np.ndarray, p_target: np.ndarray) -> np.ndarray:
        X, _, _ = self._unpack(z)
        x0 = np.asarray(x0, dtype=float).reshape(self.n_x)
        p_target = np.asarray(p_target, dtype=float).reshape(3)
        return np.concatenate([X[0] - x0, X[-1, 0:3] - p_target], axis=0)

    def _constraint_path(self, z: np.ndarray) -> np.ndarray:
        X, U, tf = self._unpack(z)

        # Enforce >=0 style constraints
        g_list = []

        # State constraints at nodes
        for k in range(self.gpm.N):
            v_xy = X[k, 3:5]
            Vh = float(np.linalg.norm(v_xy))
            g_list.append(Vh - self.config.Vh_min)
            g_list.append(self.config.Vh_max - Vh)

            roll, _, _ = quat_to_rpy(X[k, 6:10])
            g_list.append(self.config.roll_max_rad - abs(float(roll)))

            yaw_rate = float(X[k, 12])  # w_z
            g_list.append(self.config.yaw_rate_max - abs(yaw_rate))

            # Terrain clearance constraint: altitude >= terrain_height + clearance.
            if self._terrain is not None:
                terrain_h = float(self._terrain.height_m(float(X[k, 0]), float(X[k, 1])))
                altitude = float(-X[k, 2])
                g_list.append(altitude - (terrain_h + float(self._terrain_clearance_m)))

            # No-fly circles: distance_to_center >= radius + clearance.
            for zone in self._no_fly_circles:
                g_list.append(float(zone.signed_distance_m(float(X[k, 0]), float(X[k, 1]))))

            # No-fly polygons: signed distance >= 0.
            for zone in self._no_fly_polygons:
                g_list.append(float(zone.signed_distance_m(float(X[k, 0]), float(X[k, 1]))))

        # Control rate constraints (finite difference)
        tau = self.gpm.tau
        dt_half = 0.5 * tf
        for k in range(self.gpm.N - 1):
            dt = float(dt_half * (tau[k + 1] - tau[k]))
            if dt <= 1e-9:
                continue
            du = (U[k + 1] - U[k]) / dt
            for i in range(self.n_u):
                g_list.append(self.config.delta_rate_max - abs(float(du[i])))

        # Terminal upwind heading constraint (hard-ish): enforce terminal ground-track aligned to a desired heading.
        if self._terminal_heading_hat_xy is not None:
            v_xy = X[-1, 3:5]
            hat = np.asarray(self._terminal_heading_hat_xy, dtype=float).reshape(2)
            tol = float(self.config.terminal_heading_tol_rad)
            cross = float(v_xy[0] * hat[1] - v_xy[1] * hat[0])
            dot = float(v_xy[0] * hat[0] + v_xy[1] * hat[1])
            cross_limit = float(self.config.Vh_max) * float(np.sin(tol))
            dot_min = float(self.config.Vh_min) * float(np.cos(tol))
            # |cross| <= Vh_max*sin(tol)
            g_list.append(cross_limit - cross)
            g_list.append(cross_limit + cross)
            # dot >= Vh_min*cos(tol)
            g_list.append(dot - dot_min)

        return np.array(g_list, dtype=float)

    def _bounds(self) -> list[tuple[Optional[float], Optional[float]]]:
        N = self.gpm.N
        nx = self.n_x
        nu = self.n_u
        bounds: list[tuple[Optional[float], Optional[float]]] = []
        # States: unbounded
        bounds.extend([(None, None)] * (N * nx))
        # Controls: [0,1]
        bounds.extend([(0.0, 1.0)] * (N * nu))
        # Final time
        bounds.append((self.config.tf_min, self.config.tf_max))
        return bounds

    def solve(
        self,
        x0: np.ndarray,
        p_target: np.ndarray,
        tf_guess: float,
        warm_start: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
        *,
        terrain: TerrainModel | None = None,
        terrain_clearance_m: float = 0.0,
        no_fly_circles: Sequence[NoFlyCircle] | None = None,
        no_fly_polygons: Sequence[NoFlyPolygon] | None = None,
        terminal_heading_hat_xy: Optional[np.ndarray] = None,
    ) -> tuple[Trajectory, SolverInfo]:
        x0 = np.asarray(x0, dtype=float).reshape(self.n_x)
        p_target = np.asarray(p_target, dtype=float).reshape(3)

        z0 = warm_start if warm_start is not None else self._initial_guess(x0, p_target, tf_guess)
        self._u_ref = None if u_ref is None else np.asarray(u_ref, dtype=float).reshape(self.gpm.N, self.n_u)
        self._terrain = terrain if terrain is not None else None
        self._terrain_clearance_m = float(terrain_clearance_m)
        self._no_fly_circles = tuple(no_fly_circles or ())
        self._no_fly_polygons = tuple(no_fly_polygons or ())
        self._terminal_heading_hat_xy = None if terminal_heading_hat_xy is None else np.asarray(terminal_heading_hat_xy, dtype=float).reshape(2)
        # Default terrain to "flat" if a clearance is requested but no terrain is provided.
        if self._terrain is None and self._terrain_clearance_m > 0.0:
            self._terrain = FlatTerrain(height0_m=0.0)

        try:
            constraints = [
                {"type": "eq", "fun": self._constraint_dynamics},
                {"type": "eq", "fun": self._constraint_boundary, "args": (x0, p_target)},
                {"type": "ineq", "fun": self._constraint_path},
            ]

            t_start = time.perf_counter()
            max_time = float(self.config.max_solve_time_s)
            callback = None
            if max_time > 0.0:
                def _timeout_cb(_xk):  # noqa: ANN001 - scipy callback signature
                    if time.perf_counter() - t_start > max_time:
                        raise _SolveTimeout(f"gpm_solve_timeout>{max_time:.2f}s")
                callback = _timeout_cb

            method = str(self.config.method)
            options: dict = {"maxiter": self.config.maxiter}
            if method.lower() in {"slsqp"}:
                options["ftol"] = self.config.ftol
            elif method.lower() in {"trust-constr", "trust-constr".replace("-", "_")}:
                # SciPy trust-constr uses gtol/xtol/barrier_tol (no ftol).
                options["gtol"] = self.config.ftol
                options["xtol"] = self.config.ftol
                options["barrier_tol"] = self.config.ftol
            else:
                # Best-effort for other methods.
                options["ftol"] = self.config.ftol

            try:
                result = scipy.optimize.minimize(
                    fun=self._objective,
                    x0=z0,
                    args=(x0, p_target),
                    method=method,
                    bounds=self._bounds(),
                    constraints=constraints,
                    options=options,
                    callback=callback,
                )
            except _SolveTimeout as e:
                raise TimeoutError(str(e)) from e
            solve_time = time.perf_counter() - t_start
            self.last_solution_z = np.asarray(result.x, dtype=float).copy()

            X, U, tf = self._unpack(result.x)

            # Build trajectory
            tau = self.gpm.tau
            waypoints = []
            for k in range(self.gpm.N):
                t_k = self.gpm.tau_to_time(float(tau[k]), 0.0, tf)
                waypoints.append(Waypoint(t=t_k, state=State.from_vector(X[k], t=t_k)))
            controls = [Control(float(U[k, 0]), float(U[k, 1])).clipped() for k in range(self.gpm.N)]
            traj = Trajectory(waypoints=waypoints, controls=controls, trajectory_type=TrajectoryType.DIRECT)

            # Compute violations
            # NOTE: the constraint evaluation below uses the current environment context.
            dyn_violation = np.max(np.abs(self._constraint_dynamics(result.x))) if result.x.size else float("inf")
            bnd_violation = np.max(np.abs(self._constraint_boundary(result.x, x0, p_target))) if result.x.size else float("inf")
            path_violation = np.min(self._constraint_path(result.x)) if result.x.size else -float("inf")
            # For inequality constraints g(z) >= 0: negative means violation
            max_violation = float(max(dyn_violation, bnd_violation, max(0.0, -path_violation)))

            terminal_error = float(np.linalg.norm(X[-1, 0:3] - p_target))
            info = SolverInfo(
                success=bool(result.success),
                status=int(result.status),
                message=str(result.message),
                iterations=int(result.nit) if hasattr(result, "nit") else -1,
                cost=float(result.fun) if np.isfinite(result.fun) else float("inf"),
                solve_time=float(solve_time),
                max_violation=max_violation,
                terminal_error_m=terminal_error,
            )

            return traj, info
        finally:
            # Clear per-solve context (important if SciPy throws).
            self._u_ref = None
            self._terrain = None
            self._terrain_clearance_m = 0.0
            self._no_fly_circles = ()
            self._terminal_heading_hat_xy = None
