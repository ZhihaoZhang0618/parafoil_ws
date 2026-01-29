from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from numpy.polynomial.legendre import Legendre


def _barycentric_weights(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float).reshape(-1)
    n = int(nodes.shape[0])
    w = np.ones(n, dtype=float)
    for j in range(n):
        diff = nodes[j] - np.delete(nodes, j)
        if np.any(np.abs(diff) < 1e-14):
            raise ValueError("Duplicate collocation nodes detected")
        w[j] = 1.0 / np.prod(diff)
    return w


def _differentiation_matrix(nodes: np.ndarray) -> np.ndarray:
    """
    Compute Lagrange differentiation matrix for arbitrary distinct nodes.

    Uses barycentric formula:
      D_ij = w_j/(w_i*(x_i-x_j)) for i!=j
      D_ii = -sum_{j!=i} D_ij
    """
    x = np.asarray(nodes, dtype=float).reshape(-1)
    n = int(x.shape[0])
    w = _barycentric_weights(x)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            D[i, j] = (w[j] / w[i]) / (x[i] - x[j])
        D[i, i] = -np.sum(D[i, :])
    return D


def _lgl_nodes_and_weights(N: int) -> Tuple[np.ndarray, np.ndarray]:
    if N < 2:
        raise ValueError("LGL requires N>=2 (including endpoints)")

    # Interior nodes are roots of derivative of P_{N-1}
    P = Legendre.basis(N - 1)
    dP = P.deriv()
    interior = dP.roots()
    tau = np.concatenate([[-1.0], interior, [1.0]]).astype(float)

    # Quadrature weights: w_i = 2/(N(N-1) * P_{N-1}(tau_i)^2)
    P_vals = P(tau)
    weights = 2.0 / (N * (N - 1) * (P_vals**2))
    return tau, weights.astype(float)


@dataclass(frozen=True)
class GPMCollocation:
    """Gaussian Pseudospectral Method (LG/LGL) collocation utilities."""

    N: int
    scheme: str = "LGL"  # "LG" or "LGL"

    def __post_init__(self) -> None:
        if self.N < 2:
            raise ValueError("N must be >= 2")
        if self.scheme not in {"LG", "LGL"}:
            raise ValueError("scheme must be 'LG' or 'LGL'")

    @property
    def tau(self) -> np.ndarray:
        tau, _ = self.nodes_and_weights()
        return tau

    @property
    def weights(self) -> np.ndarray:
        _, w = self.nodes_and_weights()
        return w

    def nodes_and_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.scheme == "LG":
            tau, w = np.polynomial.legendre.leggauss(self.N)
            return tau.astype(float), w.astype(float)
        return _lgl_nodes_and_weights(self.N)

    @property
    def D(self) -> np.ndarray:
        return _differentiation_matrix(self.tau)

    @staticmethod
    def tau_to_time(tau: float, t0: float, tf: float) -> float:
        return float(0.5 * (tf - t0) * (tau + 1.0) + t0)

    def discretize_dynamics(
        self,
        f: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        x_nodes: np.ndarray,
        u_nodes: np.ndarray,
        t0: float,
        tf: float,
    ) -> np.ndarray:
        """
        Return dynamics defects stacked as (N*nx,) array:
          (2/(tf-t0)) * D @ X - f(X_k, U_k, t_k) == 0
        """
        X = np.asarray(x_nodes, dtype=float)
        U = np.asarray(u_nodes, dtype=float)
        if X.shape[0] != self.N or U.shape[0] != self.N:
            raise ValueError("x_nodes and u_nodes must have shape (N, ...)")

        dt_half = 0.5 * (tf - t0)
        if dt_half <= 0.0:
            raise ValueError("tf must be > t0")

        D = self.D
        dX_dtau = D @ X  # (N,nx)
        dX_dt = (1.0 / dt_half) * dX_dtau

        defects = []
        for k in range(self.N):
            t_k = self.tau_to_time(float(self.tau[k]), t0, tf)
            f_k = np.asarray(f(X[k], U[k], t_k), dtype=float).reshape(-1)
            defects.append(dX_dt[k] - f_k)
        return np.concatenate(defects, axis=0)

    def integrate_cost(
        self,
        L: Callable[[np.ndarray, np.ndarray, float], float],
        x_nodes: np.ndarray,
        u_nodes: np.ndarray,
        t0: float,
        tf: float,
    ) -> float:
        X = np.asarray(x_nodes, dtype=float)
        U = np.asarray(u_nodes, dtype=float)
        if X.shape[0] != self.N or U.shape[0] != self.N:
            raise ValueError("x_nodes and u_nodes must have shape (N, ...)")

        dt_half = 0.5 * (tf - t0)
        if dt_half <= 0.0:
            return float("inf")

        running = 0.0
        tau, w = self.nodes_and_weights()
        for k in range(self.N):
            t_k = self.tau_to_time(float(tau[k]), t0, tf)
            running += float(w[k]) * float(L(X[k], U[k], t_k))
        return float(dt_half * running)

