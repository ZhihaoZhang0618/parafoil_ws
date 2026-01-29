import numpy as np
from numpy.testing import assert_allclose

from parafoil_planner_v3.optimization.gpm_collocation import GPMCollocation


def test_lg_nodes_weights_sum_to_2():
    gpm = GPMCollocation(N=10, scheme="LG")
    tau, w = gpm.nodes_and_weights()
    assert tau.shape == (10,)
    assert w.shape == (10,)
    assert_allclose(np.sum(w), 2.0, atol=1e-12)


def test_lgl_nodes_include_endpoints_and_weights_sum_to_2():
    gpm = GPMCollocation(N=12, scheme="LGL")
    tau, w = gpm.nodes_and_weights()
    assert tau.shape == (12,)
    assert w.shape == (12,)
    assert_allclose(tau[0], -1.0, atol=1e-12)
    assert_allclose(tau[-1], 1.0, atol=1e-12)
    assert_allclose(np.sum(w), 2.0, atol=1e-10)


def test_differentiation_matrix_exact_for_polynomial():
    gpm = GPMCollocation(N=10, scheme="LGL")
    x = gpm.tau
    f = x**5 - 2.0 * x**3 + 0.1 * x
    df = 5.0 * x**4 - 6.0 * x**2 + 0.1
    df_hat = gpm.D @ f
    assert_allclose(df_hat, df, atol=1e-10)


def test_integrate_cost_constant_is_duration():
    gpm = GPMCollocation(N=8, scheme="LG")
    X = np.zeros((gpm.N, 13))
    U = np.zeros((gpm.N, 2))

    def L(_x, _u, _t):
        return 1.0

    t0, tf = 3.0, 17.0
    J = gpm.integrate_cost(L, X, U, t0, tf)
    assert_allclose(J, tf - t0, atol=1e-10)

