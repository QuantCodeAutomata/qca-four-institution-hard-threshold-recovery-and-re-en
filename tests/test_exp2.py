"""Tests for Exp 2: Rank-K closure and finite-population convergence."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from low_rank_limit import (
    ARCH1, ARCH2, ARCH3, ARCH4, ARCHITECTURES,
    solve_limit_feedback,
    compute_limiting_threshold_fraction,
    compute_groupwise_threshold_fraction,
)
from low_rank_finite import (
    largest_remainder_round,
    make_quantile_populations,
    simulate_finite_hard_grouped,
    compute_finite_slopes,
)
from monte_carlo_convergence import (
    fit_convergence_slope,
    run_deterministic_study,
)
from metrics import uniform_threshold_fraction_error, feedback_path_error


# ---------------------------------------------------------------------------
# largest_remainder_round
# ---------------------------------------------------------------------------

def test_largest_remainder_round_sums_to_N():
    props = np.array([0.25, 0.75])
    counts = largest_remainder_round(props, 100)
    assert counts.sum() == 100


def test_largest_remainder_round_three_groups():
    props = np.array([0.4, 0.35, 0.25])
    counts = largest_remainder_round(props, 400)
    assert counts.sum() == 400
    assert all(c >= 0 for c in counts)


# ---------------------------------------------------------------------------
# make_quantile_populations
# ---------------------------------------------------------------------------

def test_make_quantile_populations_shape():
    groups = ARCH1["groups"]
    N, K = 100, ARCH1["K"]
    X0, g_idx, a_mat, b_mat = make_quantile_populations(groups, N, K)
    assert X0.shape == (N,)
    assert g_idx.shape == (N,)
    assert a_mat.shape == (N, K)
    assert b_mat.shape == (N, K)


def test_make_quantile_populations_mean_close_to_group_mean():
    """For a single group, sample mean should be close to group mean."""
    groups = [{"p": 1.0, "m": 0.24, "sigma": 0.18, "a": [0.7], "b": [0.7]}]
    X0, _, _, _ = make_quantile_populations(groups, 1000, 1)
    assert abs(float(np.mean(X0)) - 0.24) < 0.01


# ---------------------------------------------------------------------------
# compute_finite_slopes
# ---------------------------------------------------------------------------

def test_compute_finite_slopes_no_distress():
    """No distress -> slopes all equal mu."""
    N, K = 4, 1
    a = np.ones((N, K)) * 0.5
    b = np.ones((N, K)) * 0.5
    distress = np.zeros(N)
    slopes, hat_beta = compute_finite_slopes(None, distress, a, b, mu=0.1, K=K, N=N)
    np.testing.assert_allclose(slopes, 0.1 * np.ones(N), atol=1e-15)
    np.testing.assert_allclose(hat_beta, np.zeros(K), atol=1e-15)


def test_compute_finite_slopes_all_distressed():
    """All distressed -> hat_beta = mean(b, axis=0), slopes = mu - a.dot(hat_beta)/K."""
    N, K = 4, 2
    a = np.tile([0.8, 0.2], (N, 1))
    b = np.tile([0.7, 0.3], (N, 1))
    distress = np.ones(N)
    slopes, hat_beta = compute_finite_slopes(None, distress, a, b, mu=0.0, K=K, N=N)
    expected_beta = np.array([0.7, 0.3])
    np.testing.assert_allclose(hat_beta, expected_beta, atol=1e-12)
    expected_slopes = -(a @ expected_beta) / K
    np.testing.assert_allclose(slopes, expected_slopes, atol=1e-12)


# ---------------------------------------------------------------------------
# solve_limit_feedback: basic properties
# ---------------------------------------------------------------------------

def test_solve_limit_feedback_beta0_zero():
    """Initial condition beta(0) = 0."""
    sol = solve_limit_feedback(ARCH1["groups"], ARCH1["mu"], ARCH1["r"], ARCH1["T"], ARCH1["K"])
    np.testing.assert_allclose(sol.y[:, 0], np.zeros(ARCH1["K"]), atol=1e-12)


def test_solve_limit_feedback_beta_nonneg_increasing():
    """beta(t) must be nondecreasing and nonneg for all architectures."""
    for arch in ARCHITECTURES:
        sol = solve_limit_feedback(arch["groups"], arch["mu"], arch["r"], arch["T"], arch["K"])
        assert np.all(sol.y >= -1e-10), f"Negative beta detected in {arch}"
        diff = np.diff(sol.y, axis=1)
        assert np.all(diff >= -1e-8), f"Non-monotone beta in {arch}"


def test_solve_limit_feedback_output_shape():
    """Output shape should be (K, n_eval)."""
    for arch in ARCHITECTURES:
        sol = solve_limit_feedback(arch["groups"], arch["mu"], arch["r"], arch["T"], arch["K"])
        assert sol.y.shape[0] == arch["K"]
        assert sol.y.shape[1] == 1001


# ---------------------------------------------------------------------------
# compute_limiting_threshold_fraction: range [0,1]
# ---------------------------------------------------------------------------

def test_limiting_h_in_unit_interval():
    """h(t) must be in [0, 1] for all architectures."""
    for arch in ARCHITECTURES:
        sol = solve_limit_feedback(arch["groups"], arch["mu"], arch["r"], arch["T"], arch["K"])
        h = compute_limiting_threshold_fraction(sol.y, arch["groups"], arch["mu"], arch["r"], sol.t, arch["K"])
        assert np.all(h >= -1e-10)
        assert np.all(h <= 1 + 1e-10)


# ---------------------------------------------------------------------------
# Terminal threshold fractions: close to expected values
# ---------------------------------------------------------------------------

EXPECTED_H_TERMINAL = [0.4062, 0.2226, 0.2417, 0.1304]

@pytest.mark.parametrize("arch, expected", list(zip(ARCHITECTURES, EXPECTED_H_TERMINAL)))
def test_terminal_h_close_to_expected(arch, expected):
    """Terminal threshold fraction within 2% of reported value."""
    sol = solve_limit_feedback(arch["groups"], arch["mu"], arch["r"], arch["T"], arch["K"])
    h = compute_limiting_threshold_fraction(sol.y, arch["groups"], arch["mu"], arch["r"], sol.t, arch["K"])
    h_T = float(h[-1])
    assert abs(h_T - expected) < 0.02, f"h(T)={h_T:.4f}, expected {expected}"


# ---------------------------------------------------------------------------
# simulate_finite_hard_grouped: sanity checks
# ---------------------------------------------------------------------------

def test_finite_simulation_output_shapes():
    """Dense output shapes are correct."""
    arch = ARCH1
    N, K = 50, arch["K"]
    X0, _, a_mat, b_mat = make_quantile_populations(arch["groups"], N, K)
    t_eval = np.linspace(0, arch["T"], 201)
    X_dense, h_dense, beta_dense = simulate_finite_hard_grouped(
        X0, a_mat, b_mat, arch["mu"], K, N, arch["T"], t_eval
    )
    assert X_dense.shape == (N, 201)
    assert h_dense.shape == (201,)
    assert beta_dense.shape == (K, 201)


def test_finite_simulation_h_in_unit_interval():
    """h_t^N must stay in [0, 1]."""
    arch = ARCH1
    N, K = 50, arch["K"]
    X0, _, a_mat, b_mat = make_quantile_populations(arch["groups"], N, K)
    t_eval = np.linspace(0, arch["T"], 101)
    _, h_dense, _ = simulate_finite_hard_grouped(
        X0, a_mat, b_mat, arch["mu"], K, N, arch["T"], t_eval
    )
    assert np.all(h_dense >= -1e-10)
    assert np.all(h_dense <= 1 + 1e-10)


def test_finite_beta_nonneg():
    """Finite feedback beta^N(t) must be nonneg (distress indicators are 0/1)."""
    arch = ARCH1
    N, K = 50, arch["K"]
    X0, _, a_mat, b_mat = make_quantile_populations(arch["groups"], N, K)
    t_eval = np.linspace(0, arch["T"], 101)
    _, _, beta_dense = simulate_finite_hard_grouped(
        X0, a_mat, b_mat, arch["mu"], K, N, arch["T"], t_eval
    )
    assert np.all(beta_dense >= -1e-10)


# ---------------------------------------------------------------------------
# Deterministic convergence: error decreases with N
# ---------------------------------------------------------------------------

def test_deterministic_errors_decrease_with_N():
    """Threshold fraction errors should decrease as N doubles."""
    arch = ARCH1
    result = run_deterministic_study(arch, [100, 400])
    assert result["h_errors"][1] < result["h_errors"][0], (
        "Error did not decrease from N=100 to N=400"
    )


def test_deterministic_terminal_fraction_close():
    """Deterministic N=400 terminal fraction should be close to limiting value."""
    arch = ARCH1
    result = run_deterministic_study(arch, [400])
    # limiting terminal fraction ~0.4062; finite at N=400 should be close
    assert abs(result["h_errors"][0]) < 0.01


# ---------------------------------------------------------------------------
# fit_convergence_slope: basic OLS sanity
# ---------------------------------------------------------------------------

def test_fit_slope_exact_half():
    """Perfect N^{-1/2} decay should yield slope -0.5."""
    N_list = [100, 400, 1600]
    errors = [c * N ** (-0.5) for N, c in zip(N_list, [1.0, 1.0, 1.0])]
    slope = fit_convergence_slope(errors, N_list)
    assert abs(slope - (-0.5)) < 0.01


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def test_uniform_threshold_error_zero_for_identical():
    """Error is zero when paths are identical."""
    h = np.linspace(0, 0.5, 100)
    t = np.linspace(0, 2, 100)
    assert uniform_threshold_fraction_error(h, h, t) == 0.0


def test_uniform_threshold_error_positive():
    """Error is positive when paths differ."""
    h_a = np.ones(50) * 0.3
    h_b = np.ones(50) * 0.4
    t = np.linspace(0, 1, 50)
    err = uniform_threshold_fraction_error(h_a, h_b, t)
    assert abs(err - 0.1) < 1e-12


def test_feedback_path_error_zero():
    beta = np.ones((2, 100)) * 0.5
    assert feedback_path_error(beta, beta) == 0.0
