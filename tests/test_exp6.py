"""Unit tests for Experiment 6: Positive-Factor Truncation."""

import numpy as np
import pytest

from positive_factor_kernel import (
    a_factor,
    b_factor,
    cumulative_mass,
    factor_weight,
    initial_profile_exp6,
)
from positive_factor_truncation import (
    precompute_factor_arrays,
    solve_factored_graphon,
)


# ─── Kernel definitions ───────────────────────────────────────────────────────

def test_factor_weights_decreasing():
    """Weights must decrease strictly with m (since 0.35 * m^{-1.8} is decreasing)."""
    weights = [factor_weight(m) for m in range(1, 10)]
    for i in range(len(weights) - 1):
        assert weights[i] > weights[i + 1], (
            f"weight({i+1})={weights[i]:.4f} is not > weight({i+2})={weights[i+1]:.4f}"
        )


def test_factor_weights_positive():
    """All weights should be strictly positive."""
    for m in range(1, 31):
        assert factor_weight(m) > 0.0


def test_a_factor_at_zero():
    """a_1(0) = 1 + 0.45*sin(0) + 0.15*cos(2*pi*2*0) = 1.15."""
    val = float(a_factor(1, 0.0))
    assert val == pytest.approx(1.15, rel=1e-10)


def test_a_factor_vectorised():
    """a_m should accept arrays and return arrays of the same shape."""
    u = np.linspace(0.0, 1.0, 50)
    result = a_factor(3, u)
    assert result.shape == (50,)
    assert np.all(np.isfinite(result))


def test_b_factor_vectorised():
    """b_m should accept arrays and return arrays of the same shape."""
    v = np.linspace(0.0, 1.0, 50)
    result = b_factor(2, v)
    assert result.shape == (50,)
    assert np.all(np.isfinite(result))


# ─── Initial profile ──────────────────────────────────────────────────────────

def test_initial_profile_exp6():
    """x0(0.5) = 0.13 + 0.08*sin(pi) - 0.04*cos(3*pi) = 0.17."""
    val = float(initial_profile_exp6(0.5))
    assert val == pytest.approx(0.17, abs=1e-12)


def test_initial_profile_exp6_at_zero():
    """x0(0) = 0.13 + 0 - 0.04*cos(0) = 0.13 - 0.04 = 0.09."""
    val = float(initial_profile_exp6(0.0))
    assert val == pytest.approx(0.09, abs=1e-12)


def test_initial_profile_exp6_range():
    """Profile should stay in a reasonable range on [0, 1]."""
    u = np.linspace(0.0, 1.0, 201)
    x0 = initial_profile_exp6(u)
    assert x0.min() > -0.1
    assert x0.max() < 0.3


# ─── Cumulative mass ──────────────────────────────────────────────────────────

def test_cumulative_mass_increasing():
    """Cumulative mass fraction should be non-decreasing with K."""
    masses = [cumulative_mass(K, M_total=30) for K in range(1, 31)]
    for i in range(len(masses) - 1):
        assert masses[i] <= masses[i + 1]


def test_cumulative_mass_full():
    """Cumulative mass at K=M_total should equal 1."""
    assert cumulative_mass(30, M_total=30) == pytest.approx(1.0)


def test_cumulative_mass_range():
    """Cumulative mass must be in (0, 1] for any valid K."""
    for K in [1, 4, 12, 24, 30]:
        cm = cumulative_mass(K, M_total=30)
        assert 0.0 < cm <= 1.0


# ─── Factor array precomputation ──────────────────────────────────────────────

def test_precompute_factor_arrays_shapes():
    """Precomputed arrays must have the expected shapes."""
    N, K = 40, 6
    u = (np.arange(N) + 0.5) / N
    A, B, w = precompute_factor_arrays(u, K)
    assert A.shape == (N, K)
    assert B.shape == (N, K)
    assert w.shape == (K,)


def test_precompute_factor_arrays_values():
    """First column of A_matrix should match a_1(u_grid) exactly."""
    N, K = 30, 3
    u = (np.arange(N) + 0.5) / N
    A, _, _ = precompute_factor_arrays(u, K)
    expected = a_factor(1, u)
    assert np.allclose(A[:, 0], expected)


def test_precompute_weights_match():
    """Weights array must match factor_weight(m) for m=1..K."""
    _, _, w = precompute_factor_arrays(np.linspace(0, 1, 10), K=5)
    for idx, m in enumerate(range(1, 6)):
        assert w[idx] == pytest.approx(factor_weight(m))


# ─── ODE solver convergence ───────────────────────────────────────────────────

def _small_solve(K: int, N: int = 50, T: float = 0.5, mu: float = -0.006, eps: float = 0.02):
    """Helper: solve the factored ODE on a small grid and return X_t."""
    u = (np.arange(N) + 0.5) / N
    x0 = initial_profile_exp6(u)
    t_eval = np.linspace(0.0, T, 26)
    A, B, w = precompute_factor_arrays(u, K)
    _, X_t = solve_factored_graphon(
        A, B, w, x0, mu, T, eps, t_eval, K=K, N=N,
        rtol=1e-7, atol=1e-9, max_step=0.02,
    )
    return X_t


def test_K4_less_accurate_than_K12():
    """K=4 approximation should have larger state error than K=12 vs K=30 reference."""
    N, T = 50, 0.5
    u = (np.arange(N) + 0.5) / N
    x0 = initial_profile_exp6(u)
    t_eval = np.linspace(0.0, T, 26)
    mu, eps = -0.006, 0.02

    # Reference K=30
    A_ref, B_ref, w_ref = precompute_factor_arrays(u, 30)
    _, X_ref = solve_factored_graphon(
        A_ref, B_ref, w_ref, x0, mu, T, eps, t_eval, K=30, N=N,
        rtol=1e-8, atol=1e-10, max_step=0.02,
    )

    def state_L1(X_t):
        return float(np.max(np.mean(np.abs(X_t - X_ref), axis=0)))

    err4 = state_L1(_small_solve(4, N=N, T=T, mu=mu, eps=eps))
    err12 = state_L1(_small_solve(12, N=N, T=T, mu=mu, eps=eps))

    assert err4 > err12, (
        f"Expected err(K=4)={err4:.3e} > err(K=12)={err12:.3e}"
    )


def test_K24_close_to_K30():
    """K=24 error vs K=30 should be much smaller than K=4 error."""
    N, T = 50, 0.5
    u = (np.arange(N) + 0.5) / N
    x0 = initial_profile_exp6(u)
    t_eval = np.linspace(0.0, T, 26)
    mu, eps = -0.006, 0.02

    A_ref, B_ref, w_ref = precompute_factor_arrays(u, 30)
    _, X_ref = solve_factored_graphon(
        A_ref, B_ref, w_ref, x0, mu, T, eps, t_eval, K=30, N=N,
        rtol=1e-8, atol=1e-10, max_step=0.02,
    )

    def state_L1(K):
        X_t = _small_solve(K, N=N, T=T, mu=mu, eps=eps)
        return float(np.max(np.mean(np.abs(X_t - X_ref), axis=0)))

    err4 = state_L1(4)
    err24 = state_L1(24)

    assert err24 < err4, (
        f"Expected err(K=24)={err24:.3e} < err(K=4)={err4:.3e}"
    )
    # K=24 should capture nearly all mass and be very close
    assert cumulative_mass(24, M_total=30) > 0.95
