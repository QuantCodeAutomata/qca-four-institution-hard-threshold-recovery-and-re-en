"""Tests for Exp 4: Osgood boundary ramp regularization (osgood_ramp.py)."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.osgood_ramp import (
    F_linear, F_log, F_sqrt,
    G_eps_linear, G_eps_log, G_eps_sqrt,
    solve_beta_eps,
    hard_solution_linear, hard_solution_log, hard_solution_sqrt,
    compute_sup_error, fit_slope,
)

T = 1.0  # time horizon matching the experiment (log slope = exp(-T) ~ 0.368 for T=1)

# ---------------------------------------------------------------------------
# Profile functions F
# ---------------------------------------------------------------------------

def test_F_linear_at_zero():
    assert F_linear(np.array([0.0]))[0] == 0.0


def test_F_linear_values():
    a = np.array([0.0, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(F_linear(a), a, atol=1e-15)


def test_F_log_continuity():
    """F_log(0) must be 0 (continuity at a=0)."""
    assert F_log(np.array([0.0]))[0] == 0.0


def test_F_log_positive_values():
    """F_log(a) = a*log(e/a) is positive for a in (0, e)."""
    a = np.array([0.1, 0.5, 1.0])
    vals = F_log(a)
    assert np.all(vals > 0)
    np.testing.assert_allclose(vals, a * np.log(np.e / a), atol=1e-15)


def test_F_sqrt_at_zero():
    assert F_sqrt(np.array([0.0]))[0] == 0.0


def test_F_sqrt_values():
    a = np.array([0.0, 0.25, 1.0, 4.0])
    np.testing.assert_allclose(F_sqrt(a), np.sqrt(a), atol=1e-15)


def test_F_sqrt_negative_clipped():
    """F_sqrt clips negative inputs to zero."""
    assert F_sqrt(np.array([-1.0]))[0] == 0.0


# ---------------------------------------------------------------------------
# Averaged kernels G_eps
# ---------------------------------------------------------------------------

def test_G_eps_linear_formula():
    """G_eps_linear(eps, b) = b + eps/2."""
    eps, b = 0.1, 0.5
    assert abs(G_eps_linear(eps, b) - 0.55) < 1e-15


def test_G_eps_linear_at_b0():
    eps = 0.2
    assert abs(G_eps_linear(eps, 0.0) - 0.1) < 1e-15


def test_G_eps_sqrt_formula():
    """G_eps_sqrt(eps, b) = (2/(3*eps))*[(b+eps)^1.5 - b^1.5]."""
    eps, b = 0.01, 0.25
    expected = (2.0 / (3.0 * eps)) * ((b + eps) ** 1.5 - b ** 1.5)
    assert abs(G_eps_sqrt(eps, b) - expected) < 1e-15


def test_G_eps_sqrt_b0():
    """G_eps_sqrt(eps, 0) = (2/3)*sqrt(eps)."""
    eps = 0.04
    expected = (2.0 / 3.0) * np.sqrt(eps)
    assert abs(G_eps_sqrt(eps, 0.0) - expected) < 1e-14


def test_G_eps_log_positive():
    """G_eps_log must be positive for eps>0, b>=0."""
    for eps in [0.1, 0.01, 0.001]:
        for b in [0.0, 0.1, 1.0]:
            val = G_eps_log(eps, b)
            assert val > 0, f"G_eps_log({eps}, {b}) = {val} not positive"


def test_G_eps_log_converges_to_F():
    """G_eps_log(eps, b) -> F_log(b) as eps -> 0 for b > 0."""
    b = 0.5
    f_exact = float(F_log(np.array([b]))[0])
    for eps in [1e-3, 1e-4, 1e-5]:
        g = G_eps_log(eps, b)
        assert abs(g - f_exact) < 5 * eps, (
            f"G_eps_log({eps}, {b}) = {g}, F_log({b}) = {f_exact}"
        )


# ---------------------------------------------------------------------------
# Hard solutions
# ---------------------------------------------------------------------------

def test_hard_solution_linear_zero():
    """Hard solution for linear is identically zero."""
    t = np.linspace(0, T, 100)
    np.testing.assert_array_equal(hard_solution_linear(t), np.zeros(100))


def test_hard_solution_log_zero():
    """Hard solution for log is identically zero."""
    t = np.linspace(0, T, 100)
    np.testing.assert_array_equal(hard_solution_log(t), np.zeros(100))


def test_hard_solution_sqrt():
    """Greatest hard solution for sqrt: beta(t) = t^2/4."""
    t = np.array([0.0, 1.0, 2.0])
    expected = np.array([0.0, 0.25, 1.0])
    np.testing.assert_allclose(hard_solution_sqrt(t), expected, atol=1e-15)


def test_hard_solution_sqrt_ode():
    """Verify beta(t)=t^2/4 satisfies dot{beta}=sqrt(beta) (greatest solution)."""
    t = np.linspace(0.01, T, 200)
    beta = hard_solution_sqrt(t)
    dbeta_dt = np.gradient(beta, t)
    # np.gradient has lower accuracy at endpoints (one-sided differences); skip them
    np.testing.assert_allclose(dbeta_dt[1:-1], F_sqrt(beta)[1:-1], rtol=1e-3)


# ---------------------------------------------------------------------------
# ODE solver sanity checks
# ---------------------------------------------------------------------------

def test_solve_beta_linear_formula():
    """beta_eps(t) for linear should equal (eps/2)*(exp(t)-1)."""
    eps = 0.1
    t, beta = solve_beta_eps(G_eps_linear, eps, T)
    expected = (eps / 2.0) * (np.exp(t) - 1.0)
    np.testing.assert_allclose(beta, expected, rtol=1e-7, atol=1e-9)


def test_solve_beta_starts_at_zero():
    """beta_eps(0) = 0 for all three profiles."""
    for G_eps in [G_eps_linear, G_eps_log, G_eps_sqrt]:
        _, beta = solve_beta_eps(G_eps, 0.01, T)
        assert abs(beta[0]) < 1e-14, f"Initial condition not zero: {beta[0]}"


def test_solve_beta_nonnegative():
    """beta_eps(t) >= 0 for all t (positivity)."""
    for G_eps in [G_eps_linear, G_eps_log, G_eps_sqrt]:
        _, beta = solve_beta_eps(G_eps, 0.01, T)
        assert float(np.min(beta)) >= -1e-12, f"Negative beta_eps detected: {np.min(beta)}"


# ---------------------------------------------------------------------------
# Convergence slopes
# ---------------------------------------------------------------------------

def test_convergence_slope_linear():
    """Estimated slope for linear profile should be near 1.0 (within 0.1)."""
    eps_list = [2.0 ** (-k) for k in range(9, 13)]
    errors = []
    for eps in eps_list:
        t, beta = solve_beta_eps(G_eps_linear, eps, T)
        beta_hard = hard_solution_linear(t)
        errors.append(compute_sup_error(t, beta, beta_hard))
    slope = fit_slope(errors, eps_list)
    assert abs(slope - 1.0) < 0.1, f"Linear slope = {slope:.4f}, expected ~1.0"


def test_convergence_slope_sqrt():
    """Estimated slope for sqrt profile should be near 0.5 (within 0.1)."""
    eps_list = [2.0 ** (-k) for k in range(9, 13)]
    errors = []
    for eps in eps_list:
        t, beta = solve_beta_eps(G_eps_sqrt, eps, T)
        beta_hard = hard_solution_sqrt(t)
        errors.append(compute_sup_error(t, beta, beta_hard))
    slope = fit_slope(errors, eps_list)
    assert abs(slope - 0.5) < 0.1, f"Sqrt slope = {slope:.4f}, expected ~0.5"


def test_log_error_decreasing():
    """Sup errors for log profile should decrease as eps decreases."""
    eps_list = [2.0 ** (-k) for k in range(4, 10)]
    errors = []
    for eps in eps_list:
        t, beta = solve_beta_eps(G_eps_log, eps, T)
        beta_hard = hard_solution_log(t)
        errors.append(compute_sup_error(t, beta, beta_hard))
    # Errors should be monotonically decreasing
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1], (
            f"Error not decreasing at index {i}: {errors[i]:.4e} vs {errors[i+1]:.4e}"
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def test_compute_sup_error_identical():
    """sup error between identical arrays is 0."""
    t = np.linspace(0, T, 100)
    beta = t ** 2 / 4.0
    assert compute_sup_error(t, beta, beta) == 0.0


def test_compute_sup_error_known():
    """sup |f(t) - g(t)| = 1 for f = t, g = t - 1 on [0, 2]."""
    t = np.linspace(0, T, 1000)
    err = compute_sup_error(t, t, t - 1.0)
    assert abs(err - 1.0) < 1e-10


def test_fit_slope_exact():
    """fit_slope recovers exact slope for power-law data."""
    eps_list = [2.0 ** (-k) for k in range(4, 10)]
    gamma = 0.75
    errors = [eps ** gamma for eps in eps_list]
    slope = fit_slope(errors, eps_list)
    assert abs(slope - gamma) < 1e-10
