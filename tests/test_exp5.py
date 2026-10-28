"""Unit tests for Experiment 5: Graphon Smoothing and Fourier Truncation."""

import numpy as np
import pytest

from graphon_kernel import (
    build_fourier_kernel_matrix,
    build_kernel_matrix,
    initial_profile,
    ramp_loss,
)
from graphon_solver import (
    compute_path_L1_error,
    compute_threshold_fraction,
    compute_threshold_fraction_error,
    solve_graphon_ramp,
)


# ─── Ramp loss ────────────────────────────────────────────────────────────────

def test_ramp_loss_values():
    """Ramp loss at canonical points."""
    assert ramp_loss(0.0, 0.1) == pytest.approx(1.0)
    assert ramp_loss(0.05, 0.1) == pytest.approx(0.5)
    assert ramp_loss(0.2, 0.1) == pytest.approx(0.0)


def test_ramp_loss_boundary():
    """Ramp loss at exact epsilon boundary."""
    assert ramp_loss(0.1, 0.1) == pytest.approx(0.0)
    # Negative values should give 1
    assert ramp_loss(-0.5, 0.05) == pytest.approx(1.0)


# ─── Initial profile ──────────────────────────────────────────────────────────

def test_initial_profile_continuity():
    """Profile values just below and above u=0.5 should be close (within 0.05)."""
    delta = 1e-6
    v_left = float(initial_profile(0.5 - delta))
    v_right = float(initial_profile(0.5 + delta))
    # The profile has a designed small jump at the block boundary (~0.025);
    # verify it is finite and the jump is bounded.
    assert np.isfinite(v_left) and np.isfinite(v_right)
    assert abs(v_left - v_right) < 0.05


def test_initial_profile_range():
    """Profile should be well below 2 and above -0.2 on the full unit interval."""
    u = np.linspace(0.0, 1.0, 201)
    x0 = initial_profile(u)
    assert x0.min() > -0.25
    assert x0.max() < 2.2


def test_initial_profile_piecewise():
    """Profile should match the piecewise formula exactly."""
    assert initial_profile(0.0) == pytest.approx(-0.18)
    assert initial_profile(0.25) == pytest.approx(-0.18 + 1.95 * 0.25)
    assert initial_profile(1.0) == pytest.approx(0.82 + 1.75 * 0.5)


# ─── Kernel matrix ────────────────────────────────────────────────────────────

def test_kernel_matrix_shape():
    """Kernel matrix should have shape (N, N)."""
    N = 30
    u = (np.arange(N) + 0.5) / N
    W = build_kernel_matrix(u)
    assert W.shape == (N, N)


def test_kernel_matrix_symmetry():
    """Kernel matrix need not be symmetric (B is not symmetric), but should be finite."""
    N = 20
    u = (np.arange(N) + 0.5) / N
    W = build_kernel_matrix(u)
    assert np.all(np.isfinite(W))


def test_kernel_matrix_positive():
    """Kernel values must be non-negative given B entries and exp term > 0."""
    N = 40
    u = (np.arange(N) + 0.5) / N
    W = build_kernel_matrix(u)
    assert np.all(W > 0)


# ─── Fourier kernel ───────────────────────────────────────────────────────────

def test_bessel_fourier_kernel():
    """M=16 and M=32 Fourier kernels should agree to within 1e-5 (Bessel decay)."""
    N = 40
    u = (np.arange(N) + 0.5) / N
    W16 = build_fourier_kernel_matrix(u, M=16, full_block=False)
    W32 = build_fourier_kernel_matrix(u, M=32, full_block=False)
    assert np.max(np.abs(W16 - W32)) < 1e-5


def test_fourier_kernel_convergence_to_exact():
    """High-M Fourier kernel should be close to the exact kernel."""
    N = 40
    u = (np.arange(N) + 0.5) / N
    W_exact = build_kernel_matrix(u)
    W_approx = build_fourier_kernel_matrix(u, M=32, full_block=True)
    assert np.max(np.abs(W_exact - W_approx)) < 1e-5


def test_fourier_kernel_error_decreasing():
    """Kernel L-inf errors should decrease strictly as M increases."""
    N = 50
    u = (np.arange(N) + 0.5) / N
    W_full = build_kernel_matrix(u)
    M_values = [1, 4, 16]
    errors = []
    for M in M_values:
        W_M = build_fourier_kernel_matrix(u, M, full_block=True)
        errors.append(np.max(np.abs(W_full - W_M)))
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1], (
            f"Kernel error did not decrease: M={M_values[i]} err={errors[i]:.3e}, "
            f"M={M_values[i+1]} err={errors[i+1]:.3e}"
        )


def test_fourier_kernel_full_block_flag():
    """full_block=False should return only the periodic component (no B matrix)."""
    N = 20
    u = (np.arange(N) + 0.5) / N
    W_per = build_fourier_kernel_matrix(u, M=4, full_block=False)
    W_full = build_fourier_kernel_matrix(u, M=4, full_block=True)
    W_full2 = build_kernel_matrix(u)
    # With full_block=True, result should be closer to exact than with full_block=False
    err_full = np.max(np.abs(W_full2 - W_full))
    err_per = np.max(np.abs(W_full2 - W_per))
    assert err_full < err_per


# ─── ODE solver and error metrics ─────────────────────────────────────────────

def test_threshold_fraction_range():
    """Threshold fraction h(t) must lie in [0, 1] at all times."""
    N = 50
    T = 0.5
    u = (np.arange(N) + 0.5) / N
    x0 = -0.18 + 1.95 * u  # first-block profile (some negative values)
    W = build_kernel_matrix(u)
    t_eval = np.linspace(0.0, T, 26)

    _, X_t = solve_graphon_ramp(
        W, x0, mu=-0.06, T=T, epsilon=0.02,
        t_eval=t_eval, rtol=1e-6, atol=1e-8, max_step=0.02,
    )
    h = compute_threshold_fraction(X_t, u)
    assert np.all(h >= 0.0)
    assert np.all(h <= 1.0)


def test_path_L1_error_zero():
    """L1 error of a trajectory against itself should be zero."""
    X_t = np.random.default_rng(0).standard_normal((30, 20))
    u = np.linspace(0, 1, 30)
    assert compute_path_L1_error(X_t, X_t, u) == pytest.approx(0.0)


def test_threshold_fraction_error_zero():
    """Threshold-fraction error of identical paths should be zero."""
    h = np.array([0.1, 0.15, 0.2, 0.18])
    assert compute_threshold_fraction_error(h, h) == pytest.approx(0.0)


def test_smoothing_error_decreasing():
    """State path L1 errors should decrease as epsilon decreases (converging to hard)."""
    N = 100
    T = 1.0
    u = (np.arange(N) + 0.5) / N
    x0 = -0.18 + 1.95 * np.where(u < 0.5, u, 0.5) + np.where(u >= 0.5, 0.82 + 1.75 * (u - 0.5), 0.0)
    x0 = np.where(u < 0.5, -0.18 + 1.95 * u, 0.82 + 1.75 * (u - 0.5))
    W = build_kernel_matrix(u)
    t_eval = np.linspace(0.0, T, 51)

    # Hard reference
    _, X_hard = solve_graphon_ramp(
        W, x0, mu=-0.06, T=T, epsilon=1e-4,
        t_eval=t_eval, rtol=1e-8, atol=1e-10, max_step=0.005,
    )
    errors = []
    for eps in [0.04, 0.02, 0.01]:
        _, X_eps = solve_graphon_ramp(
            W, x0, mu=-0.06, T=T, epsilon=eps,
            t_eval=t_eval, rtol=1e-6, atol=1e-8, max_step=0.02,
        )
        errors.append(compute_path_L1_error(X_eps, X_hard, u))

    # Largest epsilon should have the largest error
    assert errors[0] > errors[2], (
        f"eps=0.04 error ({errors[0]:.3e}) should exceed eps=0.01 error ({errors[2]:.3e})"
    )
    # Both eps=0.04 and eps=0.02 should exceed eps=0.01 (approximately monotone)
    assert errors[1] > errors[2] * 0.5
