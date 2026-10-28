"""Tests for Exp 1: hard-threshold event-driven simulation (hard_finite.py)."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hard_finite import (
    compute_slopes,
    next_event_time,
    simulate_event_driven,
    analytic_X4,
    analytic_H4,
)

# ---------------------------------------------------------------------------
# Shared experiment parameters
# ---------------------------------------------------------------------------
N = 4
T = 5.0
MU = 0.1
X0 = np.array([-0.1, -1.0, 0.3, 0.04])
E_OVER_N = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.2, 0.0, 0.0],
    [0.2, 0.0, 0.3, 0.0],
])
E = N * E_OVER_N  # actual exposure matrix


# ---------------------------------------------------------------------------
# compute_slopes
# ---------------------------------------------------------------------------

def test_compute_slopes_no_distress():
    """When no institution is distressed (d=0), every slope equals mu."""
    d = np.zeros(N)
    v = compute_slopes(X0, d, E, MU, N)
    assert v.shape == (N,)
    np.testing.assert_allclose(v, MU * np.ones(N), atol=1e-15)


def test_compute_slopes_all_distressed():
    """With d=1 everywhere, verify slopes match formula v_i = mu - (1/N)*sum_j e_{ij}."""
    d = np.ones(N)
    v = compute_slopes(X0, d, E, MU, N)
    # Manually: row sums of E divided by N
    expected = MU - E @ d / N
    np.testing.assert_allclose(v, expected, atol=1e-15)


def test_compute_slopes_initial_state():
    """Verify slopes at t=0 with d=[1,1,0,0] match the paper setup."""
    d = (X0 <= 0).astype(float)
    assert list(d) == [1.0, 1.0, 0.0, 0.0]
    v = compute_slopes(X0, d, E, MU, N)
    # v_0 = 0.1 - 0/4 = 0.1
    # v_1 = 0.1 - 0/4 = 0.1
    # v_2 = 0.1 - (0.8*1)/4 = 0.1 - 0.2 = -0.1
    # v_3 = 0.1 - (0.8*1)/4 = 0.1 - 0.2 = -0.1
    expected = np.array([0.1, 0.1, -0.1, -0.1])
    np.testing.assert_allclose(v, expected, atol=1e-15)


# ---------------------------------------------------------------------------
# next_event_time
# ---------------------------------------------------------------------------

def test_next_event_time_positive():
    """Institution with X=0.5, v=-1.0 crosses zero at tau=0.5."""
    X = np.array([0.5])
    v = np.array([-1.0])
    tau = next_event_time(X, v)
    assert abs(tau - 0.5) < 1e-15


def test_next_event_time_no_crossing():
    """Institution with X=0.5, v=1.0: moving away from zero, tau=inf."""
    X = np.array([0.5])
    v = np.array([1.0])
    tau = next_event_time(X, v)
    assert tau == np.inf


def test_next_event_time_negative_no_recovery():
    """Institution with X=-0.5, v=-1.0: moving more negative, tau=inf."""
    X = np.array([-0.5])
    v = np.array([-1.0])
    tau = next_event_time(X, v)
    assert tau == np.inf


def test_next_event_time_recovery():
    """Institution with X=-0.5, v=1.0: crosses zero at tau=0.5."""
    X = np.array([-0.5])
    v = np.array([1.0])
    tau = next_event_time(X, v)
    assert abs(tau - 0.5) < 1e-15


def test_next_event_time_multiple_picks_minimum():
    """With two institutions, returns the smaller positive crossing time."""
    X = np.array([1.0, 2.0])
    v = np.array([-1.0, -1.0])
    tau = next_event_time(X, v)
    assert abs(tau - 1.0) < 1e-15


# ---------------------------------------------------------------------------
# Full simulation: event times
# ---------------------------------------------------------------------------

def test_event_times_match_expected():
    """Simulated event times must match [0.4, 1.0, 1.6, 3.0, 3.7] within 1e-8."""
    res = simulate_event_driven(X0, E, MU, N, T)
    sim_times = res["t_events"][:-1]  # drop T sentinel
    expected  = [0.4, 1.0, 1.6, 3.0, 3.7]
    assert len(sim_times) == len(expected), (
        f"Expected {len(expected)} events, got {len(sim_times)}: {sim_times}"
    )
    for s, e_t in zip(sim_times, expected):
        assert abs(s - e_t) < 1e-8, f"Event mismatch: sim={s}, expected={e_t}"


# ---------------------------------------------------------------------------
# Full simulation: H_4 terminal value
# ---------------------------------------------------------------------------

def test_H4_terminal_value():
    """H_4(5) (institution index 3) must equal 2.5 within 1e-10."""
    res = simulate_event_driven(X0, E, MU, N, T)
    H4 = res["H_events"][-1][3]
    assert abs(H4 - 2.5) < 1e-10, f"H_4(5) = {H4}, expected 2.5"


# ---------------------------------------------------------------------------
# Full simulation: X_4 path vs analytic formula
# ---------------------------------------------------------------------------

def test_X4_analytic_match():
    """Simulated X_4(t) must match analytic_X4(t) within 1e-10 on the dense grid."""
    res = simulate_event_driven(X0, E, MU, N, T)
    t_dense  = res["t_dense"]
    X4_sim   = res["X_dense"][3, :]
    X4_exact = analytic_X4(t_dense)
    max_err  = float(np.max(np.abs(X4_sim - X4_exact)))
    assert max_err < 1e-10, f"X_4 path max error = {max_err:.2e}"


def test_H4_analytic_match():
    """Simulated H_4(t) must match analytic_H4(t) within 1e-10 on the dense grid."""
    res = simulate_event_driven(X0, E, MU, N, T)
    t_dense  = res["t_dense"]
    H4_sim   = res["H_dense"][3, :]
    H4_exact = analytic_H4(t_dense)
    max_err  = float(np.max(np.abs(H4_sim - H4_exact)))
    assert max_err < 1e-10, f"H_4 path max error = {max_err:.2e}"


# ---------------------------------------------------------------------------
# Analytic formula correctness at key times
# ---------------------------------------------------------------------------

def test_analytic_X4_key_times():
    """Check analytic_X4 at the five event times where X_4 = 0, and at t=0."""
    # t=0: X_4 = 0.04
    assert abs(analytic_X4(np.array([0.0]))[0] - 0.04) < 1e-15
    # Crossing times: X_4 = 0 at t in {0.4, 1.6} (from above→below and below→above)
    for t_cross in [0.4, 1.6, 3.7]:
        val = float(analytic_X4(np.array([t_cross]))[0])
        assert abs(val) < 1e-14, f"X_4({t_cross}) = {val}, expected 0"
    # At t=5: X_4 = 0.04 + 0.5 - 0.2 - 0.6 = -0.26
    assert abs(analytic_X4(np.array([5.0]))[0] - (-0.26)) < 1e-14


def test_analytic_H4_key_times():
    """Check analytic_H4 at boundary times."""
    # H_4(0) = 0
    assert analytic_H4(np.array([0.0]))[0] == 0.0
    # H_4(0.4) = 0  (institution just entering distress at t=0.4)
    assert abs(analytic_H4(np.array([0.4]))[0]) < 1e-15
    # H_4(1.6) = 1.2  (total distress in [0.4, 1.6])
    assert abs(analytic_H4(np.array([1.6]))[0] - 1.2) < 1e-15
    # H_4(3.7) = 1.2  (no distress in [1.6, 3.7])
    assert abs(analytic_H4(np.array([3.7]))[0] - 1.2) < 1e-15
    # H_4(5) = 2.5
    assert abs(analytic_H4(np.array([5.0]))[0] - 2.5) < 1e-15


def test_dense_output_shapes():
    """Dense output arrays have the right shapes."""
    res = simulate_event_driven(X0, E, MU, N, T)
    assert res["t_dense"].shape == (1000,)
    assert res["X_dense"].shape == (N, 1000)
    assert res["H_dense"].shape == (N, 1000)
    assert res["h_dense"].shape == (1000,)


def test_h_dense_bounds():
    """h_dense = (1/N)*sum_i 1{X_i<=0} is always in [0, 1]."""
    res = simulate_event_driven(X0, E, MU, N, T)
    h = res["h_dense"]
    assert float(np.min(h)) >= 0.0 - 1e-12
    assert float(np.max(h)) <= 1.0 + 1e-12
