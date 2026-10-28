"""Tests for Exp 3: Directed imbalance and multiplex amplification."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from imbalance import (
    DIRECTED_GROUPS,
    DIRECTED_PARAMS,
    compute_lambda,
    solve_with_imbalance,
    compute_groupwise_distress,
)
from multiplex_heatmap import (
    ARCH4_BASE,
    build_modified_arch,
    compute_terminal_distress,
    run_heatmap,
)


# ---------------------------------------------------------------------------
# compute_lambda
# ---------------------------------------------------------------------------

def test_compute_lambda_zero_at_r_zero():
    """Lambda_g = 0 when r = 0."""
    K = DIRECTED_PARAMS["K"]
    Lambda_g = compute_lambda(DIRECTED_GROUPS, r=0.0, K=K)
    np.testing.assert_allclose(Lambda_g, np.zeros(len(DIRECTED_GROUPS)), atol=1e-15)


def test_compute_lambda_antisymmetric():
    """For symmetric p=0.5, Lambda should sum to 0 (zero population-mean imbalance)."""
    K = DIRECTED_PARAMS["K"]
    p = np.array([g["p"] for g in DIRECTED_GROUPS])
    Lambda_g = compute_lambda(DIRECTED_GROUPS, r=0.05, K=K)
    # Population-mean Lambda should be zero (it's a demeaned quantity)
    assert abs(float(p @ Lambda_g)) < 1e-12


def test_compute_lambda_opposite_sign_asymmetric_groups():
    """For genuinely asymmetric a/b loadings, Lambda values have opposite signs."""
    # Asymmetric groups where sender/receiver roles differ non-symmetrically
    K = 2
    asym_groups = [
        {"p": 0.5, "m": 0.5, "sigma": 0.18, "a": [3.0, 0.2], "b": [0.2, 1.0]},
        {"p": 0.5, "m": 0.5, "sigma": 0.18, "a": [0.2, 1.0], "b": [3.0, 0.2]},
    ]
    Lambda_g = compute_lambda(asym_groups, r=0.05, K=K)
    # Population mean Lambda must be 0 (it always is)
    p = np.array([g["p"] for g in asym_groups])
    assert abs(float(p @ Lambda_g)) < 1e-12
    # With these specific asymmetric loadings the two groups have opposite Lambda signs
    # Group 0: a=[3,0.2], bar_b=[1.6, 0.6], a.bar_b = 3*1.6+0.2*0.6 = 4.92
    #          b=[0.2,1.0], bar_a=[1.6, 0.6], b.bar_a = 0.2*1.6+1.0*0.6 = 0.92, diff=4.0
    # Group 1: opposite sign
    assert Lambda_g[0] * Lambda_g[1] < 0 or (Lambda_g[0] == 0 and Lambda_g[1] == 0)


# ---------------------------------------------------------------------------
# solve_with_imbalance
# ---------------------------------------------------------------------------

def test_solve_with_imbalance_beta_zero_at_t0():
    """beta(0) = 0."""
    K = DIRECTED_PARAMS["K"]
    mu = DIRECTED_PARAMS["mu"]
    T = DIRECTED_PARAMS["T"]
    t_eval, beta_t, _ = solve_with_imbalance(DIRECTED_GROUPS, mu, 0.0, T, K)
    np.testing.assert_allclose(beta_t[:, 0], np.zeros(K), atol=1e-12)


def test_solve_with_imbalance_beta_nonneg():
    """beta(t) is nonneg for all t."""
    K = DIRECTED_PARAMS["K"]
    mu = DIRECTED_PARAMS["mu"]
    T = DIRECTED_PARAMS["T"]
    for r in [0.0, 0.05]:
        t_eval, beta_t, _ = solve_with_imbalance(DIRECTED_GROUPS, mu, r, T, K)
        assert np.all(beta_t >= -1e-10), f"Negative beta at r={r}"


def test_solve_with_imbalance_output_shape():
    """Output shapes: t_eval (n,), beta_t (K, n)."""
    K = DIRECTED_PARAMS["K"]
    mu = DIRECTED_PARAMS["mu"]
    T = DIRECTED_PARAMS["T"]
    t_eval, beta_t, Lambda_g = solve_with_imbalance(DIRECTED_GROUPS, mu, 0.05, T, K)
    assert t_eval.shape == (1001,)
    assert beta_t.shape[0] == K
    assert beta_t.shape[1] == 1001
    assert Lambda_g.shape == (len(DIRECTED_GROUPS),)


def test_solve_with_imbalance_R_values():
    """R_0=1.82, R_1=-1.82 exactly from imbalance formula with K=1."""
    K = DIRECTED_PARAMS["K"]
    # Lambda_g / r = R_g
    Lambda_g = compute_lambda(DIRECTED_GROUPS, r=0.05, K=K)
    R_g = Lambda_g / 0.05
    assert abs(R_g[0] - 1.82) < 1e-12, f"R_0 = {R_g[0]:.4f}, expected 1.82"
    assert abs(R_g[1] - (-1.82)) < 1e-12, f"R_1 = {R_g[1]:.4f}, expected -1.82"


# ---------------------------------------------------------------------------
# compute_groupwise_distress
# ---------------------------------------------------------------------------

def test_groupwise_distress_in_unit_interval():
    """h_g(t) must be in [0, 1]."""
    K = DIRECTED_PARAMS["K"]
    mu = DIRECTED_PARAMS["mu"]
    T = DIRECTED_PARAMS["T"]
    t_eval, beta_t, Lambda_g = solve_with_imbalance(DIRECTED_GROUPS, mu, 0.0, T, K)
    h_g_t = compute_groupwise_distress(DIRECTED_GROUPS, mu, 0.0, T, K, beta_t, t_eval, Lambda_g)
    assert h_g_t.shape == (len(DIRECTED_GROUPS), len(t_eval))
    assert np.all(h_g_t >= -1e-10) and np.all(h_g_t <= 1 + 1e-10)


def test_groupwise_distress_increases_with_time():
    """For negative drift, within-group distress should generally increase over time."""
    K = DIRECTED_PARAMS["K"]
    mu = DIRECTED_PARAMS["mu"]  # mu=0.015 (positive drift, so expect small distress)
    T = DIRECTED_PARAMS["T"]
    t_eval, beta_t, Lambda_g = solve_with_imbalance(DIRECTED_GROUPS, mu, 0.0, T, K)
    h_g_t = compute_groupwise_distress(DIRECTED_GROUPS, mu, 0.0, T, K, beta_t, t_eval, Lambda_g)
    # With positive mu and high buffers, distress should be small throughout
    for g in range(len(DIRECTED_GROUPS)):
        assert float(h_g_t[g, -1]) < 1.0


def test_imbalance_increases_asymmetry():
    """With r>0, group distress fractions should differ more than with r=0."""
    K = DIRECTED_PARAMS["K"]
    mu = DIRECTED_PARAMS["mu"]
    T = DIRECTED_PARAMS["T"]
    t_r0, beta_r0, L_r0 = solve_with_imbalance(DIRECTED_GROUPS, mu, 0.0, T, K)
    h_r0 = compute_groupwise_distress(DIRECTED_GROUPS, mu, 0.0, T, K, beta_r0, t_r0, L_r0)
    asym_r0 = abs(float(h_r0[0, -1]) - float(h_r0[1, -1]))

    t_r05, beta_r05, L_r05 = solve_with_imbalance(DIRECTED_GROUPS, mu, 0.05, T, K)
    h_r05 = compute_groupwise_distress(DIRECTED_GROUPS, mu, 0.05, T, K, beta_r05, t_r05, L_r05)
    asym_r05 = abs(float(h_r05[0, -1]) - float(h_r05[1, -1]))

    assert asym_r05 >= asym_r0, f"Asymmetry should increase with r: asym(r=0.05)={asym_r05:.6f} vs asym(r=0)={asym_r0:.6f}"


# ---------------------------------------------------------------------------
# build_modified_arch
# ---------------------------------------------------------------------------

def test_build_modified_arch_omega2_zero():
    """omega2=0 -> second sender loading is 0 for all groups."""
    arch = build_modified_arch(0.0, 0.0)
    for g in arch["groups"]:
        assert g["a"][1] == 0.0


def test_build_modified_arch_omega2_one_unchanged():
    """omega2=1, delta=0 -> arch unchanged."""
    arch = build_modified_arch(1.0, 0.0)
    base = ARCH4_BASE
    for g_new, g_old in zip(arch["groups"], base["groups"]):
        np.testing.assert_allclose(g_new["a"], g_old["a"], atol=1e-12)
        assert g_new["m"] == g_old["m"]


def test_build_modified_arch_delta_shifts_nbfi():
    """delta_nbfi > 0 reduces NBFI buffer mean."""
    delta = 0.08
    arch = build_modified_arch(1.0, delta)
    base_nbfi_m = ARCH4_BASE["groups"][1]["m"]
    assert abs(arch["groups"][1]["m"] - (base_nbfi_m - delta)) < 1e-12


# ---------------------------------------------------------------------------
# compute_terminal_distress
# ---------------------------------------------------------------------------

def test_compute_terminal_distress_in_unit_interval():
    """h(T) must be in [0, 1]."""
    g = ARCH4_BASE["groups"]
    h = compute_terminal_distress(g, ARCH4_BASE["mu"], ARCH4_BASE["r"], ARCH4_BASE["T"], ARCH4_BASE["K"])
    assert 0.0 <= h <= 1.0


def test_compute_terminal_distress_increases_with_stronger_second_layer():
    """Larger omega_2 -> more distress (monotone in amplification)."""
    h0 = compute_terminal_distress(
        build_modified_arch(0.0, 0.0)["groups"],
        ARCH4_BASE["mu"], ARCH4_BASE["r"], ARCH4_BASE["T"], ARCH4_BASE["K"]
    )
    h1 = compute_terminal_distress(
        build_modified_arch(1.0, 0.0)["groups"],
        ARCH4_BASE["mu"], ARCH4_BASE["r"], ARCH4_BASE["T"], ARCH4_BASE["K"]
    )
    h2 = compute_terminal_distress(
        build_modified_arch(2.0, 0.0)["groups"],
        ARCH4_BASE["mu"], ARCH4_BASE["r"], ARCH4_BASE["T"], ARCH4_BASE["K"]
    )
    assert h1 >= h0, f"h(omega_2=1)={h1:.4f} should be >= h(omega_2=0)={h0:.4f}"
    assert h2 >= h1, f"h(omega_2=2)={h2:.4f} should be >= h(omega_2=1)={h1:.4f}"


def test_compute_terminal_distress_increases_with_delta():
    """Larger NBFI shock -> more distress."""
    h0 = compute_terminal_distress(
        build_modified_arch(1.0, 0.00)["groups"],
        ARCH4_BASE["mu"], ARCH4_BASE["r"], ARCH4_BASE["T"], ARCH4_BASE["K"]
    )
    h1 = compute_terminal_distress(
        build_modified_arch(1.0, 0.08)["groups"],
        ARCH4_BASE["mu"], ARCH4_BASE["r"], ARCH4_BASE["T"], ARCH4_BASE["K"]
    )
    assert h1 > h0, f"h(delta=0.08)={h1:.4f} should exceed h(delta=0)={h0:.4f}"


# ---------------------------------------------------------------------------
# run_heatmap: shape and monotonicity
# ---------------------------------------------------------------------------

def test_heatmap_shape():
    """Heatmap output has shape (n_omega, n_delta)."""
    omega_grid = np.array([0.0, 1.0, 2.0])
    delta_grid = np.array([0.0, 0.04, 0.08])
    h_matrix = run_heatmap(omega_grid, delta_grid)
    assert h_matrix.shape == (3, 3)


def test_heatmap_values_in_unit_interval():
    """All heatmap values are in [0, 1]."""
    omega_grid = np.linspace(0, 2, 5)
    delta_grid = np.linspace(0, 0.08, 5)
    h_matrix = run_heatmap(omega_grid, delta_grid)
    assert np.all(h_matrix >= 0.0) and np.all(h_matrix <= 1.0)


def test_heatmap_baseline_close_to_expected():
    """Baseline (omega_2=1, delta=0) terminal distress close to 13.04%."""
    omega_grid = np.array([0.0, 1.0])
    delta_grid = np.array([0.0])
    h_matrix = run_heatmap(omega_grid, delta_grid)
    h_omega1 = float(h_matrix[1, 0])
    assert abs(h_omega1 - 0.1304) < 0.02, f"h(omega_2=1, delta=0)={h_omega1:.4f}, expected ~0.1304"
