# Custom -- Context7 found no library equivalent for this finite-N convergence study
"""
Monte Carlo finite-N convergence study comparing finite populations to limiting dynamics.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from low_rank_finite import (
    make_quantile_populations,
    simulate_finite_hard_grouped,
)
from low_rank_limit import (
    solve_limit_feedback,
    compute_limiting_threshold_fraction,
)
from metrics import uniform_threshold_fraction_error, feedback_path_error


def sample_population(
    groups: list[dict[str, Any]],
    N: int,
    K: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw an independently sampled finite population.

    Group labels drawn multinomially from p_g.
    Initial buffers drawn from group Gaussians.
    Loadings are group-constant.

    Returns
    -------
    X0, group_idx, a_matrix, b_matrix
    """
    p = np.array([g["p"] for g in groups])
    # Draw group labels
    group_labels = rng.choice(len(groups), size=N, p=p / p.sum())
    # Draw initial buffers
    X0 = np.array([
        groups[g]["m"] + groups[g]["sigma"] * rng.standard_normal()
        for g in group_labels
    ])
    a_matrix = np.array([groups[g]["a"] for g in group_labels], dtype=float)
    b_matrix = np.array([groups[g]["b"] for g in group_labels], dtype=float)
    return X0, group_labels, a_matrix, b_matrix


def compute_error_vs_limit(
    h_finite: np.ndarray,
    h_limit: np.ndarray,
    t_eval: np.ndarray,
) -> float:
    """Uniform threshold fraction error between finite and limiting paths."""
    return uniform_threshold_fraction_error(h_finite, h_limit, t_eval)


def run_deterministic_study(
    arch: dict[str, Any],
    N_list: list[int],
) -> dict[str, Any]:
    """
    Run deterministic quantile-matched convergence study.

    Parameters
    ----------
    arch   : architecture dict with 'K', 'mu', 'r', 'T', 'groups'
    N_list : list of population sizes to test

    Returns
    -------
    dict with keys 'N_list', 'h_errors', 'beta_errors', 'h_limit_terminal'
    """
    K = arch["K"]
    mu = arch["mu"]
    r = arch["r"]
    T = arch["T"]
    groups = arch["groups"]

    # Solve limiting dynamics
    sol = solve_limit_feedback(groups, mu, r, T, K)
    t_eval = sol.t
    beta_t = sol.y  # (K, n_t)
    h_limit = compute_limiting_threshold_fraction(beta_t, groups, mu, r, t_eval, K)
    h_limit_terminal = float(h_limit[-1])

    h_errors = []
    beta_errors = []

    for N in N_list:
        X0, _, a_mat, b_mat = make_quantile_populations(groups, N, K)
        X_dense, h_finite, beta_finite = simulate_finite_hard_grouped(
            X0, a_mat, b_mat, mu, K, N, T, t_eval
        )
        # Interpolate limiting to same grid (already on t_eval)
        h_err = uniform_threshold_fraction_error(h_finite, h_limit, t_eval)
        b_err = feedback_path_error(beta_finite, beta_t)
        h_errors.append(h_err)
        beta_errors.append(b_err)

    return {
        "N_list": N_list,
        "h_errors": h_errors,
        "beta_errors": beta_errors,
        "h_limit_terminal": h_limit_terminal,
    }


def run_monte_carlo(
    arch: dict[str, Any],
    N_list: list[int],
    n_replications: int,
    seed: int = 20261026,
) -> dict[str, Any]:
    """
    Monte Carlo independent-sampling convergence study.

    Parameters
    ----------
    arch            : architecture dict
    N_list          : list of population sizes
    n_replications  : number of independent samples per N
    seed            : NumPy PCG64 seed

    Returns
    -------
    dict with N_list, mean_h_errors, std_h_errors, ci95_h_errors
    """
    K = arch["K"]
    mu = arch["mu"]
    r = arch["r"]
    T = arch["T"]
    groups = arch["groups"]

    # Solve limiting dynamics once
    sol = solve_limit_feedback(groups, mu, r, T, K)
    t_eval = sol.t
    beta_t = sol.y
    h_limit = compute_limiting_threshold_fraction(beta_t, groups, mu, r, t_eval, K)

    rng = np.random.default_rng(seed)

    mean_h_errors = []
    std_h_errors = []
    ci95_h_errors = []

    for N in N_list:
        errors = []
        for _ in range(n_replications):
            X0, _, a_mat, b_mat = sample_population(groups, N, K, rng)
            _, h_finite, _ = simulate_finite_hard_grouped(
                X0, a_mat, b_mat, mu, K, N, T, t_eval
            )
            errors.append(compute_error_vs_limit(h_finite, h_limit, t_eval))
        errors = np.array(errors)
        mean_h_errors.append(float(errors.mean()))
        std_h_errors.append(float(errors.std()))
        se = errors.std() / np.sqrt(len(errors))
        ci95_h_errors.append((float(errors.mean() - 1.96 * se), float(errors.mean() + 1.96 * se)))

    return {
        "N_list": N_list,
        "mean_h_errors": mean_h_errors,
        "std_h_errors": std_h_errors,
        "ci95_h_errors": ci95_h_errors,
    }


def fit_convergence_slope(errors: list[float], N_list: list[int]) -> float:
    """
    Fit log-log slope of error vs N using OLS.

    Returns
    -------
    slope (expected near -0.5 for CLT convergence)
    """
    log_N = np.log(np.array(N_list, dtype=float))
    log_err = np.log(np.array(errors, dtype=float))
    A = np.column_stack([log_N, np.ones_like(log_N)])
    result = np.linalg.lstsq(A, log_err, rcond=None)
    return float(result[0][0])
