"""Reduction comparison and channel-attribution utilities.

Provides tools to quantify the error introduced by scenario-based and
SVD-based channel truncations relative to the full K-channel model.
"""
import numpy as np


def compute_max_path_error(X_approx, X_full, t_eval=None):
    """Compute worst-case buffer-path error across all banks and times.

    max_i  sup_t  |X_i^approx(t) - X_i^full(t)|

    Parameters
    ----------
    X_approx : ndarray, shape (N, T)
    X_full   : ndarray, shape (N, T)
    t_eval   : ndarray, shape (T,) – time grid (not used in current calculation,
               kept for API compatibility).

    Returns
    -------
    float
    """
    return float(np.max(np.abs(X_approx - X_full)))


def compare_reductions(full_result, scenario_results, svd_results):
    """Compute max path errors for every reduction level.

    Parameters
    ----------
    full_result : dict
        Output of :func:`solve_sovereign_full`.
    scenario_results : dict
        Mapping  q → result dict from :func:`solve_sovereign_reduced_scenario`.
    svd_results : dict
        Mapping  q → result dict from :func:`solve_sovereign_reduced_svd`.

    Returns
    -------
    errors : dict
        Keys ``'scenario_q{q}'`` and ``'svd_q{q}'`` with scalar error values.
    """
    X_full = full_result['X_t']
    t_eval = full_result['t']

    errors = {}
    for q, res in sorted(scenario_results.items()):
        errors[f'scenario_q{q}'] = compute_max_path_error(res['X_t'], X_full, t_eval)
    for q, res in sorted(svd_results.items()):
        errors[f'svd_q{q}'] = compute_max_path_error(res['X_t'], X_full, t_eval)
    return errors


def compute_attribution(B_T, a_matrix, lambda_E):
    """Compute per-channel attribution shares of total buffer reduction.

    Total reduction attributed to channel k:
      C_k = lambda_E * B_T[k] * sum_i a_{ik}

    Attribution share:
      share_k = C_k / sum_l C_l

    Parameters
    ----------
    B_T : ndarray, shape (K,)
        Cumulative feedback integrals at terminal time T.
    a_matrix : ndarray, shape (N, K)
    lambda_E : float

    Returns
    -------
    shares : ndarray, shape (K,)
        Attribution shares (sum to 1).
    """
    contrib = lambda_E * a_matrix.sum(axis=0) * B_T   # (K,)
    total = contrib.sum()
    if total == 0.0:
        return np.zeros_like(contrib)
    return contrib / total
