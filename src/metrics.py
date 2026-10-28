# Custom -- Context7 found no library equivalent for these specialized contagion metrics
"""Error and convergence metrics for finite vs limiting dynamics comparison."""
from __future__ import annotations

import numpy as np


def uniform_threshold_fraction_error(
    h_finite: np.ndarray, h_limit: np.ndarray, t_eval: np.ndarray
) -> float:
    """
    Uniform (sup-norm) threshold fraction error.

    Parameters
    ----------
    h_finite : (n_t,) finite population threshold fraction
    h_limit  : (n_t,) limiting threshold fraction (interpolated to same grid)
    t_eval   : (n_t,) time grid (unused but kept for API consistency)

    Returns
    -------
    error : sup_t |h_finite(t) - h_limit(t)|
    """
    return float(np.max(np.abs(h_finite - h_limit)))


def feedback_path_error(
    beta_finite: np.ndarray, beta_limit: np.ndarray
) -> float:
    """
    Sup-norm feedback path error in L1 over channels.

    Parameters
    ----------
    beta_finite : (K, n_t) finite feedback path
    beta_limit  : (K, n_t) limiting feedback path

    Returns
    -------
    error : max_t (1/K) * sum_k |beta_finite_k(t) - beta_limit_k(t)|
    """
    K = beta_finite.shape[0]
    diff = np.abs(beta_finite - beta_limit)  # (K, n_t)
    return float(np.max(np.mean(diff, axis=0)))


def wasserstein_1d(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """
    1D Wasserstein-1 (earth mover) distance between two sample sets.

    Parameters
    ----------
    samples_a, samples_b : 1D arrays of samples

    Returns
    -------
    W1 distance
    """
    a_sorted = np.sort(samples_a)
    b_sorted = np.sort(samples_b)
    # Pad shorter array if needed
    n = max(len(a_sorted), len(b_sorted))
    if len(a_sorted) != len(b_sorted):
        q = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
        a_quant = np.quantile(samples_a, q)
        b_quant = np.quantile(samples_b, q)
        return float(np.mean(np.abs(a_quant - b_quant)))
    return float(np.mean(np.abs(a_sorted - b_sorted)))
