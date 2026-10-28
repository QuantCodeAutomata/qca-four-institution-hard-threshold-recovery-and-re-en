# Custom -- Context7 found no library equivalent for low-rank mean-field feedback ODE
"""
Limiting K-dimensional feedback ODE solver for Gaussian-mixture low-rank architectures.

Model (r=0): X_t(z) = x + mu*t - (1/K)*sum_k a_k * beta_k(t)
Feedback ODE: d/dt beta_k = sum_g p_g * b_{g,k} * Phi((-m_g - mu*t + (1/K)*a_g.dot(beta(t))) / sigma_g)
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
from typing import Any


def solve_limit_feedback(
    groups: list[dict[str, Any]],
    mu: float,
    r: float,
    T: float,
    K: int,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    n_eval: int = 1001,
) -> Any:
    """
    Solve the K-dimensional limiting feedback ODE for Gaussian-mixture architectures.

    Parameters
    ----------
    groups : list of dicts, each with keys 'p', 'm', 'sigma', 'a' (K-vector), 'b' (K-vector)
    mu     : drift parameter
    r      : imbalance parameter (0 in these experiments)
    T      : time horizon
    K      : rank (number of feedback channels)
    rtol, atol : ODE solver tolerances

    Returns
    -------
    scipy OdeSolution object with attributes t (n_eval,) and y (K x n_eval)
    """
    t_eval = np.linspace(0.0, T, n_eval)

    # Precompute group arrays
    p = np.array([g["p"] for g in groups])          # (G,)
    m = np.array([g["m"] for g in groups])          # (G,)
    sigma = np.array([g["sigma"] for g in groups])  # (G,)
    a_g = np.array([g["a"] for g in groups])        # (G, K)
    b_g = np.array([g["b"] for g in groups])        # (G, K)

    # Imbalance contribution Lambda_g = (r/K) * sum_k (a_{g,k}*bar_b_k - b_{g,k}*bar_a_k)
    bar_a = p @ a_g  # (K,)
    bar_b = p @ b_g  # (K,)
    Lambda_g = (r / K) * (a_g @ bar_b - b_g @ bar_a)  # (G,)

    def rhs(t: float, beta: np.ndarray) -> np.ndarray:
        # Argument to Phi for each group
        # arg_g = (-m_g - mu*t - t*Lambda_g + (1/K)*a_g.dot(beta)) / sigma_g
        inner = (1.0 / K) * (a_g @ beta)  # (G,)
        arg = (-m - mu * t - t * Lambda_g + inner) / sigma  # (G,)
        # d/dt beta_k = sum_g p_g * b_{g,k} * Phi(arg_g)
        Phi_g = norm.cdf(arg)  # (G,)
        dbeta = (p * Phi_g) @ b_g  # (K,) via einsum over groups
        return dbeta

    sol = solve_ivp(
        rhs,
        [0.0, T],
        np.zeros(K),
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    return sol


def compute_limiting_threshold_fraction(
    beta_t: np.ndarray,
    groups: list[dict[str, Any]],
    mu: float,
    r: float,
    t_eval: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Compute limiting aggregate threshold fraction h(t) = sum_g p_g * Phi(arg_g(t)).

    Parameters
    ----------
    beta_t : (K, n_t) feedback path
    groups : group dicts
    mu     : drift
    r      : imbalance
    t_eval : (n_t,) time points
    K      : rank

    Returns
    -------
    h_t : (n_t,) limiting distress fraction
    """
    p = np.array([g["p"] for g in groups])
    m = np.array([g["m"] for g in groups])
    sigma = np.array([g["sigma"] for g in groups])
    a_g = np.array([g["a"] for g in groups])
    b_g = np.array([g["b"] for g in groups])
    bar_a = p @ a_g
    bar_b = p @ b_g
    Lambda_g = (r / K) * (a_g @ bar_b - b_g @ bar_a)  # (G,)

    n_t = len(t_eval)
    h_t = np.zeros(n_t)
    for idx, t in enumerate(t_eval):
        inner = (1.0 / K) * (a_g @ beta_t[:, idx])  # (G,)
        arg = (-m - mu * t - t * Lambda_g + inner) / sigma
        h_t[idx] = float(p @ norm.cdf(arg))
    return h_t


def compute_groupwise_threshold_fraction(
    beta_t: np.ndarray,
    groups: list[dict[str, Any]],
    mu: float,
    r: float,
    t_eval: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Compute within-group threshold fractions h_g(t) = Phi(arg_g(t)).

    Returns
    -------
    h_g_t : (G, n_t) per-group distress fractions
    """
    p = np.array([g["p"] for g in groups])
    m = np.array([g["m"] for g in groups])
    sigma = np.array([g["sigma"] for g in groups])
    a_g = np.array([g["a"] for g in groups])
    b_g = np.array([g["b"] for g in groups])
    bar_a = p @ a_g
    bar_b = p @ b_g
    Lambda_g = (r / K) * (a_g @ bar_b - b_g @ bar_a)

    n_t = len(t_eval)
    G = len(groups)
    h_g_t = np.zeros((G, n_t))
    for idx, t in enumerate(t_eval):
        inner = (1.0 / K) * (a_g @ beta_t[:, idx])
        arg = (-m - mu * t - t * Lambda_g + inner) / sigma
        h_g_t[:, idx] = norm.cdf(arg)
    return h_g_t


# Four architecture families
ARCH1 = {
    "K": 1, "mu": -0.005, "r": 0, "T": 2.0,
    "groups": [{"p": 1.0, "m": 0.24, "sigma": 0.18, "a": [0.7], "b": [0.7]}],
}

ARCH2 = {
    "K": 2, "mu": -0.008, "r": 0, "T": 2.5,
    "groups": [
        {"p": 0.25, "m": 0.30, "sigma": 0.18, "a": [0.8, 0.2], "b": [0.8, 0.3]},
        {"p": 0.75, "m": 0.23, "sigma": 0.18, "a": [0.3, 0.7], "b": [0.2, 0.7]},
    ],
}

ARCH3 = {
    "K": 2, "mu": -0.008, "r": 0, "T": 2.3,
    "groups": [
        {"p": 0.40, "m": 0.22, "sigma": 0.16, "a": [0.9, 0.1], "b": [0.8, 0.1]},
        {"p": 0.35, "m": 0.25, "sigma": 0.18, "a": [0.1, 0.9], "b": [0.1, 0.8]},
        {"p": 0.25, "m": 0.20, "sigma": 0.16, "a": [0.6, 0.6], "b": [0.6, 0.6]},
    ],
}

ARCH4 = {
    "K": 2, "mu": -0.006, "r": 0, "T": 2.2,
    "groups": [
        {"p": 0.70, "m": 0.28, "sigma": 0.18, "a": [0.8, 0.25], "b": [0.7, 0.3]},
        {"p": 0.30, "m": 0.24, "sigma": 0.16, "a": [0.35, 0.90], "b": [0.25, 0.9]},
    ],
}

ARCHITECTURES = [ARCH1, ARCH2, ARCH3, ARCH4]
