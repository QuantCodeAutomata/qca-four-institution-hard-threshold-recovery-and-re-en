# Custom -- Context7 found no library equivalent for directed imbalance contagion
"""
Directed imbalance experiment (Exp 3, Part A).

Two-group low-rank system with r != 0 showing asymmetry between net lenders
and net borrowers.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
from typing import Any


def compute_lambda(
    groups: list[dict[str, Any]],
    r: float,
    K: int,
) -> np.ndarray:
    """
    Compute per-group imbalance contribution.

    Lambda_g = (r/K) * sum_k (a_{g,k} * bar_b_k - b_{g,k} * bar_a_k)
    where bar_a_k = sum_g p_g * a_{g,k} and bar_b_k = sum_g p_g * b_{g,k}.

    Parameters
    ----------
    groups : list of group dicts with 'p', 'a', 'b'
    r      : imbalance parameter
    K      : rank

    Returns
    -------
    Lambda_g : (G,) per-group imbalance contributions
    """
    p = np.array([g["p"] for g in groups])
    a_g = np.array([g["a"] for g in groups], dtype=float)  # (G, K)
    b_g = np.array([g["b"] for g in groups], dtype=float)  # (G, K)
    bar_a = p @ a_g  # (K,)
    bar_b = p @ b_g  # (K,)
    Lambda_g = (r / K) * (a_g @ bar_b - b_g @ bar_a)  # (G,)
    return Lambda_g


def solve_with_imbalance(
    groups: list[dict[str, Any]],
    mu: float,
    r: float,
    T: float,
    K: int,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    n_eval: int = 1001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve limiting feedback ODE with imbalance (r != 0).

    Modified state: X_t(g) = m_g + mu*t + t*Lambda_g - (1/K)*a_g.dot(beta(t))
    ODE: d/dt beta_k = sum_g p_g * b_{g,k} * Phi(arg_g(t))
    where arg_g(t) = (-m_g - mu*t - t*Lambda_g + (1/K)*a_g.dot(beta(t))) / sigma_g

    Returns
    -------
    t_eval : (n_eval,)
    beta_t : (K, n_eval)
    Lambda_g : (G,) per-group imbalance terms
    """
    p = np.array([g["p"] for g in groups])
    m = np.array([g["m"] for g in groups])
    sigma = np.array([g["sigma"] for g in groups])
    a_g = np.array([g["a"] for g in groups], dtype=float)
    b_g = np.array([g["b"] for g in groups], dtype=float)
    Lambda_g = compute_lambda(groups, r, K)

    def rhs(t: float, beta: np.ndarray) -> np.ndarray:
        inner = (1.0 / K) * (a_g @ beta)  # (G,)
        arg = (-m - mu * t - t * Lambda_g + inner) / sigma
        Phi_g = norm.cdf(arg)
        return (p * Phi_g) @ b_g  # (K,)

    t_eval = np.linspace(0.0, T, n_eval)
    sol = solve_ivp(
        rhs,
        [0.0, T],
        np.zeros(K),
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    return sol.t, sol.y, Lambda_g


def compute_groupwise_distress(
    groups: list[dict[str, Any]],
    mu: float,
    r: float,
    T: float,
    K: int,
    beta_t: np.ndarray,
    t_eval: np.ndarray,
    Lambda_g: np.ndarray,
) -> np.ndarray:
    """
    Compute within-group distress fractions h_g(t).

    h_g(t) = Phi((-m_g - mu*t - t*Lambda_g + (1/K)*a_g.dot(beta(t))) / sigma_g)

    Returns
    -------
    h_g_t : (G, n_t)
    """
    m = np.array([g["m"] for g in groups])
    sigma = np.array([g["sigma"] for g in groups])
    a_g = np.array([g["a"] for g in groups], dtype=float)

    G, n_t = len(groups), len(t_eval)
    h_g_t = np.zeros((G, n_t))
    for idx, t in enumerate(t_eval):
        inner = (1.0 / K) * (a_g @ beta_t[:, idx])
        arg = (-m - mu * t - t * Lambda_g + inner) / sigma
        h_g_t[:, idx] = norm.cdf(arg)
    return h_g_t


# Two-group directed imbalance architecture (Part A)
# K=1 scalar loading: a=2.0, b=0.6 for lender; a=0.6, b=2.0 for borrower.
# This yields R_0 = (1/K)*(a_0*bar_b - b_0*bar_a) = 1*1.3*1.4 = 1.82 ✓
# and R_1 = -1.82 ✓, consistent with the reported imbalance terms.
DIRECTED_GROUPS = [
    {"p": 0.5, "m": 0.50, "sigma": 0.18, "a": [2.0], "b": [0.6]},
    {"p": 0.5, "m": 0.45, "sigma": 0.18, "a": [0.6], "b": [2.0]},
]
DIRECTED_PARAMS = {"mu": 0.015, "T": 2.5, "K": 1}
