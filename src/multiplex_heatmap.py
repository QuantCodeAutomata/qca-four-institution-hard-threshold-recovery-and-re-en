# Custom -- Context7 found no library equivalent for multiplex amplification heatmap
"""
Multiplex amplification experiment (Exp 3, Part B).

Bank-NBFI architecture with omega_2 scaling of second receiver loading
and Delta_NBFI shift of NBFI initial buffers.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
from typing import Any


# Baseline bank-NBFI architecture (ARCH4 from exp2)
ARCH4_BASE = {
    "K": 2,
    "mu": -0.006,
    "r": 0,
    "T": 2.2,
    "groups": [
        {"p": 0.70, "m": 0.28, "sigma": 0.18, "a": [0.8, 0.25], "b": [0.7, 0.3]},
        {"p": 0.30, "m": 0.24, "sigma": 0.16, "a": [0.35, 0.90], "b": [0.25, 0.9]},
    ],
}


def build_modified_arch(
    omega2: float,
    delta_nbfi: float,
    base_arch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build modified architecture for multiplex amplification study.

    Modifications:
    - Replace a_{g,2} by omega_2 * a_{g,2} for all groups
    - Replace NBFI mean buffer 0.24 by 0.24 - delta_nbfi (NBFI is group index 1)

    Parameters
    ----------
    omega2      : scaling factor for second sender loading
    delta_nbfi  : buffer shock for NBFIs (NBFI mean = base_mean - delta_nbfi)
    base_arch   : base architecture (defaults to ARCH4_BASE)

    Returns
    -------
    Modified arch dict with same structure
    """
    if base_arch is None:
        base_arch = ARCH4_BASE

    import copy
    arch = copy.deepcopy(base_arch)
    for g_idx, g in enumerate(arch["groups"]):
        a = list(g["a"])
        a[1] = omega2 * a[1]  # scale second sender loading
        g["a"] = a
        # NBFI is group index 1 (p=0.30 in baseline)
        if g_idx == 1:
            g["m"] = g["m"] - delta_nbfi
    return arch


def compute_terminal_distress(
    groups: list[dict[str, Any]],
    mu: float,
    r: float,
    T: float,
    K: int,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> float:
    """
    Solve limiting feedback ODE and return terminal threshold fraction h(T).

    Parameters
    ----------
    groups : list of group dicts
    mu     : drift
    r      : imbalance
    T      : horizon
    K      : rank

    Returns
    -------
    h_T : terminal aggregate distress fraction
    """
    p = np.array([g["p"] for g in groups])
    m = np.array([g["m"] for g in groups])
    sigma = np.array([g["sigma"] for g in groups])
    a_g = np.array([g["a"] for g in groups], dtype=float)
    b_g = np.array([g["b"] for g in groups], dtype=float)

    # Imbalance contributions (r=0 in these experiments, so Lambda=0)
    bar_a = p @ a_g
    bar_b = p @ b_g
    Lambda_g = (r / K) * (a_g @ bar_b - b_g @ bar_a)

    def rhs(t: float, beta: np.ndarray) -> np.ndarray:
        inner = (1.0 / K) * (a_g @ beta)
        arg = (-m - mu * t - t * Lambda_g + inner) / sigma
        return (p * norm.cdf(arg)) @ b_g

    sol = solve_ivp(
        rhs,
        [0.0, T],
        np.zeros(K),
        method="DOP853",
        t_eval=[T],
        rtol=rtol,
        atol=atol,
    )
    beta_T = sol.y[:, -1]
    inner = (1.0 / K) * (a_g @ beta_T)
    arg = (-m - mu * T - T * Lambda_g + inner) / sigma
    h_T = float(p @ norm.cdf(arg))
    return h_T


def run_heatmap(
    omega2_grid: np.ndarray,
    delta_grid: np.ndarray,
    mu: float = -0.006,
    r: float = 0.0,
    T: float = 2.2,
    K: int = 2,
    base_arch: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Compute terminal distress fraction on a 2D parameter grid.

    Parameters
    ----------
    omega2_grid : (n_omega,) values of omega_2
    delta_grid  : (n_delta,) values of Delta_NBFI
    mu, r, T, K : model parameters

    Returns
    -------
    h_matrix : (n_omega, n_delta) terminal h(T) values
    """
    n_omega = len(omega2_grid)
    n_delta = len(delta_grid)
    h_matrix = np.zeros((n_omega, n_delta))

    for i, omega2 in enumerate(omega2_grid):
        for j, delta in enumerate(delta_grid):
            arch = build_modified_arch(omega2, delta, base_arch)
            h_matrix[i, j] = compute_terminal_distress(
                arch["groups"], mu, r, T, K
            )

    return h_matrix
