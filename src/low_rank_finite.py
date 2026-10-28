# Custom -- Context7 found no library equivalent for finite grouped hard-threshold contagion
"""
Finite grouped hard-threshold dynamics for deterministic quantile-matched populations.

Grouped event-driven solver exploiting equal loadings within groups.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import Any


def largest_remainder_round(proportions: np.ndarray, N: int) -> np.ndarray:
    """
    Assign group counts by largest-remainder rounding of N * p_g.

    Parameters
    ----------
    proportions : (G,) group proportions summing to 1
    N           : total population size

    Returns
    -------
    counts : (G,) integer counts summing to N
    """
    exact = N * proportions
    floors = np.floor(exact).astype(int)
    remainders = exact - floors
    shortfall = N - floors.sum()
    # assign remaining slots to groups with largest remainders
    order = np.argsort(-remainders)
    floors[order[:shortfall]] += 1
    assert floors.sum() == N
    return floors


def make_quantile_populations(
    groups: list[dict[str, Any]], N: int, K: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create deterministic quantile-matched populations.

    Within group g with count n_g, assign buffers:
        x_{g,j} = m_g + sigma_g * Phi^{-1}((j - 0.5) / n_g),  j = 1, ..., n_g

    Parameters
    ----------
    groups : list of group dicts with 'p', 'm', 'sigma', 'a', 'b'
    N      : total population size
    K      : rank

    Returns
    -------
    X0        : (N,) initial buffer values
    group_idx : (N,) group assignment indices in 0..G-1
    a_matrix  : (N, K) sender loadings
    b_matrix  : (N, K) receiver loadings
    """
    props = np.array([g["p"] for g in groups])
    counts = largest_remainder_round(props, N)

    X0_list, gidx_list, a_list, b_list = [], [], [], []
    for g_idx, (g, n_g) in enumerate(zip(groups, counts)):
        if n_g == 0:
            continue
        quantiles = (np.arange(1, n_g + 1) - 0.5) / n_g
        x_g = g["m"] + g["sigma"] * norm.ppf(quantiles)
        X0_list.append(x_g)
        gidx_list.append(np.full(n_g, g_idx, dtype=int))
        a_list.append(np.tile(np.array(g["a"], dtype=float), (n_g, 1)))
        b_list.append(np.tile(np.array(g["b"], dtype=float), (n_g, 1)))

    X0 = np.concatenate(X0_list)
    group_idx = np.concatenate(gidx_list)
    a_matrix = np.vstack(a_list)
    b_matrix = np.vstack(b_list)
    return X0, group_idx, a_matrix, b_matrix


def compute_finite_slopes(
    X: np.ndarray,
    distress: np.ndarray,
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    mu: float,
    K: int,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute slopes for the finite hard-threshold dynamics.

    hat_beta_k = (1/N) * sum_j b_{j,k} * distress_j
    slope_i    = mu - (1/K) * a_i . hat_beta

    Parameters
    ----------
    X        : (N,) current state (unused, for interface consistency)
    distress : (N,) binary distress indicators
    a_matrix : (N, K)
    b_matrix : (N, K)
    mu       : drift
    K        : rank
    N        : population size

    Returns
    -------
    slopes   : (N,) per-institution slopes
    hat_beta : (K,) empirical feedback
    """
    hat_beta = (b_matrix.T @ distress) / N  # (K,)
    slopes = mu - (a_matrix @ hat_beta) / K  # (N,)
    return slopes, hat_beta


def simulate_finite_hard_grouped(
    X0: np.ndarray,
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    mu: float,
    K: int,
    N: int,
    T: float,
    t_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Event-driven finite hard-threshold simulation (grouped populations).

    Hard threshold: X_i <= 0 means distressed.
    On each interval between events, slopes are constant.
    Slopes are strictly negative in monotone architectures.

    Parameters
    ----------
    X0       : (N,) initial buffers
    a_matrix : (N, K)
    b_matrix : (N, K)
    mu       : drift
    K        : rank
    N        : population size
    T        : time horizon
    t_eval   : (M,) output times

    Returns
    -------
    X_dense    : (N, M) state paths
    h_dense    : (M,) aggregate threshold fraction
    beta_dense : (K, M) feedback paths
    """
    X = X0.copy()
    distress = (X <= 0).astype(float)

    t_cur = 0.0
    slopes, hat_beta = compute_finite_slopes(X, distress, a_matrix, b_matrix, mu, K, N)

    n_eval = len(t_eval)
    X_dense = np.zeros((N, n_eval))
    h_dense = np.zeros(n_eval)
    beta_dense = np.zeros((K, n_eval))

    # Cumulative beta at current t_cur: integral of hat_beta over intervals
    # We'll track cumulative beta separately
    beta_cum = np.zeros(K)  # cumulative int_0^t hat_beta(s) ds * (not used for X directly)
    # Actually X_i(t) = X0_i + mu*t - (1/K)*a_i.dot(beta(t)) is the limiting form;
    # for finite dynamics, X_i(t) = X_i(t_prev) + slope_i * (t - t_prev) on each interval.
    # The feedback beta^N(t) = cumulative sum_g b_{g,k} * distress_count_g(s)/N ds

    # We track X linearly between events
    eval_ptr = 0

    # Maximum event loop (avoid infinite loops)
    max_events = 10 * N + 100

    for _ in range(max_events):
        # Fill output for all t_eval in [t_cur, next_event_time)
        # Compute candidate crossing times
        crossings = []
        for i in range(N):
            if slopes[i] < 0 and X[i] > 0:
                tau = -X[i] / slopes[i]
                if tau > 1e-14:
                    crossings.append(tau)
            elif slopes[i] > 0 and X[i] < 0:
                tau = -X[i] / slopes[i]
                if tau > 1e-14:
                    crossings.append(tau)
            elif slopes[i] > 0 and X[i] == 0:
                # at boundary, recovering
                pass

        if crossings:
            dt_event = min(crossings)
        else:
            dt_event = T - t_cur + 1.0  # no more events, run to T

        t_next = min(t_cur + dt_event, T)

        # Fill outputs in [t_cur, t_next]
        while eval_ptr < n_eval and t_eval[eval_ptr] <= t_next + 1e-14:
            dt = t_eval[eval_ptr] - t_cur
            dt = max(dt, 0.0)
            X_t = X + slopes * dt
            X_dense[:, eval_ptr] = X_t
            h_dense[eval_ptr] = np.mean(X_t <= 0)
            beta_dense[:, eval_ptr] = beta_cum + hat_beta * dt
            eval_ptr += 1

        if t_next >= T - 1e-14:
            break

        # Advance to event
        dt_advance = t_next - t_cur
        X = X + slopes * dt_advance
        beta_cum = beta_cum + hat_beta * dt_advance
        t_cur = t_next

        # Update distress set at event time (closed convention: X<=0)
        distress = (X <= 0).astype(float)
        slopes, hat_beta = compute_finite_slopes(X, distress, a_matrix, b_matrix, mu, K, N)

    # Fill any remaining output points at T
    while eval_ptr < n_eval:
        dt = T - t_cur
        X_t = X + slopes * dt
        X_dense[:, eval_ptr] = X_t
        h_dense[eval_ptr] = np.mean(X_t <= 0)
        beta_dense[:, eval_ptr] = beta_cum + hat_beta * dt
        eval_ptr += 1

    return X_dense, h_dense, beta_dense
