"""Exp 1: Event-driven finite-network simulator for hard-threshold recovery dynamics."""

import numpy as np
from typing import Dict, List


def compute_slopes(
    X: np.ndarray, d: np.ndarray, e: np.ndarray, mu: float, N: int
) -> np.ndarray:
    """Compute piecewise-linear slopes for all institutions.

    v_i = mu - (1/N) * sum_j e_{ij} * d_j

    Parameters
    ----------
    X : (N,) array, current buffer levels (unused in formula but kept for interface)
    d : (N,) array, distress indicators (d_j = 1 if X_j <= 0)
    e : (N, N) array, exposure matrix
    mu : float, common drift
    N : int, number of institutions

    Returns
    -------
    v : (N,) array of slopes
    """
    return mu - (e @ d) / N


def next_event_time(X: np.ndarray, v: np.ndarray) -> float:
    """Compute the next time an institution crosses zero.

    For each i, the candidate crossing time is tau_i = -X_i / v_i when tau_i > 0
    (i.e. X_i and v_i have opposite signs, so the institution is heading toward zero).

    Parameters
    ----------
    X : (N,) array, current buffer levels
    v : (N,) array, current slopes

    Returns
    -------
    float : minimum positive crossing time, or np.inf if none within finite horizon
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.where(np.abs(v) > 0, -X / v, np.inf)
    tau = np.where(tau > 1e-14, tau, np.inf)
    return float(np.min(tau))


def simulate_event_driven(
    x0: np.ndarray,
    e: np.ndarray,
    mu: float,
    N: int,
    T: float,
) -> Dict:
    """Run event-driven simulation for the hard-threshold contagion model.

    On each interval the buffer evolves as X_i(t) = X_i(t_prev) + v_i*(t-t_prev)
    and the occupation as H_i(t) = H_i(t_prev) + d_i*(t-t_prev).

    Parameters
    ----------
    x0 : (N,) array, initial buffer levels
    e  : (N, N) array, exposure matrix (actual values, not e/N)
    mu : float, common drift
    N  : int, number of institutions
    T  : float, time horizon

    Returns
    -------
    dict with keys:
      't_events'  : list of event times (times where a crossing occurs)
      'X_events'  : list of (N,) arrays, buffer state at each event time
      'v_events'  : list of (N,) arrays, slopes on each interval [t_{k-1}, t_k]
      'H_events'  : list of (N,) arrays, cumulative occupation at each event time
      't_dense'   : (1000,) array of times on [0, T]
      'X_dense'   : (N, 1000) array of buffer paths
      'H_dense'   : (N, 1000) array of occupation paths
      'h_dense'   : (1000,) array, (1/N) * sum_i 1{X_i(t) <= 0}
    """
    X = np.array(x0, dtype=float)
    H = np.zeros(N, dtype=float)
    t = 0.0

    # Initial distress set: d_j = 1 if X_j <= 0
    d = (X <= 0.0).astype(float)

    # Storage for event-based output (include t=0 as first "event" for bookkeeping)
    seg_t_start: List[float] = []      # start time of each interval
    seg_X_start: List[np.ndarray] = [] # X at start of each interval
    seg_H_start: List[np.ndarray] = [] # H at start of each interval
    seg_v:       List[np.ndarray] = [] # slopes on each interval
    seg_d_start: List[np.ndarray] = [] # distress vector at start of each interval

    t_events: List[float] = []
    X_events: List[np.ndarray] = []
    H_events: List[np.ndarray] = []

    while True:
        v = compute_slopes(X, d, e, mu, N)
        tau = next_event_time(X, v)

        # Record current interval
        seg_t_start.append(t)
        seg_X_start.append(X.copy())
        seg_H_start.append(H.copy())
        seg_v.append(v.copy())
        seg_d_start.append(d.copy())

        # Stop if no crossing before T
        if tau == np.inf or t + tau > T - 1e-14:
            break

        # Advance to crossing time
        t_new = t + tau
        X_new = X + v * tau
        H_new = H + d * tau

        # Update distress: strict inequality first, then fix boundary (X=0) cases
        d_new = (X_new < -1e-14).astype(float)
        for i in range(N):
            if abs(X_new[i]) < 1e-12:
                # Institution i is at the boundary; determine direction
                d_new[i] = 1.0 if v[i] < 0.0 else 0.0

        t, X, H, d = t_new, X_new, H_new, d_new

        t_events.append(t)
        X_events.append(X.copy())
        H_events.append(H.copy())

    # Final interval to T
    v_final = compute_slopes(X, d, e, mu, N)
    seg_t_start.append(t)
    seg_X_start.append(X.copy())
    seg_H_start.append(H.copy())
    seg_v.append(v_final.copy())
    seg_d_start.append(d.copy())

    X_final = X + v_final * (T - t)
    H_final = H + d * (T - t)
    t_events.append(T)
    X_events.append(X_final.copy())
    H_events.append(H_final.copy())

    # Dense output on 1000 points in [0, T]
    t_dense = np.linspace(0.0, T, 1000)
    X_dense = np.zeros((N, 1000))
    H_dense = np.zeros((N, 1000))

    # seg_t_start now has one entry per segment; add T as the endpoint
    seg_ends = [seg_t_start[k + 1] for k in range(len(seg_t_start) - 1)] + [T]

    for k, (t0, t_end) in enumerate(zip(seg_t_start, seg_ends)):
        mask = (t_dense >= t0 - 1e-14) & (t_dense <= t_end + 1e-14)
        if not np.any(mask):
            continue
        dt_k = t_dense[mask] - t0
        X_dense[:, mask] = seg_X_start[k][:, None] + seg_v[k][:, None] * dt_k[None, :]
        H_dense[:, mask] = seg_H_start[k][:, None] + seg_d_start[k][:, None] * dt_k[None, :]

    h_dense = np.mean(X_dense <= 0.0, axis=0)

    # Build v_events: slopes on each interval between consecutive events
    # v_events[k] = slopes on interval (t_events[k-1], t_events[k]) for k>=1
    # and v_events[0] = slopes on (0, t_events[0])
    v_events = [sv.copy() for sv in seg_v]

    return {
        "t_events": t_events,
        "X_events": X_events,
        "v_events": v_events,
        "H_events": H_events,
        "t_dense": t_dense,
        "X_dense": X_dense,
        "H_dense": H_dense,
        "h_dense": h_dense,
    }


def analytic_X4(t: np.ndarray) -> np.ndarray:
    """Exact buffer path for institution 4 (1-indexed) = institution 3 (0-indexed).

    X_4(t) = 0.04 + 0.1*t - 0.2*min(t, 1) - 0.3*max(t-3, 0)

    Slope changes at t=1 (institution 0 recovers, removing its contribution to v_3)
    and t=3 (institution 2 enters distress, adding 0.3 to v_3's drag).
    """
    t = np.asarray(t, dtype=float)
    return 0.04 + 0.1 * t - 0.2 * np.minimum(t, 1.0) - 0.3 * np.maximum(t - 3.0, 0.0)


def analytic_H4(t: np.ndarray) -> np.ndarray:
    """Exact cumulative occupation for institution 4 (1-indexed).

    H_4(t) = min(max(t-0.4, 0), 1.2) + max(t-3.7, 0)

    Institution 3 is distressed on [0.4, 1.6] (duration 1.2) and [3.7, 5] (ongoing).
    """
    t = np.asarray(t, dtype=float)
    return np.minimum(np.maximum(t - 0.4, 0.0), 1.2) + np.maximum(t - 3.7, 0.0)
