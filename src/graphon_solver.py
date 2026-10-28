"""ODE solver and error metrics for the graphon dynamics (Experiment 5).

Continuous-time distress dynamics on a 2000-node discretisation of [0,1]:

    dX_i/dt = mu - (1/N) * sum_j W[i,j] * ell_eps(X_j)

where ell_eps is the positive-side ramp loss.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def _ramp_vec(X: np.ndarray, epsilon: float) -> np.ndarray:
    """Vectorised ramp loss for an array X."""
    return np.where(X <= 0.0, 1.0, np.clip(1.0 - X / epsilon, 0.0, 1.0))


def solve_graphon_ramp(
    W_matrix: np.ndarray,
    x0: np.ndarray,
    mu: float,
    T: float,
    epsilon: float,
    t_eval: np.ndarray,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the N-dimensional graphon distress ODE with ramp smoothing.

    dX_i/dt = mu - (1/N) * sum_j W[i,j] * ell_eps(X_j)

    Parameters
    ----------
    W_matrix : (N, N) precomputed kernel matrix
    x0       : (N,) initial state
    mu       : drift parameter
    T        : terminal time
    epsilon  : ramp smoothing width
    t_eval   : 1-D array of output times in [0, T]
    rtol, atol, max_step : solver tolerances and step bound

    Returns
    -------
    t : (len(t_eval),) array of evaluation times
    X : (N, len(t_eval)) array of state trajectories
    """
    N = len(x0)
    W = W_matrix  # precomputed; do NOT copy inside the closure

    def rhs(t: float, X: np.ndarray) -> np.ndarray:
        ell = _ramp_vec(X, epsilon)
        return mu - W @ ell / N

    sol = solve_ivp(
        rhs,
        [0.0, T],
        x0.copy(),
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )
    return sol.t, sol.y  # sol.y has shape (N, len(t_eval))


def compute_threshold_fraction(
    X_t: np.ndarray,
    u_grid: np.ndarray,
) -> np.ndarray:
    """Fraction of nodes in distress (X_i <= 0) at each time.

    Parameters
    ----------
    X_t    : (N, T_steps) state trajectory
    u_grid : (N,) spatial grid (unused numerically, included for API clarity)

    Returns
    -------
    h : (T_steps,) array with h(t) = (1/N) * sum_i 1{X_i(t) <= 0}
    """
    return np.mean(X_t <= 0.0, axis=0)


def compute_path_L1_error(
    X_t_a: np.ndarray,
    X_t_b: np.ndarray,
    u_grid: np.ndarray,
) -> float:
    """Aligned-label L1 path error between two state trajectories.

    Returns sup_t (1/N) * sum_i |X_i^a(t) - X_i^b(t)|

    Parameters
    ----------
    X_t_a, X_t_b : (N, T_steps) state trajectories on the same t_eval grid
    u_grid       : (N,) spatial grid

    Returns
    -------
    err : scalar supremum over t of the spatial L1 error
    """
    l1_per_t = np.mean(np.abs(X_t_a - X_t_b), axis=0)  # (T_steps,)
    return float(np.max(l1_per_t))


def compute_threshold_fraction_error(
    h_a: np.ndarray,
    h_b: np.ndarray,
) -> float:
    """Supremum-norm error between two threshold-fraction paths.

    Returns sup_t |h_a(t) - h_b(t)|
    """
    return float(np.max(np.abs(h_a - h_b)))
