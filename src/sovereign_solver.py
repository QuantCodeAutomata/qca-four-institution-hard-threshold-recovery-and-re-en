"""Sovereign contagion ODE solvers.

Three variants are implemented:

1. **Full factor model** (K-dimensional state):
   dB_k/dt = (1/N) sum_j b_{jk} ell^-_eps(x0_j - lambda_E sum_l a_{jl} B_l)
   X_i(t) = x0_i - lambda_E sum_k a_{ik} B_k(t)

2. **Scenario-truncated model** (q-dimensional state, channels from ordered list):
   Same ODE retaining only the first q channels.

3. **SVD-truncated model** (N-dimensional direct state):
   dX_i/dt = -(1/N) sum_j E_q[i,j] ell^-_eps(X_j)
"""
import numpy as np
from scipy.integrate import solve_ivp

from sovereign_factors import compute_svd_reduction


def negative_ramp_loss(x, epsilon):
    """Smooth negative-ramp loss function.

    ell^-_epsilon(x) = max(0, min(1, -x / epsilon))

    Parameters
    ----------
    x : float or ndarray
    epsilon : float  (> 0)

    Returns
    -------
    float or ndarray of same shape as x, values in [0, 1].
    """
    return np.clip(-x / epsilon, 0.0, 1.0)


def _default_t_eval(T):
    return np.linspace(0.0, T, 601)


def solve_sovereign_full(
    a_matrix, b_matrix, lambda_E, x0,
    mu=0, r=0, T=3, epsilon=0.05,
    t_eval=None, rtol=2e-10, atol=2e-12,
):
    """Solve the full K-channel factor-structured contagion ODE.

    State vector: B (K,) cumulative feedback integrals.
    Bank buffers: X_i(t) = x0_i - lambda_E * (a_matrix @ B(t))[i].

    Parameters
    ----------
    a_matrix : ndarray, shape (N, K)   portfolio-share loadings
    b_matrix : ndarray, shape (N, K)   relative-exposure loadings
    lambda_E : float
    x0 : ndarray, shape (N,)           initial buffers
    mu, r : float                      individual drift / recovery (default 0)
    T : float                          terminal time
    epsilon : float                    ramp smoothing parameter
    t_eval : ndarray or None
    rtol, atol : float                 DOP853 tolerances

    Returns
    -------
    dict with keys 't', 'B_t', 'X_t', 'h_t', 'mean_distress_rate'.
    """
    N, K = a_matrix.shape
    if t_eval is None:
        t_eval = _default_t_eval(T)

    def rhs(t, B):
        X = x0 - lambda_E * (a_matrix @ B)
        ramp = negative_ramp_loss(X, epsilon)
        return (1.0 / N) * (b_matrix.T @ ramp)

    sol = solve_ivp(
        rhs, [0.0, T], np.zeros(K),
        method='DOP853', t_eval=t_eval,
        rtol=rtol, atol=atol, dense_output=False,
    )

    B_t = sol.y                                              # (K, T)
    X_t = x0[:, None] - lambda_E * (a_matrix @ B_t)         # (N, T)
    h_t = np.mean(negative_ramp_loss(X_t, epsilon), axis=0)  # (T,)

    return {
        't': sol.t,
        'B_t': B_t,
        'X_t': X_t,
        'h_t': h_t,
        'mean_distress_rate': float(np.mean(h_t)),
    }


def solve_sovereign_reduced_scenario(
    a_matrix, b_matrix, lambda_E, x0, channel_order, q,
    mu=0, r=0, T=3, epsilon=0.05, t_eval=None,
):
    """Scenario-truncated solver: retain the first q channels.

    Initial buffers x0 are kept unchanged from the full model; only the ODE
    feedback is restricted to the retained channels.

    Parameters
    ----------
    channel_order : list of int
        Channel indices ordered by importance (from :func:`scenario_channel_order`).
    q : int
        Number of channels to retain.

    Returns
    -------
    dict with same structure as :func:`solve_sovereign_full`.
    """
    N, K = a_matrix.shape
    if t_eval is None:
        t_eval = _default_t_eval(T)

    retained = channel_order[:q]
    a_q = a_matrix[:, retained]   # (N, q)
    b_q = b_matrix[:, retained]   # (N, q)

    def rhs(t, B_q):
        X = x0 - lambda_E * (a_q @ B_q)
        ramp = negative_ramp_loss(X, epsilon)
        return (1.0 / N) * (b_q.T @ ramp)

    sol = solve_ivp(
        rhs, [0.0, T], np.zeros(q),
        method='DOP853', t_eval=t_eval,
        rtol=1e-9, atol=1e-11, dense_output=False,
    )

    B_q_t = sol.y                                            # (q, T)
    X_t = x0[:, None] - lambda_E * (a_q @ B_q_t)            # (N, T)
    h_t = np.mean(negative_ramp_loss(X_t, epsilon), axis=0)

    # Reconstruct full-dimension B_t (zeros for non-retained channels)
    B_t = np.zeros((K, len(sol.t)))
    for pos, ch in enumerate(retained):
        B_t[ch] = B_q_t[pos]

    return {
        't': sol.t,
        'B_t': B_t,
        'X_t': X_t,
        'h_t': h_t,
        'mean_distress_rate': float(np.mean(h_t)),
    }


def solve_sovereign_reduced_svd(
    e_matrix, x0, q,
    mu=0, r=0, T=3, epsilon=0.05, t_eval=None,
):
    """SVD-truncated solver: use rank-q approximation of e_matrix.

    Evolves the N-dimensional state X directly:
      dX_i/dt = -(1/N) sum_j E_q[i,j] ell^-_eps(X_j)

    Parameters
    ----------
    e_matrix : ndarray, shape (N, N)
    x0 : ndarray, shape (N,)
    q : int

    Returns
    -------
    dict with same structure as :func:`solve_sovereign_full`
    (B_t is None — no natural B representation for the SVD model).
    """
    N = e_matrix.shape[0]
    if t_eval is None:
        t_eval = _default_t_eval(T)

    E_q = compute_svd_reduction(e_matrix, q)

    def rhs(t, X):
        ramp = negative_ramp_loss(X, epsilon)
        return -(1.0 / N) * (E_q @ ramp)

    sol = solve_ivp(
        rhs, [0.0, T], x0.copy(),
        method='DOP853', t_eval=t_eval,
        rtol=1e-9, atol=1e-11, dense_output=False,
    )

    X_t = sol.y                                              # (N, T)
    h_t = np.mean(negative_ramp_loss(X_t, epsilon), axis=0)

    return {
        't': sol.t,
        'B_t': None,
        'X_t': X_t,
        'h_t': h_t,
        'mean_distress_rate': float(np.mean(h_t)),
    }
