"""Positive-factor truncation study for Experiment 6.

Exploits the rank-K structure of W(u,v) = sum_{m=1}^K weight_m * a_m(u) * b_m(v)
to reduce ODE cost from O(N^2) to O(N*K) per function evaluation.

At each time step the interaction term is:
    sum_{m=1}^K weight_m * a_m(u_i) * (1/N) * sum_j b_m(u_j) * ell_eps(X_j)
  = A_matrix @ (weights * q(t))

where q_m(t) = (1/N) * B_matrix[:, m] . ell_eps(X)  is a length-K aggregate.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from positive_factor_kernel import a_factor, b_factor, factor_weight, initial_profile_exp6


def precompute_factor_arrays(
    u_grid: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build factor arrays for a K-term truncation.

    Parameters
    ----------
    u_grid : (N,) uniform grid on [0, 1]
    K      : number of factors to include (factors 1..K)

    Returns
    -------
    A_matrix : (N, K)  where A_matrix[i, m] = a_{m+1}(u_i)
    B_matrix : (N, K)  where B_matrix[i, m] = b_{m+1}(u_i)
    weights  : (K,)    where weights[m] = factor_weight(m+1)
    """
    N = len(u_grid)
    A_matrix = np.empty((N, K), dtype=float)
    B_matrix = np.empty((N, K), dtype=float)
    weights = np.empty(K, dtype=float)
    for idx, m in enumerate(range(1, K + 1)):
        A_matrix[:, idx] = a_factor(m, u_grid)
        B_matrix[:, idx] = b_factor(m, u_grid)
        weights[idx] = factor_weight(m)
    return A_matrix, B_matrix, weights


def solve_factored_graphon(
    A_matrix: np.ndarray,
    B_matrix: np.ndarray,
    weights: np.ndarray,
    x0_vec: np.ndarray,
    mu: float,
    T: float,
    epsilon: float,
    t_eval: np.ndarray,
    K: int,
    N: int,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the factored graphon ODE via DOP853.

    dX_i/dt = mu - sum_{m=1}^K weight_m * a_m(u_i) * (1/N) * sum_j b_m(u_j)*ell_eps(X_j)
            = mu - A @ (weights * q(t))
    where q = (1/N) * B.T @ ell_eps(X).

    Parameters
    ----------
    A_matrix, B_matrix : (N, K) precomputed factor arrays
    weights            : (K,) factor weights
    x0_vec             : (N,) initial condition
    mu                 : drift
    T                  : terminal time
    epsilon            : ramp smoothing width
    t_eval             : evaluation times
    K, N               : redundant (for safety checks / clarity)
    rtol, atol         : solver tolerances
    max_step           : maximum step size

    Returns
    -------
    t : (len(t_eval),) evaluation times
    X : (N, len(t_eval)) state trajectories
    """
    assert A_matrix.shape == (N, K) and B_matrix.shape == (N, K)

    def rhs(t: float, X: np.ndarray) -> np.ndarray:
        ell = np.where(X <= 0.0, 1.0, np.clip(1.0 - X / epsilon, 0.0, 1.0))
        q = B_matrix.T @ ell / N          # (K,)
        return mu - A_matrix @ (weights * q)

    sol = solve_ivp(
        rhs,
        [0.0, T],
        x0_vec.copy(),
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )
    return sol.t, sol.y


def _kernel_L1_error(
    A_ref: np.ndarray,
    B_ref: np.ndarray,
    weights_ref: np.ndarray,
    K: int,
) -> float:
    """L1 kernel approximation error using the factored tail structure.

    ||W_{K_ref} - W_K||_1 = (1/N^2) * sum_{i,j} |sum_{m=K+1}^{K_ref} w_m * a_m(u_i) * b_m(u_j)|

    Built as an (N, N) matrix via a low-rank outer-product sum.
    """
    K_ref = A_ref.shape[1]
    N = A_ref.shape[0]
    if K >= K_ref:
        return 0.0
    A_tail = A_ref[:, K:]                          # (N, K_ref-K)
    B_tail = B_ref[:, K:]                          # (N, K_ref-K)
    w_tail = weights_ref[K:]                       # (K_ref-K,)
    # delta[i,j] = sum_m (A_tail[i,m] * w_tail[m]) * B_tail[j,m]
    delta = (A_tail * w_tail[None, :]) @ B_tail.T  # (N, N)
    return float(np.mean(np.abs(delta)))


def run_truncation_study(
    K_list: list[int] | None = None,
    K_ref: int = 30,
    N: int = 2000,
    T: float = 2.0,
    mu: float = -0.006,
    epsilon: float = 0.02,
) -> dict:
    """Run the positive-factor truncation convergence study.

    Parameters
    ----------
    K_list : truncation orders to test (default [4, 12, 24])
    K_ref  : reference (full) truncation order
    N      : number of discretisation points
    T      : terminal time
    mu     : drift
    epsilon: ramp smoothing width

    Returns
    -------
    results : dict with keys:
        'u_grid'       : (N,) grid
        't_eval'       : evaluation times
        'X_ref'        : (N, T_steps) reference trajectory (K=K_ref)
        'h_ref'        : (T_steps,) threshold fraction for reference
        'K_ref'        : K_ref value
        'K_results'    : list of dicts per K in K_list:
            {'K', 'cumulative_mass', 'kernel_L1_err', 'state_L1_err', 'X_t', 'h_t'}
    """
    if K_list is None:
        K_list = [4, 12, 24]

    from positive_factor_kernel import cumulative_mass

    u_grid = (np.arange(N) + 0.5) / N
    x0 = initial_profile_exp6(u_grid)
    t_eval = np.linspace(0.0, T, 201)

    # Reference (K = K_ref)
    print(f"[exp6] precomputing reference arrays (K={K_ref}) …")
    A_ref, B_ref, w_ref = precompute_factor_arrays(u_grid, K_ref)

    print(f"[exp6] solving reference ODE (K={K_ref}, eps={epsilon}) …")
    _, X_ref = solve_factored_graphon(
        A_ref, B_ref, w_ref, x0, mu, T, epsilon, t_eval, K=K_ref, N=N,
    )
    h_ref = np.mean(X_ref <= 0.0, axis=0)

    K_results = []
    for K in K_list:
        print(f"[exp6] solving K={K} ODE …")
        A_K = A_ref[:, :K]
        B_K = B_ref[:, :K]
        w_K = w_ref[:K]

        _, X_K = solve_factored_graphon(
            A_K, B_K, w_K, x0, mu, T, epsilon, t_eval, K=K, N=N,
        )
        h_K = np.mean(X_K <= 0.0, axis=0)

        kern_err = _kernel_L1_error(A_ref, B_ref, w_ref, K)
        state_err = float(np.max(np.mean(np.abs(X_K - X_ref), axis=0)))
        cm = cumulative_mass(K, M_total=K_ref)

        print(
            f"  K={K:2d}: cum_mass={cm:.4f}, "
            f"kern_L1={kern_err:.3e}, state_L1={state_err:.3e}"
        )
        K_results.append({
            "K": K,
            "cumulative_mass": cm,
            "kernel_L1_err": kern_err,
            "state_L1_err": state_err,
            "X_t": X_K,
            "h_t": h_K,
        })

    return {
        "u_grid": u_grid,
        "t_eval": t_eval,
        "X_ref": X_ref,
        "h_ref": h_ref,
        "K_ref": K_ref,
        "K_results": K_results,
    }
