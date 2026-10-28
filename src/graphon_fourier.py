"""Graphon Fourier-truncation study for Experiment 5.

Builds the exact kernel and M-truncated Fourier-Bessel approximations.
For each M, runs the smoothed (eps=0.01) ODE and measures:
  - kernel L-inf approximation error
  - state-path L1 error vs the full-kernel dynamics
"""

from __future__ import annotations

import numpy as np

from graphon_kernel import (
    build_fourier_kernel_matrix,
    build_kernel_matrix,
    initial_profile,
)
from graphon_solver import compute_path_L1_error, solve_graphon_ramp


def run_fourier_experiment(
    N_grid: int = 2000,
    T: float = 2.0,
    mu: float = -0.06,
    M_list: list[int] | None = None,
    epsilon: float = 0.01,
) -> dict:
    """Run the Fourier-truncation convergence study.

    Parameters
    ----------
    N_grid   : number of discretisation points on [0, 1]
    T        : terminal time
    mu       : drift
    M_list   : list of truncation orders to test (default [1, 2, 4, 8, 16])
    epsilon  : ramp width used in all ODE solves

    Returns
    -------
    results : dict with keys:
        'u_grid'      : (N,) grid
        't_eval'      : evaluation times
        'X_full'      : (N, T_steps) full-kernel reference trajectory
        'M_results'   : list of dicts, one per M:
            {'M', 'kernel_linf_err', 'state_L1_err', 'X_t', 'h_t'}
    """
    if M_list is None:
        M_list = [1, 2, 4, 8, 16]

    u_grid = (np.arange(N_grid) + 0.5) / N_grid
    x0 = initial_profile(u_grid)
    t_eval = np.linspace(0.0, T, 201)

    # Full (exact) kernel
    print(f"[fourier] building exact {N_grid}×{N_grid} kernel …")
    W_full = build_kernel_matrix(u_grid)

    print(f"[fourier] solving full-kernel reference (eps={epsilon}) …")
    _, X_full = solve_graphon_ramp(
        W_full, x0, mu, T, epsilon, t_eval,
        rtol=1e-9, atol=1e-11, max_step=0.01,
    )

    M_results = []
    for M in M_list:
        print(f"[fourier] building M={M} Fourier kernel …")
        W_M = build_fourier_kernel_matrix(u_grid, M, full_block=True)

        kernel_linf_err = float(np.max(np.abs(W_full - W_M)))

        print(f"[fourier] solving M={M} ODE …")
        _, X_M = solve_graphon_ramp(
            W_M, x0, mu, T, epsilon, t_eval,
            rtol=1e-9, atol=1e-11, max_step=0.01,
        )
        state_err = compute_path_L1_error(X_M, X_full, u_grid)

        from graphon_solver import compute_threshold_fraction
        h_t = compute_threshold_fraction(X_M, u_grid)

        print(
            f"  M={M:2d}: kernel_Linf={kernel_linf_err:.3e}, "
            f"state_L1_err={state_err:.3e}"
        )
        M_results.append({
            "M": M,
            "kernel_linf_err": kernel_linf_err,
            "state_L1_err": state_err,
            "X_t": X_M,
            "h_t": h_t,
        })

    return {
        "u_grid": u_grid,
        "t_eval": t_eval,
        "X_full": X_full,
        "M_results": M_results,
    }
