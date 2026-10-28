"""Graphon ramp-smoothing study for Experiment 5.

Compares hard-threshold dynamics (approximated via a very fine epsilon=1e-4)
against positive-ramp smoothed dynamics at several epsilon values.

All solves share the same graphon kernel (built once) and initial condition.
"""

from __future__ import annotations

import numpy as np

from graphon_kernel import build_kernel_matrix, initial_profile
from graphon_solver import (
    compute_path_L1_error,
    compute_threshold_fraction,
    compute_threshold_fraction_error,
    solve_graphon_ramp,
)


def run_smoothing_experiment(
    N_grid: int = 2000,
    T: float = 2.0,
    mu: float = -0.06,
    epsilons: list[float] | None = None,
) -> dict:
    """Run the ramp-smoothing convergence study.

    Parameters
    ----------
    N_grid   : number of discretisation points on [0, 1]
    T        : terminal time
    mu       : drift
    epsilons : ramp widths to test (default [0.04, 0.02, 0.01])

    Returns
    -------
    results : dict with keys:
        'u_grid'           : (N,) grid
        't_eval'           : (T_steps,) common evaluation times
        'X_hard'           : (N, T_steps) hard-reference trajectory (eps=1e-4)
        'h_hard'           : (T_steps,) threshold fraction for reference
        'eps_results'      : list of dicts, one per epsilon:
            {'epsilon', 'X_t', 'h_t', 'state_L1_err', 'h_err'}
    """
    if epsilons is None:
        epsilons = [0.04, 0.02, 0.01]

    u_grid = (np.arange(N_grid) + 0.5) / N_grid
    x0 = initial_profile(u_grid)
    t_eval = np.linspace(0.0, T, 201)

    print(f"[smoothing] building {N_grid}×{N_grid} kernel matrix …")
    W = build_kernel_matrix(u_grid)

    # Hard reference: very fine ramp approximation (epsilon = 1e-4)
    eps_ref = 1e-4
    print(f"[smoothing] solving hard reference (eps={eps_ref}) …")
    _, X_hard = solve_graphon_ramp(
        W, x0, mu, T, eps_ref, t_eval,
        rtol=1e-9, atol=1e-11, max_step=0.005,
    )
    h_hard = compute_threshold_fraction(X_hard, u_grid)

    eps_results = []
    for eps in epsilons:
        print(f"[smoothing] solving eps={eps} …")
        _, X_t = solve_graphon_ramp(
            W, x0, mu, T, eps, t_eval,
            rtol=1e-9, atol=1e-11, max_step=0.01,
        )
        h_t = compute_threshold_fraction(X_t, u_grid)
        state_err = compute_path_L1_error(X_t, X_hard, u_grid)
        h_err = compute_threshold_fraction_error(h_t, h_hard)
        eps_results.append({
            "epsilon": eps,
            "X_t": X_t,
            "h_t": h_t,
            "state_L1_err": state_err,
            "h_err": h_err,
        })
        print(
            f"  eps={eps:.2f}: h(T)={h_t[-1]:.4f}, "
            f"state_L1_err={state_err:.3e}, h_err={h_err:.3e}"
        )

    return {
        "u_grid": u_grid,
        "t_eval": t_eval,
        "X_hard": X_hard,
        "h_hard": h_hard,
        "eps_results": eps_results,
    }
