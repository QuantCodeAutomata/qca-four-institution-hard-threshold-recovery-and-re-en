"""Sensitivity analysis over (x_star, lambda_E, epsilon) parameter grids."""
import sys
import os
from itertools import product

import numpy as np
import pandas as pd

# Allow imports from src/ when running as a script
sys.path.insert(0, os.path.dirname(__file__))

from sovereign_factors import build_loadings, build_scenario_buffers
from sovereign_solver import solve_sovereign_full


def run_sensitivity_grid(
    a_matrix, b_matrix, S_matrix, bucket_names,
    shocks_dict, x_star_list, lambda_E_list, epsilon_list,
):
    """Run the full model over a parameter grid and collect summary statistics.

    For each combination of (x_star, lambda_E, epsilon):
    - Re-build loadings with the given lambda_E
    - Re-build initial buffers with the given x_star
    - Solve the full model
    - Record terminal count below zero and mean distress rate

    Parameters
    ----------
    a_matrix, b_matrix : ndarray, shape (N, K)
        Pre-computed loadings (used only for shape reference; re-built internally).
    S_matrix : ndarray, shape (N, K)
        Raw holdings matrix.
    bucket_names : list of str
    shocks_dict : dict   country code → eta value
    x_star_list : list of float
    lambda_E_list : list of float
    epsilon_list : list of float

    Returns
    -------
    pd.DataFrame with columns:
        x_star, lambda_E, epsilon, terminal_below_zero, mean_distress_rate
    """
    records = []
    for x_star, lambda_E, epsilon in product(x_star_list, lambda_E_list, epsilon_list):
        a_mat, b_mat, _ = build_loadings(S_matrix, lambda_E=lambda_E)
        x0 = build_scenario_buffers(a_mat, shocks_dict, bucket_names, x_star=x_star)
        result = solve_sovereign_full(a_mat, b_mat, lambda_E, x0, epsilon=epsilon)

        X_T = result['X_t'][:, -1]
        records.append({
            'x_star': x_star,
            'lambda_E': lambda_E,
            'epsilon': epsilon,
            'terminal_below_zero': int(np.sum(X_T < 0)),
            'mean_distress_rate': result['mean_distress_rate'],
        })

    return pd.DataFrame(records)
