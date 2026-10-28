"""
Experiment 3: Directed Imbalance and Multiplex Amplification
in Low-Rank Contagion Systems.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from imbalance import (
    DIRECTED_GROUPS,
    DIRECTED_PARAMS,
    compute_lambda,
    solve_with_imbalance,
    compute_groupwise_distress,
)
from multiplex_heatmap import (
    ARCH4_BASE,
    run_heatmap,
    compute_terminal_distress,
    build_modified_arch,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_directed_imbalance():
    """Part A: directed imbalance experiment."""
    print("\n=== Part A: Directed Imbalance Experiment ===")
    K = DIRECTED_PARAMS["K"]
    mu = DIRECTED_PARAMS["mu"]
    T = DIRECTED_PARAMS["T"]

    results = {}
    for r_val in [0.0, 0.05]:
        t_eval, beta_t, Lambda_g = solve_with_imbalance(
            DIRECTED_GROUPS, mu, r_val, T, K
        )
        h_g_t = compute_groupwise_distress(
            DIRECTED_GROUPS, mu, r_val, T, K, beta_t, t_eval, Lambda_g
        )
        h_g_T = h_g_t[:, -1]
        p = np.array([g["p"] for g in DIRECTED_GROUPS])
        h_T = float(p @ h_g_T)

        # Report imbalance terms
        Lambda_computed = compute_lambda(DIRECTED_GROUPS, r_val, K)

        print(f"\n  r = {r_val}:")
        print(f"    Lambda_g (per group): {Lambda_computed}")
        print(f"    Terminal h_g(T) by group: {h_g_T}")
        print(f"    Terminal h(T) aggregate: {h_T:.6f}")
        print(f"    Group 1 (net lender) terminal distress: {h_g_T[0]*100:.4f}%")
        print(f"    Group 2 (net borrower) terminal distress: {h_g_T[1]*100:.4f}%")

        results[f"r_{r_val}"] = {
            "Lambda_g": Lambda_computed.tolist(),
            "h_g_T": h_g_T.tolist(),
            "h_T": h_T,
        }

    # Expected (from paper): r=0.05: group1 ~0.091%, group2 ~13.47%
    print("\n  Expected (r=0.05): net lenders ~0.091%, net borrowers ~13.47%")
    return results


def run_multiplex_amplification():
    """Part B: multiplex amplification heatmap."""
    print("\n=== Part B: Multiplex Amplification Heatmap ===")
    omega2_grid = np.arange(0.0, 3.05, 0.05)
    delta_grid = np.arange(0.0, 0.161, 0.004)

    print(f"  Grid: omega_2 in [0, 3] x Delta_NBFI in [0, 0.16] ({len(omega2_grid)} x {len(delta_grid)} points)")
    h_matrix = run_heatmap(omega2_grid, delta_grid)
    print(f"  Heatmap computed. Shape: {h_matrix.shape}")

    # Report key values
    idx_omega0 = 0  # omega2=0
    idx_omega1 = np.argmin(np.abs(omega2_grid - 1.0))
    idx_omega2 = np.argmin(np.abs(omega2_grid - 2.0))
    idx_delta0 = 0  # delta=0
    idx_delta008 = np.argmin(np.abs(delta_grid - 0.08))

    print(f"\n  Baseline (delta=0):")
    print(f"    omega_2=0: h(T) = {h_matrix[idx_omega0, idx_delta0]:.4f} (expected ~10.06%)")
    print(f"    omega_2=1: h(T) = {h_matrix[idx_omega1, idx_delta0]:.4f} (expected ~13.04%)")
    print(f"\n  With Delta_NBFI=0.08:")
    print(f"    omega_2=0: h(T) = {h_matrix[idx_omega0, idx_delta008]:.4f} (expected ~14.27%)")
    print(f"    omega_2=1: h(T) = {h_matrix[idx_omega1, idx_delta008]:.4f} (expected ~23.40%)")
    print(f"    omega_2=2: h(T) = {h_matrix[idx_omega2, idx_delta008]:.4f} (expected ~43.23%)")

    results = {
        "omega2_grid": omega2_grid.tolist(),
        "delta_grid": delta_grid.tolist(),
        "baseline_omega0": float(h_matrix[idx_omega0, idx_delta0]),
        "baseline_omega1": float(h_matrix[idx_omega1, idx_delta0]),
        "delta008_omega0": float(h_matrix[idx_omega0, idx_delta008]),
        "delta008_omega1": float(h_matrix[idx_omega1, idx_delta008]),
        "delta008_omega2": float(h_matrix[idx_omega2, idx_delta008]),
    }
    return results, h_matrix, omega2_grid, delta_grid


def plot_imbalance(imbalance_results: dict):
    """Plot group distress fractions over time for r=0 and r=0.05."""
    K = DIRECTED_PARAMS["K"]
    mu = DIRECTED_PARAMS["mu"]
    T = DIRECTED_PARAMS["T"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["steelblue", "tomato"]
    group_labels = ["Group 1 (net lender a=[2.0,0.6])", "Group 2 (net borrower a=[0.6,2.0])"]

    for ax, r_val, title in zip(axes, [0.0, 0.05], ["r=0 (no imbalance)", "r=0.05 (directed imbalance)"]):
        t_eval, beta_t, Lambda_g = solve_with_imbalance(DIRECTED_GROUPS, mu, r_val, T, K)
        h_g_t = compute_groupwise_distress(DIRECTED_GROUPS, mu, r_val, T, K, beta_t, t_eval, Lambda_g)
        for g_idx, (color, lbl) in enumerate(zip(colors, group_labels)):
            ax.plot(t_eval, h_g_t[g_idx] * 100, color=color, lw=1.5, label=lbl)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Within-group distress fraction (%)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_imbalance.png"), dpi=120)
    plt.close()


def plot_heatmap(h_matrix: np.ndarray, omega2_grid: np.ndarray, delta_grid: np.ndarray):
    """Plot 2D heatmap of terminal distress fractions."""
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(
        h_matrix.T * 100,
        origin="lower",
        aspect="auto",
        extent=[omega2_grid[0], omega2_grid[-1], delta_grid[0], delta_grid[-1]],
        cmap="hot_r",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Terminal distress h(T) (%)", fontsize=11)
    ax.set_xlabel("omega_2 (second-layer scaling)", fontsize=11)
    ax.set_ylabel("Delta_NBFI (NBFI buffer shock)", fontsize=11)
    ax.set_title("Multiplex Amplification: Terminal Distress h(T)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_heatmap.png"), dpi=120)
    plt.close()


def main():
    print("=" * 70)
    print("EXPERIMENT 3: Directed Imbalance and Multiplex Amplification")
    print("=" * 70)

    imbalance_results = run_directed_imbalance()
    heatmap_results, h_matrix, omega2_grid, delta_grid = run_multiplex_amplification()

    plot_imbalance(imbalance_results)
    plot_heatmap(h_matrix, omega2_grid, delta_grid)

    metrics = {
        "directed_imbalance": imbalance_results,
        "multiplex_amplification": {
            k: v for k, v in heatmap_results.items()
            if k not in ("omega2_grid", "delta_grid")
        },
    }
    with open(os.path.join(RESULTS_DIR, "exp3_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to results/exp3_metrics.json")
    print("Plots saved to results/exp3_*.png")


if __name__ == "__main__":
    main()
