"""
Experiment 2: Exact Rank-K Closure and Finite-Population Convergence
in Gaussian-Mixture Low-Rank Systems.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from low_rank_limit import (
    ARCHITECTURES,
    solve_limit_feedback,
    compute_limiting_threshold_fraction,
)
from low_rank_finite import make_quantile_populations, simulate_finite_hard_grouped
from monte_carlo_convergence import (
    run_deterministic_study,
    run_monte_carlo,
    fit_convergence_slope,
)
from metrics import uniform_threshold_fraction_error, feedback_path_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ARCH_NAMES = [
    "Rank-one mean field",
    "Core-periphery",
    "Overlapping CCP",
    "Bank-NBFI multiplex",
]

EXPECTED_H_TERMINAL = [0.4062, 0.2226, 0.2417, 0.1304]
EXPECTED_ERRORS_N400 = [0.001276, 0.002543, 0.003410, 0.002546]
EXPECTED_ERRORS_N1600 = [0.000315, 0.000627, 0.000886, 0.000622]


def run_limiting_dynamics():
    """Solve and report limiting terminal threshold fractions."""
    print("\n=== Limiting Terminal Threshold Fractions ===")
    print(f"{'Architecture':<30} {'h(T) computed':>15} {'h(T) expected':>15}")
    print("-" * 62)
    results = {}
    for arch, name, expected in zip(ARCHITECTURES, ARCH_NAMES, EXPECTED_H_TERMINAL):
        sol = solve_limit_feedback(arch["groups"], arch["mu"], arch["r"], arch["T"], arch["K"])
        h = compute_limiting_threshold_fraction(sol.y, arch["groups"], arch["mu"], arch["r"], sol.t, arch["K"])
        h_T = float(h[-1])
        print(f"{name:<30} {h_T:>15.4f} {expected:>15.4f}")
        results[name] = {"h_terminal": h_T, "expected": expected}
    return results


def run_deterministic_convergence():
    """Run deterministic quantile-matched populations for N=400 and N=1600."""
    print("\n=== Deterministic Quantile-Matched Threshold Fraction Errors ===")
    N_list = [400, 1600]
    print(f"{'Architecture':<30} {'N=400 computed':>15} {'N=400 expected':>15} {'N=1600 computed':>15} {'N=1600 expected':>15}")
    print("-" * 92)
    results = {}
    for arch, name, exp400, exp1600 in zip(
        ARCHITECTURES, ARCH_NAMES, EXPECTED_ERRORS_N400, EXPECTED_ERRORS_N1600
    ):
        det = run_deterministic_study(arch, N_list)
        err400 = det["h_errors"][0]
        err1600 = det["h_errors"][1]
        print(f"{name:<30} {err400:>15.6f} {exp400:>15.6f} {err1600:>15.6f} {exp1600:>15.6f}")
        results[name] = {
            "N400_error": err400,
            "N400_expected": exp400,
            "N1600_error": err1600,
            "N1600_expected": exp1600,
        }
    return results


def run_mc_slope_study():
    """Run Monte Carlo convergence and fit slopes."""
    print("\n=== Monte Carlo Convergence Slopes ===")
    N_list = [100, 200, 400, 800, 1600]
    n_reps = 30  # sufficient for slope estimation

    slope_results = {}
    arch = ARCHITECTURES[0]  # rank-one for main slope study
    mc = run_monte_carlo(arch, N_list, n_reps, seed=20261026)

    slope = fit_convergence_slope(mc["mean_h_errors"], N_list)
    print(f"Rank-one arch Monte Carlo slope (h error vs N): {slope:.4f} (expected ~-0.465)")
    slope_results["rank_one_h_slope"] = slope
    slope_results["N_list"] = N_list
    slope_results["mean_h_errors"] = mc["mean_h_errors"]
    return slope_results, mc


def plot_limiting_paths():
    """Plot limiting threshold fraction paths for all architectures."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, arch, name in zip(axes, ARCHITECTURES, ARCH_NAMES):
        sol = solve_limit_feedback(arch["groups"], arch["mu"], arch["r"], arch["T"], arch["K"])
        h = compute_limiting_threshold_fraction(sol.y, arch["groups"], arch["mu"], arch["r"], sol.t, arch["K"])
        ax.plot(sol.t, h, "b-", lw=1.5, label="Limiting h(t)")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("h(t)")
        ax.set_ylim(0, None)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_limiting_paths.png"), dpi=120)
    plt.close()


def plot_mc_convergence(mc_result: dict, slope_result: dict):
    """Plot log-log convergence of Monte Carlo errors."""
    fig, ax = plt.subplots(figsize=(7, 5))
    N_arr = np.array(mc_result["N_list"], dtype=float)
    err_arr = np.array(mc_result["mean_h_errors"])
    ax.loglog(N_arr, err_arr, "bo-", label="MC mean error")
    # Reference slope -0.5
    ref = err_arr[0] * (N_arr / N_arr[0]) ** (-0.5)
    ax.loglog(N_arr, ref, "r--", label="Slope -0.5 (CLT)")
    ax.set_xlabel("N")
    ax.set_ylabel("Mean h-error")
    ax.set_title("Rank-one Monte Carlo Convergence")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_mc_convergence.png"), dpi=120)
    plt.close()


def main():
    print("=" * 70)
    print("EXPERIMENT 2: Rank-K Closure and Finite-Population Convergence")
    print("=" * 70)

    limit_results = run_limiting_dynamics()
    det_results = run_deterministic_convergence()
    slope_results, mc_result = run_mc_slope_study()
    plot_limiting_paths()
    plot_mc_convergence(mc_result, slope_results)

    # Save metrics
    metrics = {
        "limiting_terminal_fractions": limit_results,
        "deterministic_convergence": det_results,
        "mc_convergence": {
            "slope": slope_results["rank_one_h_slope"],
            "N_list": slope_results["N_list"],
            "mean_errors": slope_results["mean_h_errors"],
        },
    }
    with open(os.path.join(RESULTS_DIR, "exp2_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to results/exp2_metrics.json")
    print("Plots saved to results/exp2_*.png")


if __name__ == "__main__":
    main()
