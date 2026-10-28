"""Experiment 5: Graphon Smoothing and Fourier Truncation.

Runs two sub-experiments:
  A) Ramp-smoothing convergence: compares eps=0.04, 0.02, 0.01 vs hard reference
  B) Fourier-truncation convergence: compares M=1,2,4,8,16 vs exact kernel

Saves:
  results/exp5_smoothing_paths.png
  results/exp5_smoothing_errors.png
  results/exp5_fourier_errors.png
  results/exp5_metrics.json
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow importing from src/ regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphon_smoothing import run_smoothing_experiment
from graphon_fourier import run_fourier_experiment

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Sub-experiment A: ramp smoothing ────────────────────────────────────────

def run_smoothing(N_grid: int = 2000) -> dict:
    print("\n=== Experiment 5A: Ramp Smoothing ===")
    res = run_smoothing_experiment(
        N_grid=N_grid, T=2.0, mu=-0.06, epsilons=[0.04, 0.02, 0.01]
    )

    t = res["t_eval"]
    h_hard = res["h_hard"]

    # --- Plot: threshold-fraction paths ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, h_hard, "k-", lw=2, label=r"hard ($\varepsilon=10^{-4}$)")
    colors = ["#d73027", "#fc8d59", "#91bfdb"]
    for er, col in zip(res["eps_results"], colors):
        eps = er["epsilon"]
        ax.plot(t, er["h_t"], color=col, lw=1.5,
                label=rf"$\varepsilon={eps}$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"distress fraction $h(t)$")
    ax.set_title("Exp 5A – Ramp Smoothing: distress fraction paths")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp5_smoothing_paths.png"), dpi=150)
    plt.close(fig)

    # --- Plot: state L1 errors vs epsilon ---
    eps_vals = [er["epsilon"] for er in res["eps_results"]]
    state_errs = [er["state_L1_err"] for er in res["eps_results"]]
    h_errs = [er["h_err"] for er in res["eps_results"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog(eps_vals, state_errs, "bo-", ms=6)
    axes[0].set_xlabel(r"$\varepsilon$")
    axes[0].set_ylabel("state path L1 error")
    axes[0].set_title("State-path error vs smoothing width")
    axes[0].grid(which="both", alpha=0.3)

    axes[1].loglog(eps_vals, h_errs, "rs-", ms=6)
    axes[1].set_xlabel(r"$\varepsilon$")
    axes[1].set_ylabel(r"$\sup_t|h^\varepsilon - h^{\rm hard}|$")
    axes[1].set_title("Threshold-fraction error vs smoothing width")
    axes[1].grid(which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp5_smoothing_errors.png"), dpi=150)
    plt.close(fig)

    # Summary print
    print(f"\n  hard h(T)  = {h_hard[-1]:.4f}")
    for er in res["eps_results"]:
        print(
            f"  eps={er['epsilon']:.2f}: h(T)={er['h_t'][-1]:.4f}, "
            f"state_L1={er['state_L1_err']:.3e}, h_err={er['h_err']:.3e}"
        )

    return {
        "h_hard_T": float(h_hard[-1]),
        "smoothing": [
            {
                "epsilon": er["epsilon"],
                "h_T": float(er["h_t"][-1]),
                "state_L1_err": float(er["state_L1_err"]),
                "h_err": float(er["h_err"]),
            }
            for er in res["eps_results"]
        ],
    }


# ─── Sub-experiment B: Fourier truncation ────────────────────────────────────

def run_fourier(N_grid: int = 2000) -> dict:
    print("\n=== Experiment 5B: Fourier Truncation ===")
    res = run_fourier_experiment(
        N_grid=N_grid, T=2.0, mu=-0.06,
        M_list=[1, 2, 4, 8, 16], epsilon=0.01,
    )

    M_vals = [r["M"] for r in res["M_results"]]
    k_errs = [r["kernel_linf_err"] for r in res["M_results"]]
    s_errs = [r["state_L1_err"] for r in res["M_results"]]

    # --- Plot: kernel and state errors vs M ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(M_vals, k_errs, "go-", ms=6)
    axes[0].set_xlabel("Fourier terms $M$")
    axes[0].set_ylabel(r"$\|W - W_M\|_\infty$")
    axes[0].set_title("Kernel L-inf error vs M")
    axes[0].set_xticks(M_vals)
    axes[0].grid(which="both", alpha=0.3)

    axes[1].semilogy(M_vals, s_errs, "mo-", ms=6)
    axes[1].set_xlabel("Fourier terms $M$")
    axes[1].set_ylabel("state path L1 error")
    axes[1].set_title("State-path L1 error vs M")
    axes[1].set_xticks(M_vals)
    axes[1].grid(which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp5_fourier_errors.png"), dpi=150)
    plt.close(fig)

    # Summary print
    print("\n  {:>4s} | {:>14s} | {:>14s}".format("M", "kernel_Linf", "state_L1"))
    print("  " + "-" * 38)
    for r in res["M_results"]:
        print(
            f"  {r['M']:4d} | {r['kernel_linf_err']:14.3e} | {r['state_L1_err']:14.3e}"
        )

    return {
        "fourier": [
            {
                "M": r["M"],
                "kernel_linf_err": float(r["kernel_linf_err"]),
                "state_L1_err": float(r["state_L1_err"]),
            }
            for r in res["M_results"]
        ]
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    metrics: dict = {}
    metrics.update(run_smoothing(N_grid=2000))
    metrics.update(run_fourier(N_grid=2000))

    out_path = os.path.join(RESULTS_DIR, "exp5_metrics.json")
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nMetrics saved to {out_path}")
    print("Exp 5 complete.")
