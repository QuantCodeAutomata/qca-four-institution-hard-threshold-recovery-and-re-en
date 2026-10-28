"""Experiment 6: Positive-Factor Truncation of an Infinite-Rank Nonnegative Graphon.

Runs the factor-truncation convergence study with K=4, 12, 24 vs K=30 reference.

Saves:
  results/exp6_paths.png
  results/exp6_errors.png
  results/exp6_metrics.json
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from positive_factor_truncation import run_truncation_study

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main() -> None:
    print("\n=== Experiment 6: Positive-Factor Truncation ===")
    res = run_truncation_study(
        K_list=[4, 12, 24],
        K_ref=30,
        N=2000,
        T=2.0,
        mu=-0.006,
        epsilon=0.02,
    )

    t = res["t_eval"]
    h_ref = res["h_ref"]
    K_ref = res["K_ref"]

    # ── Plot: threshold fraction paths ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, h_ref, "k-", lw=2.2, label=f"reference ($K={K_ref}$)")
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    for kr, col in zip(res["K_results"], colors):
        ax.plot(t, kr["h_t"], color=col, lw=1.5, ls="--",
                label=f"$K={kr['K']}$ (mass={kr['cumulative_mass']:.3f})")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"distress fraction $h(t)$")
    ax.set_title("Exp 6 – Factor truncation: distress-fraction paths")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp6_paths.png"), dpi=150)
    plt.close(fig)

    # ── Plot: kernel and state errors vs K ──────────────────────────────────
    K_vals = [kr["K"] for kr in res["K_results"]]
    k_errs = [kr["kernel_L1_err"] for kr in res["K_results"]]
    s_errs = [kr["state_L1_err"] for kr in res["K_results"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(K_vals, k_errs, "go-", ms=7)
    axes[0].set_xlabel("truncation $K$")
    axes[0].set_ylabel(r"kernel L1 error $\|W_K - W_{30}\|_1$")
    axes[0].set_title("Kernel L1 error vs K")
    axes[0].set_xticks(K_vals)
    axes[0].grid(which="both", alpha=0.3)

    axes[1].semilogy(K_vals, s_errs, "mo-", ms=7)
    axes[1].set_xlabel("truncation $K$")
    axes[1].set_ylabel("state path L1 error")
    axes[1].set_title("State-path L1 error vs K")
    axes[1].set_xticks(K_vals)
    axes[1].grid(which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp6_errors.png"), dpi=150)
    plt.close(fig)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n  reference: K={K_ref}, h(T)={h_ref[-1]:.4f}")
    print(f"\n  {'K':>4} | {'cum_mass':>9} | {'kern_L1':>12} | {'state_L1':>12}")
    print("  " + "-" * 46)
    for kr in res["K_results"]:
        print(
            f"  {kr['K']:4d} | {kr['cumulative_mass']:9.4f} | "
            f"{kr['kernel_L1_err']:12.3e} | {kr['state_L1_err']:12.3e}"
        )

    # Verify monotone convergence
    s_errs_vals = [kr["state_L1_err"] for kr in res["K_results"]]
    monotone = all(s_errs_vals[i] > s_errs_vals[i + 1] for i in range(len(s_errs_vals) - 1))
    print(f"\n  Errors decrease monotonically with K: {monotone}")

    # ── Save metrics ────────────────────────────────────────────────────────
    metrics = {
        "K_ref": K_ref,
        "h_ref_T": float(h_ref[-1]),
        "truncations": [
            {
                "K": kr["K"],
                "cumulative_mass": float(kr["cumulative_mass"]),
                "kernel_L1_err": float(kr["kernel_L1_err"]),
                "state_L1_err": float(kr["state_L1_err"]),
                "h_T": float(kr["h_t"][-1]),
            }
            for kr in res["K_results"]
        ],
    }
    out_path = os.path.join(RESULTS_DIR, "exp6_metrics.json")
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nMetrics saved to {out_path}")
    print("Exp 6 complete.")


if __name__ == "__main__":
    main()
