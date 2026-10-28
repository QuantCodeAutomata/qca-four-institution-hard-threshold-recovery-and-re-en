"""Exp 4: Hard-Threshold Regularization at the Osgood Boundary.

Three ODE profiles F on [0, inf):
  F_linear(a) = a           -> unique hard solution beta = 0
  F_log(a)    = a*log(e/a)  -> unique hard solution beta = 0  (Osgood)
  F_sqrt(a)   = sqrt(a)     -> greatest hard solution beta = t^2/4

Positive-ramp regularisation: G_eps(b) = (1/eps)*int_0^eps F(b+s) ds

Convergence exponents (expected for T=1):
  linear : gamma ~ 1.00
  log    : gamma ~ 0.365  (= exp(-1) theoretically)
  sqrt   : gamma ~ 0.50
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.osgood_ramp import (
    G_eps_linear, G_eps_log, G_eps_sqrt,
    solve_beta_eps,
    hard_solution_linear, hard_solution_log, hard_solution_sqrt,
    compute_sup_error, fit_slope,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Time horizon: log-profile slope = exp(-T); T=1 gives exp(-1) ~0.368 ~0.365
T = 1.0

PROFILES = {
    "linear": {
        "G_eps":     G_eps_linear,
        "hard":      hard_solution_linear,
        "color":     "tab:blue",
        "label":     r"$F(a)=a$",
        "exp_slope": 1.00,
    },
    "log": {
        "G_eps":     G_eps_log,
        "hard":      hard_solution_log,
        "color":     "tab:orange",
        "label":     r"$F(a)=a\log(e/a)$",
        "exp_slope": 0.365,
    },
    "sqrt": {
        "G_eps":     G_eps_sqrt,
        "hard":      hard_solution_sqrt,
        "color":     "tab:green",
        "label":     r"$F(a)=\sqrt{a}$",
        "exp_slope": 0.50,
    },
}

# Epsilon grids
EPS_PLOT = [2.0 ** (-k) for k in range(4, 13)]   # 2^{-4} ... 2^{-12}
EPS_FIT  = [2.0 ** (-k) for k in range(9, 13)]   # 2^{-9} ... 2^{-12}


def run() -> dict:
    print("=" * 60)
    print("EXP 4: Regularization at the Osgood Boundary")
    print("=" * 60)
    print(f"Time horizon T = {T}")
    print(f"Fitting eps range: 2^-9 to 2^-12\n")

    all_errors: dict = {name: [] for name in PROFILES}
    fitted_slopes: dict = {}

    # ------------------------------------------------------------------
    # Compute errors for all profiles and epsilon values
    # ------------------------------------------------------------------
    for name, cfg in PROFILES.items():
        for eps in EPS_PLOT:
            t, beta_eps = solve_beta_eps(cfg["G_eps"], eps, T)
            beta_hard   = cfg["hard"](t)
            err = compute_sup_error(t, beta_eps, beta_hard)
            all_errors[name].append(err)

        fine_errors = [
            all_errors[name][EPS_PLOT.index(eps)] for eps in EPS_FIT
        ]
        slope = fit_slope(fine_errors, EPS_FIT)
        fitted_slopes[name] = slope

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    print(f"{'Profile':<10}  {'Expected gamma':>14}  {'Fitted gamma':>12}  {'Residual':>10}")
    print("-" * 55)
    for name, cfg in PROFILES.items():
        s = fitted_slopes[name]
        exp_s = cfg["exp_slope"]
        print(f"{name:<10}  {exp_s:>14.3f}  {s:>12.4f}  {abs(s-exp_s):>10.4f}")

    # ------------------------------------------------------------------
    # Plot 1: Trajectory families
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    t_ref = np.linspace(0.0, T, 500)

    for ax, (name, cfg) in zip(axes, PROFILES.items()):
        beta_hard = cfg["hard"](t_ref)
        ax.plot(t_ref, beta_hard, "k--", lw=2.2, label="Hard solution", zorder=5)

        n_show = min(5, len(EPS_PLOT))
        for k, eps in enumerate(EPS_PLOT[:n_show]):
            t_sol, beta_eps = solve_beta_eps(cfg["G_eps"], eps, T)
            alpha_val = 0.4 + 0.55 * k / max(n_show - 1, 1)
            ax.plot(
                t_sol, beta_eps,
                color=cfg["color"],
                alpha=alpha_val,
                lw=1.4,
                label=f"$\\varepsilon=2^{{-{4+k}}}$",
            )
        ax.set_xlabel("$t$", fontsize=12)
        ax.set_ylabel(r"$\beta_\varepsilon(t)$", fontsize=12)
        ax.set_title(cfg["label"], fontsize=12)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.set_xlim(0, T)

    fig.suptitle("Exp 4 - Ramp-Regularised Trajectories vs. Hard Solutions", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp4_trajectories.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: Log-log convergence
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, cfg in PROFILES.items():
        errors = all_errors[name]
        ax.loglog(
            EPS_PLOT, errors,
            "o-", color=cfg["color"], lw=1.8, ms=5,
            label=f"{cfg['label']}  ($\\hat{{\\gamma}}={fitted_slopes[name]:.3f}$)",
        )

    eps_arr = np.array(EPS_PLOT, dtype=float)
    ref_y0 = all_errors["linear"][0]
    ax.loglog(eps_arr, ref_y0 * (eps_arr / EPS_PLOT[0]) ** 1.0,
              "k:", lw=1.2, alpha=0.6, label="slope 1")
    ref_y0s = all_errors["sqrt"][0]
    ax.loglog(eps_arr, ref_y0s * (eps_arr / EPS_PLOT[0]) ** 0.5,
              "k--", lw=1.2, alpha=0.6, label="slope 0.5")

    ax.set_xlabel(r"$\varepsilon$", fontsize=13)
    ax.set_ylabel(r"$\sup_t\,|\beta_\varepsilon(t)-\beta(t)|$", fontsize=13)
    ax.set_title("Exp 4 - Log-Log Convergence to Hard Solution", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp4_convergence.png"), dpi=150)
    plt.close(fig)

    print("\nPlots saved to results/exp4_trajectories.png and results/exp4_convergence.png")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metrics = {
        "T": T,
        "eps_plot":  EPS_PLOT,
        "eps_fit":   EPS_FIT,
        "errors": {
            name: [float(e) for e in all_errors[name]]
            for name in PROFILES
        },
        "fitted_slopes": {name: float(v) for name, v in fitted_slopes.items()},
        "expected_slopes": {name: cfg["exp_slope"] for name, cfg in PROFILES.items()},
    }
    metrics_path = os.path.join(RESULTS_DIR, "exp4_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


if __name__ == "__main__":
    run()
