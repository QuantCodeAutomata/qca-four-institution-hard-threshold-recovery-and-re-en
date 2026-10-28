"""Exp 1: Four-Institution Hard-Threshold Recovery and Re-Entry Dynamics.

N=4 institutions, T=5, mu=0.1.
Initial buffers: x0 = [-0.1, -1.0, 0.3, 0.04]  (0-indexed: 0,1,2,3)
Exposure matrix (e/N):
    [[0,   0,   0,   0  ],
     [0,   0,   0,   0  ],
     [0,   0.2, 0,   0  ],
     [0.2, 0,   0.3, 0  ]]

Expected event times: 0.4, 1.0, 1.6, 3.0, 3.7
Expected H_4(5) = 2.5  (institution 4 = index 3)
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hard_finite import (
    simulate_event_driven,
    analytic_X4,
    analytic_H4,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run() -> dict:
    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    N = 4
    T = 5.0
    mu = 0.1

    x0 = np.array([-0.1, -1.0, 0.3, 0.04])

    # e/N matrix then scale by N
    eoverN = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0],
        [0.2, 0.0, 0.3, 0.0],
    ])
    e = N * eoverN  # actual exposure matrix

    # ------------------------------------------------------------------
    # Run event-driven simulation
    # ------------------------------------------------------------------
    res = simulate_event_driven(x0, e, mu, N, T)

    t_events = res["t_events"]
    X_events = res["X_events"]
    H_events = res["H_events"]
    t_dense  = res["t_dense"]
    X_dense  = res["X_dense"]
    H_dense  = res["H_dense"]
    h_dense  = res["h_dense"]

    # ------------------------------------------------------------------
    # Verify event times  (exclude the final t=T sentinel)
    # ------------------------------------------------------------------
    expected_times = [0.4, 1.0, 1.6, 3.0, 3.7]
    sim_times = t_events[:-1]  # drop the T=5.0 appended at the end

    print("=" * 60)
    print("EXP 1: Hard-Threshold Recovery and Re-Entry Dynamics")
    print("=" * 60)
    print(f"\nEvent times (simulated):  {[round(t, 10) for t in sim_times]}")
    print(f"Event times (expected):   {expected_times}")

    tol_times = 1e-10
    for sim_t, exp_t in zip(sim_times, expected_times):
        err = abs(sim_t - exp_t)
        status = "OK" if err < tol_times else f"FAIL (err={err:.2e})"
        print(f"  t={exp_t}: simulated={sim_t:.14f}  {status}")

    assert len(sim_times) == len(expected_times), (
        f"Wrong number of events: got {len(sim_times)}, expected {len(expected_times)}"
    )
    for sim_t, exp_t in zip(sim_times, expected_times):
        assert abs(sim_t - exp_t) < tol_times, (
            f"Event time mismatch: simulated {sim_t}, expected {exp_t}"
        )

    # ------------------------------------------------------------------
    # Verify H_4(T) = 2.5  (institution index 3)
    # ------------------------------------------------------------------
    H4_terminal = H_events[-1][3]
    print(f"\nH_4(T=5) simulated:  {H4_terminal:.14f}")
    print(f"H_4(T=5) expected:   2.5")
    assert abs(H4_terminal - 2.5) < 1e-10, (
        f"H_4(5) mismatch: simulated {H4_terminal}, expected 2.5"
    )
    print("H_4(T) check: OK")

    # ------------------------------------------------------------------
    # Compare X_4 path to analytic formula on dense grid
    # ------------------------------------------------------------------
    X4_sim = X_dense[3, :]
    X4_analytic = analytic_X4(t_dense)
    max_X4_err = float(np.max(np.abs(X4_sim - X4_analytic)))
    print(f"\nMax |X_4_sim - X_4_analytic| over dense grid: {max_X4_err:.2e}")
    assert max_X4_err < 1e-10, f"X_4 path mismatch: {max_X4_err}"
    print("X_4 analytic match: OK")

    # ------------------------------------------------------------------
    # Compare H_4 path to analytic formula on dense grid
    # ------------------------------------------------------------------
    H4_sim = H_dense[3, :]
    H4_analytic = analytic_H4(t_dense)
    max_H4_err = float(np.max(np.abs(H4_sim - H4_analytic)))
    print(f"Max |H_4_sim - H_4_analytic| over dense grid: {max_H4_err:.2e}")
    assert max_H4_err < 1e-10, f"H_4 occupation mismatch: {max_H4_err}"
    print("H_4 analytic match: OK")

    # ------------------------------------------------------------------
    # Plot 1: Buffer paths X_i(t)
    # ------------------------------------------------------------------
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    labels = [f"$X_{i+1}(t)$" for i in range(N)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(N):
        ax.plot(t_dense, X_dense[i], color=colors[i], lw=1.8, label=labels[i])
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    for et in expected_times:
        ax.axvline(et, color="gray", lw=0.6, ls=":", alpha=0.7)
    ax.set_xlabel("$t$", fontsize=13)
    ax.set_ylabel("Buffer level $X_i(t)$", fontsize=13)
    ax.set_title("Exp 1 — Hard-Threshold Recovery: Buffer Paths", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim(0, T)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_paths.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: Occupation H_i(t) and system-level h_t = (1/N) sum 1{X_i<=0}
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1 = axes[0]
    for i in range(N):
        ax1.plot(t_dense, H_dense[i], color=colors[i], lw=1.8, label=labels[i])
    ax1.set_ylabel("Occupation $H_i(t)$", fontsize=13)
    ax1.set_title("Exp 1 — Cumulative Occupation Paths", fontsize=13)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.set_xlim(0, T)

    ax2 = axes[1]
    ax2.plot(t_dense, h_dense, color="purple", lw=2.0, label=r"$h_t = \frac{1}{N}\sum_i \mathbf{1}\{X_i\leq 0\}$")
    for et in expected_times:
        ax2.axvline(et, color="gray", lw=0.6, ls=":", alpha=0.7)
    ax2.set_xlabel("$t$", fontsize=13)
    ax2.set_ylabel("Fraction distressed $h_t$", fontsize=13)
    ax2.set_title("System-Level Distress Fraction", fontsize=13)
    ax2.legend(loc="upper left", fontsize=11)
    ax2.set_xlim(0, T)
    ax2.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_occupation.png"), dpi=150)
    plt.close(fig)

    print("\nPlots saved to results/exp1_paths.png and results/exp1_occupation.png")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metrics = {
        "event_times_simulated": [float(t) for t in sim_times],
        "event_times_expected": expected_times,
        "event_time_max_abs_err": float(
            max(abs(s - e) for s, e in zip(sim_times, expected_times))
        ),
        "H4_terminal_simulated": float(H4_terminal),
        "H4_terminal_expected": 2.5,
        "H4_terminal_abs_err": float(abs(H4_terminal - 2.5)),
        "max_X4_path_err": float(max_X4_err),
        "max_H4_path_err": float(max_H4_err),
    }
    metrics_path = os.path.join(RESULTS_DIR, "exp1_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


if __name__ == "__main__":
    run()
