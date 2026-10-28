"""Experiment 7: Empirical Sovereign-Portfolio Contagion and Channel-Reduction Comparison.

Uses EBA 2025 EU-wide transparency exercise data (item 2520810, period 202506,
portfolio 0, maturity 8).  Falls back to a realistic synthetic dataset when the
real file is unavailable.

Outputs
-------
results/exp7_contagion_paths.png
results/exp7_reduction_comparison.png
results/exp7_sensitivity.png
results/exp7_metrics.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Resolve src/ regardless of where the script is invoked
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), 'src')
_RESULTS = os.path.join(os.path.dirname(_HERE), 'results')
_DATA = os.path.join(os.path.dirname(_HERE), 'data')
sys.path.insert(0, _SRC)

from eba_loader import load_eba_data
from sovereign_factors import (
    build_loadings,
    build_scenario_buffers,
    scenario_channel_order,
)
from sovereign_solver import (
    solve_sovereign_full,
    solve_sovereign_reduced_scenario,
    solve_sovereign_reduced_svd,
)
from sovereign_reduction import compare_reductions, compute_attribution
from sovereign_sensitivity import run_sensitivity_grid

os.makedirs(_RESULTS, exist_ok=True)

# ── Baseline shock scenario ────────────────────────────────────────────────────
SHOCKS = {'IT': 0.45, 'FR': 0.35, 'DE': 0.15, 'NL': 0.05}
X_STAR = 0.40
LAMBDA_E = 0.075
EPSILON = 0.05
T = 3.0

# ── 1. Load data ───────────────────────────────────────────────────────────────
EBA_CANDIDATES = [
    os.path.join(_DATA, 'eba_2025.csv'),
    os.path.join(_DATA, 'eba_2025.xlsx'),
]
real_data_used = False
for fpath in EBA_CANDIDATES:
    if os.path.exists(fpath):
        S, bank_names, bucket_names = load_eba_data(filepath=fpath)
        real_data_used = True
        print(f'[Exp 7] Loaded real EBA data from {fpath}')
        break

if not real_data_used:
    S, bank_names, bucket_names = load_eba_data()
    print('[Exp 7] Real EBA data not found – using synthetic dataset')

N, K = S.shape
print(f'[Exp 7] Dataset: {N} banks × {K} sovereign buckets')

# ── 2. Build loadings and verify row-mean property ────────────────────────────
a_mat, b_mat, e_mat = build_loadings(S, lambda_E=LAMBDA_E)

row_means = e_mat.mean(axis=1)
row_mean_err = float(np.max(np.abs(row_means - LAMBDA_E)))
print(f'[Exp 7] Row-mean property error (max |mean_j e_ij - lambda_E|): {row_mean_err:.2e}')

# ── 3. Scenario buffers ────────────────────────────────────────────────────────
x0 = build_scenario_buffers(a_mat, SHOCKS, bucket_names, x_star=X_STAR)
n_initial_distressed = int(np.sum(x0 < 0))
print(f'[Exp 7] Initially distressed (x0 < 0): {n_initial_distressed}')

# ── 4. Channel ordering ────────────────────────────────────────────────────────
ch_order = scenario_channel_order(SHOCKS, bucket_names, a_mat, S)
print(f'[Exp 7] Top 5 channels: {[bucket_names[k] for k in ch_order[:5]]}')

# ── 5. Full 38-channel model ───────────────────────────────────────────────────
print('[Exp 7] Solving full model...')
full_result = solve_sovereign_full(
    a_mat, b_mat, LAMBDA_E, x0, T=T, epsilon=EPSILON,
)
n_terminal_distressed = int(np.sum(full_result['X_t'][:, -1] < 0))
print(f'[Exp 7] Terminal distressed: {n_terminal_distressed}')
print(f'[Exp 7] Mean distress rate: {full_result["mean_distress_rate"]:.4f}')

# ── 6. Scenario-truncated reductions ──────────────────────────────────────────
scenario_qs = [2, 32]
scenario_results = {}
for q in scenario_qs:
    print(f'[Exp 7] Scenario truncation q={q}...')
    scenario_results[q] = solve_sovereign_reduced_scenario(
        a_mat, b_mat, LAMBDA_E, x0, ch_order, q, T=T, epsilon=EPSILON,
    )

# ── 7. SVD-truncated reductions ────────────────────────────────────────────────
svd_qs = [2, 12, 32]
svd_results = {}
for q in svd_qs:
    print(f'[Exp 7] SVD truncation q={q}...')
    svd_results[q] = solve_sovereign_reduced_svd(
        e_mat, x0, q, T=T, epsilon=EPSILON,
    )

# ── 8. Max path errors ────────────────────────────────────────────────────────
errors = compare_reductions(full_result, scenario_results, svd_results)
print('\n[Exp 7] Max path errors:')
for key, val in sorted(errors.items()):
    print(f'  {key}: {val:.3e}')

# ── 9. Channel attribution ────────────────────────────────────────────────────
B_T = full_result['B_t'][:, -1]
attr = compute_attribution(B_T, a_mat, LAMBDA_E)
top5_attr = sorted(
    [(bucket_names[k], attr[k]) for k in range(K)],
    key=lambda x: -x[1],
)[:5]
print('\n[Exp 7] Top-5 channel attributions:')
for bname, share in top5_attr:
    print(f'  {bname}: {share:.3f}')

# ── 10. Sensitivity analysis ──────────────────────────────────────────────────
print('\n[Exp 7] Running sensitivity grid...')
sens_df = run_sensitivity_grid(
    a_mat, b_mat, S, bucket_names,
    shocks_dict=SHOCKS,
    x_star_list=[0.36, 0.38, 0.40, 0.42],
    lambda_E_list=[0.0375, 0.075, 0.15],
    epsilon_list=[EPSILON],
)
print(sens_df.to_string(index=False))

# ── 11. Save plots ─────────────────────────────────────────────────────────────

# — Plot A: Contagion paths ——————————————————————————————————————————————————
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Experiment 7: Sovereign Contagion Dynamics', fontsize=13)

t = full_result['t']
X_t = full_result['X_t']
h_t = full_result['h_t']

ax = axes[0]
for i in range(N):
    col = 'tab:red' if x0[i] < 0 else 'tab:blue'
    ax.plot(t, X_t[i], color=col, alpha=0.15, linewidth=0.5)
ax.axhline(0, color='k', linestyle='--', linewidth=0.8, label='x = 0')
ax.set_xlabel('Time')
ax.set_ylabel('Buffer X_i(t)')
ax.set_title(f'Bank buffer paths (red = initially distressed, n={n_initial_distressed})')
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(t, h_t, color='tab:orange', linewidth=1.5, label='Full model')
ax.plot(t, full_result['h_t'], color='tab:orange', linewidth=1.5)
for q in scenario_qs:
    ax.plot(t, scenario_results[q]['h_t'], linestyle='--',
            linewidth=1.0, label=f'Scenario q={q}')
for q in svd_qs:
    ax.plot(t, svd_results[q]['h_t'], linestyle=':',
            linewidth=1.0, label=f'SVD q={q}')
ax.set_xlabel('Time')
ax.set_ylabel('Mean distress rate h(t)')
ax.set_title('Aggregate distress intensity')
ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(_RESULTS, 'exp7_contagion_paths.png'), dpi=150)
plt.close(fig)

# — Plot B: Reduction comparison ————————————————————————————————————————————
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Experiment 7: Channel-Reduction Error Comparison', fontsize=13)

# Left: max path error bar chart
ax = axes[0]
labels = list(errors.keys())
vals = [errors[k] for k in labels]
colors = ['tab:blue' if 'scenario' in l else 'tab:orange' for l in labels]
bars = ax.bar(range(len(labels)), vals, color=colors)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax.set_yscale('log')
ax.set_ylabel('Max path error  (log scale)')
ax.set_title('Max path error by reduction type and rank')
# Add value labels on bars
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.2,
            f'{val:.2e}', ha='center', va='bottom', fontsize=7)

# Right: buffer paths – full vs best scenario and SVD
ax = axes[1]
ax.plot(t, X_t.mean(axis=0), color='k', linewidth=1.5, label='Full (mean)')
if 2 in scenario_results:
    ax.plot(t, scenario_results[2]['X_t'].mean(axis=0), '--',
            color='tab:blue', linewidth=1.0, label='Scenario q=2 (mean)')
if 32 in scenario_results:
    ax.plot(t, scenario_results[32]['X_t'].mean(axis=0), '-.',
            color='tab:blue', linewidth=1.0, label='Scenario q=32 (mean)')
if 2 in svd_results:
    ax.plot(t, svd_results[2]['X_t'].mean(axis=0), '--',
            color='tab:orange', linewidth=1.0, label='SVD q=2 (mean)')
if 32 in svd_results:
    ax.plot(t, svd_results[32]['X_t'].mean(axis=0), ':',
            color='tab:orange', linewidth=1.0, label='SVD q=32 (mean)')
ax.set_xlabel('Time')
ax.set_ylabel('Mean buffer')
ax.set_title('Mean buffer paths: full vs reductions')
ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(_RESULTS, 'exp7_reduction_comparison.png'), dpi=150)
plt.close(fig)

# — Plot C: Sensitivity analysis ————————————————————————————————————————————
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Experiment 7: Sensitivity Analysis', fontsize=13)

for ax, metric, title, fmt in zip(
    axes,
    ['terminal_below_zero', 'mean_distress_rate'],
    ['Terminal banks below zero', 'Mean distress rate'],
    ['.0f', '.3f'],
):
    pivot = sens_df.pivot_table(
        index='x_star', columns='lambda_E', values=metric, aggfunc='mean'
    )
    im = ax.imshow(
        pivot.values, aspect='auto', cmap='YlOrRd',
        vmin=pivot.values.min(), vmax=pivot.values.max(),
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{v:.4f}' for v in pivot.columns], rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{v:.2f}' for v in pivot.index])
    ax.set_xlabel('lambda_E')
    ax.set_ylabel('x_star')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f'{pivot.iloc[i, j]:{fmt}}',
                    ha='center', va='center', fontsize=8, color='black')

plt.tight_layout()
fig.savefig(os.path.join(_RESULTS, 'exp7_sensitivity.png'), dpi=150)
plt.close(fig)

# ── 12. Save metrics ──────────────────────────────────────────────────────────
metrics = {
    'data_source': 'real' if real_data_used else 'synthetic',
    'N_banks': N,
    'N_buckets': K,
    'n_initial_distressed': n_initial_distressed,
    'n_terminal_distressed': n_terminal_distressed,
    'mean_distress_rate': full_result['mean_distress_rate'],
    'row_mean_property_error': row_mean_err,
    'channel_order_top5': [bucket_names[k] for k in ch_order[:5]],
    'max_path_errors': {k: float(v) for k, v in errors.items()},
    'top5_attribution': {bname: float(share) for bname, share in top5_attr},
    'sensitivity': sens_df.to_dict(orient='records'),
}

metrics_path = os.path.join(_RESULTS, 'exp7_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f'\n[Exp 7] Metrics saved to {metrics_path}')
print('[Exp 7] Plots saved to results/')
print('[Exp 7] Done.')
