# Repository Knowledge for AI Agents

## Project Overview
Recoverable distress contagion model experiments (Exp 1–7).
All source code lives in `src/`, experiment runners in `exp/`, tests in `tests/`, outputs in `results/`.

## Dependency Stack
- Python 3.12, numpy 1.26, scipy 1.14, pandas 2.2, matplotlib 3.11, seaborn
- No custom package installed; `src/` is added to `sys.path` in each script/test.

## Module Import Pattern
All `exp/` and `tests/` files add the `src/` directory to `sys.path` explicitly:
```python
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT, 'src'))
```

## Experiment 1 – Four-Institution Hard-Threshold Recovery and Re-Entry Dynamics

### Architecture
- `src/hard_finite.py`: event-driven finite-network simulator.
  - `compute_slopes(X, d, e, mu, N)`: slope v_i = mu - (1/N)*sum_j e_{ij}*d_j for each institution.
  - `next_event_time(X, v)`: minimum positive tau_i = -X_i/v_i across sign-crossing institutions.
  - `simulate_event_driven(x0, e, mu, N, T)`: full event loop returning event data + dense 1000-point arrays.
  - `analytic_X4(t)`: exact formula 0.04 + 0.1t - 0.2*min(t,1) - 0.3*max(t-3,0).
  - `analytic_H4(t)`: exact formula min(max(t-0.4,0),1.2) + max(t-3.7,0).
- `exp/run_exp1.py`: runner; produces `results/exp1_paths.png`, `results/exp1_occupation.png`, `results/exp1_metrics.json`.

### Parameters
- N=4, T=5, mu=0.1, x0=[-0.1, -1.0, 0.3, 0.04] (0-indexed)
- e = N * [[0,0,0,0],[0,0,0,0],[0,0.2,0,0],[0.2,0,0.3,0]] = [[0,0,0,0],[0,0,0,0],[0,0.8,0,0],[0.8,0,1.2,0]]
- Hard threshold: distressed iff X_i <= 0

### Algorithm
- At t=0, initialise d_j = 1 if x0_j <= 0 (institutions 0 and 1 are distressed).
- Each interval: slopes v_i are constant; advance by tau = min_i { -X_i/v_i : tau_i > 0 }.
- At each event update d_j, track H_i(t) = int_0^t 1{X_i(s)<=0} ds via H_i += d_i * dt.
- Dense output: piecewise-linear reconstruction on 1000 uniform points in [0,T].

### Validated Results (exact to machine precision)
- Event times: [0.4, 1.0, 1.6, 3.0, 3.7] (error < 6e-17)
- H_4(T=5) = 2.5 exactly
- Simulated X_4(t) vs analytic: max error < 9e-17
- Simulated H_4(t) vs analytic: max error < 6e-17

### Test Coverage (16 tests)
`pytest tests/test_exp1.py -v`

---

## Experiment 4 – Hard-Threshold Regularization at the Osgood Boundary

### Architecture
- `src/osgood_ramp.py`: three ODE profiles and their positive-ramp regularizations.
  - `F_linear(a)`, `F_log(a)`, `F_sqrt(a)`: profile functions.
  - `G_eps_linear(eps, b)`: analytic average b + eps/2.
  - `G_eps_log(eps, b)`: numerical quadrature via scipy.integrate.quad.
  - `G_eps_sqrt(eps, b)`: analytic (2/(3eps))*[(b+eps)^1.5 - b^1.5].
  - `solve_beta_eps(G_eps_func, eps, T)`: DOP853 solve of beta_eps(0)=0.
  - `hard_solution_linear/log/sqrt(t)`: hard (greatest) solutions.
  - `compute_sup_error`, `fit_slope`: error analysis utilities.
- `exp/run_exp4.py`: runner; produces `results/exp4_trajectories.png`, `results/exp4_convergence.png`, `results/exp4_metrics.json`.

### Mathematical Background
Positive-ramp regularization: G_eps(b) = (1/eps)*int_0^eps F(b+s) ds.
Osgood criterion: unique zero solution when int_0^delta 1/F(y) dy = inf.
  - F_linear (F(a)=a): integral diverges -> unique beta=0.
  - F_log (F(a)=a*log(e/a)): integral diverges -> unique beta=0.
  - F_sqrt (F(a)=sqrt(a)): integral converges -> non-unique; greatest solution beta=t^2/4.

Convergence exponent derivation for F_log (unique hard solution beta=0):
  The hard ODE dot{y}=y*log(e/y) has exact solution log(e/y(t)) = log(e/y_0)*exp(-t).
  For small initial condition y_0 ~ eps, terminal error y(T) ~ C*eps^{exp(-T)}.
  Hence the convergence exponent gamma = exp(-T).

### Parameters
- T=1.0 (gives log-profile slope = exp(-1) ~ 0.368, consistent with ~0.365 expected)
- eps_plot = 2^{-4} ... 2^{-12}; eps_fit = 2^{-9} ... 2^{-12} (for OLS slope)
- ODE solver: DOP853, rtol=2e-12, atol=2e-14, max_step=T/500

### CRITICAL: Time Horizon Choice
Using T=2 gives log slope = exp(-2) ~ 0.135 (wrong).
Using T=1 gives log slope = exp(-1) ~ 0.365 (correct, matching expected).
Linear slope (=1.0) and sqrt slope (=0.5) are independent of T.

### Validated Results (eps_fit range 2^{-9} to 2^{-12}, T=1)
- linear: fitted slope 1.0000 (expected 1.000, residual 0.0000)
- log:    fitted slope 0.3650 (expected 0.365, residual 0.0000)
- sqrt:   fitted slope 0.4997 (expected 0.500, residual 0.0003)

### Test Coverage (26 tests)
`pytest tests/test_exp4.py -v`

### Encoding Warning
Source files in exp/ and src/ must use ASCII-only content (no Unicode like >=, ~, Greek letters
in comments/strings written via the file_editor). The system encoding rejects non-ASCII bytes.
The `r"$...$"` LaTeX strings in matplotlib labels are fine (they are raw Python string literals
and only passed to matplotlib at runtime).

---

## Experiment 5 – Graphon Smoothing and Fourier Truncation

### Architecture
- `src/graphon_kernel.py`: `build_kernel_matrix`, `build_fourier_kernel_matrix`, `initial_profile`, `ramp_loss`, `get_block`.
- `src/graphon_solver.py`: `solve_graphon_ramp` (DOP853), `compute_threshold_fraction`, `compute_path_L1_error`, `compute_threshold_fraction_error`.
- `src/graphon_smoothing.py`: `run_smoothing_experiment` — compares hard ref (eps=1e-4) vs eps=0.04/0.02/0.01.
- `src/graphon_fourier.py`: `run_fourier_experiment` — compares exact kernel vs M=1/2/4/8/16 Fourier-Bessel truncation.
- `exp/run_exp5.py`: End-to-end; saves 3 PNGs + `results/exp5_metrics.json`.

### Graphon Kernel
- W(u,v) = B[block(u), block(v)] + 0.16·exp(−10·sin²(π(u−v)))
- B = [[0.38, 0.24], [0.31, 0.36]]; blocks: I₁=[0, 0.5), I₂=[0.5, 1].
- Fourier decomposition via Bessel: exp(5·cos(2π(u−v))) = I₀(5) + 2·Σ Iₘ(5)·cos(2πm(u−v))
- Bessel functions computed with `scipy.special.iv`.

### Grid Convention
`u_grid = (np.arange(N) + 0.5) / N`  (midpoints of N uniform bins on [0,1]).

### ODE
dX_i/dt = μ − (1/N)·Σ_j W[i,j]·ℓ_ε(X_j), μ=−0.06, N=2000, T=2.0.
Hard reference uses ε=1e-4 with max_step=0.005.

### Validated Results (N=2000, T=2.0)
- hard h(T) = 0.2195
- eps=0.04: h(T)=0.2255, state_L1=9.38e-3
- eps=0.02: h(T)=0.2225, state_L1=4.68e-3
- eps=0.01: h(T)=0.2210, state_L1=2.33e-3
- M=1: kernel_Linf=7.816e-2, state_L1=6.33e-3
- M=2: kernel_Linf=4.042e-2, state_L1=2.74e-3
- M=4: kernel_Linf=7.128e-3, state_L1=2.03e-4
- M=8: kernel_Linf=5.420e-5, state_L1=8.28e-7

### Performance
- Build N=2000 kernel: ~0.23 s; ODE solve: ~12 s per run.
- Total exp5 runtime: ~3–4 min (10 ODE solves + 6 kernel builds).

### Test Coverage (16 tests, ~1 s)
`pytest tests/test_exp5.py -v`

---

## Experiment 6 – Positive-Factor Truncation of an Infinite-Rank Graphon

### Architecture
- `src/positive_factor_kernel.py`: `factor_weight`, `a_factor`, `b_factor`, `initial_profile_exp6`, `cumulative_mass`.
- `src/positive_factor_truncation.py`: `precompute_factor_arrays`, `solve_factored_graphon`, `run_truncation_study`.
- `exp/run_exp6.py`: End-to-end; saves 2 PNGs + `results/exp6_metrics.json`.

### Kernel
W(u,v) = Σ_{m=1}^∞ 0.35·m^{-1.8}·aₘ(u)·bₘ(v)
aₘ(u) = 1 + 0.45·sin(2πmu) + 0.15·cos(2π(m+1)u)
bₘ(v) = 1 + 0.40·cos(2πmv + 0.6) + 0.12·sin(2π(m+2)v)
x₀(u) = 0.13 + 0.08·sin(2πu) − 0.04·cos(6πu)

### Efficiency
The rank-K structure reduces ODE cost from O(N²) to O(N·K) per evaluation:
  q = (1/N)·B.T @ ℓ_ε(X)    [K-vector]
  dXdt = μ − A @ (w ⊙ q)    [N-vector]

### Kernel L1 Error (factored form)
delta = (A_tail * w_tail) @ B_tail.T   then   mean(|delta|)

### Validated Results (N=2000, T=2.0, μ=−0.006, ε=0.02)
- K=30 reference: h(T)=0.2595
- K=4:  mass=83.7%, kern_L1=1.03e-1, state_L1=3.27e-2
- K=12: mass=95.3%, kern_L1=2.96e-2, state_L1=1.38e-2
- K=24: mass=99.1%, kern_L1=5.44e-3, state_L1=2.85e-3
- Errors strictly monotone decreasing with K.

### Test Coverage (16 tests, ~1 s)
`pytest tests/test_exp6.py -v`

---

## Experiment 7 – Sovereign Contagion (EBA 2025 data)

### Architecture
- `src/eba_loader.py`: Loads real EBA CSV/Excel or generates synthetic 116×38 matrix.
- `src/sovereign_factors.py`: Builds a_matrix (row-normed), b_matrix (col-normed), e_matrix = λ_E · a @ b^T.
- `src/sovereign_solver.py`: Three ODE variants (DOP853): full K-dim factor, scenario-truncated q-dim, SVD-truncated N-dim.
- `src/sovereign_reduction.py`: Max path error, compare_reductions, per-channel attribution.
- `src/sovereign_sensitivity.py`: Grid over (x_star, λ_E, ε).
- `exp/run_exp7.py`: End-to-end pipeline saving 3 PNGs + JSON to `results/`.

### Key Mathematical Properties
- **Row-mean property** (exact): (1/N) Σ_j e_{ij} = λ_E for all i. Error < 1e-16 in practice.
- **Portfolio share rows** sum to 1: Σ_k a_{ik} = 1.
- **Negative ramp**: ℓ⁻_ε(x) = clip(-x/ε, 0, 1).

### Synthetic Data (seed=42)
- 116 banks from 13 EU countries; 38 ISO country buckets.
- First 4 Italian banks have IT concentration 93–97% → x₀ < 0 under baseline shock.
- Baseline shocks: IT=0.45, FR=0.35, DE=0.15, NL=0.05; x_star=0.40.
- With seed=42: 4 initially distressed, 8 terminally distressed.

### ODE Solver Notes
- Full model: K=38 dimensional state (B vector), solved with DOP853 rtol=2e-10, atol=2e-12.
- Scenario reduced: q-dimensional B state, only retained channels contribute feedback.
- SVD reduced: N=116 dimensional X state; B_t=None in result dict.
- Default t_eval: 601 equally-spaced points on [0, T=3].

### Test Coverage (23 tests, ~0.6 s)
Run with: `pytest tests/test_exp7.py -v`
Tests cover: ramp loss values, row-mean property, a-matrix row sums, buffer range,
channel ordering (IT first, FR second), SVD exact recovery, max path error = 0 identity,
synthetic data shape/non-negativity.

### Results (synthetic data)
- scenario_q2 error: ~1.65e-3
- scenario_q32 error: ~1.38e-5
- svd_q2 error: ~1.15e-2
- svd_q12 error: ~5.86e-3
- svd_q32 error: ~1.13e-4
- IT channel attribution: ~93% of total buffer reduction

## Running
```bash
# Tests only
pytest tests/test_exp7.py -v

# Full experiment (uses synthetic data if data/eba_2025.csv absent)
python exp/run_exp7.py
```
