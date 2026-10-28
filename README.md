# Recoverable Distress Contagion: Finite Networks and Graphon Limits

This repository implements experiments from the recoverable distress contagion model,
covering finite hard-threshold networks, low-rank Gaussian-mixture closures, graphon
dynamics, regularization at the Osgood boundary, and empirical sovereign-portfolio contagion.

## Repository Structure

```
repo/
├── data/                # Data loading scripts and storage
├── src/                 # Core implementation modules
│   ├── hard_finite.py           # Exp 1: Event-driven finite-network simulator
│   ├── low_rank_limit.py        # Exp 2: Limiting K-dim feedback ODE solver
│   ├── low_rank_finite.py       # Exp 2: Finite grouped hard-threshold solver
│   ├── monte_carlo_convergence.py  # Exp 2: Monte Carlo convergence study
│   ├── metrics.py               # Exp 2: Error/convergence metrics
│   ├── imbalance.py             # Exp 3: Directed imbalance experiment
│   ├── multiplex_heatmap.py     # Exp 3: Multiplex amplification heatmap
│   ├── osgood_ramp.py           # Exp 4: Osgood boundary regularization
│   ├── graphon_kernel.py        # Exp 5: Graphon kernel construction
│   ├── graphon_solver.py        # Exp 5: Graphon dynamics solver
│   ├── graphon_smoothing.py     # Exp 5: Positive-ramp smoothing
│   ├── graphon_fourier.py       # Exp 5: Fourier truncation
│   ├── positive_factor_kernel.py  # Exp 6: Positive-factor kernel
│   ├── positive_factor_truncation.py  # Exp 6: Factor truncation study
│   ├── eba_loader.py            # Exp 7: EBA transparency data loader
│   ├── sovereign_factors.py     # Exp 7: Sovereign factor construction
│   ├── sovereign_solver.py      # Exp 7: Sovereign contagion solver
│   ├── sovereign_reduction.py   # Exp 7: SVD and scenario reductions
│   └── sovereign_sensitivity.py # Exp 7: Sensitivity analysis
├── exp/                 # Experiment scripts
│   ├── run_exp1.py
│   ├── run_exp2.py
│   ├── run_exp3.py
│   ├── run_exp4.py
│   ├── run_exp5.py
│   ├── run_exp6.py
│   └── run_exp7.py
├── tests/               # pytest tests
│   ├── test_exp1.py
│   ├── test_exp2.py
│   ├── test_exp3.py
│   ├── test_exp4.py
│   ├── test_exp5.py
│   ├── test_exp6.py
│   └── test_exp7.py
├── results/             # Artifacts (tables, plots, logs)
│   └── RESULTS.md
├── README.md
└── .gitignore
```

## Experiments

1. **Exp 1**: Four-Institution Hard-Threshold Recovery and Re-Entry Dynamics
2. **Exp 2**: Exact Rank-K Closure and Finite-Population Convergence
3. **Exp 3**: Directed Imbalance and Multiplex Amplification
4. **Exp 4**: Hard-Threshold Regularization at the Osgood Boundary
5. **Exp 5**: Graphon Smoothing and Fourier Truncation
6. **Exp 6**: Positive-Factor Truncation of an Infinite-Rank Graphon
7. **Exp 7**: Empirical Sovereign-Portfolio Contagion (EBA 2025 data)

## Installation

```bash
pip install numpy scipy matplotlib seaborn pandas requests
```

## Running Experiments

```bash
python exp/run_exp1.py
python exp/run_exp2.py
# ...etc
```

## Running Tests

```bash
pytest tests/ -v
```

## Results

See `results/RESULTS.md` for a summary of all experiment findings.
