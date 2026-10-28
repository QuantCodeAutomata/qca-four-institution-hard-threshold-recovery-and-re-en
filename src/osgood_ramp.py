"""Exp 4: Hard-threshold regularization at the Osgood boundary.

Three scalar ODE profiles F on [0, inf) are studied:
  F_linear(a) = a
  F_log(a)    = a * log(e/a)  (with F(0)=0 by continuity)
  F_sqrt(a)   = sqrt(a)

The hard ODE is  dot{beta} = F(beta),  beta(0) = 0.
Uniqueness classification (Osgood criterion: int_0^delta 1/F(a) da = inf iff unique):
  F_linear : Lipschitz   -> unique solution beta = 0
  F_log    : Osgood      -> unique solution beta = 0
  F_sqrt   : non-Osgood  -> non-unique; positive-ramp limit selects beta(t) = t^2/4

The positive-ramp regularisation replaces F by the averaged kernel
  G_eps(b) = (1/eps) * int_0^eps F(b + s) ds

and solves  dot{beta_eps} = G_eps(beta_eps),  beta_eps(0) = 0.
Convergence rate:  sup_t |beta_eps(t) - beta_hard(t)| = O(eps^gamma)
  linear: gamma ~ 1.00
  log:    gamma ~ 0.365
  sqrt:   gamma ~ 0.50
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, List


# ---------------------------------------------------------------------------
# Profile functions F
# ---------------------------------------------------------------------------

def F_linear(a: np.ndarray) -> np.ndarray:
    """Linear profile: F(a) = a."""
    return np.asarray(a, dtype=float)


def F_log(a: np.ndarray) -> np.ndarray:
    """Logarithmic profile: F(a) = a * log(e/a), with F(0) = 0 by continuity."""
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = a * np.log(np.e / a)
    return np.where(a > 0.0, val, 0.0)


def F_sqrt(a: np.ndarray) -> np.ndarray:
    """Square-root profile: F(a) = sqrt(a)."""
    return np.sqrt(np.maximum(np.asarray(a, dtype=float), 0.0))


# ---------------------------------------------------------------------------
# Averaged kernels G_eps  (positive-ramp regularisation)
# G_eps(b) = (1/eps) * int_0^eps F(b+s) ds
# ---------------------------------------------------------------------------

def G_eps_linear(eps: float, b: float) -> float:
    """Ramp-averaged F_linear: (1/eps) * int_0^eps (b+s) ds = b + eps/2."""
    return b + eps / 2.0


def _log_antideriv(u: float) -> float:
    """Antiderivative A(u) of F_log(u) = u*log(e/u) = u*(1-log u).

    A(u) = u^2 * (3/4 - log(u)/2),  A(0) = 0.
    Satisfies A'(u) = u*(1 - log u) = u*log(e/u).
    """
    if u <= 0.0:
        return 0.0
    return u * u * (0.75 - 0.5 * np.log(u))


def G_eps_log(eps: float, b: float) -> float:
    """Ramp-averaged F_log: (1/eps) * int_b^{b+eps} u*log(e/u) du (analytic)."""
    return (_log_antideriv(b + eps) - _log_antideriv(max(b, 0.0))) / eps


def G_eps_sqrt(eps: float, b: float) -> float:
    """Ramp-averaged F_sqrt: (2/(3*eps)) * [(b+eps)^{3/2} - b^{3/2}]."""
    b = max(b, 0.0)
    return (2.0 / (3.0 * eps)) * ((b + eps) ** 1.5 - b ** 1.5)


# ---------------------------------------------------------------------------
# ODE solver
# ---------------------------------------------------------------------------

def solve_beta_eps(
    G_eps_func: Callable[[float, float], float],
    eps: float,
    T: float,
    n_points: int = 2001,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve dot{beta_eps} = G_eps_func(eps, beta_eps), beta_eps(0)=0 on [0, T].

    Parameters
    ----------
    G_eps_func : callable(eps, b) -> float
    eps        : regularisation parameter
    T          : time horizon
    n_points   : number of output time points
    rtol, atol : tolerances for DOP853 solver

    Returns
    -------
    t      : (n_points,) time array
    beta   : (n_points,) solution array
    """
    t_eval = np.linspace(0.0, T, n_points)

    def rhs(t, y):  # noqa: ARG001
        return [G_eps_func(eps, max(y[0], 0.0))]

    sol = solve_ivp(
        rhs,
        [0.0, T],
        [0.0],
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=T / 500.0,
    )
    return sol.t, sol.y[0]


# ---------------------------------------------------------------------------
# Hard (greatest) solutions
# ---------------------------------------------------------------------------

def hard_solution_linear(t: np.ndarray) -> np.ndarray:
    """Greatest hard solution for F_linear: beta(t) = 0 (unique by Lipschitz)."""
    return np.zeros_like(np.asarray(t, dtype=float))


def hard_solution_log(t: np.ndarray) -> np.ndarray:
    """Greatest hard solution for F_log: beta(t) = 0 (unique by Osgood criterion)."""
    return np.zeros_like(np.asarray(t, dtype=float))


def hard_solution_sqrt(t: np.ndarray) -> np.ndarray:
    """Greatest hard solution for F_sqrt: beta(t) = t^2/4 (positive-ramp selection)."""
    t = np.asarray(t, dtype=float)
    return t ** 2 / 4.0


# ---------------------------------------------------------------------------
# Error and slope utilities
# ---------------------------------------------------------------------------

def compute_sup_error(
    t: np.ndarray,
    beta_eps: np.ndarray,
    beta_hard: np.ndarray,
) -> float:
    """Compute sup_t |beta_eps(t) - beta_hard(t)|."""
    return float(np.max(np.abs(beta_eps - beta_hard)))


def fit_slope(errors: List[float], eps_list: List[float]) -> float:
    """Fit log-log slope of error vs epsilon by OLS.

    log(error) = gamma * log(eps) + intercept

    Returns
    -------
    gamma : estimated convergence exponent
    """
    log_eps = np.log(np.array(eps_list, dtype=float))
    log_err = np.log(np.array(errors, dtype=float))
    A = np.column_stack([log_eps, np.ones(len(log_eps))])
    result = np.linalg.lstsq(A, log_err, rcond=None)
    return float(result[0][0])
