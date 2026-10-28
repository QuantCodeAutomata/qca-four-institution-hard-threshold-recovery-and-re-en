"""Positive-factor graphon kernel for Experiment 6.

Infinite-rank graphon:
    W(u, v) = sum_{m=1}^inf  weight_m * a_m(u) * b_m(v)

where
    weight_m = 0.35 * m^{-1.8}
    a_m(u)   = 1 + 0.45*sin(2*pi*m*u) + 0.15*cos(2*pi*(m+1)*u)
    b_m(v)   = 1 + 0.40*cos(2*pi*m*v + 0.6) + 0.12*sin(2*pi*(m+2)*v)

Initial condition:
    x0(u) = 0.13 + 0.08*sin(2*pi*u) - 0.04*cos(6*pi*u)
"""

from __future__ import annotations

import numpy as np


def factor_weight(m: int) -> float:
    """Amplitude weight for factor m: 0.35 * m^{-1.8}."""
    return 0.35 * m ** (-1.8)


def a_factor(m: int, u: np.ndarray | float) -> np.ndarray:
    """Left factor a_m(u) = 1 + 0.45*sin(2*pi*m*u) + 0.15*cos(2*pi*(m+1)*u)."""
    u = np.asarray(u, dtype=float)
    return 1.0 + 0.45 * np.sin(2.0 * np.pi * m * u) + 0.15 * np.cos(2.0 * np.pi * (m + 1) * u)


def b_factor(m: int, v: np.ndarray | float) -> np.ndarray:
    """Right factor b_m(v) = 1 + 0.40*cos(2*pi*m*v + 0.6) + 0.12*sin(2*pi*(m+2)*v)."""
    v = np.asarray(v, dtype=float)
    return 1.0 + 0.40 * np.cos(2.0 * np.pi * m * v + 0.6) + 0.12 * np.sin(2.0 * np.pi * (m + 2) * v)


def initial_profile_exp6(u: np.ndarray | float) -> np.ndarray:
    """Initial state profile for Exp 6.

    x0(u) = 0.13 + 0.08*sin(2*pi*u) - 0.04*cos(6*pi*u)
    """
    u = np.asarray(u, dtype=float)
    return 0.13 + 0.08 * np.sin(2.0 * np.pi * u) - 0.04 * np.cos(6.0 * np.pi * u)


def cumulative_mass(K: int, M_total: int = 30) -> float:
    """Fraction of total kernel weight captured by the first K factors.

    Returns sum_{m=1}^K weight_m / sum_{m=1}^M_total weight_m
    """
    total = sum(factor_weight(m) for m in range(1, M_total + 1))
    partial = sum(factor_weight(m) for m in range(1, K + 1))
    return partial / total
