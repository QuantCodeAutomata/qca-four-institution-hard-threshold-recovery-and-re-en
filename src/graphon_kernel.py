"""Graphon kernel construction for Experiment 5.

The graphon is defined on I = [0, 1] with two blocks:
    I_1 = [0, 1/2)  (block 0)
    I_2 = [1/2, 1]  (block 1)

Full kernel:
    W(u, v) = B[block(u), block(v)] + 0.16 * exp(-10 * sin^2(pi*(u-v)))

where B = [[0.38, 0.24],
           [0.31, 0.36]].

Fourier (Bessel) decomposition of the periodic component:
    0.16 * exp(-10*sin^2(pi*(u-v)))
    = 0.16 * exp(-5) * exp(5*cos(2*pi*(u-v)))
    = 0.16 * exp(-5) * [I_0(5) + 2*sum_{m>=1} I_m(5)*cos(2*pi*m*(u-v))]

where I_m is the modified Bessel function of the first kind.
"""

from __future__ import annotations

import numpy as np
from scipy.special import iv


# Block connectivity matrix
B_MATRIX = np.array([[0.38, 0.24],
                     [0.31, 0.36]])


def get_block(u: float) -> int:
    """Return block index: 0 if u < 0.5, else 1."""
    return 0 if u < 0.5 else 1


def build_kernel_matrix(u_grid: np.ndarray) -> np.ndarray:
    """Build the dense N×N exact graphon kernel matrix.

    W[i, j] = B[block(u_i), block(u_j)] + 0.16 * exp(-10 * sin^2(pi*(u_i - u_j)))

    Parameters
    ----------
    u_grid : array of shape (N,)

    Returns
    -------
    W : array of shape (N, N)
    """
    u = np.asarray(u_grid, dtype=float)
    diff = u[:, None] - u[None, :]                          # (N, N)
    W_per = 0.16 * np.exp(-10.0 * np.sin(np.pi * diff) ** 2)

    blocks = (u >= 0.5).astype(int)
    W_block = B_MATRIX[blocks[:, None], blocks[None, :]]    # (N, N)

    return W_block + W_per


def build_fourier_kernel_matrix(
    u_grid: np.ndarray,
    M: int,
    full_block: bool = True,
) -> np.ndarray:
    """Build the M-truncated Fourier (Bessel) approximation of the kernel.

    Periodic component approximation:
        W_sm^(M)(u,v) = 0.16*exp(-5)*[I_0(5) + 2*sum_{m=1}^M I_m(5)*cos(2*pi*m*(u-v))]

    Parameters
    ----------
    u_grid     : array of shape (N,)
    M          : number of Fourier terms (harmonics 1..M)
    full_block : if True, add the block component B[block(u), block(v)];
                 if False, return only the periodic M-truncated part.

    Returns
    -------
    W_M : array of shape (N, N)
    """
    u = np.asarray(u_grid, dtype=float)
    diff = u[:, None] - u[None, :]           # (N, N)

    scale = 0.16 * np.exp(-5.0)
    W_sm = scale * float(iv(0, 5.0)) * np.ones((len(u), len(u)))
    for m in range(1, M + 1):
        W_sm += scale * 2.0 * float(iv(m, 5.0)) * np.cos(2.0 * np.pi * m * diff)

    if not full_block:
        return W_sm

    blocks = (u >= 0.5).astype(int)
    W_block = B_MATRIX[blocks[:, None], blocks[None, :]]
    return W_block + W_sm


def initial_profile(u: np.ndarray | float) -> np.ndarray:
    """Piecewise-linear initial state profile.

    x0(u) = -0.18 + 1.95*u          for u < 0.5
            0.82 + 1.75*(u - 0.5)   for u >= 0.5
    """
    u = np.asarray(u, dtype=float)
    return np.where(u < 0.5, -0.18 + 1.95 * u, 0.82 + 1.75 * (u - 0.5))


def ramp_loss(x: float, epsilon: float) -> float:
    """Positive-side ramp loss (scalar).

    ell_epsilon(x) = 1            if x <= 0
                     1 - x/eps    if 0 < x < eps
                     0            if x >= eps
    """
    if x <= 0.0:
        return 1.0
    elif x < epsilon:
        return 1.0 - x / epsilon
    else:
        return 0.0
