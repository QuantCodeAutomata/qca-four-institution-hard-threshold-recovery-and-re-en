"""Sovereign factor model construction.

Builds the common-exposure matrix E from sovereign holdings data using the
factor decomposition:  E_{ij} = lambda_E * sum_k  a_{ik} * b_{jk}

where
  a_{ik} = s_{ik} / sum_l s_{il}           (bank i's portfolio share in bucket k)
  b_{ik} = s_{ik} / (N^{-1} sum_j s_{jk})  (bank i's relative exposure to bucket k)

This factorisation satisfies the row-mean property:
  (1/N) sum_j E_{ij} = lambda_E  for all i.
"""
import numpy as np


def build_loadings(S_matrix, lambda_E=0.075):
    """Construct factor loadings and common-exposure matrix.

    Parameters
    ----------
    S_matrix : ndarray, shape (N, K)
        Holdings matrix (EUR millions). Rows = banks, columns = sovereign buckets.
    lambda_E : float
        Common-exposure scaling parameter.

    Returns
    -------
    a_matrix : ndarray, shape (N, K)
        Row-normalised portfolio shares:  a_{ik} = s_{ik} / sum_l s_{il}.
    b_matrix : ndarray, shape (N, K)
        Column-normalised relative exposures:  b_{ik} = s_{ik} / (col_mean_k).
    e_matrix : ndarray, shape (N, N)
        Common-exposure matrix E = lambda_E * a @ b^T.
        Satisfies (1/N) sum_j E_{ij} = lambda_E for all i.
    """
    N, K = S_matrix.shape
    row_sums = S_matrix.sum(axis=1, keepdims=True)          # (N, 1)
    a_matrix = S_matrix / row_sums                           # (N, K)

    col_means = S_matrix.sum(axis=0, keepdims=True) / N     # (1, K)  per-capita average
    b_matrix = S_matrix / col_means                          # (N, K)

    e_matrix = lambda_E * (a_matrix @ b_matrix.T)            # (N, N)
    return a_matrix, b_matrix, e_matrix


def build_scenario_buffers(a_matrix, shocks_dict, bucket_names, x_star=0.40):
    """Compute initial buffer vector for a shock scenario.

    Parameters
    ----------
    a_matrix : ndarray, shape (N, K)
        Portfolio-share matrix from :func:`build_loadings`.
    shocks_dict : dict
        Mapping of ISO country code → shock weight eta (e.g. ``{'IT': 0.45}``).
    bucket_names : list of str
        Ordered list of bucket labels matching columns of ``a_matrix``.
    x_star : float
        Stress-test capital ratio threshold.

    Returns
    -------
    x0 : ndarray, shape (N,)
        Initial buffer  x_i = x_star - sum_{k in shocked} eta_k * a_{ik}.
    """
    K = len(bucket_names)
    eta = np.zeros(K)
    for country, shock in shocks_dict.items():
        if country in bucket_names:
            eta[bucket_names.index(country)] = shock
    return x_star - a_matrix @ eta


def scenario_channel_order(shocks_dict, bucket_names, a_matrix, S_matrix):
    """Order sovereign channels for scenario-based truncation.

    Channels are ranked by shock weight (eta_k) descending; ties broken by
    aggregate holdings sum_i s_{ik} descending.

    Parameters
    ----------
    shocks_dict : dict
        Mapping of country code → eta value.
    bucket_names : list of str
    a_matrix : ndarray, shape (N, K)   (not used directly, kept for API symmetry)
    S_matrix : ndarray, shape (N, K)

    Returns
    -------
    order : list of int
        Indices into bucket_names, sorted from most to least important.
    """
    K = len(bucket_names)
    eta = np.zeros(K)
    for country, shock in shocks_dict.items():
        if country in bucket_names:
            eta[bucket_names.index(country)] = shock

    col_sums = S_matrix.sum(axis=0)
    order = sorted(range(K), key=lambda k: (eta[k], col_sums[k]), reverse=True)
    return order


def compute_svd_reduction(e_matrix, q):
    """Compute rank-q SVD approximation of the common-exposure matrix.

    Parameters
    ----------
    e_matrix : ndarray, shape (N, N)
    q : int
        Rank of approximation.

    Returns
    -------
    E_q : ndarray, shape (N, N)
        Signed rank-q approximation  E_q = U[:, :q] diag(s[:q]) Vt[:q, :].
        May contain negative entries.
    """
    U, s, Vt = np.linalg.svd(e_matrix, full_matrices=True)
    E_q = (U[:, :q] * s[:q]) @ Vt[:q, :]
    return E_q
