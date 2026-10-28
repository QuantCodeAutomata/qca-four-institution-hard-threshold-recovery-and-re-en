"""Tests for Experiment 7: Sovereign-Portfolio Contagion.

All tests operate on the synthetic dataset so they run without real EBA data.
"""
import sys
import os

import numpy as np
import pytest

# Make src/ importable whether tests are run from the project root or tests/
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT, 'src'))

from eba_loader import generate_synthetic_eba_data
from sovereign_factors import (
    build_loadings,
    build_scenario_buffers,
    scenario_channel_order,
    compute_svd_reduction,
)
from sovereign_solver import negative_ramp_loss, solve_sovereign_full
from sovereign_reduction import compute_max_path_error


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def synthetic_data():
    S, bank_names, bucket_names = generate_synthetic_eba_data(seed=42)
    return S, bank_names, bucket_names


@pytest.fixture(scope='module')
def loadings(synthetic_data):
    S, bank_names, bucket_names = synthetic_data
    a, b, e = build_loadings(S, lambda_E=0.075)
    return a, b, e


@pytest.fixture(scope='module')
def baseline_shocks():
    return {'IT': 0.45, 'FR': 0.35, 'DE': 0.15, 'NL': 0.05}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNegativeRampLoss:
    def test_zero_input(self):
        """ell^-_eps(0) = 0 for any epsilon."""
        assert negative_ramp_loss(0.0, epsilon=0.05) == 0.0

    def test_full_distress(self):
        """ell^-_eps(-eps) = 1 (fully distressed at x = -epsilon)."""
        assert negative_ramp_loss(-0.05, epsilon=0.05) == pytest.approx(1.0)

    def test_half_distress(self):
        """ell^-_eps(-eps/2) = 0.5 (linear interior)."""
        assert negative_ramp_loss(-0.025, epsilon=0.05) == pytest.approx(0.5)

    def test_positive_input(self):
        """ell^-_eps(x) = 0 for x > 0."""
        assert negative_ramp_loss(1.0, epsilon=0.05) == 0.0

    def test_deep_negative(self):
        """ell^-_eps(x) = 1 for x << -epsilon (capped at 1)."""
        assert negative_ramp_loss(-100.0, epsilon=0.05) == pytest.approx(1.0)

    def test_vectorised(self):
        """Works element-wise on arrays."""
        x = np.array([0.1, 0.0, -0.025, -0.05, -1.0])
        result = negative_ramp_loss(x, epsilon=0.05)
        expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(result, expected)


class TestRowMeanProperty:
    def test_row_mean_equals_lambda_E(self, loadings):
        """(1/N) sum_j e_{ij} = lambda_E for every bank i."""
        _, _, e = loadings
        N = e.shape[0]
        row_means = e.mean(axis=1)
        np.testing.assert_allclose(row_means, 0.075, rtol=1e-10,
                                   err_msg='Row-mean property violated')


class TestLoadingsRowSum:
    def test_a_rows_sum_to_one(self, loadings):
        """Portfolio shares a_{ik} sum to 1 for each bank."""
        a, _, _ = loadings
        row_sums = a.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-12,
                                   err_msg='a_matrix rows do not sum to 1')


class TestScenarioBuffers:
    def test_buffers_in_reasonable_range(self, loadings, synthetic_data, baseline_shocks):
        """x0 values should be in (-1, 2) for any realistic scenario."""
        a, _, _ = loadings
        S, bank_names, bucket_names = synthetic_data
        x0 = build_scenario_buffers(a, baseline_shocks, bucket_names, x_star=0.40)
        assert np.all(x0 > -1.0), f'Some x0 below -1: min={x0.min():.4f}'
        assert np.all(x0 < 2.0), f'Some x0 above 2: max={x0.max():.4f}'

    def test_buffers_depend_on_x_star(self, loadings, synthetic_data, baseline_shocks):
        """Larger x_star shifts all buffers up by the same amount."""
        a, _, _ = loadings
        S, bank_names, bucket_names = synthetic_data
        x0_40 = build_scenario_buffers(a, baseline_shocks, bucket_names, x_star=0.40)
        x0_42 = build_scenario_buffers(a, baseline_shocks, bucket_names, x_star=0.42)
        np.testing.assert_allclose(x0_42 - x0_40, 0.02, atol=1e-12)


class TestChannelOrdering:
    def test_italy_first(self, synthetic_data, loadings, baseline_shocks):
        """Italy channel must be first in baseline scenario ordering."""
        S, bank_names, bucket_names = synthetic_data
        a, _, _ = loadings
        order = scenario_channel_order(baseline_shocks, bucket_names, a, S)
        assert bucket_names[order[0]] == 'IT', (
            f'Expected IT first, got {bucket_names[order[0]]}'
        )

    def test_france_second(self, synthetic_data, loadings, baseline_shocks):
        """France channel must be second (largest non-IT shock weight)."""
        S, bank_names, bucket_names = synthetic_data
        a, _, _ = loadings
        order = scenario_channel_order(baseline_shocks, bucket_names, a, S)
        assert bucket_names[order[1]] == 'FR', (
            f'Expected FR second, got {bucket_names[order[1]]}'
        )

    def test_shocked_countries_precede_others(self, synthetic_data, loadings, baseline_shocks):
        """All shocked countries should appear before any unshocked country."""
        S, bank_names, bucket_names = synthetic_data
        a, _, _ = loadings
        order = scenario_channel_order(baseline_shocks, bucket_names, a, S)
        shocked = set(baseline_shocks.keys()) & set(bucket_names)
        shocked_positions = [i for i, k in enumerate(order) if bucket_names[k] in shocked]
        unshocked_positions = [i for i, k in enumerate(order) if bucket_names[k] not in shocked]
        assert max(shocked_positions) < min(unshocked_positions), (
            'Shocked channels do not all precede unshocked channels'
        )


class TestSVDReduction:
    def test_full_rank_svd_equals_original(self, loadings):
        """Rank-N SVD approximation should recover e_matrix exactly."""
        _, _, e = loadings
        N = e.shape[0]
        E_N = compute_svd_reduction(e, q=N)
        np.testing.assert_allclose(E_N, e, atol=1e-8,
                                   err_msg='Full-rank SVD does not recover original matrix')

    def test_low_rank_shape(self, loadings):
        """SVD reduction returns same shape as input for any q."""
        _, _, e = loadings
        N = e.shape[0]
        for q in [1, 5, 10]:
            E_q = compute_svd_reduction(e, q)
            assert E_q.shape == (N, N), f'Wrong shape for q={q}: {E_q.shape}'

    def test_rank_increases_accuracy(self, loadings):
        """Higher rank approximation should have smaller Frobenius error."""
        _, _, e = loadings
        errors = []
        for q in [2, 10, 30]:
            E_q = compute_svd_reduction(e, q)
            errors.append(np.linalg.norm(E_q - e, 'fro'))
        assert errors[0] >= errors[1] >= errors[2], (
            f'Frobenius errors not monotone: {errors}'
        )


class TestMaxPathError:
    def test_zero_error_same_input(self, loadings, synthetic_data, baseline_shocks):
        """Max path error of a trajectory vs itself should be exactly 0."""
        a, b, _ = loadings
        S, bank_names, bucket_names = synthetic_data
        x0 = build_scenario_buffers(a, baseline_shocks, bucket_names, x_star=0.40)
        result = solve_sovereign_full(a, b, 0.075, x0, T=1.0,
                                      t_eval=np.linspace(0, 1, 51))
        err = compute_max_path_error(result['X_t'], result['X_t'])
        assert err == 0.0

    def test_positive_error_different_inputs(self):
        """Max path error is positive when arrays differ."""
        X1 = np.ones((5, 10))
        X2 = np.zeros((5, 10))
        assert compute_max_path_error(X1, X2) == pytest.approx(1.0)


class TestSyntheticDataShape:
    def test_shape(self, synthetic_data):
        """Synthetic dataset must have exactly 116 banks and 38 buckets."""
        S, bank_names, bucket_names = synthetic_data
        assert S.shape == (116, 38), f'Wrong shape: {S.shape}'
        assert len(bank_names) == 116
        assert len(bucket_names) == 38

    def test_non_negative(self, synthetic_data):
        """Holdings matrix must be non-negative."""
        S, _, _ = synthetic_data
        assert np.all(S >= 0), f'Negative holdings found: min={S.min()}'

    def test_all_banks_have_positive_total(self, synthetic_data):
        """Every bank should have positive total holdings."""
        S, bank_names, _ = synthetic_data
        row_sums = S.sum(axis=1)
        assert np.all(row_sums > 0), 'Some banks have zero total holdings'

    def test_italy_bucket_exists(self, synthetic_data):
        """Bucket list must contain 'IT'."""
        _, _, bucket_names = synthetic_data
        assert 'IT' in bucket_names

    def test_france_bucket_exists(self, synthetic_data):
        """Bucket list must contain 'FR'."""
        _, _, bucket_names = synthetic_data
        assert 'FR' in bucket_names
