"""EBA 2025 EU-wide transparency data loader with synthetic data fallback.

The real dataset (item 2520810, period 202506, portfolio 0, maturity 8) can be
downloaded from https://www.eba.europa.eu/risk-and-data-analysis/transparency-exercise
and placed at data/eba_2025.csv (or .xlsx).
"""
import numpy as np
import pandas as pd
from pathlib import Path

BUCKET_NAMES = [
    'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
    'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT',
    'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK', 'AL', 'BA', 'CH',
    'ME', 'MK', 'NO', 'RS', 'TR', 'GB', 'US', 'XK',
]  # 38 EU/European sovereign buckets

# Relative market weights used when distributing non-home holdings
_SOVEREIGN_BASE_WEIGHTS = {
    'IT': 10.0, 'FR': 8.0,  'DE': 7.0,  'ES': 5.0,  'NL': 4.0,
    'BE': 3.0,  'AT': 2.5,  'GB': 3.0,  'US': 2.0,  'CH': 1.5,
    'PT': 2.0,  'GR': 1.5,  'IE': 1.5,  'SE': 2.0,  'DK': 1.5,
    'FI': 1.5,  'LU': 0.8,  'PL': 1.0,  'NO': 1.0,  'HU': 0.8,
    'CZ': 0.8,  'RO': 0.7,  'HR': 0.5,  'SK': 0.4,  'SI': 0.4,
    'BG': 0.4,  'EE': 0.2,  'LT': 0.2,  'LV': 0.2,  'CY': 0.3,
    'MT': 0.2,  'TR': 0.5,  'AL': 0.1,  'BA': 0.1,  'ME': 0.1,
    'MK': 0.1,  'RS': 0.1,  'XK': 0.05,
}

# Bank groups: (home_country, n_banks, avg_holdings_EUR_mn, conc_center, conc_half_range)
_BANK_GROUPS = [
    ('IT', 30, 1500.0, 0.76, 0.11),   # first 4 get special high-concentration treatment
    ('FR', 20, 1400.0, 0.72, 0.08),
    ('DE', 20, 1400.0, 0.66, 0.08),
    ('ES', 10, 1000.0, 0.72, 0.08),
    ('NL',  8,  900.0, 0.62, 0.08),
    ('BE',  6,  700.0, 0.62, 0.08),
    ('AT',  5,  600.0, 0.65, 0.08),
    ('PL',  4,  500.0, 0.65, 0.08),
    ('SE',  4,  500.0, 0.65, 0.08),
    ('GR',  3,  400.0, 0.72, 0.08),
    ('PT',  3,  400.0, 0.72, 0.08),
    ('IE',  2,  500.0, 0.60, 0.08),
    ('DK',  1,  600.0, 0.65, 0.08),
]


def load_eba_data(filepath=None):
    """Load EBA transparency exercise data or fall back to synthetic data.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to CSV or Excel file with columns: institution_name, country_bucket,
        holdings, item, period, portfolio_code, maturity_code.

    Returns
    -------
    S : ndarray, shape (N_banks, N_buckets)
        Holdings matrix (EUR millions).
    bank_names : list of str
    bucket_names : list of str
    """
    if filepath is not None:
        path = Path(filepath)
        if path.exists():
            df = pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_excel(path)

            df = df[df['item'] == 2520810]
            df = df[df['period'] == 202506]
            df = df[df['portfolio_code'] == 0]
            df = df[df['maturity_code'] == 8]
            df = df[df['institution_name'] != 'All other banks']

            # Replace small negative residuals (≤ EUR 1 000 mn in absolute value) with zero
            df = df.copy()
            small_neg = (df['holdings'] < 0) & (df['holdings'] >= -1000)
            df.loc[small_neg, 'holdings'] = 0.0

            pivot = df.pivot_table(
                index='institution_name', columns='country_bucket',
                values='holdings', aggfunc='sum', fill_value=0.0,
            )

            # Exclude institutions with zero total holdings
            pivot = pivot.loc[pivot.sum(axis=1) > 0]

            # Retain buckets with ≥ 0.1 % of aggregate holdings
            col_sums = pivot.sum(axis=0)
            pivot = pivot.loc[:, col_sums >= 0.001 * col_sums.sum()]

            return pivot.values.astype(float), list(pivot.index), list(pivot.columns)

    return generate_synthetic_eba_data()


def generate_synthetic_eba_data(N_banks=116, N_buckets=38, seed=42):
    """Generate synthetic EBA-like sovereign holdings matrix.

    Mimics the structure of the EBA 2025 dataset:
    - 116 banks from 13 EU countries
    - 38 sovereign buckets (ISO country codes)
    - Home-country bias with Italian bucket having the largest aggregate holdings
    - 4 Italian banks with very high IT concentration → negative initial buffers
      under the baseline shock scenario (IT 45 %, FR 35 %, DE 15 %, NL 5 %)

    Returns
    -------
    S : ndarray, shape (116, 38)
    bank_names : list of str (length 116)
    bucket_names : list of str (length 38)
    """
    rng = np.random.default_rng(seed)
    bucket_names = BUCKET_NAMES

    base_w = np.array([_SOVEREIGN_BASE_WEIGHTS[b] for b in bucket_names], dtype=float)
    base_w /= base_w.sum()

    S = np.zeros((N_banks, N_buckets))
    bank_names = []
    idx = 0

    for country, n_banks, avg_h, conc_c, conc_r in _BANK_GROUPS:
        home_k = bucket_names.index(country)
        for j in range(n_banks):
            bank_names.append(f'{country}_Bank_{j + 1:02d}')
            total_h = avg_h * rng.uniform(0.75, 1.25)

            # First 4 Italian banks: extreme home bias → guaranteed negative initial buffer
            if country == 'IT' and j < 4:
                home_share = rng.uniform(0.93, 0.97)
            else:
                raw = conc_c + rng.uniform(-conc_r, conc_r)
                home_share = float(np.clip(raw, 0.30, 0.88))

            S[idx, home_k] = home_share * total_h

            # Distribute the remainder by market weights (home bucket excluded)
            rest_w = base_w.copy()
            rest_w[home_k] = 0.0
            rest_w /= rest_w.sum()
            alpha = np.maximum(rest_w * 4.0, 1e-6)
            gamma_draws = rng.gamma(alpha)
            gamma_draws /= gamma_draws.sum()
            S[idx] += gamma_draws * (1.0 - home_share) * total_h

            idx += 1

    return S, bank_names, bucket_names
