# Data Directory

This directory holds input data for the experiments.  
The EBA 2025 EU-wide stress test transparency exercise data is **not distributed
in this repository** and must be downloaded separately.

---

## Experiment 7 – EBA 2025 Transparency Exercise Data

### What you need

The pipeline uses sovereign-bond holdings reported by individual EU institutions
in the 2025 EBA EU-wide transparency exercise.  Specifically it filters to:

| Filter field    | Required value |
|-----------------|---------------|
| `item`          | `2520810`     |
| `period`        | `202506`      |
| `portfolio_code`| `0`           |
| `maturity_code` | `8`           |

The row of interest is labelled *"Sovereign exposures (domestic and foreign)"*
net of collateral and guarantees, at the bank-by-sovereign level.

### How to obtain the data

1. Visit the EBA transparency exercise page:  
   <https://www.eba.europa.eu/risk-and-data-analysis/transparency-exercise>

2. Download the **2025 EU-wide Transparency Exercise – Data** package
   (usually a ZIP archive containing CSV/Excel files).

3. Locate the file containing institution-level sovereign holdings.
   It typically has columns including at least:

   | Column name        | Description                                  |
   |--------------------|----------------------------------------------|
   | `institution_name` | Bank or banking group name                   |
   | `country_bucket`   | ISO 3166-1 alpha-2 country code              |
   | `holdings`         | Holdings in EUR millions                     |
   | `item`             | EBA item identifier (filter to `2520810`)    |
   | `period`           | Reporting period as YYYYMM (filter `202506`) |
   | `portfolio_code`   | Portfolio classification (filter `0`)        |
   | `maturity_code`    | Maturity bucket code (filter `8`)            |

4. Place the file in this directory as **`eba_2025.csv`** (or `eba_2025.xlsx`):

   ```
   data/
   └── eba_2025.csv      ← place the file here
   ```

5. Run the experiment:

   ```bash
   python exp/run_exp7.py
   ```

### Synthetic fallback

If the real data file is absent, `src/eba_loader.py` automatically generates a
**synthetic dataset** with the same dimensionality (116 banks × 38 sovereign
buckets) and qualitative properties:

* Italian sovereign bucket has the largest aggregate holdings.
* French sovereign bucket is second largest.
* Home-country bias patterns consistent with EBA stylised facts.
* Four Italian banks are initially distressed under the baseline shock scenario
  (`IT` 45 %, `FR` 35 %, `DE` 15 %, `NL` 5 %; `x_star` = 0.40).

The synthetic data is deterministic given the random seed (default `seed=42`).

### Expected real-data results (paper benchmark)

When using the actual EBA 2025 dataset the experiment should reproduce
approximately:

| Metric                             | Value     |
|------------------------------------|-----------|
| Banks initially distressed         | 4         |
| Banks terminally distressed        | 4         |
| Scenario q=2 max path error        | ~7.20 × 10⁻⁴ |
| SVD rank-2 max path error          | ~2.34 × 10⁻²  |
| SVD rank-12 max path error         | ~1.12 × 10⁻²  |
| SVD rank-32 max path error         | ~4.95 × 10⁻⁵  |

Deviations from these figures when using the synthetic data are expected.
