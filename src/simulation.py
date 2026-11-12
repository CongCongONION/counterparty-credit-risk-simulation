"""Part 3a/3b - correlated Monte Carlo paths for the prototype netting set.

One starting level and one volatility are kept per underlying.  Paths are
built in log space on a monthly grid: equity and FX follow driftless
geometric Brownian motion, while rates, commodity and credit factors use
the exact Ornstein-Uhlenbeck transition so they revert towards their own
starting level.  A single Cholesky factor of the target correlation
matrix correlates the shocks, and the generator is consumed step by step
in a fixed order so a given seed always reproduces the same paths.
"""

import numpy as np
import pandas as pd

from .config import N_PATHS, RANDOM_SEED

# Mean-reversion speed per asset class; zero keeps the factor a pure GBM.
KAPPA_BY_ASSET_CLASS = {
    "Rates": 1.0,
    "Commodity": 0.3,
    "Credit": 0.3,
    "Equity": 0.0,
    "FX": 0.0,
}

GENERIC_CROSS_CORR = 0.10


def build_risk_factors(live_trades):
    """Collapse live trades to one spot / averaged vol per underlying."""
    per_factor_class = live_trades.groupby("Underlying_Risk_Factor")["Asset_Class"].nunique()
    if (per_factor_class > 1).any():
        raise ValueError("An underlying maps to more than one asset class.")
    per_factor_spot = live_trades.groupby("Underlying_Risk_Factor")["Spot_Price"].nunique()
    if (per_factor_spot > 1).any():
        raise ValueError("An underlying has inconsistent starting levels.")

    table = live_trades.groupby("Underlying_Risk_Factor").agg(
        Asset_Class=("Asset_Class", "first"),
        Spot=("Spot_Price", "first"),
        Vol=("Volatility", "mean"),
        N_Trades=("Trade_ID", "count"),
    )
    table = table.sort_values("Asset_Class", kind="stable")
    if (table["Spot"] <= 0).any() or (table["Vol"] <= 0).any():
        raise ValueError("Log-space simulation needs positive spots and vols.")
    return table


def build_time_grid(valuation_date, horizon_end):
    """Monthly dates anchored on the valuation date, ending at the horizon."""
    grid = pd.date_range(
        start=valuation_date, end=horizon_end, freq=pd.DateOffset(months=1)
    )
    if grid[-1] < horizon_end:
        grid = grid.append(pd.DatetimeIndex([horizon_end]))
    return grid.drop_duplicates().sort_values()


def years_from(grid, valuation_date):
    return np.array([(d - valuation_date).days / 365.0 for d in grid])


def build_correlation(factors):
    """Stylised target correlation: small generic co-movement, two overrides."""
    n = len(factors)
    corr = np.full((n, n), GENERIC_CROSS_CORR)
    np.fill_diagonal(corr, 1.0)

    def override(first, second, value):
        if first in factors and second in factors:
            i, j = factors.index(first), factors.index(second)
            corr[i, j] = corr[j, i] = value

    override("WTI_CRUDE", "GOLD", 0.30)     # two commodity underlyings move together
    override("SPX", "ITRAXX.EU", -0.50)     # equity down / credit wider: the WWR link
    if np.linalg.eigvalsh(corr).min() <= 0:
        raise ValueError("Target correlation matrix is not positive definite.")
    return corr


def simulate_paths(risk_factors, t_years, n_paths=N_PATHS, seed=RANDOM_SEED):
    """Simulate level paths with shape (n_paths, n_dates, n_factors)."""
    factors = risk_factors.index.tolist()
    spots = risk_factors.loc[factors, "Spot"].to_numpy(dtype=float)
    vols = risk_factors.loc[factors, "Vol"].to_numpy(dtype=float)
    kappas = (
        risk_factors.loc[factors, "Asset_Class"]
        .map(KAPPA_BY_ASSET_CLASS)
        .to_numpy(dtype=float)
    )
    if np.isnan(kappas).any():
        raise ValueError("An asset class has no mean-reversion speed assigned.")

    corr = build_correlation(factors)
    chol = np.linalg.cholesky(corr)

    n_steps = len(t_years) - 1
    log_start = np.log(spots)
    log_paths = np.zeros((n_paths, n_steps + 1, len(factors)))
    log_paths[:, 0, :] = log_start

    reverting = kappas > 0
    safe_kappa = np.where(reverting, kappas, 1.0)  # placeholder where GBM applies

    rng = np.random.default_rng(seed)
    for step in range(n_steps):
        dt = t_years[step + 1] - t_years[step]
        if dt <= 0:
            raise ValueError("Time steps must be positive.")
        normals = rng.standard_normal((n_paths, len(factors)))
        shocks = normals @ chol.T
        current = log_paths[:, step, :]

        gbm_centre = current - 0.5 * vols**2 * dt
        gbm_scale = vols * np.sqrt(dt)

        ou_centre = log_start + (current - log_start) * np.exp(-safe_kappa * dt)
        ou_var = (vols**2) / (2 * safe_kappa) * (1.0 - np.exp(-2 * safe_kappa * dt))
        ou_scale = np.sqrt(ou_var)

        centre = np.where(reverting, ou_centre, gbm_centre)
        scale = np.where(reverting, ou_scale, gbm_scale)
        log_paths[:, step + 1, :] = centre + scale * shocks

    return np.exp(log_paths)
