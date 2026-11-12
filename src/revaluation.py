"""Part 3c - revaluing the prototype trades along the simulated paths.

Three deliberately simple pricers are used, each calibrated so that the
value at the valuation date reproduces the trade's reported MTM exactly:

* linear underlyings (equity, FX, commodity): a forward-style position
  whose contract level is backed out of the reported MTM;
* rates and credit: an annuity-style position whose sensitivity scales
  with remaining maturity and whose contract level is likewise implied;
* the swaption: an undiscounted Black-76 call whose position size is
  scaled to the reported MTM.

Trades contribute nothing after their maturity date.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

MTM_COL = "MTM (RC)"


def black76_call(forward, strike, vol, tau):
    """Undiscounted Black-76 call value; intrinsic value once tau runs out."""
    forward = np.asarray(forward, dtype=float)
    if tau <= 0:
        return np.maximum(forward - strike, 0.0)
    width = vol * np.sqrt(tau)
    d1 = (np.log(forward / strike) + 0.5 * vol**2 * tau) / width
    d2 = d1 - width
    return forward * norm.cdf(d1) - strike * norm.cdf(d2)


def _level_scale(asset_class, spot):
    """Bring a quoted rate / spread level into decimal units by magnitude."""
    if asset_class == "Rates":
        return 0.01 if abs(spot) > 1.0 else 1.0
    if asset_class == "Credit":
        if abs(spot) > 10.0:
            return 0.0001
        if abs(spot) >= 1.0:
            return 0.01
    return 1.0


def _years_remaining(maturity, grid):
    return np.maximum(np.array([(maturity - d).days / 365.0 for d in grid]), 0.0)


def revalue_trades(live_trades, paths, risk_factors, grid):
    """Return (trade values array, method labels, t=0 check frame).

    The value array has shape (n_paths, n_dates, n_trades) and follows the
    row order of ``live_trades``.
    """
    factors = risk_factors.index.tolist()
    n_paths, n_dates, _ = paths.shape
    values = np.zeros((n_paths, n_dates, len(live_trades)))
    methods = []

    for col, (_, trade) in enumerate(live_trades.iterrows()):
        asset = trade["Asset_Class"]
        sign = 1.0 if trade["Direction"] == "Long" else -1.0
        notional = float(trade["Notional_Amount"])
        mtm_today = float(trade[MTM_COL])
        factor_pos = factors.index(trade["Underlying_Risk_Factor"])
        start_level = float(risk_factors.loc[trade["Underlying_Risk_Factor"], "Spot"])
        level_path = paths[:, :, factor_pos]
        still_alive = np.asarray(grid <= trade["Maturity_Date"])
        years_left = _years_remaining(trade["Maturity_Date"], grid)

        if bool(trade["Is_Option"]):
            strike = float(trade["Strike_Price"])
            vol = float(trade["Volatility"])
            opening_price = black76_call(start_level, strike, vol, years_left[0])
            position_scale = mtm_today / (sign * opening_price)
            priced = np.zeros_like(level_path)
            for t_pos, tau in enumerate(years_left):
                priced[:, t_pos] = sign * position_scale * black76_call(
                    level_path[:, t_pos], strike, vol, tau
                )
            label = "Black-76-style option"
        elif asset in ("Rates", "Credit"):
            scale = _level_scale(asset, start_level)
            level_today = start_level * scale
            path_levels = level_path * scale
            opening_annuity = years_left[0]
            contract_level = level_today - mtm_today / (
                sign * notional * opening_annuity
            )
            priced = (
                sign
                * notional
                * years_left[np.newaxis, :]
                * (path_levels - contract_level)
            )
            label = "Annuity-based swap" if asset == "Rates" else "Annuity-based CDS"
        else:
            units = notional / start_level
            contract_level = start_level - mtm_today / (sign * units)
            priced = sign * units * (level_path - contract_level)
            label = "Forward-style linear"

        values[:, :, col] = np.where(still_alive[np.newaxis, :], priced, 0.0)
        methods.append(label)

    if not np.isfinite(values).all():
        raise ValueError("Revaluation produced non-finite values.")

    check = pd.DataFrame(
        {
            "Trade_ID": live_trades["Trade_ID"].to_numpy(),
            "Asset_Class": live_trades["Asset_Class"].to_numpy(),
            "Method": methods,
            "MTM_RC": live_trades[MTM_COL].to_numpy(dtype=float),
            "Simulated_t0": values[0, 0, :],
        }
    )
    check["Match"] = np.isclose(check["MTM_RC"], check["Simulated_t0"], atol=1.0)
    return values, methods, check


def summarise_exposure(exposure, grid, t_years):
    """EE / PFE profile and the time-weighted EPE of an exposure array."""
    ee = exposure.mean(axis=0)
    profile = pd.DataFrame(
        {
            "Date": grid.values,
            "t_years": t_years,
            "EE": ee,
            "PFE_95": np.percentile(exposure, 95, axis=0),
            "PFE_99": np.percentile(exposure, 99, axis=0),
        }
    )
    epe = float(np.trapezoid(ee, t_years) / t_years[-1])
    return profile, epe
