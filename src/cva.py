"""Part 5 - credit valuation adjustment from an expected-exposure profile.

The counterparty spread proxy is the median spread across the prototype
set's live trades, with the interquartile spreads kept as sensitivity
scenarios.  A flat hazard rate follows from the credit triangle
(hazard = spread / LGD), survival decays exponentially, and CVA is the
LGD-weighted sum of discounted expected exposure times marginal default
probability, period by period.
"""

import numpy as np
import pandas as pd

from .config import LGD, RISK_FREE_RATE


def spread_statistics(live_trades):
    """Median spread plus the 25th / 75th percentile sensitivity anchors."""
    spreads = pd.to_numeric(
        live_trades["Counterparty_Credit_Spread_bps"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(spreads).all() or (spreads <= 0).any():
        raise ValueError("Live-trade spreads must be positive and finite.")
    return {
        "median_bps": float(np.median(spreads)),
        "p25_bps": float(np.percentile(spreads, 25)),
        "p75_bps": float(np.percentile(spreads, 75)),
        "n_trades": int(len(spreads)),
    }


def flat_cva(ee_profile, t_years, spread_bps, lgd=LGD, risk_free=RISK_FREE_RATE):
    """Flat-hazard CVA of one EE profile, with its period-level pieces."""
    hazard = (spread_bps / 10_000.0) / lgd
    survival = np.exp(-hazard * t_years)
    marginal_pd = survival[:-1] - survival[1:]
    mean_ee = 0.5 * (ee_profile[:-1] + ee_profile[1:])
    mid_times = 0.5 * (t_years[:-1] + t_years[1:])
    discount = np.exp(-risk_free * mid_times)
    pieces = lgd * mean_ee * discount * marginal_pd
    return {
        "cva": float(pieces.sum()),
        "hazard_rate": float(hazard),
        "survival": survival,
        "marginal_pd": marginal_pd,
        "mean_ee": mean_ee,
        "discount_factor": discount,
        "contribution": pieces,
    }


def yearly_breakdown(grid, t_years, result):
    """Aggregate the period contributions of a flat-CVA result by year."""
    period_year = np.maximum(np.ceil(t_years[1:]).astype(int), 1)
    frame = pd.DataFrame(
        {
            "Start_Date": grid[:-1].values,
            "End_Date": grid[1:].values,
            "Year": period_year,
            "Avg_EE": result["mean_ee"],
            "Discount_Factor": result["discount_factor"],
            "Marginal_PD": result["marginal_pd"],
            "CVA_Contribution": result["contribution"],
        }
    )
    yearly = frame.groupby("Year").agg(
        Avg_EE=("Avg_EE", "mean"),
        Marginal_PD=("Marginal_PD", "sum"),
        CVA_Contribution=("CVA_Contribution", "sum"),
    )
    total = result["cva"]
    yearly["Pct_of_CVA"] = 100.0 * yearly["CVA_Contribution"] / total if total > 0 else 0.0
    return frame, yearly
