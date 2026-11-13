"""Version 2, Part 1 - Monte Carlo exposure and CVA for every netting set.

The v1 machinery was written around one prototype agreement.  This module
applies the same construction set by set:

* live trades are those maturing after the valuation date;
* one spot and one averaged volatility are kept per underlying;
* the correlation construction, asset-class mean-reversion speeds and
  monthly grid rule are unchanged;
* the prototype CSA-005 keeps seed 42 and the v1 draw order exactly;
* the other agreements, in agreement-ID order, use seeds 43, 44, ...;
* CSA threshold/MTA are the modal values on the set's collateralised live
  trades (the template's treatment of inconsistently recorded terms),
  with the smallest value breaking a tie deterministically;
* CVA uses each set's median live-trade spread, LGD 60% and a flat 4%
  risk-free rate.

Each set is simulated independently.  Portfolio figures are therefore
sums of set-level marginal EPE/CVA figures, not a joint portfolio loss
distribution; no cross-set dependence or diversification is introduced.
"""

import numpy as np
import pandas as pd

from . import collateral as collateral_mod
from . import cva as cva_mod
from . import revaluation as reval_mod
from . import simulation as sim_mod
from .config import (
    MPOR_BUSINESS_DAYS,
    N_PATHS,
    PROTOTYPE_NETTING_SET,
    RANDOM_SEED,
    VALUATION_DATE,
)


def _modal_value(values):
    """Most frequent value; smallest value wins a tie, deterministically."""
    counts = pd.Series(values).value_counts()
    top = counts.max()
    return float(counts[counts == top].index.min())


def infer_csa_terms(live_trades):
    """Infer one agreement-level threshold/MTA pair from collateralised trades."""
    flagged = live_trades[
        live_trades["CSA_Flag"].astype(bool)
        & live_trades["Threshold"].notna()
        & live_trades["MTA"].notna()
    ]
    if flagged.empty:
        raise ValueError("No collateralised live trades from which to infer CSA terms.")
    return _modal_value(flagged["Threshold"]), _modal_value(flagged["MTA"])


def seed_for_set(netting_set, all_sets):
    """Prototype seed 42; other sets take 43, 44, ... in agreement order."""
    if netting_set == PROTOTYPE_NETTING_SET:
        return RANDOM_SEED
    others = [name for name in sorted(all_sets) if name != PROTOTYPE_NETTING_SET]
    return RANDOM_SEED + 1 + others.index(netting_set)


def simulate_netting_set(
    live_trades,
    *,
    n_paths=N_PATHS,
    seed=RANDOM_SEED,
    threshold,
    mta,
    mpor_days=MPOR_BUSINESS_DAYS,
    correlation_shift=0.0,
    risk_factors=None,
):
    """Run paths, revaluation, collateral and flat CVA for one set."""
    live = live_trades.copy()
    factors = (
        sim_mod.build_risk_factors(live)
        if risk_factors is None
        else risk_factors.copy()
    )
    horizon_end = pd.Timestamp(live["Maturity_Date"].max())
    grid = sim_mod.build_time_grid(VALUATION_DATE, horizon_end)
    t_years = sim_mod.years_from(grid, VALUATION_DATE)
    paths = sim_mod.simulate_paths(
        factors,
        t_years,
        n_paths=n_paths,
        seed=seed,
        correlation_shift=correlation_shift,
    )

    trade_values, _methods, t0_check = reval_mod.revalue_trades(
        live, paths, factors, grid
    )
    if not t0_check["Match"].all():
        raise AssertionError("t=0 revaluation failed for a full-book netting set.")
    exposure = np.maximum(trade_values.sum(axis=2), 0.0)
    profile_uncollat, epe_uncollat = reval_mod.summarise_exposure(
        exposure, grid, t_years
    )

    held, exposure_collat = collateral_mod.simulate_collateral(
        exposure,
        t_years,
        threshold=threshold,
        mta=mta,
        mpor_days=mpor_days,
    )
    profile_collat, epe_collat = reval_mod.summarise_exposure(
        exposure_collat, grid, t_years
    )

    spread_stats = cva_mod.spread_statistics(live)
    anchor = spread_stats["median_bps"]
    cva_uncollat = cva_mod.flat_cva(
        profile_uncollat["EE"].to_numpy(dtype=float), t_years, anchor
    )["cva"]
    cva_collat = cva_mod.flat_cva(
        profile_collat["EE"].to_numpy(dtype=float), t_years, anchor
    )["cva"]

    return {
        "live_trades": live,
        "risk_factors": factors,
        "grid": grid,
        "t_years": t_years,
        "paths": paths,
        "trade_values": trade_values,
        "t0_check": t0_check,
        "exposure": exposure,
        "profile_uncollat": profile_uncollat,
        "epe_uncollat": epe_uncollat,
        "collateral_held": held,
        "exposure_collat": exposure_collat,
        "profile_collat": profile_collat,
        "epe_collat": epe_collat,
        "spread_stats": spread_stats,
        "cva_uncollat": cva_uncollat,
        "cva_collat": cva_collat,
        # The valuation-date balance is path-invariant: at t=0 every path
        # has the same net MTM, hence the same day-one required balance.
        "vm_at_valuation": float(held[0, 0]),
    }


def run_full_book(live_book, n_paths=N_PATHS):
    """Simulate all agreements and build set / counterparty summaries."""
    all_sets = sorted(live_book["Netting_Agreement_ID"].unique())
    rows = []
    details = {}

    for netting_set in all_sets:
        live = live_book[live_book["Netting_Agreement_ID"] == netting_set].copy()
        counterparty = str(live["Counterparty"].iloc[0])
        if live["Counterparty"].nunique() != 1:
            raise ValueError(f"{netting_set} maps to more than one counterparty.")
        threshold, mta = infer_csa_terms(live)
        seed = seed_for_set(netting_set, all_sets)
        result = simulate_netting_set(
            live,
            n_paths=n_paths,
            seed=seed,
            threshold=threshold,
            mta=mta,
        )
        details[netting_set] = {**result, "threshold": threshold, "mta": mta, "seed": seed}

        profile_u = result["profile_uncollat"]
        profile_c = result["profile_collat"]
        rows.append(
            {
                "Counterparty": counterparty,
                "Netting_Agreement_ID": netting_set,
                "N_Trades": int(len(live)),
                "N_Risk_Factors": int(len(result["risk_factors"])),
                "Horizon_Years": float(result["t_years"][-1]),
                "Seed": seed,
                "Threshold": threshold,
                "MTA": mta,
                "MPOR_Business_Days": MPOR_BUSINESS_DAYS,
                "Credit_Spread_bps": result["spread_stats"]["median_bps"],
                "EPE_Uncollateralized": result["epe_uncollat"],
                "EPE_Collateralized": result["epe_collat"],
                "Peak_PFE95_Uncollateralized": float(profile_u["PFE_95"].max()),
                "Peak_PFE95_Collateralized": float(profile_c["PFE_95"].max()),
                "CVA_Uncollateralized": result["cva_uncollat"],
                "CVA_Collateralized": result["cva_collat"],
                "VM_At_Valuation": result["vm_at_valuation"],
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        "CVA_Collateralized", ascending=False
    ).reset_index(drop=True)

    counterparty = (
        summary.groupby("Counterparty")
        .agg(
            N_Netting_Sets=("Netting_Agreement_ID", "nunique"),
            N_Trades=("N_Trades", "sum"),
            EPE_Uncollateralized=("EPE_Uncollateralized", "sum"),
            EPE_Collateralized=("EPE_Collateralized", "sum"),
            CVA_Uncollateralized=("CVA_Uncollateralized", "sum"),
            CVA_Collateralized=("CVA_Collateralized", "sum"),
        )
        .reset_index()
        .sort_values("CVA_Collateralized", ascending=False)
    )
    total_collat = counterparty["CVA_Collateralized"].sum()
    counterparty["CVA_Collateralized_Share_Pct"] = (
        100.0 * counterparty["CVA_Collateralized"] / total_collat
        if total_collat > 0
        else 0.0
    )
    return {"summary": summary, "counterparty": counterparty, "details": details}
