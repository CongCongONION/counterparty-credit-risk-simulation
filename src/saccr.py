"""Part 2 - simplified SA-CCR for the whole book.

For every live trade an add-on is built as

    effective notional x supervisory delta x supervisory factor x maturity factor

where the effective notional carries a supervisory duration for rates and
credit trades.  Trade add-ons are then combined inside hedging sets (with
Basel supervisory correlations for equity, credit and commodity), summed
to the netting set, scaled by the PFE multiplier, and finally

    EAD = 1.4 x (RC + PFE).

Collateral is deliberately ignored at this stage, matching the reference
methodology; it enters later through the Monte Carlo collateral model.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import VALUATION_DATE

MPOR_YEARS = 10 / 250  # ten business days expressed in years

# Supervisory correlations applied inside a hedging set.
HEDGING_CORR = {"Commodity": 0.40, "Equity": 0.80, "Credit": 0.80}

RATE_OR_CREDIT = ("Rates", "Credit")


def _option_kind(category):
    return "Put" if "Put" in str(category) else "Call"


def _supervisory_delta(row):
    """Signed supervisory delta: +/-1 for linear trades, Black-style for options."""
    if not bool(row["Is_Option"]):
        return 1.0 if row["Direction"] == "Long" else -1.0
    spot = float(row["Spot_Price"])
    strike = float(row["Strike_Price"])
    vol = float(row["Volatility"])
    tenor = float(row["T"])
    d1 = (np.log(spot / strike) + 0.5 * vol**2 * tenor) / (vol * np.sqrt(tenor))
    if row["Option_Type"] == "Call":
        unit = norm.cdf(d1)
    else:
        unit = -norm.cdf(-d1)
    return unit if row["Direction"] == "Long" else -unit


def _supervisory_factor(row):
    asset = row["Asset_Class"]
    if asset == "Rates":
        return 0.005
    if asset == "FX":
        return 0.04
    if asset == "Credit":
        return 0.0038 if row["Underlying_Risk_Factor"] in ("CDX.IG", "ITRAXX.EU") else 0.0106
    if asset == "Equity":
        return 0.20
    if asset == "Commodity":
        return 0.18
    raise ValueError(f"No supervisory factor for asset class {asset!r}")


def _hedging_set_key(row):
    if row["Asset_Class"] == "FX":
        return row["Underlying_Risk_Factor"]
    if row["Asset_Class"] == "Rates":
        return row["Currency"]
    return row["Asset_Class"]


def add_trade_terms(live):
    """Attach T, delta, duration, factor, maturity factor and trade add-on."""
    out = live.copy()
    out["T"] = (out["Maturity_Date"] - VALUATION_DATE).dt.days / 365.0
    if (out["T"] <= 0).any():
        raise ValueError("A matured trade leaked into the SA-CCR population.")
    out["Option_Type"] = out["Category"].map(_option_kind)
    out["Delta"] = out.apply(_supervisory_delta, axis=1)

    is_duration_asset = out["Asset_Class"].isin(RATE_OR_CREDIT)
    out["SD"] = np.where(
        is_duration_asset,
        (1.0 - np.exp(-0.05 * out["T"])) / 0.05,
        np.nan,
    )
    out["SF"] = out.apply(_supervisory_factor, axis=1)

    floored_maturity = np.maximum(out["T"], MPOR_YEARS)
    out["MF"] = np.where(
        out["CSA_Flag"].astype(bool),
        1.5 * np.sqrt(MPOR_YEARS),
        np.sqrt(np.minimum(floored_maturity, 1.0)),
    )
    out["Effective_Notional"] = np.where(
        is_duration_asset,
        out["Notional_Amount"] * out["SD"],
        out["Notional_Amount"],
    )
    out["Trade_AddOn"] = (
        out["Effective_Notional"] * out["Delta"] * out["SF"] * out["MF"]
    )
    out["Hedging_Set"] = out.apply(_hedging_set_key, axis=1)
    return out


def _combine_hedging_set(group, asset_class):
    """Aggregate trade add-ons inside one hedging set.

    FX and rates hedging sets use a plain absolute sum.  For equity,
    credit and commodity the Basel form blends a systematic component
    (entity sums scaled by rho) with an idiosyncratic component.
    """
    if asset_class in ("FX", "Rates"):
        return abs(float(group["Trade_AddOn"].sum()))
    rho = HEDGING_CORR[asset_class]
    per_entity = group.groupby("Underlying_Risk_Factor")["Trade_AddOn"].sum()
    systematic = (rho * per_entity.sum()) ** 2
    idiosyncratic = (1.0 - rho**2) * float((per_entity**2).sum())
    return float(np.sqrt(systematic + idiosyncratic))


def hedging_set_addons(enriched):
    rows = []
    group_cols = ["Counterparty", "Netting_Agreement_ID", "Asset_Class", "Hedging_Set"]
    for (cpty, agreement, asset_class, hedging_set), group in enriched.groupby(group_cols):
        rows.append(
            {
                "Counterparty": cpty,
                "Netting_Agreement_ID": agreement,
                "Asset_Class": asset_class,
                "Hedging_Set": hedging_set,
                "HedgingSet_AddOn": _combine_hedging_set(group, asset_class),
            }
        )
    table = pd.DataFrame(rows)
    if (table["HedgingSet_AddOn"] < 0).any():
        raise ValueError("Hedging-set add-ons must not be negative.")
    return table


def ead_table(netting, hedging):
    """Netting-set EAD: multiplier-scaled PFE plus floored replacement cost."""
    total_addon = (
        hedging.groupby(["Counterparty", "Netting_Agreement_ID"])["HedgingSet_AddOn"]
        .sum()
        .rename("Total_AddOn")
    )
    table = netting[["Net_MTM", "RC"]].join(total_addon)
    exponent = table["Net_MTM"] / (2 * 0.95 * table["Total_AddOn"])
    table["Multiplier"] = np.minimum(1.0, 0.05 + 0.95 * np.exp(exponent))
    table["PFE"] = table["Multiplier"] * table["Total_AddOn"]
    table["EAD"] = 1.4 * (table["RC"] + table["PFE"])
    return table.sort_values("EAD", ascending=False)
