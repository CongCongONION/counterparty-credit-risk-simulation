"""Version 2, Part 2 - two SA-CCR refinements omitted from v1.

1. Interest-rate maturity buckets.  The maturity-bucket structure follows
   CRE52.53: trades are placed in buckets below one year, from one to five
   years, and above five years according to remaining maturity.  For this
   educational v2 implementation the bucket effective notionals are then
   combined with the specified CRE52-style quadratic supervisory
   correlation aggregation using a uniform pairwise rho of 50%:

       AddOn = sqrt( sum_b D_b^2 + 2 rho sum_{b<k} D_b D_k )

   where D_b is the sum of the v1 trade add-ons in bucket b (those trade
   add-ons already contain the supervisory factor, delta, supervisory
   duration and maturity factor).  CRE52.54-52.55 supply the effective
   notional / add-on architecture; the uniform 50% pairwise correlation
   is the documented simplification adopted here rather than a claim
   that Basel prescribes identical pairwise bucket correlations.

2. Collateralised replacement cost.  For each CSA agreement,

       RC_collat = max(V - VM, TH + MTA - NICA, 0),  NICA = 0,

   in the simplified CRE52 form.  VM is the valuation-date balance from
   the Monte Carlo collateral model: every path starts with the day-one
   required balance max(RC - TH, 0), so that balance is path-invariant.
   The PFE multiplier for the collateralised variant is evaluated at
   V - VM, keeping the multiplier's value input consistent with the
   collateralised replacement-cost calculation.
"""

import numpy as np
import pandas as pd

CROSS_BUCKET_RHO = 0.50
BUCKETS = ("<1y", "1-5y", ">5y")


def maturity_bucket(tenor_years):
    if tenor_years < 1.0:
        return "<1y"
    if tenor_years <= 5.0:
        return "1-5y"
    return ">5y"


def _aggregate_buckets(bucket_amounts):
    amounts = np.array([bucket_amounts.get(name, 0.0) for name in BUCKETS])
    quadratic = float(amounts @ amounts)
    for i in range(len(amounts)):
        for j in range(i + 1, len(amounts)):
            quadratic += 2.0 * CROSS_BUCKET_RHO * amounts[i] * amounts[j]
    return float(np.sqrt(max(quadratic, 0.0)))


def rates_bucket_table(enriched):
    """Per-currency IR bucket amounts with v1 and v2 hedging-set add-ons."""
    rates = enriched[enriched["Asset_Class"] == "Rates"].copy()
    rates["Maturity_Bucket"] = rates["T"].map(maturity_bucket)
    rows = []
    group_cols = ["Counterparty", "Netting_Agreement_ID", "Currency"]
    for (counterparty, agreement, currency), group in rates.groupby(group_cols):
        bucket_amounts = group.groupby("Maturity_Bucket")["Trade_AddOn"].sum().to_dict()
        v2 = _aggregate_buckets(bucket_amounts)
        rows.append(
            {
                "Counterparty": counterparty,
                "Netting_Agreement_ID": agreement,
                "Currency": currency,
                "Bucket_LT_1Y_AddOn": bucket_amounts.get("<1y", 0.0),
                "Bucket_1_5Y_AddOn": bucket_amounts.get("1-5y", 0.0),
                "Bucket_GT_5Y_AddOn": bucket_amounts.get(">5y", 0.0),
                "IR_AddOn_v1": abs(float(group["Trade_AddOn"].sum())),
                "IR_AddOn_v2_Bucketed": v2,
                "IR_Bucket_Gross_AddOn": float(
                    sum(abs(value) for value in bucket_amounts.values())
                ),
            }
        )
    return pd.DataFrame(rows)


def _multiplier(value, addon):
    if addon <= 0:
        return 1.0
    return float(min(1.0, 0.05 + 0.95 * np.exp(value / (2 * 0.95 * addon))))


def build_saccr_v2(bundle, fullbook):
    """Build the per-set v1 / v2 / v2-plus-collateral-RC comparison."""
    ead = bundle["ead"].reset_index()
    hedging = bundle["hedging"]
    bucket_table = rates_bucket_table(bundle["enriched"])

    # v1 non-IR add-on per set (all hedging sets except Rates).
    non_ir = (
        hedging[hedging["Asset_Class"] != "Rates"]
        .groupby("Netting_Agreement_ID")["HedgingSet_AddOn"]
        .sum()
    )
    ir_v2 = bucket_table.groupby("Netting_Agreement_ID")["IR_AddOn_v2_Bucketed"].sum()
    ir_gross = bucket_table.groupby("Netting_Agreement_ID")["IR_Bucket_Gross_AddOn"].sum()

    details = fullbook["details"]
    rows = []
    for _, base in ead.iterrows():
        agreement = base["Netting_Agreement_ID"]
        addon_v1 = float(base["Total_AddOn"])
        addon_v2 = float(non_ir.get(agreement, 0.0) + ir_v2.get(agreement, 0.0))
        value = float(base["Net_MTM"])
        rc = float(base["RC"])
        vm = float(details[agreement]["vm_at_valuation"])
        threshold = float(details[agreement]["threshold"])
        mta = float(details[agreement]["mta"])

        mult_v1 = _multiplier(value, addon_v1)
        pfe_v1 = mult_v1 * addon_v1
        ead_v1 = 1.4 * (rc + pfe_v1)

        mult_v2 = _multiplier(value, addon_v2)
        pfe_v2 = mult_v2 * addon_v2
        ead_v2 = 1.4 * (rc + pfe_v2)

        rc_collat = max(value - vm, threshold + mta, 0.0)
        mult_collat = _multiplier(value - vm, addon_v2)
        pfe_collat = mult_collat * addon_v2
        ead_collat = 1.4 * (rc_collat + pfe_collat)

        rows.append(
            {
                "Counterparty": base["Counterparty"],
                "Netting_Agreement_ID": agreement,
                "Net_MTM": value,
                "AddOn_v1": addon_v1,
                "AddOn_v2_Bucketed": addon_v2,
                "IR_Bucket_Gross_AddOn": float(ir_gross.get(agreement, 0.0)),
                "RC": rc,
                "VM_At_Valuation": vm,
                "Threshold": threshold,
                "MTA": mta,
                "RC_Collateralized": rc_collat,
                "Multiplier_v1": mult_v1,
                "PFE_v1": pfe_v1,
                "EAD_v1": ead_v1,
                "Multiplier_v2": mult_v2,
                "PFE_v2": pfe_v2,
                "EAD_v2": ead_v2,
                "Multiplier_v2_Collateralized_RC": mult_collat,
                "PFE_v2_Collateralized_RC": pfe_collat,
                "EAD_v2_Collateralized_RC": ead_collat,
                "EAD_v2_Minus_v1": ead_v2 - ead_v1,
                "EAD_Collat_Minus_v2": ead_collat - ead_v2,
            }
        )

    comparison = pd.DataFrame(rows).sort_values("EAD_v1", ascending=False).reset_index(drop=True)
    return {"comparison": comparison, "rates_buckets": bucket_table}
