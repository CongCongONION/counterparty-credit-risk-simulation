"""Version 2, Part 3 - convergence, stress and sensitivity analytics.

All analytics are run on the CSA-005 prototype and reuse the v1/v2 path,
revaluation and collateral machinery rather than a parallel model.
"""

import numpy as np
import pandas as pd

from . import collateral as collateral_mod
from . import cva as cva_mod
from . import fullbook as fullbook_mod
from . import revaluation as reval_mod
from .config import LGD, PROTOTYPE_NETTING_SET, RANDOM_SEED, RISK_FREE_RATE

PATH_COUNTS = (1000, 2500, 5000, 10000)
MPOR_SCENARIOS = (5, 10, 20)
SPREAD_SHIFTS_BPS = (-100.0, 0.0, 100.0, 300.0)
VOL_STRESS_MULTIPLIER = 1.5
CORRELATION_STRESS_SHIFT = 0.20


def _summary_row(label, result):
    return {
        "Scenario": label,
        "EPE_Uncollateralized": result["epe_uncollat"],
        "EPE_Collateralized": result["epe_collat"],
        "Peak_PFE95_Uncollateralized": float(
            result["profile_uncollat"]["PFE_95"].max()
        ),
        "Peak_PFE95_Collateralized": float(result["profile_collat"]["PFE_95"].max()),
        "CVA_Collateralized": result["cva_collat"],
        "CVA_Uncollateralized": result["cva_uncollat"],
    }


def run_csa_analytics(bundle):
    """Convergence, two stresses, and an MPOR x spread sensitivity grid."""
    live = bundle["csa_live"].copy()
    threshold, mta = fullbook_mod.infer_csa_terms(live)
    anchor = float(bundle["spread_stats"]["median_bps"])

    # ---- (a) Monte Carlo convergence --------------------------------
    convergence_rows = []
    convergence_results = {}
    for n_paths in PATH_COUNTS:
        result = fullbook_mod.simulate_netting_set(
            live,
            n_paths=n_paths,
            seed=RANDOM_SEED,
            threshold=threshold,
            mta=mta,
        )
        convergence_results[n_paths] = result
        convergence_rows.append(
            {
                "N_Paths": n_paths,
                "EPE_Uncollateralized": result["epe_uncollat"],
                "EPE_Collateralized": result["epe_collat"],
                "CVA_Collateralized": result["cva_collat"],
            }
        )
    convergence = pd.DataFrame(convergence_rows)
    reference = convergence[convergence["N_Paths"] == 10000].iloc[0]
    for column in ("EPE_Uncollateralized", "EPE_Collateralized", "CVA_Collateralized"):
        convergence[f"{column}_Pct_Vs_10000"] = (
            100.0 * (convergence[column] - reference[column]) / reference[column]
        )

    # The 5,000-path run is the common base for the stress comparison.
    base = convergence_results[5000]
    base_row = _summary_row("Base", base)

    # ---- (b) Stressed exposure --------------------------------------
    stressed_trades = live.copy()
    stressed_trades["Volatility"] = (
        stressed_trades["Volatility"].astype(float) * VOL_STRESS_MULTIPLIER
    )
    vol_result = fullbook_mod.simulate_netting_set(
        stressed_trades,
        n_paths=5000,
        seed=RANDOM_SEED,
        threshold=threshold,
        mta=mta,
    )
    corr_result = fullbook_mod.simulate_netting_set(
        live,
        n_paths=5000,
        seed=RANDOM_SEED,
        threshold=threshold,
        mta=mta,
        correlation_shift=CORRELATION_STRESS_SHIFT,
    )
    stress = pd.DataFrame(
        [
            base_row,
            _summary_row("Volatility x1.5", vol_result),
            _summary_row("Cross-factor correlation +0.20 (PSD repaired)", corr_result),
        ]
    )
    for column in (
        "EPE_Uncollateralized",
        "EPE_Collateralized",
        "Peak_PFE95_Uncollateralized",
        "Peak_PFE95_Collateralized",
        "CVA_Collateralized",
    ):
        stress[f"{column}_Pct_Vs_Base"] = (
            100.0 * (stress[column] - base_row[column]) / base_row[column]
        )

    # ---- (c) MPOR x spread sensitivity ------------------------------
    # The base exposure array is held fixed; only the margin lag and the
    # CVA spread change, so every cell is directly comparable.
    exposure = bundle["exposure"]
    t_years = bundle["t_years"]
    sensitivity_rows = []
    for mpor in MPOR_SCENARIOS:
        _held, exposure_collat = collateral_mod.simulate_collateral(
            exposure,
            t_years,
            threshold=threshold,
            mta=mta,
            mpor_days=mpor,
        )
        profile_collat, epe_collat = reval_mod.summarise_exposure(
            exposure_collat, bundle["grid"], t_years
        )
        ee_collat = profile_collat["EE"].to_numpy(dtype=float)
        for shift in SPREAD_SHIFTS_BPS:
            spread = max(anchor + shift, 1.0)
            cva = cva_mod.flat_cva(
                ee_collat, t_years, spread, lgd=LGD, risk_free=RISK_FREE_RATE
            )["cva"]
            sensitivity_rows.append(
                {
                    "Netting_Agreement_ID": PROTOTYPE_NETTING_SET,
                    "MPOR_Business_Days": mpor,
                    "Spread_Shift_bps": shift,
                    "Credit_Spread_bps": spread,
                    "EPE_Collateralized": epe_collat,
                    "CVA_Collateralized": cva,
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    return {
        "convergence": convergence,
        "stress": stress,
        "sensitivity": sensitivity,
        "stress_profiles": {
            "Base": base["profile_uncollat"],
            "Volatility x1.5": vol_result["profile_uncollat"],
            "Correlation +0.20": corr_result["profile_uncollat"],
        },
        "threshold": threshold,
        "mta": mta,
        "anchor_bps": anchor,
    }
