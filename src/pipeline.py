"""End-to-end orchestration: book -> SA-CCR -> simulation -> collateral
-> CVA -> wrong-way risk, returning every artefact in one bundle dict."""

import numpy as np
import pandas as pd

from . import book as book_mod
from . import collateral as collateral_mod
from . import cva as cva_mod
from . import revaluation as reval_mod
from . import saccr as saccr_mod
from . import simulation as sim_mod
from . import wwr as wwr_mod
from .config import (
    BASE_BETA,
    BETA_SCENARIOS,
    ITRAXX_FACTOR,
    LGD,
    PROTOTYPE_NETTING_SET,
    RISK_FREE_RATE,
    VALUATION_DATE,
)


def run_pipeline(data_path):
    bundle = {}

    # ---- Full book: netting (Part 1) and simplified SA-CCR (Part 2) ----
    whole_book = book_mod.load_book(data_path)
    live_book = book_mod.live_trades(whole_book, VALUATION_DATE)
    netting = book_mod.netting_summary(live_book)
    enriched = saccr_mod.add_trade_terms(live_book)
    hedging = saccr_mod.hedging_set_addons(enriched)
    ead = saccr_mod.ead_table(netting, hedging)
    bundle.update(
        whole_book=whole_book,
        live_book=live_book,
        netting=netting,
        enriched=enriched,
        hedging=hedging,
        ead=ead,
    )

    # ---- Prototype set: scope (trades still open on the valuation date) ----
    csa_book = whole_book[whole_book["Netting_Agreement_ID"] == PROTOTYPE_NETTING_SET].copy()
    csa_live = csa_book[csa_book["Maturity_Date"] >= VALUATION_DATE].copy()
    csa_live["Sign"] = csa_live["Direction"].map({"Long": 1.0, "Short": -1.0})
    bundle["csa_book"] = csa_book
    bundle["csa_live"] = csa_live

    # ---- Monte Carlo paths (Part 3a/3b) ----
    risk_factors = sim_mod.build_risk_factors(csa_live)
    horizon_end = pd.Timestamp(csa_live["Maturity_Date"].max())
    grid = sim_mod.build_time_grid(VALUATION_DATE, horizon_end)
    t_years = sim_mod.years_from(grid, VALUATION_DATE)
    paths = sim_mod.simulate_paths(risk_factors, t_years)
    bundle.update(
        risk_factors=risk_factors, grid=grid, t_years=t_years, paths=paths,
        correlation=sim_mod.build_correlation(risk_factors.index.tolist()),
    )

    # ---- Revaluation and uncollateralised exposure (Part 3c) ----
    trade_values, methods, t0_check = reval_mod.revalue_trades(
        csa_live, paths, risk_factors, grid
    )
    if not t0_check["Match"].all():
        raise AssertionError("t=0 revaluation failed to reproduce reported MTMs.")
    netted_mtm = trade_values.sum(axis=2)
    exposure = np.maximum(netted_mtm, 0.0)
    profile_uncollat, epe_uncollat = reval_mod.summarise_exposure(exposure, grid, t_years)
    bundle.update(
        trade_values=trade_values,
        methods=methods,
        t0_check=t0_check,
        netted_mtm=netted_mtm,
        exposure=exposure,
        profile_uncollat=profile_uncollat,
        epe_uncollat=epe_uncollat,
    )

    # ---- Collateral (Part 4) ----
    held, exposure_collat = collateral_mod.simulate_collateral(exposure, t_years)
    profile_collat, epe_collat = reval_mod.summarise_exposure(
        exposure_collat, grid, t_years
    )
    bundle.update(
        collateral_held=held,
        exposure_collat=exposure_collat,
        profile_collat=profile_collat,
        epe_collat=epe_collat,
    )

    # ---- CVA (Part 5) ----
    spread_stats = cva_mod.spread_statistics(csa_live)
    anchor = spread_stats["median_bps"]
    ee_collat = profile_collat["EE"].to_numpy(dtype=float)
    ee_uncollat = profile_uncollat["EE"].to_numpy(dtype=float)
    base = cva_mod.flat_cva(ee_collat, t_years, anchor)
    uncollat = cva_mod.flat_cva(ee_uncollat, t_years, anchor)
    low = cva_mod.flat_cva(ee_collat, t_years, spread_stats["p25_bps"])
    high = cva_mod.flat_cva(ee_collat, t_years, spread_stats["p75_bps"])
    breakdown, yearly = cva_mod.yearly_breakdown(grid, t_years, base)
    bundle.update(
        spread_stats=spread_stats,
        cva_base=base,
        cva_collat=base["cva"],
        cva_uncollat=uncollat["cva"],
        cva_low=low["cva"],
        cva_high=high["cva"],
        cva_breakdown=breakdown,
        cva_yearly=yearly,
    )

    # ---- Wrong-way risk (Part 6) ----
    factors = risk_factors.index.tolist()
    index_pos = factors.index(ITRAXX_FACTOR)
    index_spot = float(risk_factors.loc[ITRAXX_FACTOR, "Spot"])
    index_ratio = paths[:, :, index_pos] / index_spot

    flat_check = wwr_mod.joint_and_independent_cva(
        exposure_collat, index_ratio, t_years, anchor, beta=0.0
    )
    beta_runs = {
        beta: wwr_mod.joint_and_independent_cva(
            exposure_collat, index_ratio, t_years, anchor, beta=beta
        )
        for beta in BETA_SCENARIOS
    }
    base_run = beta_runs[BASE_BETA]
    uncollat_run = wwr_mod.joint_and_independent_cva(
        exposure, index_ratio, t_years, anchor, beta=BASE_BETA
    )
    terciles, corr_collat, _spreads, _epes = wwr_mod.tercile_evidence(
        base_run["spread_path"], exposure_collat, t_years
    )
    _t2, corr_uncollat, _s2, _e2 = wwr_mod.tercile_evidence(
        base_run["spread_path"], exposure, t_years
    )
    bundle.update(
        index_ratio=index_ratio,
        wwr_flat= flat_check,
        wwr_beta_runs=beta_runs,
        wwr_joint=base_run["joint"],
        wwr_independent=base_run["independent"],
        wwr_addon=base_run["joint"] - base_run["independent"],
        wwr_joint_uncollat=uncollat_run["joint"],
        wwr_independent_uncollat=uncollat_run["independent"],
        wwr_addon_uncollat=uncollat_run["joint"] - uncollat_run["independent"],
        wwr_terciles=terciles,
        wwr_corr=corr_collat,
        wwr_corr_uncollat=corr_uncollat,
    )
    return bundle


def headline_metrics(bundle):
    """Flat dict of the numbers the report and reconciliation care about."""
    netting = bundle["netting"]
    ead = bundle["ead"]
    csa_row = ead.xs(PROTOTYPE_NETTING_SET, level="Netting_Agreement_ID")
    return {
        "n_trades_total": int(len(bundle["whole_book"])),
        "n_trades_live": int(len(bundle["live_book"])),
        "n_netting_sets": int(len(netting)),
        "n_counterparties": int(bundle["live_book"]["Counterparty"].nunique()),
        "gross_exposure": float(netting["Gross_Exposure"].sum()),
        "replacement_cost": float(netting["RC"].sum()),
        "netting_benefit_pct": float(
            100.0
            * (netting["Gross_Exposure"].sum() - netting["RC"].sum())
            / netting["Gross_Exposure"].sum()
        ),
        "ead_total": float(ead["EAD"].sum()),
        "ead_csa005": float(csa_row["EAD"].iloc[0]),
        "csa_live_trades": int(len(bundle["csa_live"])),
        "epe_uncollat": float(bundle["epe_uncollat"]),
        "epe_collat": float(bundle["epe_collat"]),
        "epe_reduction_pct": float(
            100.0 * (1.0 - bundle["epe_collat"] / bundle["epe_uncollat"])
        ),
        "spread_median_bps": float(bundle["spread_stats"]["median_bps"]),
        "spread_p25_bps": float(bundle["spread_stats"]["p25_bps"]),
        "spread_p75_bps": float(bundle["spread_stats"]["p75_bps"]),
        "cva_collat": float(bundle["cva_collat"]),
        "cva_uncollat": float(bundle["cva_uncollat"]),
        "cva_low": float(bundle["cva_low"]),
        "cva_high": float(bundle["cva_high"]),
        "wwr_independent": float(bundle["wwr_independent"]),
        "wwr_joint": float(bundle["wwr_joint"]),
        "wwr_addon": float(bundle["wwr_addon"]),
        "wwr_addon_pct": float(
            100.0 * bundle["wwr_addon"] / bundle["wwr_independent"]
        ),
        "wwr_beta_0_5": float(bundle["wwr_beta_runs"][0.5]["joint"]),
        "wwr_beta_1_0": float(bundle["wwr_beta_runs"][1.0]["joint"]),
        "wwr_beta_1_5": float(bundle["wwr_beta_runs"][1.5]["joint"]),
        "wwr_addon_uncollat": float(bundle["wwr_addon_uncollat"]),
        "wwr_addon_uncollat_pct": float(
            100.0
            * bundle["wwr_addon_uncollat"]
            / bundle["wwr_independent_uncollat"]
        ),
        "lgd": LGD,
        "risk_free_rate": RISK_FREE_RATE,
    }
