"""Reconciliation of the replication against the reference run's results.

Each check compares a quantity produced by this implementation with the
corresponding verified figure from the template project's baseline run.
Deterministic stages (netting, SA-CCR) are held to the dollar; Monte
Carlo stages share the seed and draw order with the reference, so they
are held to a tight 0.5% band and in practice land far inside it.
Structural checks (floors, ordering, internal identities) must hold
exactly.
"""

import numpy as np

from .pipeline import headline_metrics

BASELINE = {
    "n_trades_live": 95,
    "n_netting_sets": 8,
    "n_counterparties": 6,
    "gross_exposure": 117_512_366.15,
    "replacement_cost": 84_953_641.80,
    "netting_benefit_pct": 27.7,
    # Exact values from the template run itself (its log prints them
    # rounded: $478.5M total and $82.06M for CSA-005).
    "ead_total": 478_547_962.19,
    "ead_csa005": 82_064_083.32,
    "csa_live_trades": 8,
    "epe_uncollat": 9_769_587.0,
    "epe_collat": 794_518.0,
    "spread_median_bps": 264.85,
    "cva_collat": 142_759.0,
    "cva_uncollat": 1_769_585.0,
    "cva_low": 47_551.0,
    "cva_high": 184_333.0,
    "wwr_independent": 145_335.0,
    "wwr_joint": 146_252.0,
    "wwr_addon": 916.0,
    "wwr_beta_0_5": 143_828.0,
    "wwr_beta_1_5": 150_058.0,
    "wwr_addon_uncollat": 25_698.0,
}

MC_REL_TOL = 0.005  # 0.5% for Monte Carlo quantities


def _close(name, actual, expected, abs_tol=None, rel_tol=None):
    diff = abs(actual - expected)
    if abs_tol is not None:
        ok = diff <= abs_tol
        detail = f"actual={actual:,.2f} baseline={expected:,.2f} |diff|={diff:,.2f} tol={abs_tol:,.2f}"
    else:
        ok = diff <= rel_tol * abs(expected)
        detail = (
            f"actual={actual:,.2f} baseline={expected:,.2f} "
            f"rel.diff={diff / abs(expected) * 100:.4f}% tol={rel_tol * 100:.2f}%"
        )
    return (name, bool(ok), detail)


def check_bundle(bundle):
    """Run every check against a pipeline bundle; return result tuples."""
    m = headline_metrics(bundle)
    results = []

    # --- Counts -------------------------------------------------------
    results.append(_close("live trade count", m["n_trades_live"], BASELINE["n_trades_live"], abs_tol=0))
    results.append(_close("netting set count", m["n_netting_sets"], BASELINE["n_netting_sets"], abs_tol=0))
    results.append(_close("counterparty count", m["n_counterparties"], BASELINE["n_counterparties"], abs_tol=0))
    results.append(_close("CSA-005 live trades", m["csa_live_trades"], BASELINE["csa_live_trades"], abs_tol=0))

    # --- Part 1 identities ---------------------------------------------
    netting = bundle["netting"]
    identity_gap = float(
        np.abs(
            (netting["Gross_Exposure"] - netting["Netting_Benefit"]).to_numpy()
            - netting["RC"].to_numpy()
        ).max()
    )
    results.append((
        "netting identity gross - benefit = RC per set",
        identity_gap <= 0.01,
        f"max gap ${identity_gap:,.4f}",
    ))
    floor_ok = bool(((netting["RC"] - np.maximum(netting["Net_MTM"], 0.0)).abs() <= 1e-6).all())
    results.append(("RC equals max(net MTM, 0) per set", floor_ok, "floored replacement cost"))
    results.append(_close("gross exposure total", m["gross_exposure"], BASELINE["gross_exposure"], abs_tol=1.0))
    results.append(_close("replacement cost total", m["replacement_cost"], BASELINE["replacement_cost"], abs_tol=1.0))
    results.append(_close("netting benefit %", m["netting_benefit_pct"], BASELINE["netting_benefit_pct"], abs_tol=0.15))

    # --- Part 2 SA-CCR ---------------------------------------------------
    results.append(_close("SA-CCR EAD total (to the dollar)", m["ead_total"], BASELINE["ead_total"], abs_tol=1.0))
    results.append(_close("SA-CCR EAD CSA-005 (to the dollar)", m["ead_csa005"], BASELINE["ead_csa005"], abs_tol=1.0))

    # --- Part 3 revaluation ---------------------------------------------
    t0 = bundle["t0_check"]
    results.append((
        "t=0 revaluation reproduces every MTM",
        bool(t0["Match"].all()),
        f"{int(t0['Match'].sum())}/{len(t0)} trades match within $1",
    ))
    results.append(_close("uncollateralised EPE", m["epe_uncollat"], BASELINE["epe_uncollat"], rel_tol=MC_REL_TOL))
    results.append(_close("spread anchor (median bps)", m["spread_median_bps"], BASELINE["spread_median_bps"], abs_tol=0.01))

    # --- Part 4 collateral ----------------------------------------------
    exp_u, exp_c = bundle["exposure"], bundle["exposure_collat"]
    ordering_ok = bool((exp_c <= exp_u + 1e-9).all() and (exp_c >= 0).all())
    results.append(("collateralised exposure within [0, uncollateralised]", ordering_ok,
                    f"max excess ${float((exp_c - exp_u).max()):,.4f}"))
    results.append(_close("collateralised EPE", m["epe_collat"], BASELINE["epe_collat"], rel_tol=MC_REL_TOL))

    # --- Part 5 CVA -------------------------------------------------------
    for key, label in (
        ("cva_collat", "CVA collateralised"),
        ("cva_uncollat", "CVA uncollateralised"),
        ("cva_low", "CVA low-spread sensitivity"),
        ("cva_high", "CVA high-spread sensitivity"),
    ):
        results.append(_close(label, m[key], BASELINE[key], rel_tol=MC_REL_TOL))

    # --- Part 6 WWR -------------------------------------------------------
    flat = bundle["wwr_flat"]["joint"]
    results.append((
        "beta=0 reproduces flat-spread CVA",
        abs(flat - m["cva_collat"]) <= max(1.0, 1e-6 * m["cva_collat"]),
        f"beta0=${flat:,.2f} flat=${m['cva_collat']:,.2f}",
    ))
    results.append(_close("WWR independent CVA", m["wwr_independent"], BASELINE["wwr_independent"], rel_tol=MC_REL_TOL))
    results.append(_close("WWR joint CVA", m["wwr_joint"], BASELINE["wwr_joint"], rel_tol=MC_REL_TOL))
    results.append(_close("WWR add-on", m["wwr_addon"], BASELINE["wwr_addon"], rel_tol=0.05))
    results.append(("WWR add-on is positive (wrong-way)", m["wwr_addon"] > 0, f"add-on ${m['wwr_addon']:,.0f}"))
    results.append(_close("WWR beta=0.5 joint CVA", m["wwr_beta_0_5"], BASELINE["wwr_beta_0_5"], rel_tol=MC_REL_TOL))
    results.append(_close("WWR beta=1.5 joint CVA", m["wwr_beta_1_5"], BASELINE["wwr_beta_1_5"], rel_tol=MC_REL_TOL))
    results.append(_close("WWR uncollateralised add-on", m["wwr_addon_uncollat"], BASELINE["wwr_addon_uncollat"], rel_tol=MC_REL_TOL))
    terc = bundle["wwr_terciles"]
    mono = bool(
        terc.loc["Low spread", "Avg_EPE"]
        <= terc.loc["Mid spread", "Avg_EPE"]
        <= terc.loc["High spread", "Avg_EPE"]
    )
    results.append(("tercile EPE rises with spread tercile", mono,
                    "low <= mid <= high average EPE"))
    return results


def check_v2(bundle, v2):
    """Version 2 checks: full-book regression, aggregation and SA-CCR v2."""
    results = []
    fullbook = v2["fullbook"]
    summary = fullbook["summary"]
    counterparty = fullbook["counterparty"]

    # --- Full-book Monte Carlo -----------------------------------------
    csa = summary[summary["Netting_Agreement_ID"] == "CSA-005"].iloc[0]
    results.append(_close("v2 full-book CSA-005 uncollateralised EPE", float(csa["EPE_Uncollateralized"]), BASELINE["epe_uncollat"], rel_tol=MC_REL_TOL))
    results.append(_close("v2 full-book CSA-005 collateralised EPE", float(csa["EPE_Collateralized"]), BASELINE["epe_collat"], rel_tol=MC_REL_TOL))
    results.append(_close("v2 full-book CSA-005 uncollateralised CVA", float(csa["CVA_Uncollateralized"]), BASELINE["cva_uncollat"], rel_tol=MC_REL_TOL))
    results.append(_close("v2 full-book CSA-005 collateralised CVA", float(csa["CVA_Collateralized"]), BASELINE["cva_collat"], rel_tol=MC_REL_TOL))

    t0_ok = all(
        bool(detail["t0_check"]["Match"].all())
        for detail in fullbook["details"].values()
    )
    results.append((
        "v2 t=0 revaluation reproduces every MTM in all sets",
        t0_ok,
        f"{len(fullbook['details'])} netting sets checked",
    ))
    ordering_ok = bool(
        (summary["EPE_Collateralized"] <= summary["EPE_Uncollateralized"] + 1e-6).all()
        and (summary["EPE_Collateralized"] >= 0).all()
    )
    results.append((
        "v2 collateralised EPE within [0, uncollateralised] for every set",
        ordering_ok,
        f"max excess ${float((summary['EPE_Collateralized'] - summary['EPE_Uncollateralized']).max()):,.4f}",
    ))

    total_gap = max(
        abs(float(summary[column].sum()) - float(counterparty[column].sum()))
        for column in (
            "EPE_Uncollateralized",
            "EPE_Collateralized",
            "CVA_Uncollateralized",
            "CVA_Collateralized",
        )
    )
    results.append((
        "v2 portfolio totals equal counterparty sums",
        total_gap <= 0.01,
        f"max gap ${total_gap:,.4f} across EPE/CVA totals",
    ))
    regrouped = summary.groupby("Counterparty")["CVA_Collateralized"].sum()
    cp_gap = max(
        abs(float(regrouped.get(row.Counterparty, 0.0)) - float(row.CVA_Collateralized))
        for row in counterparty.itertuples()
    )
    results.append((
        "v2 counterparty CVA equals sum of its netting sets",
        cp_gap <= 0.01,
        f"max gap ${cp_gap:,.4f}",
    ))

    # --- SA-CCR v2 -------------------------------------------------------
    saccr = v2["saccr_v2"]["comparison"]
    identity_gap = 0.0
    for row in saccr.itertuples():
        identity_gap = max(
            identity_gap,
            abs(row.EAD_v1 - 1.4 * (row.RC + row.PFE_v1)),
            abs(row.EAD_v2 - 1.4 * (row.RC + row.PFE_v2)),
            abs(
                row.EAD_v2_Collateralized_RC
                - 1.4 * (row.RC_Collateralized + row.PFE_v2_Collateralized_RC)
            ),
        )
    results.append((
        "v2 SA-CCR EAD identity holds under all three variants",
        identity_gap <= 0.01,
        f"max gap ${identity_gap:,.6f}",
    ))
    buckets = v2["saccr_v2"]["rates_buckets"]
    bucket_ok = bool(
        (buckets["IR_AddOn_v2_Bucketed"] >= -1e-9).all()
        and (
            buckets["IR_AddOn_v2_Bucketed"]
            <= buckets["IR_Bucket_Gross_AddOn"] + 1e-6
        ).all()
    )
    results.append((
        "v2 bucketed IR add-on bounded by gross bucket add-on",
        bucket_ok,
        f"{len(buckets)} currency hedging sets checked",
    ))
    positive_vm = saccr[saccr["VM_At_Valuation"] > 0]
    collat_ok = bool(
        (
            positive_vm["EAD_v2_Collateralized_RC"]
            <= positive_vm["EAD_v2"] + 0.01
        ).all()
    )
    results.append((
        "v2 collateralised-RC EAD does not exceed v2 EAD for positive-VM sets",
        collat_ok,
        f"{len(positive_vm)} positive-VM sets checked",
    ))
    multiplier_ok = bool(
        saccr[
            [
                "Multiplier_v1",
                "Multiplier_v2",
                "Multiplier_v2_Collateralized_RC",
            ]
        ]
        .apply(lambda column: column.between(0.05, 1.0).all())
        .all()
    )
    results.append(("v2 PFE multipliers lie in [0.05, 1]", multiplier_ok, "all variants"))

    # --- CSA-005 analytics ------------------------------------------------
    analytics = v2["analytics"]
    stress = analytics["stress"].set_index("Scenario")
    vol_ok = bool(
        stress.loc["Volatility x1.5", "EPE_Uncollateralized"]
        >= stress.loc["Base", "EPE_Uncollateralized"]
        and stress.loc["Volatility x1.5", "EPE_Collateralized"]
        >= stress.loc["Base", "EPE_Collateralized"]
    )
    results.append((
        "v2 volatility stress raises CSA-005 EPE",
        vol_ok,
        "uncollateralised and collateralised EPE compared with base",
    ))

    convergence = analytics["convergence"]
    conv_ok = bool(
        convergence["N_Paths"].tolist() == [1000, 2500, 5000, 10000]
        and np.isfinite(
            convergence[
                ["EPE_Uncollateralized", "EPE_Collateralized", "CVA_Collateralized"]
            ].to_numpy(dtype=float)
        ).all()
    )
    at = convergence.set_index("N_Paths")
    spacing_detail = (
        "informational only (not forced): "
        f"uncollat EPE |5k-10k|=${abs(at.loc[5000, 'EPE_Uncollateralized'] - at.loc[10000, 'EPE_Uncollateralized']):,.0f} vs "
        f"|1k-10k|=${abs(at.loc[1000, 'EPE_Uncollateralized'] - at.loc[10000, 'EPE_Uncollateralized']):,.0f}; "
        f"collat CVA |5k-10k|=${abs(at.loc[5000, 'CVA_Collateralized'] - at.loc[10000, 'CVA_Collateralized']):,.0f} vs "
        f"|1k-10k|=${abs(at.loc[1000, 'CVA_Collateralized'] - at.loc[10000, 'CVA_Collateralized']):,.0f}"
    )
    results.append(("v2 convergence table populated and finite", conv_ok, spacing_detail))

    sensitivity = analytics["sensitivity"]
    mono_cva = bool(
        sensitivity.groupby("MPOR_Business_Days")["CVA_Collateralized"]
        .apply(lambda values: values.is_monotonic_increasing)
        .all()
    )
    epe_by_mpor = sensitivity.groupby("MPOR_Business_Days")["EPE_Collateralized"].nunique()
    sens_ok = bool(
        len(sensitivity) == 12
        and mono_cva
        and (epe_by_mpor == 1).all()
    )
    results.append((
        "v2 sensitivity grid is 3 MPOR x 4 spreads with CVA increasing in spread",
        sens_ok,
        "EPE is spread-invariant within each MPOR row group by construction",
    ))
    return results


def report(results):
    lines = []
    for name, ok, detail in results:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    failed = [r for r in results if not r[1]]
    lines.append("")
    lines.append("ALL CHECKS PASS" if not failed else f"{len(failed)} CHECK(S) FAILED")
    return "\n".join(lines), not failed


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.pipeline import run_pipeline

    data = Path(__file__).resolve().parents[1] / "data" / "CCR_Book_Simulation_v3.xlsx"
    if not data.exists():
        data = data.with_name("CCR_Book.csv")
    text, ok = report(check_bundle(run_pipeline(data)))
    print(text)
    raise SystemExit(0 if ok else 1)
