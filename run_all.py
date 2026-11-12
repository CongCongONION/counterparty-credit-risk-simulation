"""Run the full counterparty credit risk replication end to end.

Usage (from this directory):

    ../.venv/bin/python run_all.py

Writes the summary CSVs, exposure profiles, charts and a comparison
against the reference baseline into outputs/, then prints the
reconciliation verdict.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import reconciliation  # noqa: E402
from src.config import PROTOTYPE_NETTING_SET  # noqa: E402
from src.pipeline import headline_metrics, run_pipeline  # noqa: E402

DATA_PATH = ROOT / "data" / "CCR_Book_Simulation_v3.xlsx"
if not DATA_PATH.exists():  # fall back to the identical CSV export
    DATA_PATH = ROOT / "data" / "CCR_Book.csv"
OUT = ROOT / "outputs"


def write_outputs(bundle, metrics):
    OUT.mkdir(parents=True, exist_ok=True)

    bundle["netting"].reset_index().to_csv(OUT / "netting_summary.csv", index=False)
    bundle["ead"].reset_index().to_csv(OUT / "saccr_summary.csv", index=False)
    bundle["hedging"].to_csv(OUT / "saccr_hedging_sets.csv", index=False)
    bundle["t0_check"].to_csv(OUT / "t0_revaluation_check.csv", index=False)
    bundle["risk_factors"].reset_index().to_csv(OUT / "risk_factors.csv", index=False)
    bundle["profile_uncollat"].to_csv(
        OUT / "exposure_profile_uncollateralized.csv", index=False
    )
    bundle["profile_collat"].to_csv(
        OUT / "exposure_profile_collateralized.csv", index=False
    )
    bundle["cva_breakdown"].to_csv(OUT / "cva_breakdown.csv", index=False)

    horizon = float(bundle["t_years"][-1])
    cva_summary = pd.DataFrame(
        [
            {
                "Netting_Set": PROTOTYPE_NETTING_SET,
                "Credit_Spread_bps": metrics["spread_median_bps"],
                "Credit_Spread_P25_bps": metrics["spread_p25_bps"],
                "Credit_Spread_P75_bps": metrics["spread_p75_bps"],
                "LGD": metrics["lgd"],
                "Risk_Free_Rate": metrics["risk_free_rate"],
                "Hazard_Rate": bundle["cva_base"]["hazard_rate"],
                "Horizon_Years": horizon,
                "Survival_Prob_At_Horizon": float(bundle["cva_base"]["survival"][-1]),
                "EPE_Uncollateralized": metrics["epe_uncollat"],
                "EPE_Collateralized": metrics["epe_collat"],
                "CVA_Collateralized": metrics["cva_collat"],
                "CVA_Uncollateralized": metrics["cva_uncollat"],
                "CVA_Low_Spread": metrics["cva_low"],
                "CVA_High_Spread": metrics["cva_high"],
            }
        ]
    )
    cva_summary.to_csv(OUT / "cva_summary.csv", index=False)

    wwr_summary = pd.DataFrame(
        [
            {
                "Netting_Set": PROTOTYPE_NETTING_SET,
                "Credit_Spread_Anchor_bps": metrics["spread_median_bps"],
                "CVA_Step5_Flat": bundle["wwr_flat"]["joint"],
                "CVA_Independent": metrics["wwr_independent"],
                "CVA_Joint": metrics["wwr_joint"],
                "WWR_Addon": metrics["wwr_addon"],
                "WWR_Addon_Pct": metrics["wwr_addon_pct"],
                "Spread_EPE_Correlation": bundle["wwr_corr"],
                "CVA_Beta_0_5": metrics["wwr_beta_0_5"],
                "CVA_Beta_1_0": metrics["wwr_beta_1_0"],
                "CVA_Beta_1_5": metrics["wwr_beta_1_5"],
                "CVA_Independent_Uncollateralized": bundle["wwr_independent_uncollat"],
                "CVA_Joint_Uncollateralized": bundle["wwr_joint_uncollat"],
                "WWR_Addon_Uncollateralized": metrics["wwr_addon_uncollat"],
                "WWR_Addon_Pct_Uncollateralized": metrics["wwr_addon_uncollat_pct"],
                "Spread_EPE_Correlation_Uncollateralized": bundle["wwr_corr_uncollat"],
            }
        ]
    )
    wwr_summary.to_csv(OUT / "wwr_summary.csv", index=False)
    bundle["wwr_terciles"].reset_index().to_csv(OUT / "wwr_terciles.csv", index=False)
    return cva_summary, wwr_summary


def write_charts(bundle, metrics):
    t = bundle["t_years"]
    uncollat = bundle["profile_uncollat"]
    collat = bundle["profile_collat"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, uncollat["EE"] / 1e6, label="EE - uncollateralised")
    ax.plot(t, collat["EE"] / 1e6, label="EE - collateralised")
    ax.plot(t, uncollat["PFE_95"] / 1e6, "--", label="PFE 95% - uncollateralised")
    ax.plot(t, collat["PFE_95"] / 1e6, "--", label="PFE 95% - collateralised")
    ax.set_xlabel("Years from valuation date")
    ax.set_ylabel("Exposure ($m)")
    ax.set_title("CSA-005 exposure profile: collateralised vs uncollateralised")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exposure_profile.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [
        f"Low ({metrics['spread_p25_bps']:.1f} bps)",
        f"Base ({metrics['spread_median_bps']:.1f} bps)",
        f"High ({metrics['spread_p75_bps']:.1f} bps)",
    ]
    values = [metrics["cva_low"], metrics["cva_collat"], metrics["cva_high"]]
    ax.bar(labels, [v / 1e3 for v in values], color=["#7aa6c2", "#1f5fa6", "#c26d5c"])
    ax.set_ylabel("Collateralised CVA ($k)")
    ax.set_title("CSA-005 CVA sensitivity to the counterparty spread")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cva_sensitivity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    betas = [0.5, 1.0, 1.5]
    joint = [metrics["wwr_beta_0_5"], metrics["wwr_beta_1_0"], metrics["wwr_beta_1_5"]]
    independent = [
        bundle["wwr_beta_runs"][b]["independent"] for b in betas
    ]
    ax.plot(betas, [v / 1e3 for v in joint], "o-", label="Joint (path-linked) CVA")
    ax.plot(betas, [v / 1e3 for v in independent], "s--", label="Independent benchmark")
    ax.set_xlabel("Beta (spread sensitivity to the credit index)")
    ax.set_ylabel("CVA ($k)")
    ax.set_title("Wrong-way risk: joint CVA vs independent benchmark by beta")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "wwr_beta_sensitivity.png", dpi=150)
    plt.close(fig)


COMPARISON_ROWS = [
    ("Live trades (of 120)", "n_trades_live", 95, "{:,.0f}"),
    ("Gross exposure, full book ($)", "gross_exposure", 117_512_366, "{:,.0f}"),
    ("Replacement cost, full book ($)", "replacement_cost", 84_953_642, "{:,.0f}"),
    ("Netting benefit (%)", "netting_benefit_pct", 27.7, "{:.1f}"),
    ("SA-CCR EAD, full book ($)", "ead_total", 478_547_962, "{:,.0f}"),
    ("SA-CCR EAD, CSA-005 ($)", "ead_csa005", 82_064_083, "{:,.0f}"),
    ("EPE uncollateralised, CSA-005 ($)", "epe_uncollat", 9_769_587, "{:,.0f}"),
    ("EPE collateralised, CSA-005 ($)", "epe_collat", 794_518, "{:,.0f}"),
    ("EPE reduction from collateral (%)", "epe_reduction_pct", 91.9, "{:.1f}"),
    ("Credit spread anchor (bps)", "spread_median_bps", 264.85, "{:.2f}"),
    ("CVA collateralised ($)", "cva_collat", 142_759, "{:,.0f}"),
    ("CVA uncollateralised ($)", "cva_uncollat", 1_769_585, "{:,.0f}"),
    ("CVA low-spread sensitivity ($)", "cva_low", 47_551, "{:,.0f}"),
    ("CVA high-spread sensitivity ($)", "cva_high", 184_333, "{:,.0f}"),
    ("WWR independent CVA ($)", "wwr_independent", 145_335, "{:,.0f}"),
    ("WWR joint CVA ($)", "wwr_joint", 146_252, "{:,.0f}"),
    ("WWR add-on ($)", "wwr_addon", 916, "{:,.0f}"),
    ("WWR joint CVA, beta 0.5 ($)", "wwr_beta_0_5", 143_828, "{:,.0f}"),
    ("WWR joint CVA, beta 1.5 ($)", "wwr_beta_1_5", 150_058, "{:,.0f}"),
    ("WWR add-on uncollateralised ($)", "wwr_addon_uncollat", 25_698, "{:,.0f}"),
]


def write_comparison(metrics):
    lines = [
        "# Replication vs template baseline",
        "",
        "| Metric | Template baseline | This replication | Difference | % difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key, baseline, fmt in COMPARISON_ROWS:
        actual = metrics[key]
        diff = actual - baseline
        pct = 100.0 * diff / baseline if baseline else 0.0
        lines.append(
            f"| {label} | {fmt.format(baseline)} | {fmt.format(actual)} "
            f"| {diff:+,.2f} | {pct:+.4f}% |"
        )
    lines.append("")
    (OUT / "comparison_vs_template.md").write_text("\n".join(lines), encoding="utf-8")
    return lines


def main():
    bundle = run_pipeline(DATA_PATH)
    metrics = headline_metrics(bundle)
    write_outputs(bundle, metrics)
    write_charts(bundle, metrics)
    comparison = write_comparison(metrics)

    print("=== Headline results (this replication) ===")
    for label, key, _baseline, fmt in COMPARISON_ROWS:
        print(f"  {label:<42} {fmt.format(metrics[key])}")
    print()
    print("\n".join(comparison))
    print()
    results = reconciliation.check_bundle(bundle)
    text, ok = reconciliation.report(results)
    print("=== Reconciliation ===")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
