"""Pytest suite that re-derives headline results from the committed
outputs/ CSVs produced by run_all.py.

Run from the replication/ directory:

    ../.venv/bin/python -m pytest tests/ -q
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

OUT = Path(__file__).resolve().parents[1] / "outputs"


def _read(name):
    path = OUT / name
    assert path.exists(), f"{name} missing - run run_all.py first"
    return pd.read_csv(path)


def test_rc_is_floored_net_mtm():
    netting = _read("netting_summary.csv")
    assert (netting["RC"] >= 0).all()
    np.testing.assert_allclose(
        netting["RC"], np.maximum(netting["Net_MTM"], 0.0), atol=1e-6
    )
    # At least one netting set in this book has a negative net MTM, so the
    # floor must actually bind somewhere for the check to be meaningful.
    assert (netting["Net_MTM"] < 0).any()
    assert (netting.loc[netting["Net_MTM"] < 0, "RC"] == 0).all()


def test_gross_minus_benefit_equals_rc():
    netting = _read("netting_summary.csv")
    np.testing.assert_allclose(
        netting["Gross_Exposure"] - netting["Netting_Benefit"],
        netting["RC"],
        atol=0.01,
    )


def test_saccr_ead_total_and_csa005():
    saccr = _read("saccr_summary.csv")
    # EAD identity must hold row by row from the committed columns.
    np.testing.assert_allclose(
        saccr["EAD"], 1.4 * (saccr["RC"] + saccr["PFE"]), rtol=1e-9
    )
    assert saccr["EAD"].sum() == pytest.approx(478_547_962.19, abs=1.0)
    csa005 = saccr[saccr["Netting_Agreement_ID"] == "CSA-005"]
    assert len(csa005) == 1
    assert csa005["EAD"].iloc[0] == pytest.approx(82_064_083.32, abs=1.0)


def _epe(profile):
    t = profile["t_years"].to_numpy()
    return float(np.trapezoid(profile["EE"].to_numpy(), t) / t[-1])


def test_collateral_reduces_epe_by_about_92pct():
    uncollat = _read("exposure_profile_uncollateralized.csv")
    collat = _read("exposure_profile_collateralized.csv")
    epe_u, epe_c = _epe(uncollat), _epe(collat)
    assert epe_u == pytest.approx(9_769_587, rel=0.005)
    assert epe_c == pytest.approx(794_518, rel=0.005)
    reduction_pct = 100.0 * (1.0 - epe_c / epe_u)
    assert reduction_pct == pytest.approx(91.9, abs=0.5)
    # Collateralised EE can never exceed the uncollateralised profile.
    assert (collat["EE"] <= uncollat["EE"] + 1e-6).all()
    assert (collat["EE"] >= 0).all()


def test_cva_levels_and_sensitivity_ordering():
    cva = _read("cva_summary.csv").iloc[0]
    assert cva["CVA_Collateralized"] == pytest.approx(142_759, rel=0.005)
    assert cva["CVA_Uncollateralized"] == pytest.approx(1_769_585, rel=0.005)
    assert cva["CVA_Low_Spread"] < cva["CVA_Collateralized"] < cva["CVA_High_Spread"]
    assert cva["Credit_Spread_bps"] == pytest.approx(264.85, abs=0.01)


def test_wwr_addon_is_positive_and_small():
    wwr = _read("wwr_summary.csv").iloc[0]
    assert wwr["WWR_Addon"] > 0  # wrong-way, not right-way
    assert wwr["WWR_Addon"] == pytest.approx(916, rel=0.05)
    assert wwr["CVA_Joint"] > wwr["CVA_Independent"]
    # Stronger spread linkage must not lower the joint CVA in this book.
    assert wwr["CVA_Beta_0_5"] < wwr["CVA_Beta_1_0"] < wwr["CVA_Beta_1_5"]
    # Collateral shrinks the absolute add-on.
    assert abs(wwr["WWR_Addon"]) < abs(wwr["WWR_Addon_Uncollateralized"])


def test_tercile_epe_rises_with_spread():
    terciles = _read("wwr_terciles.csv").set_index("Tercile")
    assert (
        terciles.loc["Low spread", "Avg_EPE"]
        <= terciles.loc["Mid spread", "Avg_EPE"]
        <= terciles.loc["High spread", "Avg_EPE"]
    )
    assert terciles["N_Paths"].sum() == 5000
