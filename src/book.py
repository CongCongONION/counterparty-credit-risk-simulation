"""Loading the synthetic derivatives book and the Part 1 netting maths.

Netting logic: under an enforceable master agreement the mark-to-market
values inside one netting set offset each other, so current exposure is
the floored net sum while gross exposure keeps only the positive legs.
"""

import numpy as np
import pandas as pd

MTM_COL = "MTM (RC)"


def load_book(path):
    """Read the book (Excel workbook, or the identical CSV export of it)."""
    if str(path).endswith(".csv"):
        frame = pd.read_csv(path)
    else:
        frame = pd.read_excel(path, sheet_name="Derivatives Book")
    frame.columns = frame.columns.astype(str).str.strip()
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")].copy()
    frame["Maturity_Date"] = pd.to_datetime(frame["Maturity_Date"], errors="raise")
    frame["Trade_Date"] = pd.to_datetime(frame["Trade_Date"], errors="raise")
    return frame


def live_trades(frame, valuation_date):
    """Trades still open for the full-book stages (maturity after the date)."""
    return frame[frame["Maturity_Date"] > valuation_date].copy()


def netting_summary(live):
    """Per-agreement gross exposure, net MTM and replacement cost.

    Gross exposure sums only positive MTMs; replacement cost floors the
    net MTM at zero; the gap between the two is the netting benefit.
    """
    keys = ["Counterparty", "Netting_Agreement_ID"]
    positive_legs = live[live[MTM_COL] > 0]
    gross = positive_legs.groupby(keys)[MTM_COL].sum().rename("Gross_Exposure")
    net_mtm = live.groupby(keys)[MTM_COL].sum().rename("Net_MTM")

    table = pd.concat([gross, net_mtm], axis=1).fillna(0.0)
    table["RC"] = table["Net_MTM"].clip(lower=0.0)
    table["Netting_Benefit"] = table["Gross_Exposure"] - table["RC"]
    table["Netting_Benefit_Pct"] = np.where(
        table["Gross_Exposure"] > 0,
        100.0 * table["Netting_Benefit"] / table["Gross_Exposure"],
        0.0,
    )
    return table.sort_values("Netting_Benefit", ascending=False)
