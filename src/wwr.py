"""Part 6 - wrong-way risk from an index-linked counterparty spread.

The prototype counterparty's spread is tied path by path to the simulated
credit index:

    spread_path(t) = anchor x (index(t) / index(0)) ** beta

Two CVA numbers are then formed from the *same* stochastic spread
distribution.  The joint number keeps each path's own exposure paired
with its own default probabilities; the independent benchmark averages
exposure and default probability separately before multiplying, which
removes the pairing.  Their difference isolates the dependence effect.
"""

import numpy as np
import pandas as pd

from .config import LGD, RISK_FREE_RATE


def joint_and_independent_cva(
    exposure, index_ratio, t_years, anchor_bps, beta, lgd=LGD, risk_free=RISK_FREE_RATE
):
    """Pathwise-hazard CVA, kept joint and decoupled, for one beta."""
    n_paths = exposure.shape[0]
    steps = np.diff(t_years)
    mid_times = 0.5 * (t_years[:-1] + t_years[1:])
    discount = np.exp(-risk_free * mid_times)

    spread_path = anchor_bps * np.power(index_ratio, beta)
    hazard_path = (spread_path / 10_000.0) / lgd
    step_hazard = 0.5 * (hazard_path[:, :-1] + hazard_path[:, 1:])
    cumulated = np.cumsum(step_hazard * steps[np.newaxis, :], axis=1)
    cumulated = np.column_stack([np.zeros(n_paths), cumulated])
    survival_path = np.exp(-cumulated)
    marginal_pd_path = survival_path[:, :-1] - survival_path[:, 1:]

    mean_exposure = 0.5 * (exposure[:, :-1] + exposure[:, 1:])
    per_path = lgd * discount[np.newaxis, :] * mean_exposure * marginal_pd_path
    joint = float(per_path.sum(axis=1).mean())

    decoupled = lgd * discount * mean_exposure.mean(axis=0) * marginal_pd_path.mean(axis=0)
    independent = float(decoupled.sum())
    return {"joint": joint, "independent": independent, "spread_path": spread_path}


def tercile_evidence(spread_path, exposure, t_years):
    """Average spread and lifetime EPE inside low / mid / high spread terciles."""
    horizon = t_years[-1]
    path_mean_spread = np.trapezoid(spread_path, t_years, axis=1) / horizon
    path_epe = np.trapezoid(exposure, t_years, axis=1) / horizon
    ranks = pd.Series(path_mean_spread).rank(method="first")
    labels = pd.qcut(ranks, 3, labels=["Low spread", "Mid spread", "High spread"])
    frame = pd.DataFrame(
        {"Tercile": labels, "Path_Avg_Spread": path_mean_spread, "Path_EPE": path_epe}
    )
    summary = frame.groupby("Tercile", observed=True).agg(
        Avg_Spread_bps=("Path_Avg_Spread", "mean"),
        Avg_EPE=("Path_EPE", "mean"),
        N_Paths=("Path_EPE", "count"),
    )
    correlation = float(np.corrcoef(path_mean_spread, path_epe)[0, 1])
    return summary, correlation, path_mean_spread, path_epe
