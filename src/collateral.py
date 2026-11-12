"""Part 4 - variation margin for the prototype netting set.

Margin is worked on a daily business-day grid built by interpolating the
monthly exposure paths.  The required balance is the exposure above the
threshold, but cash only moves when the gap to the balance already held
reaches the minimum transfer amount.  Because this is an existing
relationship, the opening balance starts at the day-one required level.
The balance that may be netted against a monthly reporting date is the
one held a margin period of risk earlier, which is what leaves residual
exposure even under a collateral agreement.
"""

import numpy as np

from .config import BUSINESS_DAYS_PER_YEAR, MPOR_BUSINESS_DAYS, MTA, THRESHOLD


def simulate_collateral(
    exposure,
    t_years,
    threshold=THRESHOLD,
    mta=MTA,
    mpor_days=MPOR_BUSINESS_DAYS,
    days_per_year=BUSINESS_DAYS_PER_YEAR,
):
    """Return (collateral held at each report date, collateralised exposure)."""
    if (exposure < 0).any() or not np.isfinite(exposure).all():
        raise ValueError("Exposure entering the margin model must be finite and >= 0.")
    horizon = t_years[-1]
    mpor_years = mpor_days / days_per_year

    step = 1.0 / days_per_year
    daily_grid = np.arange(0.0, horizon + step / 2, step)
    daily_grid = daily_grid[daily_grid <= horizon]
    if daily_grid[-1] < horizon:
        daily_grid = np.append(daily_grid, horizon)

    # Which daily point supplies the collateral for each monthly report date.
    lookback_times = np.maximum(t_years - mpor_years, 0.0)
    lookback_pos = np.searchsorted(daily_grid, lookback_times, side="right") - 1
    lookback_pos = np.clip(lookback_pos, 0, len(daily_grid) - 1)
    reports_at = {}
    for report_pos, daily_pos in enumerate(lookback_pos):
        reports_at.setdefault(int(daily_pos), []).append(report_pos)

    balance = np.maximum(exposure[:, 0] - threshold, 0.0)
    held = np.full_like(exposure, np.nan)
    for report_pos in reports_at.get(0, []):
        held[:, report_pos] = balance

    n_dates = exposure.shape[1]
    for daily_pos in range(1, len(daily_grid)):
        moment = daily_grid[daily_pos]
        right = np.searchsorted(t_years, moment, side="right")
        if right >= n_dates:
            exposure_now = exposure[:, -1]
        else:
            left = right - 1
            weight = (moment - t_years[left]) / (t_years[right] - t_years[left])
            exposure_now = exposure[:, left] * (1.0 - weight) + exposure[:, right] * weight
        required = np.maximum(exposure_now - threshold, 0.0)
        due_to_move = np.abs(required - balance) >= mta
        balance = np.where(due_to_move, required, balance)
        for report_pos in reports_at.get(daily_pos, []):
            held[:, report_pos] = balance

    if not np.isfinite(held).all():
        raise ValueError("A reporting date never received a collateral balance.")
    if (held < 0).any():
        raise ValueError("Held collateral must not be negative in this one-way model.")

    netted = np.maximum(exposure - held, 0.0)
    return held, netted
