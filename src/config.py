"""Shared settings for the counterparty credit risk replication.

Every stage of the pipeline reads its anchor parameters from here so the
valuation date, the prototype netting set and the CSA / CVA assumptions
stay consistent across modules.
"""

import pandas as pd

VALUATION_DATE = pd.Timestamp("2025-06-15")

# Parts 3-6 zoom in on this single agreement; Parts 1-2 cover the full book.
PROTOTYPE_NETTING_SET = "CSA-005"

# Monte Carlo design.
N_PATHS = 5000
RANDOM_SEED = 42

# Variation-margin terms used for the prototype netting set.
THRESHOLD = 500_000.0
MTA = 250_000.0
MPOR_BUSINESS_DAYS = 10
BUSINESS_DAYS_PER_YEAR = 250

# CVA assumptions.
LGD = 0.60
RISK_FREE_RATE = 0.04

# Wrong-way risk design.
ITRAXX_FACTOR = "ITRAXX.EU"
BASE_BETA = 1.0
BETA_SCENARIOS = (0.5, 1.0, 1.5)
