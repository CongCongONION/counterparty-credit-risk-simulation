# Counterparty Credit Risk & Derivatives Exposure Modeling

## An independent re-implementation: netting, simplified SA-CCR, Monte Carlo exposure, collateral, CVA and wrong-way risk

**Author:** Cong Wang
**Valuation date:** 15 June 2025
**Units:** U.S. dollars

> **Attribution.** The methodology in this repository follows the template project [pedadadk2017/Counterparty-Credit-Risk-Simulation](https://github.com/pedadadk2017/Counterparty-Credit-Risk-Simulation). All data is synthetic. This is an independent re-implementation, written from the methodology for study purposes — no code was copied from the template; the model was re-derived module by module and reconciled against the template's published results (see *Reconciliation* below).

---

## Objective

Measure counterparty exposure to a synthetic derivatives book from two complementary angles:

1. **Regulatory-style, whole book** — netting-set replacement cost and a simplified SA-CCR exposure at default (EAD) for all 120 trades.
2. **Economic, one netting set in depth** — a correlated Monte Carlo simulation of the `CSA-005` netting set (Fund2) producing expected exposure (EE), expected positive exposure (EPE) and potential future exposure (PFE), then variation-margin collateral, CVA, and a wrong-way risk (WWR) study on top of the same paths.

## Framework

| Stage | What it does |
|---|---|
| **1. Netting** | Live trades (maturity after the valuation date) are grouped by netting agreement. Gross exposure sums positive MTMs; replacement cost is `RC = max(net MTM, 0)`; the difference is the netting benefit. |
| **2. Simplified SA-CCR** | Per trade: supervisory delta (Black-style for options), supervisory duration for rates/credit, supervisory factors by asset class, and a maturity factor (0.3 if margined, otherwise `sqrt(min(M, 1))`). Add-ons are aggregated inside hedging sets using Basel supervisory correlations (equity/credit 0.8, commodity 0.4; FX and rates hedging sets are summed), scaled by the PFE multiplier `min(1, 0.05 + 0.95·exp(V / (2·0.95·AddOn)))`, and `EAD = 1.4·(RC + PFE)`. Collateral is ignored at this stage. |
| **3. Monte Carlo** | 5,000 paths on a monthly grid to the longest live maturity (8.35 years) for the six `CSA-005` underlyings. Equity/FX follow GBM in log space; rates, commodity and credit use the exact mean-reverting (OU) transition. Shocks are correlated via Cholesky (SPX vs credit index −0.5; the two commodity underlyings +0.3; +0.1 elsewhere). Seed 42, drawn step by step so runs reproduce exactly. Trades are revalued with forward-style (equity/FX/commodity), annuity-scaled (rates/CDS) and Black-76 (swaption) approximations, each calibrated to reproduce the reported MTM at t = 0, and roll off at maturity. |
| **4. Collateral** | Daily business-day variation-margin simulation: threshold $500k, minimum transfer amount $250k, 10-business-day margin period of risk. The balance starts at the day-one required level; the collateral available at a monthly date is the balance held one MPOR earlier. |
| **5. CVA** | From the collateralised EE profile: `CVA = LGD · Σ DF · EE · marginal PD`, with LGD 60%, flat 4% discounting, and exponential survival driven by a flat hazard from the credit triangle. The spread anchor is the median live-trade spread of the netting set (264.85 bps); the 25th/75th percentile spreads give the sensitivity range. |
| **6. Wrong-way risk** | The counterparty spread follows `anchor · (credit-index path / index start)^beta` path by path. Joint CVA (exposure and default kept paired) is compared with an independent benchmark built from the *same* stochastic spread distribution with the pairing removed, so their difference isolates dependence. Beta ∈ {0, 0.5, 1.0, 1.5}; beta = 0 reproduces the Stage-5 CVA exactly. |

## Results (this implementation's run)

| Metric | Result |
|---|---:|
| Live trades as of 2025-06-15 | 95 of 120 (8 netting sets, 6 counterparties) |
| Full-book gross exposure | $117,512,366 |
| Full-book replacement cost | $84,953,642 |
| Netting benefit | 27.7% |
| Full-book simplified SA-CCR EAD | $478,547,962 |
| `CSA-005` simplified SA-CCR EAD | $82,064,083 |
| `CSA-005` uncollateralised EPE | $9,769,587 |
| `CSA-005` collateralised EPE | $794,518 (−91.9%) |
| CVA, collateralised | $142,759 (sensitivity $47,551 – $184,333) |
| CVA, uncollateralised | $1,769,585 |
| WWR: independent / joint CVA (beta = 1.0) | $145,335 / $146,252 |
| WWR add-on | $916 (+0.63%); uncollateralised $25,698 (+1.42%) |
| WWR joint CVA by beta (0.5 / 1.0 / 1.5) | $143,828 / $146,252 / $150,058 |

Reading of the results: SA-CCR EAD for `CSA-005` ($82.1M) sits far above its simulated lifetime EPE ($9.77M) — the two answer different questions (a conservative regulatory measure vs an average simulated profile). Collateral cuts average exposure by about 92%, yet residual exposure survives through the threshold, the MTA and the MPOR lag, leaving a still-material CVA of about $143k. Wrong-way risk is positive but modest after collateral (+0.6%), and roughly 28 times larger in dollars before collateral — margining suppresses most of the dependence effect. Average path EPE rises monotonically across spread terciles ($775,916 → $801,571 → $806,072), confirming the direction of the exposure–credit link.

## Reconciliation

`src/reconciliation.py` checks every headline figure against the template project's verified baseline run, plus structural identities (RC flooring, gross − benefit = RC per set, t = 0 revaluation of all 8 trades, collateralised exposure inside [0, uncollateralised], beta = 0 reproducing the flat CVA, monotone terciles). Current status: **ALL CHECKS PASS** — SA-CCR matches to the dollar, and all Monte Carlo quantities match the baseline to well under 0.01% (the baseline is published rounded to whole dollars; the residual differences are sub-dollar). The full side-by-side table is written to `outputs/comparison_vs_template.md` on every run.

## Repository structure

```text
.
├── run_all.py                 # end-to-end runner: outputs, charts, reconciliation
├── data/
│   ├── CCR_Book_Simulation_v3.xlsx   # synthetic 120-trade book (input only)
│   └── CCR_Book.csv                  # identical CSV export of the same book (fallback input)
├── src/
│   ├── config.py              # valuation date, CSA / CVA / WWR assumptions
│   ├── book.py                # loading + Part 1 netting
│   ├── saccr.py               # Part 2 simplified SA-CCR
│   ├── simulation.py          # Part 3a/3b correlated risk-factor paths
│   ├── revaluation.py         # Part 3c trade revaluation + EE/PFE/EPE
│   ├── collateral.py          # Part 4 variation margin
│   ├── cva.py                 # Part 5 CVA
│   ├── wwr.py                 # Part 6 wrong-way risk
│   ├── pipeline.py            # orchestration of all stages
│   └── reconciliation.py      # checks against the template baseline
├── tests/
│   └── test_outputs.py        # pytest suite re-deriving results from outputs/
├── outputs/                   # generated CSVs, charts, comparison table
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
python run_all.py            # writes outputs/ and prints the reconciliation verdict
python -m pytest tests/ -q   # re-derives headline numbers from the output CSVs
python -m src.reconciliation # standalone reconciliation (re-runs the pipeline)
```

The Monte Carlo stage is seeded (`numpy.random.default_rng(42)`, shocks drawn step by step), so repeated runs reproduce identical numbers. Simulated path arrays are not stored in the repository (they are large and fully regenerable); only summary CSVs and charts are kept in `outputs/`.

## Limitations

- All trades, counterparties, CSA terms and market parameters are synthetic; volatilities and correlations are scenario assumptions, not market calibrations.
- The SA-CCR stage is *simplified*: no interest-rate maturity buckets, no collateralised replacement cost, and a single-formula supervisory duration.
- The deep-dive simulation covers one netting set (`CSA-005`), not the full book.
- Revaluation uses product-sensitive approximations (forward-style, annuity-scaled, Black-76) rather than full curve-based pricing; rate/spread units are inferred from level magnitudes.
- Hazard and discount rates are flat; the WWR link uses a broad credit index and an assumed beta rather than a calibrated single-name model.
- Collateral modelling excludes initial margin, haircuts, collateral interest and disputes.

## Disclaimer

This project is for educational and portfolio-demonstration purposes only. It is not a regulatory capital calculation, a production valuation system, or investment advice.
