# Model Validation Memorandum

## Counterparty Credit Risk & Derivatives Exposure Modeling

**Author:** Cong Wang  
**Project status:** Educational / portfolio model; not a production or regulatory-capital model  
**Valuation date:** 15 June 2025  
**Validation basis:** Independent re-run of `python run_all.py`, output inspection, reconciliation checks and the pytest suite  
**Overall conclusion:** **Pass for the stated educational scope.** All 29 v1 reconciliation checks, all 15 v2 checks and all 12 pytest tests pass. The v1 results are unchanged. Version 2 materially widens coverage, but the model remains simplified and synthetic; it is not validated for production pricing, limit setting or regulatory capital.

---

## 1. Purpose

This memorandum documents the design, implementation checks, results and limitations of a counterparty credit risk (CCR) model for a synthetic 120-trade derivatives book. The model measures exposure in two complementary ways:

1. a regulatory-style simplified SA-CCR calculation for the full book; and
2. an economic Monte Carlo calculation of expected exposure (EE), expected positive exposure (EPE), potential future exposure (PFE) and credit valuation adjustment (CVA).

Version 1 concentrated the Monte Carlo work on the CSA-005 netting set. Version 2 extends Monte Carlo exposure and CVA to all eight netting sets, adds two SA-CCR refinements, and adds convergence, stress and sensitivity evidence for CSA-005.

## 2. Scope and data

- **Book:** 120 synthetic trades; 95 are live at the valuation date.
- **Agreements / counterparties:** 8 netting sets and 6 counterparties.
- **Asset classes:** Rates, credit, equity, FX and commodity.
- **Prototype for detailed v1 and v2 analytics:** CSA-005 (Fund2), 8 live trades and a maximum maturity of approximately 8.35 years.
- **Monte Carlo size:** 5,000 paths per netting set in the production-equivalent project run; 10,000 paths are used only for the CSA-005 convergence comparison.
- **Units:** U.S. dollars unless spreads are stated in basis points.

The workbook records some CSA fields trade by trade even though collateral terms are economically agreement-level terms. Following the template's treatment, v2 infers each agreement's threshold and minimum transfer amount (MTA) as the modal values on that set's collateralised live trades. If values tie, the smallest tied value is selected so the rule is deterministic. This reproduces CSA-005's $500,000 threshold and $250,000 MTA. It is a modelling convention, not evidence of actual legal CSA terms.

## 3. Methodology summary

### 3.1 Version 1

#### Netting

For each agreement, gross exposure is the sum of positive trade MTMs, net MTM is the signed sum, and replacement cost is:

\[
RC = \max(\text{net MTM}, 0).
\]

The netting benefit is gross exposure minus RC.

#### Simplified SA-CCR

Each live trade receives an add-on based on:

\[
\text{effective notional} \times \text{supervisory delta}
\times \text{supervisory factor} \times \text{maturity factor}.
\]

Rates and credit trades use a supervisory duration in the effective notional. Options use a Black-style supervisory delta. Equity, credit and commodity hedging sets use the v1 Basel-style systematic/idiosyncratic aggregation; FX and rates hedging sets use a signed sum followed by an absolute value. The PFE multiplier is:

\[
\min\left(1,\;0.05+0.95\exp\left(\frac{V}{2(0.95)\text{AddOn}}\right)\right),
\]

and:

\[
EAD = 1.4(RC+PFE).
\]

#### Monte Carlo, collateral, CVA and wrong-way risk

For CSA-005, one starting level and one averaged volatility are retained per underlying. Equity and FX follow driftless geometric Brownian motion in log space. Rates, commodity and credit use the exact log-space Ornstein-Uhlenbeck transition, with mean-reversion speeds of 1.0 for rates and 0.3 for commodity and credit. Shocks are correlated through a Cholesky decomposition. The base correlation is +0.10 for most pairs, +0.30 between WTI and gold, and -0.50 between SPX and the credit index.

Trades are revalued using three approximations, each calibrated to reproduce reported MTM at time zero:

- forward-style linear valuation for equity, FX and commodity trades;
- annuity-scaled valuation for rates and credit trades; and
- an undiscounted Black-76 approximation for the swaption.

Variation margin is simulated on a daily business-day grid. Required collateral is exposure above the threshold; balances move only when the required change reaches the MTA. The balance available at a monthly reporting date is lagged by the 10-business-day margin period of risk (MPOR). The opening balance is the day-one required amount.

CVA uses:

\[
CVA = LGD \sum_t DF_t \cdot EE_t \cdot \Delta PD_t,
\]

with LGD of 60%, flat discounting at 4%, an exponential survival curve and hazard equal to spread divided by LGD. The CSA-005 spread anchor is the 264.85 bp median of its live-trade spreads.

Wrong-way risk links the counterparty spread path to the simulated credit index. Joint CVA preserves the path pairing between exposure and default probability; the independent benchmark uses the same stochastic spread distribution after removing that pairing. Their difference is the isolated wrong-way-risk add-on.

### 3.2 Version 2, P1 - full-book Monte Carlo

The same simulation, revaluation, collateral and CVA construction is applied independently to every netting set. Each set has:

- its own live trades, risk factors, starting levels and averaged volatilities;
- a monthly grid ending at its own longest live maturity;
- its own modal threshold and MTA;
- a 10-business-day MPOR; and
- a CVA spread equal to the median spread on its own live trades.

CSA-005 keeps seed 42 and exactly the same random-draw order as v1. Other agreements use seeds 43-49, assigned in agreement-ID order after removing the prototype from the list (CSA-001 = 43, CSA-002 = 44, CSA-003 = 45, CSA-004 = 46, CSA-006 = 47, CSA-007 = 48, CSA-008 = 49).

Counterparty results are sums of the relevant netting-set results. This is an additive reporting aggregation. It is not a joint simulation of exposures across agreements and does not model cross-set dependence.

### 3.3 Version 2, P2 - SA-CCR refinements

#### Interest-rate maturity buckets

Rates trades are allocated by remaining maturity to the CRE52.53 buckets:

- less than 1 year;
- 1 to 5 years; and
- more than 5 years.

For each currency hedging set, `D_b` is the signed sum of v1 trade add-ons in bucket `b`. The v2 add-on is:

\[
\sqrt{\sum_b D_b^2 + 2\rho\sum_{b<k}D_bD_k},
\qquad \rho = 50\%.
\]

The maturity-bucket structure follows CRE52.53 and the effective-notional/add-on architecture follows CRE52.54-52.55 in style. The uniform 50% pairwise correlation is an explicit educational simplification used for this v2 exercise; the memorandum does not claim that Basel prescribes one identical correlation for every bucket pair.

#### Collateralised replacement cost

The simplified collateralised replacement cost is:

\[
RC_{collat} = \max(V - VM,\; TH + MTA - NICA,\; 0),
\qquad NICA = 0.
\]

`VM` is the valuation-date balance produced by the collateral model. At time zero all paths have the same net MTM and therefore the same required balance, `max(RC - TH, 0)`, so the selected VM is path-invariant and does not use future simulated information. The PFE multiplier in this variant is evaluated at `V - VM` for consistency with the collateralised value input.

### 3.4 Version 2, P3 - CSA-005 analytics

Three analyses reuse the v1/v2 machinery:

1. **Convergence:** 1,000, 2,500, 5,000 and 10,000 paths, all with seed 42 and the same draw protocol. Results are compared with the 10,000-path run.
2. **Stress:** trade volatilities are multiplied by 1.5; separately, every cross-factor correlation is shifted by +0.20. The shifted matrix is passed through an eigenvalue-clipping PSD check/repair and rescaled to unit diagonal. For CSA-005 the shifted matrix was already positive definite, so repair changed entries by no more than approximately \(4\times10^{-16}\).
3. **Sensitivity:** MPOR is varied over 5, 10 and 20 business days and the spread anchor over -100 bp, base, +100 bp and +300 bp, with a 1 bp floor. Exposure is held fixed across spread cells; only collateral lag and default-risk pricing change.

## 4. Validation approach

Validation consisted of:

- preserving the complete v1 pipeline as a regression suite;
- comparing deterministic v1 quantities to the verified template baseline to the dollar where applicable;
- comparing Monte Carlo quantities sharing seed and draw order within a 0.5% tolerance (the observed differences are sub-dollar / far below tolerance);
- checking time-zero revaluation for every trade, first for CSA-005 and in v2 for all 95 live trades across all sets;
- checking exposure ordering under collateral;
- re-deriving EAD identities from committed output columns;
- checking portfolio and counterparty aggregation from set-level outputs;
- checking stress direction and sensitivity monotonicity; and
- running a pytest suite that reads the generated CSVs rather than trusting in-memory values.

The runtime verdicts in this memorandum are from `run_all.py`, which prints the v1 and v2 reconciliation sections separately.

## 5. Complete check inventory and results

### 5.1 Version 1 reconciliation - 29 of 29 passed

| # | Check | Result |
|---:|---|---|
| 1 | Live trade count | 95; exact pass |
| 2 | Netting set count | 8; exact pass |
| 3 | Counterparty count | 6; exact pass |
| 4 | CSA-005 live trades | 8; exact pass |
| 5 | Gross - netting benefit = RC for every set | Maximum gap $0.0000 |
| 6 | RC equals max(net MTM, 0) | Pass |
| 7 | Gross exposure total | $117,512,366.15; exact to tolerance |
| 8 | Replacement cost total | $84,953,641.80; exact to tolerance |
| 9 | Netting benefit percentage | 27.71% vs 27.70% published rounded baseline |
| 10 | SA-CCR EAD total | $478,547,962.19; to the dollar |
| 11 | SA-CCR EAD, CSA-005 | $82,064,083.32; to the dollar |
| 12 | Time-zero revaluation | 8 of 8 CSA-005 trades match within $1 |
| 13 | Uncollateralised EPE | $9,769,587.42 vs $9,769,587 baseline |
| 14 | Spread anchor | 264.85 bp; exact pass |
| 15 | Collateralised exposure ordering | Every path/date lies in [0, uncollateralised exposure] |
| 16 | Collateralised EPE | $794,518.13 vs $794,518 baseline |
| 17 | Collateralised CVA | $142,758.57 vs $142,759 baseline |
| 18 | Uncollateralised CVA | $1,769,585.04 vs $1,769,585 baseline |
| 19 | Low-spread CVA sensitivity | $47,551.22 vs $47,551 baseline |
| 20 | High-spread CVA sensitivity | $184,333.38 vs $184,333 baseline |
| 21 | WWR beta = 0 identity | Joint CVA $142,758.57 equals flat-spread CVA |
| 22 | WWR independent CVA | $145,335.08 vs $145,335 baseline |
| 23 | WWR joint CVA | $146,251.56 vs $146,252 baseline |
| 24 | WWR add-on | $916.48 vs $916 baseline |
| 25 | WWR add-on sign | Positive; wrong-way direction |
| 26 | WWR beta = 0.5 joint CVA | $143,827.74 vs $143,828 baseline |
| 27 | WWR beta = 1.5 joint CVA | $150,058.28 vs $150,058 baseline |
| 28 | Uncollateralised WWR add-on | $25,698.20 vs $25,698 baseline |
| 29 | Spread-tercile EPE ordering | Low <= mid <= high average EPE |

### 5.2 Version 2 reconciliation - 15 of 15 passed

| # | Check | Result |
|---:|---|---|
| 1 | Full-book CSA-005 uncollateralised EPE regression | $9,769,587.42; pass within 0.5% |
| 2 | Full-book CSA-005 collateralised EPE regression | $794,518.13; pass within 0.5% |
| 3 | Full-book CSA-005 uncollateralised CVA regression | $1,769,585.04; pass within 0.5% |
| 4 | Full-book CSA-005 collateralised CVA regression | $142,758.57; pass within 0.5% |
| 5 | Time-zero revaluation in all sets | All trades in all 8 sets match |
| 6 | Collateralised EPE ordering in every set | Collateralised EPE in [0, uncollateralised EPE] for all 8 sets |
| 7 | Portfolio totals equal counterparty sums | Maximum EPE/CVA gap $0.0000 |
| 8 | Counterparty CVA aggregation | Each counterparty equals the sum of its sets; maximum gap $0.0000 |
| 9 | SA-CCR EAD identities | `EAD = 1.4(RC + PFE)` holds for v1, v2 and collateralised-RC variants; maximum gap below $0.01 |
| 10 | Bucketed IR add-on bound | For all 13 currency hedging sets, v2 add-on is non-negative and no larger than gross bucket add-on |
| 11 | Collateralised-RC direction for positive-VM sets | Collateralised-RC EAD <= v2 EAD for all 5 positive-VM sets |
| 12 | PFE multiplier bounds | All multipliers lie in [0.05, 1] |
| 13 | Volatility-stress direction | Both uncollateralised and collateralised EPE rise under x1.5 volatilities |
| 14 | Convergence table | Four required path counts are populated and finite; spacing is reported rather than forced (see Section 6.3) |
| 15 | Sensitivity grid | 3 x 4 grid is complete; CVA increases with spread at each MPOR and EPE is spread-invariant within MPOR |

### 5.3 Pytest - 12 of 12 passed

The seven v1 tests re-derive RC flooring, the netting identity, SA-CCR totals, collateral EPE reduction, CVA ordering, the WWR add-on and spread-tercile ordering from output CSVs. Five v2 tests cover:

1. full-book CSA-005 regression;
2. collateralised <= uncollateralised for EPE, peak PFE95 and CVA in every set;
3. all three SA-CCR v2 EAD identities and the positive-VM direction;
4. sensitivity-grid shape and CVA monotonicity in spread; and
5. convergence/stress output coherence, including the volatility-stress direction.

## 6. Findings

### 6.1 P1 - Full-book exposure and CVA

| Counterparty | Sets | EPE uncoll. | EPE collat. | CVA uncoll. | CVA collat. | Share of collat. CVA |
|---|---:|---:|---:|---:|---:|---:|
| Bank1 | 1 | $72,725,582 | $4,222,695 | $12,042,844 | $731,232 | 38.2% |
| Bank3 | 1 | $27,078,139 | $1,758,406 | $5,146,662 | $358,447 | 18.7% |
| Fund1 | 1 | $27,412,343 | $1,833,646 | $3,820,611 | $270,846 | 14.1% |
| Bank2 | 1 | $72,831,027 | $2,491,918 | $7,113,332 | $264,478 | 13.8% |
| Fund2 | 2 | $18,151,565 | $1,628,616 | $2,795,102 | $249,308 | 13.0% |
| Fund3 | 2 | $2,273,141 | $493,694 | $184,677 | $41,742 | 2.2% |
| **Total** | **8** | **$220,471,795** | **$12,428,975** | **$31,103,226** | **$1,916,052** | **100%** |

Principal observations:

- Collateral reduces summed EPE by 94.4% and summed CVA by 93.8%. These are large reductions because the model's day-one balance starts at the required level and margin is thereafter simulated without disputes, haircuts or failed transfers.
- Bank1 is the largest collateralised CVA contributor despite Bank2 having a marginally larger uncollateralised EPE. Bank1 combines the largest collateralised EPE ($4.22 million) with a 251.30 bp spread. Bank2's spread is lower at 151.50 bp and its collateralised EPE is $2.49 million.
- Fund2 and Fund3 each hold two agreements. Their counterparty figures are transparent sums of set-level calculations; they are not the result of a single joint default-time simulation.
- CSA-005's full-book row exactly preserves its v1 EPE and CVA, demonstrating that the generalisation did not disturb the prototype's path stream.

### 6.2 P2 - SA-CCR v2

| Measure | Total |
|---|---:|
| AddOn, v1 | $263,208,660 |
| AddOn, v2 bucketed | $263,861,771 |
| EAD, v1 | $478,547,962 |
| EAD, v2 bucketed | $479,462,602 |
| Change from bucketing | **+$914,640 (+0.19%)** |
| EAD, v2 plus collateralised RC | $367,877,503 |
| Change from collateralised RC vs v2 | **-$111,585,099 (-23.3%)** |

Only four agreements change under bucketing:

- CSA-007: EAD increases by $619,161;
- CSA-001: EAD increases by $349,548;
- CSA-008: EAD decreases by $25,526; and
- CSA-004: EAD decreases by $28,543.

The increases are economically interpretable. In CSA-007 and CSA-001, rates positions in different maturity buckets offset under v1's single signed sum for a currency. The quadratic bucket formula gives only partial offsetting credit, so the add-on rises. Where bucket amounts have the same sign, the 50% cross-bucket correlation is below perfect correlation and the quadratic result can be slightly below the v1 absolute sum.

For the five positive-VM agreements, collateralised RC lowers EAD in every case. The total decrease is partly offset by the three negative-MTM, zero-VM agreements (CSA-002, CSA-003 and CSA-004): their RC rises from zero to the $550,000 or $1,100,000 `TH + MTA` floor specified by the formula, adding $3,080,000 of EAD in aggregate. This is not interpreted as collateral increasing risk; it is the prescribed positive floor in the simplified margined-RC formula when no VM is held at the valuation date.

### 6.3 P3 - Convergence

| Paths | Uncoll. EPE | Difference vs 10k | Collat. EPE | Difference vs 10k | Collat. CVA | Difference vs 10k |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | $9,920,343 | +1.93% | $797,982 | +0.22% | $143,038 | +0.03% |
| 2,500 | $9,847,032 | +1.18% | $796,220 | +0.00% | $143,143 | +0.10% |
| 5,000 | $9,769,587 | +0.38% | $794,518 | -0.21% | $142,759 | -0.16% |
| 10,000 | $9,732,643 | - | $796,216 | - | $142,993 | - |

The 5,000-path production-equivalent run is adequate for the project's headline precision: collateralised EPE and CVA are each within about 0.25% of the 10,000-path estimates, and uncollateralised EPE is within 0.38%. Convergence is not monotone in every statistic. In particular, the 1,000-path collateralised CVA is $45 from the 10,000-path estimate while the 5,000-path estimate is $234 away. Because each path count draws a different cross-section (the generator's array shape changes), this is ordinary Monte Carlo noise rather than evidence that 1,000 paths are generally more accurate. The reconciliation therefore reports spacing but does not force a monotone-spacing assertion.

### 6.4 P3 - Stress results

| Scenario | Uncoll. EPE | Change | Collat. EPE | Change | Peak PFE95, uncoll. | Change | Collat. CVA | Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | $9,769,587 | - | $794,518 | - | $63,677,741 | - | $142,759 | - |
| Volatilities x1.5 | $12,285,563 | +25.8% | $987,025 | +24.2% | $86,141,484 | +35.3% | $179,098 | +25.5% |
| Correlations +0.20 | $10,251,527 | +4.9% | $814,415 | +2.5% | $67,898,026 | +6.6% | $146,644 | +2.7% |

The volatility stress is materially stronger than the uniform correlation shift. This is plausible for CSA-005: wider factor distributions directly widen most trade values, whereas a common correlation shift has offsetting effects across long/short positions and asset classes. The correlation stress also makes the SPX-credit relationship less negative (-0.50 to -0.30), moderating one wrong-way channel even as most other correlations rise.

### 6.5 P3 - MPOR and spread sensitivity

At the 264.85 bp base spread:

| MPOR | Collateralised EPE | Collateralised CVA |
|---:|---:|---:|
| 5 business days | $531,943 | $93,022 |
| 10 business days | $794,518 | $142,759 |
| 20 business days | $1,312,529 | $240,615 |

Exposure and CVA increase strongly with MPOR because the collateral balance available at a reporting date is older and less responsive to intervening exposure increases.

At the base 10-day MPOR:

| Spread shift | Spread | Collateralised CVA |
|---:|---:|---:|
| -100 bp | 164.85 bp | $92,383 |
| Base | 264.85 bp | $142,759 |
| +100 bp | 364.85 bp | $189,339 |
| +300 bp | 564.85 bp | $272,475 |

CVA is monotone in spread at every MPOR, as expected under the flat-hazard formula. The response is concave rather than linear because survival probability declines exponentially as hazard rises.

## 7. Remaining limitations

1. **Synthetic and uncalibrated inputs.** Trades, spreads, volatilities, correlations, CSA terms and market dynamics are scenario assumptions. No market calibration, backtest against realised defaults or independent price verification has been performed.
2. **Approximate valuation.** Forward-style, annuity-scaled and Black-76 approximations are not curve-based production pricers. Rates and credit level units are inferred from magnitude, and all trades are calibrated to reported time-zero MTM rather than independently priced from contractual cash flows.
3. **SA-CCR remains simplified.** The v2 IR bucket implementation uses remaining maturity as the bucket tenor, the v1 supervisory-duration approximation, and a uniform 50% pairwise bucket correlation. It does not implement the full Basel trade-date/effective-notional schedule, every prescribed bucket correlation, or all SA-CCR collateral and maturity details. Other asset classes retain v1 aggregation simplifications.
4. **Collateral is stylised.** The model excludes initial margin, haircuts, collateral currency mismatch, interest on collateral, segregation, disputes, failed margin calls and dynamic CSA renegotiation. VM in the SA-CCR v2 calculation is a model-generated day-one required balance, not an observed custodian balance. NICA is set to zero.
5. **CSA terms are inferred, not legal inputs.** The source records conflicting threshold/MTA combinations within agreements. Modal inference is transparent and deterministic, but a production model must use executed CSA documents.
6. **No joint portfolio simulation.** Netting sets are simulated independently with different seeds. Summed EPE and CVA do not capture cross-set exposure dependence, common shocks at a portfolio horizon, or diversification. Counterparty aggregation also sums set-level CVAs computed from set-specific median spread proxies rather than one counterparty hazard curve.
7. **CVA is deliberately narrow.** It uses one flat hazard and one flat discount rate per set. It excludes DVA, FVA, funding/liquidity costs, capital costs, rating migration, stochastic recovery and single-name credit calibration. Wrong-way risk remains a CSA-005 analysis only.
8. **Monte Carlo resolution.** Monthly exposure dates and daily collateral interpolation are approximations. Five thousand paths are sufficient for the reported headline stability, but tail measures and small differences between scenarios remain subject to Monte Carlo error.
9. **Stress scenarios are not forecasts.** The x1.5 volatility and +0.20 correlation shifts are transparent sensitivity designs, not historically calibrated crisis scenarios. Their results should not be interpreted as regulatory stress-test outcomes.

## 8. Conclusion

The implementation passes its full regression and internal-consistency suite. Version 2 removes the largest v1 scope limitation by producing exposure and CVA for every netting set while preserving CSA-005's v1 path stream and results. The SA-CCR extension demonstrates that maturity-bucket offsetting assumptions have a modest net effect on this book (+0.19% of EAD), whereas recognising valuation-date VM through collateralised replacement cost has a much larger effect (-23.3% from v2 EAD). CSA-005 remains materially sensitive to volatility and MPOR, and its 5,000-path estimates are stable enough for the educational conclusions drawn.

**Validation opinion:** Approved for educational demonstration and interview/project discussion, subject to the limitations above. Not approved as a production CCR, CVA or regulatory-capital model.

**Prepared by:** Cong Wang
