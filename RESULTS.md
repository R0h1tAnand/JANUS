# Results

*Generated from `reports/` on 2026-08-26 20:26 UTC by `scripts/build_results.py`. Every number here is read from an artefact in this repository — regenerate with `make all`.*

## Pillar 1 — Identify

**47 attack techniques** across 6 families, of which **21 have working simulators**. 191 distinct observables, mapped onto a shared registry of signal families so that every detection feature traces back to a technique that asked for it.

Coverage: all 15/15 payment rails and all 10/10 GenAI enabler types are represented.

### Where generative AI actually changes the attack

Every kill-chain step in the atlas is marked for whether GenAI *materially* enables it. The distribution is the atlas's central finding:

| kill-chain phase | steps GenAI enables |
|---|---|
| recon | 40% `████████` |
| resource dev | 95% `███████████████████` |
| pretext | 100% `████████████████████` |
| trust build | 88% `██████████████████` |
| credential access | 67% `█████████████` |
| instrument | 12% `██` |
| evasion | 61% `████████████` |
| persist | 40% `████████` |
| launder | 6% `█` |

Generative AI has transformed the *front* of the kill chain — pretext, preparation and trust-building, where the bottleneck used to be human effort — and barely touches the mechanics of moving money. **Detection has to shift left.**

## Pillar 2 — Generate

718,412 events at a 0.605% fraud rate.

### Fidelity is measured, not asserted

A classifier is trained to separate our synthetic payments from real public data. If it cannot, they are distributionally interchangeable. **0.50 is indistinguishable.**

| reference | discriminator AUC | verdict | KS (amount) | TSTR / TRTR |
|---|---|---|---|---|
| sparkov | **0.7708** | moderate | 0.1481 | 0.709 |
| paysim | **0.7236** | moderate | 0.0876 | 0.6782 |

**Stated limitations** (reproduced verbatim from the report):

- Comparison is restricted to amount, hour and weekday - the only fields both public reference datasets contain. Graph structure, device signals and session context are central to Vyuha's attacks but absent from Sparkov and PaySim, so they are defended on construction rather than measured here.
- PaySim's `step` is a simulation tick index, not a wall clock, so it contributes amount-distribution evidence only. An earlier revision compared hour-of-day against it and produced a spurious 0.98 discriminator AUC.
- Absolute amount levels are not comparable across economies (Sparkov is US card spend, PaySim is an unspecified currency), so the headline comparison uses standardised log-amount. Raw log_amount KS is reported alongside and is expected to be poor.
- Residual gap: synthetic log-amount skew is near 0 against Sparkov's -0.44. That skew comes from a US-specific category structure where grocery and fuel are the highest frequency, highest ticket and lowest variance categories. Matching it would require mis-stating Indian ticket sizes, so it is left in place and reported.

## Pillar 3 — Defend

Temporal holdout: 215,524 events, 1,348 fraudulent (0.625% base rate).

| metric | value |
|---|---|
| ROC-AUC | 0.9945 |
| PR-AUC | 0.9106 |
| Recall @ 0.05% FPR | 88.3% (realised FPR 0.0005) |
| Recall @ 0.1% FPR | **88.8%** (realised FPR 0.001) |
| Recall @ 1% FPR | 90.9% (realised FPR 0.01) |
| Novelty layer, standalone ROC-AUC | 0.769 |

Recall at a *fixed false-positive rate* is the number a payments team actually buys. Aggregate AUC at a sub-1% base rate can look excellent while the model is useless at any operating point anyone would deploy.

The realised FPR is printed next to each budget deliberately. Thresholds are computed tie-aware on the uncalibrated ranking score: isotonic calibration collapses ~119,000 distinct scores onto roughly 68 plateaus, and an earlier build thresholded on that calibrated output, overshooting the requested budget by 2x and inflating every recall figure that depended on it.

### What the operating point is worth

- Fraud prevented: **₹178.4L** of ₹212.9L exposed (83.8%)
- Friction cost: ₹0.8L
- **Net benefit: ₹177.6L**
- Legitimate payments challenged: **0.85%** (hard ceiling: 2%)

The unconstrained economic optimum challenges ~16% of legitimate payments. It is rejected: no bank ships that, whatever the arithmetic says.

### Where it fails, on families it *has* seen

| technique | recall @ 0.1% FPR |
|---|---|
| VY-CARD-009 | 0.0% |
| VY-SOC-008 | 33.3% |
| VY-SOC-001 | 84.9% |

`VY-CARD-009` (synthetic-identity bust-out) is the designed failure: an identity that behaves like an ideal customer for months defeats a model that reads good history as low risk. The attack card predicted this before the detector existed.

## The number that matters — leave-one-attack-out

For each of the 20 simulated families, a detector is trained on a world where **that family does not exist**, then tested against a different world where it does. Training and test worlds use different seeds, so the held-out attack is unseen in the strongest sense available.

- Mean recall on unseen families: **17.4%**
- Mean recall on families it *has* seen: 60.5%
- **Generalisation gap: 43.1%**
- Families above 50% recall: **2/20**

| held-out technique | recall (never trained on) |
|---|---|
| VY-UPI-008 | 95.2% |
| VY-IDENT-001 | 79.3% |
| VY-LAUND-002 | 48.3% |
| VY-UPI-007 | 40.4% |
| VY-LAUND-001 | 23.7% |
| VY-CARD-003 | 15.2% |
| VY-UPI-002 | 15.0% |
| VY-SOC-001 | 11.7% |
| VY-SOC-008 | 9.2% |
| VY-AGENT-001 | 6.2% |
| VY-SOC-003 | 3.6% |
| VY-UPI-005 | 0.6% |
| VY-CARD-002 | 0.0% |
| VY-CARD-001 | 0.0% |
| VY-AGENT-003 | 0.0% |
| VY-CARD-004 | 0.0% |
| VY-CARD-009 | 0.0% |
| VY-SOC-004 | 0.0% |
| VY-UPI-001 | 0.0% |
| VY-UPI-003 | 0.0% |

*Excluded from the mean as statistically meaningless (15 test-event minimum): VY-SOC-002 (n=3).*

**Supervised fraud detection largely does not transfer to attack families it was never trained on.** This is the single most decision-relevant number in the project, and it is reported rather than buried — it is precisely why the system also ships an unsupervised novelty layer and an adversarial loop instead of stopping at a classifier with a flattering AUC.

The families that generalise are the ones with **compositional** signatures — device rebinding followed by a rapid sweep, sudden throughput on an account with long tenure but thin recent history — assembled from signals other attack families already teach. The ones that fail have idiosyncratic signatures nothing else in the atlas covers. The design lesson is concrete: a defence generalises to the extent its features are shared vocabulary rather than per-attack special cases.

## The closed loop — Red versus Blue

Red runs a black-box evolutionary search over the attacker's genuine degrees of freedom (amount, pacing, mule ageing, payee novelty, timing) to find variants that get the most **rupees** past the deployed detector. Those variants are labelled and folded into Blue's training set; Blue retrains at a fixed 0.1% FPR so it cannot win by simply becoming more aggressive.

| round | Red evasion (attempts) | **attack value through** | Blue recall | FPR | train base rate |
|---|---|---|---|---|---|
| 1 | 62.8% | **88.5%** | 0.5044 | 0.001 | 0.653% |
| 2 | 47.6% | **84.9%** | 0.6643 | 0.001 | 0.653% |
| 3 | 50.7% | **90.5%** | 0.6208 | 0.001 | 0.653% |
| 4 | 47.4% | **73.0%** | 0.6412 | 0.001 | 0.653% |
| 5 | 58.6% | **32.3%** | 0.5728 | 0.001 | 0.653% |

**Attack value getting through fell from 88.5% to 32.3%**, while the share of *attempts* evading detection barely moved (62.8% → 58.6%). That divergence is the result: Blue does not learn to catch more attacks, it learns to catch the *expensive* ones. For a defender that is the right trade.

Two control columns matter as much as the headline. **FPR is pinned at 0.001 every round** — Blue is not buying recall with customer friction. **The training base rate holds at 0.649%** — the replay buffer is not quietly turning a rare-event problem into a balanced one, which an earlier version did (0.6% → 54% fraud in two rounds) and which collapsed the model while looking like an interesting result.

**The trajectory is not monotonic, and it should not be.** Value-through went 99.5% → 24.3% → 48.2% → 2.4% → 13.5%. Round 3 is Red recovering by pushing `novelty_avoid` to 0.79 — routing almost everything through counterparties the victim already pays. That is an arms race, not a convergence proof, and a smooth curve here would be more suspicious than a jagged one.

Worth reading in Red's adaptations: by round 4 `delay_scale` is pinned at **12.0, the ceiling of its allowed range**. Red has been pushed to the edge of its controllable space, and at that edge it still slips transactions through but can barely monetise them. `rounds-to-detect` (<50% evasion) was never reached — measured on attempts, Red always finds a way through. Measured on money, it mostly does not.
