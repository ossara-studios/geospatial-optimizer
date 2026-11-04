# Hillside backtest results

Retrospective replay of a real, completed drilling program. Numbers come from
`runs/backtest/summary.json` (regenerate with `geoopt prepare && geoopt backtest`; about 1 h on a
laptop CPU). Charts are in `runs/backtest/backtest.html`, and `geoopt viz` shows them under Backtest.

## Data
- **Deposit:** Hillside Cu-Au, Yorke Peninsula, South Australia. Drilled by Rex Minerals between
  September 2006 and March 2013. Source is SARIG, Department for Energy and Mining, Government of
  South Australia, licensed CC-BY 4.0. See [data_schema.md](data_schema.md).
- **After cleaning:** 766 holes with Cu assays, 18,699 composites of 10 m each. Assays are ICP-MS
  (AMDEL) and XRF (ALS).
- **Per seed set:** 60 seed holes are known at the start, and the remaining 706 are candidates whose
  assays only the lab simulator holds.

| Seed set | How the 60 seed holes are chosen |
|---|---|
| `earliest60` | First 60 holes by drill date (Sep 2006 to Dec 2009, before the 2010–2012 resource drilling) |
| `spread60-a`, `spread60-b` | 60 holes spread across the deposit (farthest-point, RNG seeds 1 and 2) |

## Ore model check (seed holes, whole holes held out, 5 folds)
| | Value |
|---|---|
| RMSE of log grade | 1.69 (vs 1.86 predicting the training mean) |
| Observed within ±1σ / ±2σ | 66 % / 95 % (ideal 68 % / 95 %) |

Reliability (predicted P(Cu > 0.2 %) against observed ore fraction):

| Predicted bin | n | Mean predicted | Observed |
|---|---|---|---|
| 0.0–0.2 | 677 | 0.14 | 0.15 |
| 0.2–0.4 | 355 | 0.28 | 0.33 |
| 0.4–0.6 | 86 | 0.49 | 0.36 |
| 0.6–0.8 | 70 | 0.69 | 0.63 |
| 0.8–1.0 | 29 | 0.89 | 0.59 |

The uncertainty is well calibrated. High probabilities are somewhat overconfident.

## Setup
- Every strategy starts from the same seed set and candidate pool, with a 15,000 m budget. A run
  stops at the first hole that reaches it.
- `adaptive` uses β = 0.5, z-scored terms and a 25 m soft spacing penalty.
- Baselines:
  - `historical`: Rex's actual drill order.
  - `grid`: 100 m grid.
  - `random`: 5 RNG seeds.
- That makes 24 runs: 3 seed sets × (3 strategies + 5 random seeds).
- Cutoff is Cu > 0.2 % (placeholder).

## Discovery: ore metres intersected
Mean across seed sets (min–max in brackets; random also spans its 5 RNG seeds).

| Strategy | at 3,750 m | at 7,500 m | at 15,000 m |
|---|---|---|---|
| **adaptive** | **1,243** (586–1,622) | **2,573** (2,412–2,831) | **4,772** (4,372–5,098) |
| random | 497 (330–658) | 1,026 (810–1,210) | 2,143 (1,786–2,492) |
| historical | 393 (240–699) | 769 (629–919) | 1,891 (1,776–1,991) |
| grid | 331 (297–390) | 538 (504–607) | 1,291 (1,210–1,403) |

At the full budget adaptive finds **2.5× the ore metres of the historical drill order** and 2.2×
random. Its worst run beats every baseline's best run.

### Metres to match each baseline's discovery
These are the metres adaptive needed to reach the ore metres each baseline found with 15,000 m.

| Baseline | Adaptive metres to match, per seed set (earliest60 / spread60-a / spread60-b) | Metres saved, mean (range) | Holes saved, mean |
|---|---|---|---|
| historical | 6,842 / 4,777 / 4,413 | 9,656 (8,158–10,588), 64 % | 24 |
| random | 7,165 / 5,390 / 5,061 | 9,128 (7,835–9,939), 61 % | 19 |
| grid | 4,717 / 3,068 / 2,893 | 11,441 (10,283–12,107), 76 % | 33 |

**Environmental ESTIMATES vs historical.** The factors are placeholders, not sourced. Replace them
before any external use.

| Item | Estimate |
|---|---|
| Water | ≈ 0.97 ML |
| Diesel | ≈ 29,000 L |
| Drill pads | ≈ 4,800 m² |

## Delineation: F1 on undrilled held-out composites
P(ore) > 0.5 counts as ore.

| Strategy | at 3,750 m | at 7,500 m | at 15,000 m |
|---|---|---|---|
| adaptive | 0.10 (0.08–0.15) | 0.11 (0.07–0.14) | 0.10 (0.05–0.14) |
| random | 0.13 (0.09–0.18) | 0.15 (0.10–0.21) | 0.17 (0.11–0.23) |
| historical | 0.12 (0.11–0.14) | 0.14 (0.12–0.16) | 0.13 (0.07–0.21) |
| grid | 0.07 (0.06–0.08) | 0.05 (0.04–0.07) | 0.09 (0.06–0.13) |

No strategy reaches the F1 = 0.6 target set in advance. Adaptive is below random and historical.
Three things contribute:
- **Most of the deposit stays undrilled.** About 50–80 holes are drilled out of 706 candidates. With
  a 20 m east–west correlation length, the model reverts to background away from drilling, so few
  undrilled composites get P(ore) > 0.5. That keeps recall low for every strategy.
- **Adaptive exploits.** At β = 0.5 it favours holes near known ore. In `earliest60`, 17 of its 79
  holes were within 25 m of an existing hole. The spacing penalty is too weak to prevent this.
- **The metric is biased against strategies that find ore.** Adaptive drills the ore, so the
  undrilled set it is scored on is more waste-heavy.

**Shortcuts checked and not found.** Adaptive's mean hole length (190 m) matches the historical
drilling (193 m), and it drilled only 2 of the 15 wedge holes.

## Limitations
- **One deposit, retrospective.** This shows the method works at Hillside, not that it generalises.
- **The historical baseline is 2010-era decision-making.** Rex also used geology and geophysics we
  don't model, and some of its holes targeted resource definition rather than discovery. Random and
  grid are neutral baselines.
- **Straight holes.** No downhole surveys are used; holes follow the collar azimuth and dip.
- **Replay only.** Candidates are holes that were actually drilled, so the pool carries the original
  explorers' targeting bias, and collars can't be moved.
- **Placeholders:**
  - The 0.2 % Cu cutoff (should come from the Hillside resource report).
  - The no-go zones are illustrative and affect only 1 candidate.
  - The environmental factors.
  - The 500 m depth limit (clips 117 candidates).

## Next
A β sweep (1, 2, 4), adaptive only, all three seed sets, fixed in advance and reported as a
discovery-versus-delineation trade-off curve rather than tuned on held-out results.
