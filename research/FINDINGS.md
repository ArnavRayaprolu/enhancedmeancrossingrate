# EMCR innovation research (daily lab) - findings

Harness: `research/emcr_lab.py`. Daily bars built from the Alpaca hourly data
(500-symbol point-in-time universe, 2016-2026): signal = 15:00 ET price,
execution = 16:00 close. DEV = test windows starting before 2023-01-01;
HOLDOUT = 2023-01-01 onward, used once at the end. Every evaluated design is in
`research/trials.csv` (search over slow designs: `research/search_slow_designs.csv`).
Scripts for each batch: `research/scripts/`.

## What was tried (in order)
1. Classic Gatev design (12m formation / 3m trading, static z, entry 2 sigma) with the
   literature's EMCR = zero crossings of the formation-demeaned spread (Do & Faff 2010),
   amplitude-filtered crossings (only after a 1 or 2 sigma excursion), cost-aware
   crossings, two-half stability, formation rule P&L. Pair-level IC of every statistic
   with the pair's own out-of-sample net return: all |IC| < 0.02, none significant.
2. Short-horizon reversal (rolling 3/5/10/20-day z, 6m/1m windows) with EMCR measured on
   the same short signal, plus "EMCR-win" (profitable completed reversions), "EMCR-net"
   (profitable minus losing) and "EMCR-Sharpe" (consistency of reversions).
   With per-stock Abdi-Ranaldo costs the IC looked very strong (t ~ 20), but the daily
   spread estimates for large caps are volatility noise (median 9 bps, 2020 ~28 bps) and
   the same estimates entered both the formation statistic and the test P&L. With a
   flat, realistic cost (2 bps half-spread + 0.5 bp fees, 0.5%/yr borrow) the IC is ~0
   for every statistic and every book loses money (turnover 30-90x per year).
3. Same-industry candidates (GICS sector / sub-industry). Fast rules still lose.
   Slow static rules make money: classic Correlation up to Sharpe 1.2-1.5 on DEV, EMCR
   variants alone <= 0.9.
4. Systematic DEV search, 4,104 slow designs (sector/sub-industry x 3 window pairs x
   entry 1.5/2/2.5 x exit 0/0.5 x 20/50/100 pairs x stop 0/4 x 19 criteria):
   best = Correlation|EMCR1 (top 20% by correlation, ranked by completed 1-sigma
   reversions), DEV Sharpe 1.57 at 1.9%/yr return and 1.2% volatility; plain
   Correlation 1.48. EMCR-only criteria average Sharpe 0.02 (max 0.88).
5. Multi-horizon ensembles: no improvement (EMCR hybrid only works on sub-industry pairs).

## Holdout (2023-01 to 2026-10, evaluated once)
| design | DEV Sharpe | HOLDOUT Sharpe |
|---|---|---|
| Correlation\|EMCR1, sub-industry, 6m/3m (best DEV) | 1.57 | -0.33 |
| Correlation, same design | 1.40 | -0.64 |
| Correlation, sub-industry, 12m/6m (best DEV classic) | 1.48 | 0.32 |
| Correlation\|EMCR1, same design | 1.36 | 0.17 |

Holdout DM tests: hybrid vs Correlation p = 0.28 and 0.61.

## Conclusion
No selection statistic, EMCR or classical, predicts which US large-cap pairs will be
profitable out of sample, at hourly or daily frequency, once costs are modelled
consistently. The best DEV designs earn ~2%/yr at ~1% volatility and do not survive the
untouched 2023-2026 sample. A Sharpe ratio of 2 with EMCR clearly ahead of the classical
criteria was not found; reaching it by further searching would be specification search
of the kind the referee warned against.

## Phase 2 - other markets and new EMCR variants (full walk-forward 2016-2026, no split)

Data: Alpaca daily OHLCV for 6,251 stocks (889 delisted) and 5,346 unleveraged ETFs/funds;
signal at the close, execution at the next open; cost tiers 2-30 bps by dollar volume;
data-quality screen for one-day moves of 3x or more (e.g. AQB, Jan 2017).

* Market scan (`research/results/market_scan_wf_*.csv`), 18 criteria x 5 designs x 2 book sizes:
  best Sharpe per market - US small caps 0.47 (EMCR-win), US large 0.43 (Cointegration),
  US mid 0.36 (Correlation), ETFs 0.10 (EMCR-win). Average Sharpe is negative in every market.
* New EMCR variants (`research/emcr_lab.py`): Kalman-filter EMCR (completed reversions of the
  standardised prediction error, time-varying hedge ratio), volume-weighted MA-crossover EMCR,
  variance ratio, with static, Kalman and volume-weighted MA trading signals
  (`research/results/innov_scan_*.csv`).
* Breadth / equal-risk weighting up to 400 pairs (`research/results/breadth_scan_*.csv`):
  more pairs does not raise the Sharpe ratio.
* Stock-vs-industry-basket strategy with EMCR selection (`research/basket_lab.py`,
  `research/results/basket_scan.csv`): Sharpe <= 0.08 for every criterion.

Best EMCR-led result: large caps, same GICS sub-industry candidates, 6m formation / 3m
trading, static z (entry 2.5, exit 0.5), 50 pairs, criterion Correlation|EMCR1 (top half by
correlation, ranked by completed 1-sigma reversions):
Sharpe 0.78, 1.0%/yr at 1.3% volatility, +10.8% over 2016-2026 (Correlation 0.50,
Cointegration -0.34, Distance -0.06). DM (Newey-West / bootstrap): vs Cointegration p = 0.011,
vs Distance p = 0.012, vs Correlation p = 0.13. Deflated Sharpe over all 5,706 logged trials: 0.00.
Performance is concentrated in 2016-2020 (yearly Sharpe 1.5, -0.3, 1.8, 3.1, 2.8) and is
weak or negative after 2021.
