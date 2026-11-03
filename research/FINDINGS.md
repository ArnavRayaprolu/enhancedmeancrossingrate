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
