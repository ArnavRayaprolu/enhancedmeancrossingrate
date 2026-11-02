# MCR pairs selection: revised model

`MCR_Pairs_Trading.ipynb` is the original model submitted with the paper
(NAJEF-D-26-00879). The `mcr/` package is the revised model: the same design
(signal, thresholds, rank-weighted capital, walk-forward), with every change
requested by the reviewer implemented in code and the protocol frozen in
`mcr/config.py` before any result on the new data was computed. Earlier
experiments are kept, unused, in `archive/`.

## Run

```bash
python scripts/download_alpaca.py assets   # on GitHub Actions: workflow alpaca-data
python scripts/run_mcr.py run --workers 4  # walk-forward backtests -> results/raw/
python scripts/run_mcr.py report           # tables -> results/*.csv, results/REPORT.md
python -m pytest -q tests                  # integrity audit
```

## Data

Alpaca Market Data, US equities, 1-hour bars 2016 onward, split and dividend
adjusted (SIP feed). Active and inactive (delisted) assets are listed; every
symbol that ranked in the 1,500 most liquid common stocks in any calendar year
is downloaded, and each walk-forward window picks its own universe from
formation data only (`mcr/data.py`, `scripts/download_alpaca.py`).

## Where each reviewer comment is handled

| Comment | Implementation |
|---|---|
| R1 MCR interpretation | `signals.ma_signal` docstring: a crossing = fast/slow MA of the ratio cross, not the ratio crossing its mean. `selection.diagnostics` relates MCR to half-life, AR(1) of z, crossings of the cointegrating residual and of the ratio about its mean, completed engine cycles, and each pair's own out-of-sample trades and net return (report section R1). |
| R2 specification search | All parameters fixed in `config.py` before any result; independent sample (2016-2026, new universe); every configuration and criterion reported; Hansen SPA and White Reality Check over all trials (`inference.spa_test`). |
| R3 Diebold-Mariano | `inference.diebold_mariano` / `report.dm_tests`: d_t = n^A_t - n^B_t, loss = -net return, H0 E[d]=0 two-sided; Bartlett HAC with the paper's rule, 2x, 4x, Andrews (1991) and one-day lags; daily-aggregated test; stationary bootstrap with Politis-White block length under the null; Holm within and across families. |
| R4 reproducible engine | `engine.states` (rules for opposite extremes, consecutive signals, missing z, flat start, forced liquidation; no upper entry bound), `engine.pair_weights` (Eq. 8 made explicit), `selection.select` (ties, no reuse, per criterion), `walkforward` docstring = pseudocode. `engine_original` reproduces the notebook's rules. |
| R5 costs and break-even | `engine.run_book`: holdings drift, trades only on target changes, per-leg turnover, final liquidation, borrow on the short leg; costs per stock from the Abdi-Ranaldo spread estimator plus fees, stress scenario with square-root impact; break-even by sum of returns and by compounded wealth. |
| R6 sample, survivorship, missing data | 10 years, ~58 windows, delisted names included, point-in-time universe, regime/year table; `complete_case` variant without forward filling. |
| R7 ratio vs cointegration | `hedged_spread` variant trades log P_A - beta log P_B with the Engle-Granger beta and beta-weighted legs. |
| R8 robustness | `config.GRID_SPEC`: MA windows, entry/exit, number of pairs, capital rule, sector-constrained S&P universe, universe size, windows, delay, engine; all rows reported. |
| R9 deflated Sharpe, CIs | `inference.deflated_sharpe` over all trials; stationary-bootstrap CIs for Sharpe and total return. |
| m1-m2 naming, zeros | `signals.crossing_rate` rules for zeros, consecutive zeros, NaN. |
| m3 percentile rank | `selection.pct_rank`: average ties, orientation, missing = 0. |
| m4 comparability | one candidate set per window for all criteria; `report.composition`. |
| m5 adjusted data | split + dividend adjusted prices used for both signal and returns. |
| m6 metrics | `inference.metrics` / `report.performance` docstrings. |
| m7 concentration | `report.concentration`: pair, sector and P&L concentration, effective number of pairs; capped capital rule. |
