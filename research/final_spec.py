"""FROZEN finalist (chosen on 2020-2023 development data only; frozen before the
2024-2026 hold-out was evaluated).  Do not edit after the hold-out run.

Strategy
--------
universe   the 100 most liquid coins by median formation-window dollar volume
           (>= 95% coverage), all pairs of them as candidates
criterion  MCR = share of formation bars at which the price ratio P_A/P_B crosses
           its own trailing 24-bar (24-hour) mean; highest MCR ranked first
selection  top 5 pairs, no coin used twice
signal     z_t = (r_t - MA_24(r)_t) / SD_24(r)_t,  r_t = P_A/P_B (hourly bars)
entry      z <= -3: long A / short B;  z >= +3: short A / long B
exit       when z has crossed 0.5 SD beyond the mean (z >= +0.5 for a long ratio
           position, z <= -0.5 for a short one); start flat, liquidate at window end
capital    1/5 of capital per pair, equal dollars in each leg, idle when flat
walk-fwd   630-bar formation, 315-bar trading window, non-overlapping
costs      one-way cost per dollar traded per leg: 6 bps (Binance USDT-M perpetuals,
           taker fee + half-spread); spot venues at 11 bps plus 10%/yr borrow on shorts

Development evidence (Binance perpetuals, 2020-01 .. 2023-12, 108 folds):
Sharpe 2.06, total return +191%; every calendar year positive (0.64, 1.96, 2.70,
3.36); 358 development trials logged in results/dev_search/trials.csv.
"""
import sys

sys.path.insert(0, ".")
from emcr.explore import Spec  # noqa: E402

RULE = dict(signal="LEVEL", window=24, entry=3.0, exit=-0.5, stop=None, max_hold=None,
            n_pairs=5, max_uses=1, top_k=100, weighting="equal")
FINAL = Spec(criterion="MCR", **RULE)
COMPARE = ["MCR", "HalfLife", "Cointegration", "Distance", "Correlation"]
HOLDOUT_VENUES = ["perp_1h", "binance_1h", "bitstamp_1h", "coinbase_1h", "okx_1h"]
