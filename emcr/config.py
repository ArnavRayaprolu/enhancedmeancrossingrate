"""Frozen study protocol, version 3 (pre-registered).

This file fixes every parameter that determines the PRIMARY results and was
committed to git before the cryptocurrency datasets were downloaded; the
commit timestamp is the pre-registration record.  Version 3 replaces version 2
(commit 323aa90) at the authors' request: the study is re-focused on
cryptocurrencies at bar lengths of one hour or less, and the selection
statistic is re-defined so that it literally counts crossings of the price
ratio through its own mean.  No hold-out crypto result had been seen when
this change was made.

The strategy (deliberately the textbook rule, no tuning)
--------------------------------------------------------
    ratio        r_t = P^A_t / P^B_t
    mean / sd    trailing 140-bar mean and standard deviation of r
    z-score      z_t = (r_t - MA_140(r)_t) / SD_140(r)_t
    entry        z_t <= -2  buy the ratio  (long A, short B)
                 z_t >= +2  sell the ratio (short A, long B)
    exit         when the ratio returns to its mean (z crosses 0)
    selection    5 pairs per formation window, no coin used twice
    capital      rank-weighted across the pairs with an open position (Eq. 8)
    walk-forward 630-bar formation window, 315-bar trading window, non-overlapping

The proposed selection criterion
--------------------------------
    MCR  (Mean-Crossing Rate) = share of consecutive formation bars at which the
         price ratio crosses its own trailing mean, i.e. sign(r_t - MA_140(r)_t)
         changes.  Higher is better.
    CCR  (Completed-Cycle Rate) = number of completed "diverge to 2 SD and come
         back to the mean" cycles per formation bar (the in-sample count of the
         round trips the rule above would have completed).  Higher is better.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

PROTOCOL_VERSION = "3.0"
PROTOCOL_FROZEN_ON = "2026-11-01"

# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
# LEVEL  : z-score of the price ratio around its own trailing mean  (PRIMARY)
# HEDGED : z-score of the Engle-Granger spread log P_A - beta log P_B  (robustness)
# MAD    : (MA_35 - MA_140) / SD_140 of the ratio, the original submission's
#          signal (robustness / continuity only)
SIGNALS = ("LEVEL", "HEDGED", "MAD")
PRIMARY_SIGNAL = "LEVEL"

# --------------------------------------------------------------------------- #
# Criteria
# --------------------------------------------------------------------------- #
CLASSICS = ("Distance", "Correlation", "Cointegration")
BENCHMARKS = CLASSICS + ("HalfLife",)
PROPOSED = ("MCR", "CCR", "MCR+Coint", "CCR+Coint")
CRITERIA = BENCHMARKS + PROPOSED
HYBRIDS = {"MCR+Coint": ("MCR", "Cointegration"), "CCR+Coint": ("CCR", "Cointegration")}

# Pre-registered primary hypotheses, tested on every dataset:
#   (P1) pair level: the fold-average rank IC between formation-window MCR and
#        out-of-sample net pair return is positive;
#   (P2) portfolio level: the MCR portfolio earns a higher mean net return than
#        each classical criterion (Diebold-Mariano, Holm over the 3 comparisons).
PRIMARY_TESTS = tuple(("MCR", b) for b in CLASSICS)


# --------------------------------------------------------------------------- #
# One backtest configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    signal: str = PRIMARY_SIGNAL
    window: int = 140              # trailing mean / sd window (bars)
    entry: float = 2.0
    exit: float = 0.0              # 0 = exit when the ratio is back at its mean
    n_pairs: int = 5
    weighting: str = "active"      # "active" (rank-weighted, Eq. 8) or "equal" (1/N)
    fast: int = 35                 # MAD only

    @property
    def key(self) -> str:
        return (f"{self.signal}|w{self.window}|e{self.entry:g}|x{self.exit:g}"
                f"|n{self.n_pairs}|{self.weighting}")

    @property
    def variant(self) -> tuple:
        """What the z-score depends on (signal, fast window, slow window)."""
        return (self.signal, self.fast if self.signal == "MAD" else 0, self.window)


PRIMARY = Config()
MAD_ORIGINAL = Config(signal="MAD", entry=1.0, exit=0.25)
HEDGED_BASE = Config(signal="HEDGED")

# Robustness grid (all results reported, none used to pick a winner).
GRID_WINDOW = (70, 140, 280)
GRID_ENTRY = (1.5, 2.0, 2.5)
GRID_EXIT = (0.0, 0.5)
GRID_N = (5, 10, 20)
GRID_WEIGHTING = ("active", "equal")


def grid() -> list[Config]:
    out = [Config("LEVEL", w, e, x, n, wt)
           for w, e, x, n, wt in product(GRID_WINDOW, GRID_ENTRY, GRID_EXIT, GRID_N,
                                         GRID_WEIGHTING)]
    for base in (HEDGED_BASE, MAD_ORIGINAL):
        for wt in GRID_WEIGHTING:
            out.append(Config(base.signal, base.window, base.entry, base.exit, base.n_pairs,
                              wt, base.fast))
    return out


def detailed() -> list[Config]:
    """Configurations stored bar by bar (performance, costs, trades, DM tests)."""
    return [PRIMARY, Config(weighting="equal"), HEDGED_BASE, MAD_ORIGINAL]


# --------------------------------------------------------------------------- #
# Walk-forward and data handling
# --------------------------------------------------------------------------- #
TRAIN_BARS = 630
TEST_BARS = 315
MIN_COVERAGE = 0.95            # share of non-missing bars in the formation window
TOP_K = 40                     # universe: top-K coins by formation-window dollar volume


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Dataset:
    name: str
    source: str                # folder under data/
    resample: str | None       # pandas offset, None = native bars
    bars_per_year: float
    cost_bps: float            # one-way cost per dollar traded, per leg (fee + half-spread)
    borrow_pa: float           # annual financing charge on short-leg notional
    label: str
    pairs: str = "all"
    top_k: int | None = TOP_K
    role: str = "holdout"


H = 24 * 365
# spot: 10 bps taker fee + 1 bp half-spread; perpetuals: 5 bps taker + 1 bp.
# Short legs pay 10% a year (spot margin borrow; for perpetuals a conservative
# stand-in for funding paid).
DATASETS = {
    "binance_1h":  Dataset("binance_1h", "binance_spot_1h", None, H, 11.0, 0.10,
                           "Binance spot, 1h", role="primary"),
    "binance_30m": Dataset("binance_30m", "binance_spot_15m", "30min", 2 * H, 11.0, 0.10,
                           "Binance spot, 30m"),
    "binance_15m": Dataset("binance_15m", "binance_spot_15m", None, 4 * H, 11.0, 0.10,
                           "Binance spot, 15m"),
    "perp_1h":     Dataset("perp_1h", "binance_perp_1h", None, H, 6.0, 0.10,
                           "Binance perpetuals, 1h"),
    "coinbase_1h": Dataset("coinbase_1h", "coinbase_1h", None, H, 11.0, 0.10, "Coinbase, 1h"),
    "bitstamp_1h": Dataset("bitstamp_1h", "bitstamp_1h", None, H, 11.0, 0.10, "Bitstamp, 1h"),
    "okx_1h":      Dataset("okx_1h", "okx_1h", None, H, 11.0, 0.10, "OKX, 1h"),
    "bybit_1h":    Dataset("bybit_1h", "bybit_1h", None, H, 11.0, 0.10, "Bybit, 1h"),
    # Original submission's sample, re-run with the revised method (continuity).
    "sp500_1h":    Dataset("sp500_1h", "sp500_1h", None, 7 * 252, 5.0, 0.005,
                           "S&P 500, 1h (original sample)", pairs="sector", top_k=None,
                           role="original"),
}

COST_GRID_BPS = (0.0, 2.0, 5.0, 11.0, 20.0, 30.0)
SEED = 20261101

# Local-only test set: 39 Binance perpetuals from a public GitHub mirror of the
# Binance archive (data/_mirror/perp39_std, not committed).  The mirror owner
# chose the symbols, so it is survivorship-biased; used only to test the code.
DATASETS["mirror_perp_1h"] = Dataset("mirror_perp_1h", "_mirror/perp39_std", None, H, 6.0,
                                     0.10, "Perpetuals mirror, 1h (code test)", role="test")
