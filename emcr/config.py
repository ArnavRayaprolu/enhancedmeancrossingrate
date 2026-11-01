"""Study protocol, version 4 (the revision's design).

Back to the original EMCR design of the submission; one trading strategy is held
fixed across every universe and bar interval, so that differences in results
measure the quality of PAIR SELECTION only.

Fixed strategy
--------------
    ratio      r_t = P^A_t / P^B_t
    signal     z_t = (MA_35(r)_t - MA_140(r)_t) / SD_140(r)_t      (bars)
    EMCR       share of consecutive formation-window bars at which z changes sign,
               i.e. the empirical rate at which the moving-average spread of the
               ratio crosses its equilibrium (zero); higher ranks first
    entry/exit |z| >= 1.00 opens a position (long the ratio if z <= -1);
               closed once z has reverted to within 0.25 of zero (or crossed it)
    selection  top N = 20 pairs, no security used twice
    capital    rank-weighted: pair i receives Phi_i / sum_j Phi_j of capital,
               Phi = formation-window selection score, fixed for the window;
               capital of a flat pair stays in cash (no re-splitting churn)
    walk-fwd   630-bar formation, 315-bar trading window, non-overlapping
    costs      venue-specific fee + half-spread per dollar traded per leg, borrow
               on short stock legs, actual funding on crypto perpetuals, drift
               re-balancing and window-end liquidation charged

Selection methods (12): Distance, Correlation, Cointegration, Half-life, EMCR,
the hybrids EMCR+X = 1/2 rk(EMCR) + 1/2 rk(X) for every classical X, the three-way
EMCR+Coint+HalfLife, EMCR+All (1/2 EMCR + 1/2 mean of the four classics), and the
filter-then-rank EMCR|Coint (Engle-Granger p < 0.10, ranked by EMCR).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

PROTOCOL_VERSION = "4.0"

# --------------------------------------------------------------------------- #
# signals (MAD = original; LEVEL and HEDGED are robustness checks for the editor)
# --------------------------------------------------------------------------- #
SIGNALS = ("MAD", "LEVEL", "HEDGED")
PRIMARY_SIGNAL = "MAD"

# --------------------------------------------------------------------------- #
# selection methods
# --------------------------------------------------------------------------- #
CLASSICS = ("Distance", "Correlation", "Cointegration", "HalfLife")
BENCHMARKS = CLASSICS
HYBRIDS = {f"EMCR+{x}": ("EMCR", x) for x in CLASSICS}
EXTENDED = ("EMCR+Coint+HalfLife", "EMCR+All", "EMCR|Coint")
PROPOSED = ("EMCR",) + tuple(HYBRIDS) + EXTENDED
CRITERIA = BENCHMARKS + PROPOSED
# Primary hypotheses: each hybrid against its own classical component, EMCR against
# each classical criterion, and each extended hybrid against cointegration
# (Diebold-Mariano, Holm-adjusted).
PRIMARY_TESTS = (tuple((f"EMCR+{x}", x) for x in CLASSICS) + tuple(("EMCR", x) for x in CLASSICS)
                 + tuple((h, "Cointegration") for h in EXTENDED))


@dataclass(frozen=True)
class Config:
    signal: str = PRIMARY_SIGNAL
    window: int = 140              # slow window (bars)
    entry: float = 1.0
    exit: float = 0.25
    n_pairs: int = 20
    weighting: str = "rank"        # "rank" (fixed rank weights) or "active" (paper Eq. 8)
    fast: int = 35                 # MAD fast window (bars)

    @property
    def key(self) -> str:
        f = f"f{self.fast}" if self.signal == "MAD" else ""
        return (f"{self.signal}|{f}w{self.window}|e{self.entry:g}|x{self.exit:g}"
                f"|n{self.n_pairs}|{self.weighting}")

    @property
    def variant(self) -> tuple:
        return (self.signal, self.fast if self.signal == "MAD" else 0, self.window)


PRIMARY = Config()
ACTIVE_WEIGHTS = Config(weighting="active")                      # the submission's Eq. 8
LEVEL_BASE = Config(signal="LEVEL", entry=2.0, exit=0.5)          # classic ratio z-score
HEDGED_BASE = Config(signal="HEDGED", entry=2.0, exit=0.5)        # cointegrating spread
MAD_ORIGINAL = Config(n_pairs=5, weighting="active")              # the submission's spec


def detailed() -> list[Config]:
    """Configurations stored bar by bar (performance, costs, trades, DM tests)."""
    return [PRIMARY, ACTIVE_WEIGHTS, LEVEL_BASE, HEDGED_BASE, MAD_ORIGINAL]


# One-at-a-time sensitivity (editor: report favourable and unfavourable settings)
GRID_N = (5, 10, 20, 30)


def grid() -> list[Config]:
    out = list(detailed())
    for n in GRID_N:
        out.append(Config(n_pairs=n))
    for f, w in ((17, 70), (70, 280)):
        out.append(Config(fast=f, window=w))
    for e, x in ((0.75, 0.25), (1.25, 0.25), (1.0, 0.0), (1.0, 0.5)):
        out.append(Config(entry=e, exit=x))
    seen, uniq = set(), []
    for c in out:
        if c.key not in seen:
            seen.add(c.key); uniq.append(c)
    return uniq


# --------------------------------------------------------------------------- #
# walk-forward and data handling
# --------------------------------------------------------------------------- #
TRAIN_BARS = 630
TEST_BARS = 315
MIN_COVERAGE = 0.95


@dataclass(frozen=True)
class Dataset:
    name: str
    source: str                # folder under data/
    resample: str | None       # pandas offset, None = native bars
    bars_per_year: float
    cost_bps: float            # one-way cost per dollar traded per leg (fee + half-spread)
    borrow_pa: float           # annual borrow fee on short-leg notional (0 if funding used)
    label: str
    pairs: str = "all"         # "all" or "sector" (same group only)
    top_k: int | None = None   # universe: top-K by formation-window dollar volume
    universe: str = ""
    interval: str = ""
    role: str = "main"


# Real-world costs (one-way, per dollar traded, per leg):
#   crypto perpetuals : 5 bps taker fee + 1 bp half-spread, actual funding paid/received
#   S&P 500 stocks    : ~0.5 bp commission + 1.5 bp half-spread + impact -> 3 bps; 0.5%/yr borrow
#   NIFTY 50 stocks   : brokerage, STT, exchange and stamp charges + spread -> 5 bps;
#                       shorts via single-stock futures (no borrow fee)
#   index funds (ETFs): ~0.5 bp commission + 1 bp half-spread -> 1.5 bps; 0.3%/yr borrow
UNIV = {
    "crypto":  dict(cost=6.0, borrow=0.0, pairs="all", top_k=100, label="Crypto"),
    "sp500":   dict(cost=3.0, borrow=0.005, pairs="sector", top_k=None, label="S&P 500"),
    "nifty50": dict(cost=5.0, borrow=0.0, pairs="all", top_k=None, label="NIFTY 50"),
    "etf":     dict(cost=1.5, borrow=0.003, pairs="all", top_k=None, label="Index funds"),
}
# bars per year by interval: crypto trades 24/7; US equities 6.5h x 252 days;
# NSE 6.25h x 250 days
BPY = {
    "crypto": {"5m": 288 * 365, "15m": 96 * 365, "30m": 48 * 365, "1h": 24 * 365,
               "4h": 6 * 365, "1d": 365},
    "us": {"5m": 78 * 252, "15m": 26 * 252, "30m": 13 * 252, "1h": 7 * 252, "1d": 252},
    "in": {"5m": 75 * 250, "15m": 25 * 250, "30m": 13 * 250, "1h": 7 * 250, "1d": 250},
}
# (universe, interval) -> (data folder, resample rule)
SOURCES = {
    ("crypto", "5m"): ("binance_perp_5m", None),
    ("crypto", "15m"): ("binance_perp_15m", None),
    ("crypto", "30m"): ("binance_perp_15m", "30min"),
    ("crypto", "1h"): ("binance_perp_1h", None),
    ("crypto", "4h"): ("binance_perp_1h", "4h"),
    ("crypto", "1d"): ("binance_perp_1h", "1D"),
    ("sp500", "5m"): ("sp500_5m", None),
    ("sp500", "15m"): ("sp500_5m", "15min"),
    ("sp500", "30m"): ("sp500_5m", "30min"),
    ("sp500", "1h"): ("sp500_1h", None),
    ("sp500", "1d"): ("sp500_1d", None),
    ("nifty50", "5m"): ("nifty50_5m", None),
    ("nifty50", "15m"): ("nifty50_5m", "15min"),
    ("nifty50", "30m"): ("nifty50_5m", "30min"),
    ("nifty50", "1h"): ("nifty50_1h", None),
    ("nifty50", "1d"): ("nifty50_1d", None),
    ("etf", "5m"): ("etf_5m", None),
    ("etf", "15m"): ("etf_5m", "15min"),
    ("etf", "30m"): ("etf_5m", "30min"),
    ("etf", "1h"): ("etf_1h", None),
    ("etf", "1d"): ("etf_1d", None),
}

DATASETS: dict[str, Dataset] = {}
for (u, iv), (src, rs) in SOURCES.items():
    p = UNIV[u]
    cal = "crypto" if u == "crypto" else ("in" if u == "nifty50" else "us")
    DATASETS[f"{u}_{iv}"] = Dataset(f"{u}_{iv}", src, rs, BPY[cal][iv], p["cost"], p["borrow"],
                                    f"{p['label']}, {iv}", p["pairs"], p["top_k"], u, iv)

# Legacy dataset names used by the development-search scripts (research/).
_H = 24 * 365
DATASETS.update({
    "perp_1h": Dataset("perp_1h", "binance_perp_1h", None, _H, 6.0, 0.0, "Binance perps, 1h",
                       top_k=100),
    "perp_15m": Dataset("perp_15m", "binance_perp_15m", None, 4 * _H, 6.0, 0.0,
                        "Binance perps, 15m", top_k=100),
    "perp_30m": Dataset("perp_30m", "binance_perp_15m", "30min", 2 * _H, 6.0, 0.0,
                        "Binance perps, 30m", top_k=100),
    "binance_1h": Dataset("binance_1h", "binance_spot_1h", None, _H, 11.0, 0.10,
                          "Binance spot, 1h", top_k=100),
    "binance_15m": Dataset("binance_15m", "binance_spot_15m", None, 4 * _H, 11.0, 0.10,
                           "Binance spot, 15m", top_k=100),
    "binance_30m": Dataset("binance_30m", "binance_spot_15m", "30min", 2 * _H, 11.0, 0.10,
                           "Binance spot, 30m", top_k=100),
    "coinbase_1h": Dataset("coinbase_1h", "coinbase_1h", None, _H, 11.0, 0.10, "Coinbase, 1h",
                           top_k=100),
    "okx_1h": Dataset("okx_1h", "okx_1h", None, _H, 11.0, 0.10, "OKX, 1h", top_k=100),
    "bitstamp_1h": Dataset("bitstamp_1h", "bitstamp_1h", None, _H, 11.0, 0.10,
                           "Bitstamp, 1h", top_k=100),
    "sp500_1h_all": Dataset("sp500_1h_all", "sp500_1h", None, 7 * 252, 5.0, 0.005,
                            "S&P 500, 1h", top_k=100),
    "nifty50_1h": Dataset("nifty50_1h", "nifty50_1h", None, 7 * 250, 5.0, 0.0, "NIFTY 50, 1h"),
})

COST_GRID_BPS = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
SEED = 20261101
