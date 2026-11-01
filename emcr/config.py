"""Frozen study protocol (pre-registered).

Every parameter that determines the PRIMARY results is fixed here, and this
file was committed to git before any of the hold-out datasets were downloaded.
The commit timestamp is the pre-registration record.

The primary specification is the one used in the original submission,
expressed in bars so that it can be carried unchanged to other markets and
sampling frequencies:

    signal        z = (MA_35(r) - MA_140(r)) / SD_140(r),  r = P_A / P_B
    entry / exit  |z| >= 1.00  /  reverted to within 0.25 of zero
    formation     630 bars      (90 trading days x 7 hourly bars)
    trading       315 bars      (45 trading days x 7 hourly bars), non-overlapping
    portfolio     5 pairs, no security reused, rank-weighted across active pairs

The S&P 500 hourly sample is the DEVELOPMENT sample (it was used to arrive at
this specification). Every other dataset / frequency is a HOLD-OUT sample on
which the frozen specification is run exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product

PROTOCOL_VERSION = "2.0"
PROTOCOL_FROZEN_ON = "2026-11-01"

# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
# MAD    : moving-average difference of the price ratio (original paper)
# LEVEL  : classic level z-score of the price ratio, (r - MA_s) / SD_s
# HEDGED : level z-score of the Engle-Granger cointegrating spread
#          s = log P_A - beta log P_B, beta estimated on the formation window
SIGNALS = ("MAD", "LEVEL", "HEDGED")
PRIMARY_SIGNAL = "MAD"

# Base entry threshold per signal; the exit threshold is entry / 4.
# MAD: the paper's 1.00 / 0.25.  LEVEL / HEDGED: the two-standard-deviation
# convention of Gatev et al. (2006) with the same 1/4 exit ratio.
BASE_ENTRY = {"MAD": 1.0, "LEVEL": 2.0, "HEDGED": 2.0}
EXIT_FRACTION = 0.25

# --------------------------------------------------------------------------- #
# Criteria
# --------------------------------------------------------------------------- #
BENCHMARKS = ("Distance", "Correlation", "Cointegration", "HalfLife")
PROPOSED = ("SCR", "CCR", "SCR+Coint", "CCR+Coint")
CRITERIA = BENCHMARKS + PROPOSED
HYBRIDS = {"SCR+Coint": ("SCR", "Cointegration"), "CCR+Coint": ("CCR", "Cointegration")}

# Pre-registered primary hypotheses: each hybrid against cointegration alone,
# on every hold-out dataset, using the primary configuration.
PRIMARY_TESTS = (("SCR+Coint", "Cointegration"), ("CCR+Coint", "Cointegration"))


# --------------------------------------------------------------------------- #
# Configuration of one backtest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    signal: str = PRIMARY_SIGNAL
    fast: int = 35                 # bars (MAD only)
    slow: int = 140                # bars
    entry_mult: float = 1.0        # multiple of BASE_ENTRY[signal]
    exit_frac: float = EXIT_FRACTION
    n_pairs: int = 5
    weighting: str = "active"      # "active" (paper, Eq. 8) or "equal" (1/N committed)

    @property
    def entry(self) -> float:
        return BASE_ENTRY[self.signal] * self.entry_mult

    @property
    def exit(self) -> float:
        return self.entry * self.exit_frac

    @property
    def key(self) -> str:
        return (f"{self.signal}|f{self.fast}s{self.slow}|e{self.entry_mult:g}"
                f"|x{self.exit_frac:g}|n{self.n_pairs}|{self.weighting}")


PRIMARY = Config()

# Sensitivity grid (reported in full, never used to pick a "winner").
GRID_MA = ((17, 70), (35, 140), (70, 280))
GRID_ENTRY = (0.75, 1.0, 1.5)
GRID_EXIT = (0.0, 0.25, 0.5)
GRID_N = (5, 10, 20)
GRID_WEIGHTING = ("active", "equal")


def grid() -> list[Config]:
    out = []
    for sig, (f, s), e, x, n, w in product(SIGNALS, GRID_MA, GRID_ENTRY, GRID_EXIT,
                                           GRID_N, GRID_WEIGHTING):
        out.append(Config(sig, f, s, e, x, n, w))
    return out


# --------------------------------------------------------------------------- #
# Walk-forward and data handling
# --------------------------------------------------------------------------- #
TRAIN_BARS = 630
TEST_BARS = 315
MIN_COVERAGE = 0.95            # share of non-missing bars in the formation window
CRYPTO_TOP_K = 40              # crypto universe: top-K by formation dollar volume


# --------------------------------------------------------------------------- #
# Datasets: costs, annualisation, pair-candidate rule
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Dataset:
    name: str
    source: str                # folder under data/
    resample: str | None       # pandas offset for resampling, None = native
    bars_per_year: float
    cost_bps: float            # one-way cost per dollar traded, per leg (bps)
    borrow_pa: float           # annual borrow fee on short-leg notional
    pairs: str                 # "sector" (same group) or "all"
    role: str                  # "development" or "holdout"
    point_in_time_members: bool = False
    top_k: int | None = None


DATASETS = {
    "sp500_1h":   Dataset("sp500_1h", "sp500_1h", None, 7 * 252, 5.0, 0.005, "sector",
                          "development", point_in_time_members=True),
    "sp500_1d":   Dataset("sp500_1d", "sp500_1d", None, 252, 5.0, 0.005, "sector",
                          "holdout", point_in_time_members=True),
    "etf_1d":     Dataset("etf_1d", "etf_1d", None, 252, 5.0, 0.005, "all", "holdout"),
    "crypto_1h":  Dataset("crypto_1h", "binance_1h", None, 24 * 365, 10.0, 0.10, "all",
                          "holdout", top_k=CRYPTO_TOP_K),
    "crypto_4h":  Dataset("crypto_4h", "binance_1h", "4h", 6 * 365, 10.0, 0.10, "all",
                          "holdout", top_k=CRYPTO_TOP_K),
    "crypto_1d":  Dataset("crypto_1d", "binance_1h", "1D", 365, 10.0, 0.10, "all",
                          "holdout", top_k=CRYPTO_TOP_K),
}

# Added after the freeze (data only, no parameter changes): Binance USDT-M
# perpetual futures from a public GitHub mirror of data.binance.vision.  The
# mirror's 39 symbols were chosen by its owner, so this sample is NOT free of
# survivorship / selection bias; it is reported as supplementary evidence.
DATASETS["perp_1h"] = Dataset("perp_1h", "binance_perp_1h", None, 24 * 365, 5.0, 0.10,
                              "all", "holdout", top_k=CRYPTO_TOP_K)
DATASETS["perp_4h"] = Dataset("perp_4h", "binance_perp_1h", "4h", 6 * 365, 5.0, 0.10,
                              "all", "holdout", top_k=CRYPTO_TOP_K)

# Cost sensitivity grid, one-way bps per leg.
COST_GRID_BPS = (0.0, 1.0, 2.5, 5.0, 10.0, 15.0, 25.0)

SEED = 20261101
