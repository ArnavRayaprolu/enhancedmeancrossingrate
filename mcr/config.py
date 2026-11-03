"""Frozen study protocol for the revised MCR model.

Everything in this file was fixed BEFORE any result on the Kaggle 2021 data was
computed (reviewer comment R2: parameters set in advance, independent sample).

ORIGINAL  reproduces the submitted notebook (MCR_Pairs_Trading.ipynb):
          (MA_35 - MA_140)/SD_140 of the price ratio, enter |z| >= 1.00,
          exit |z| <= 0.25, 5 pairs per method, within-sector candidates,
          capital re-split over active pairs by rank (Eq. 8), 3 bps round trip.
PRIMARY   the revised headline configuration. Same signal and thresholds as
          the original; the changes are the reviewer-driven engine rules
          (R4), realistic stock-specific costs (R5), point-in-time universe
          and split adjustment (R6, m5), all pair combinations of the liquid
          universe, 20 pairs with no security reused, and a capped version
          of the Eq. 8 capital rule (diversification, m7).
GRID      one-at-a-time sensitivity around PRIMARY (R8). Every row is
          reported, favourable or not.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

PROTOCOL = "R1-2026-11"
BARS_PER_DAY = 7                      # regular session, bars stamped 09:30 ... 15:30 ET
BARS_PER_YEAR = BARS_PER_DAY * 252


@dataclass(frozen=True)
class Costs:
    """One-way cost per dollar traded on each leg, plus borrow on short legs.
    spread_mult   x stock-specific half-spread (Abdi-Ranaldo 2017, formation window)
    fixed_bps     commission + exchange/regulatory fees, per dollar traded
    impact_coef   square-root impact: coef * sigma_bar * sqrt(trade $ / bar $ volume)
    borrow_pa     annual borrow fee on the short leg's market value
    """
    name: str
    flat_bps: float = 0.0             # used instead of the spread model when > 0
    spread_mult: float = 0.0
    fixed_bps: float = 0.0
    impact_coef: float = 0.0
    aum: float = 10e6
    borrow_pa: float = 0.0


COSTS = {
    # the submitted paper: 3 bps "round trip" = 1.5 bps per dollar traded, nothing else
    "original": Costs("original", flat_bps=1.5),
    # headline: each stock's estimated half-spread + 0.5 bp fees + 0.5% p.a. borrow
    "base": Costs("base", spread_mult=1.0, fixed_bps=0.5, borrow_pa=0.005),
    # stress: double spread, square-root impact for a $10m book, 2% borrow
    "stress": Costs("stress", spread_mult=2.0, fixed_bps=0.5, impact_coef=0.5,
                    aum=10e6, borrow_pa=0.02),
}
HEADLINE_COST = "base"


@dataclass(frozen=True)
class Config:
    label: str = "primary"
    universe: str = "us_liquid"       # us_liquid | sp500
    top_k: int = 250                  # liquid universe size (formation dollar volume); the
                                      # hourly download holds each year's 260 most liquid names
    min_price: float = 5.0
    min_vol: float = 0.10             # annualised formation volatility; drops SPAC shells
    min_coverage: float = 0.95        # share of formation bars with a trade
    ffill: bool = True                # forward-fill gaps inside the fold (R6)
    sector_pairs: bool = False        # True: candidates restricted to same GICS sector
    signal: str = "ratio"             # ratio: (MA_f - MA_s)/SD_s of P_A/P_B
                                      # hedged: same on log P_A - beta log P_B (R7)
    fast: int = 35
    slow: int = 140
    entry: float = 1.00
    exit: float = 0.25
    engine: str = "revised"           # revised (R4 rules) | original (notebook)
    lag: int = 0                      # bars between decision and execution
    n_pairs: int = 20
    max_uses: int = 1                 # times a security may appear in the book
    weighting: str = "capped"         # capped | active (Eq. 8) | rank | equal
    cap: float = 0.10                 # per-pair cap for 'capped' (= 2 / n_pairs)
    train_days: int = 90              # walk-forward windows of the original notebook
    test_days: int = 45

    @property
    def key(self) -> str:
        return self.label


ORIGINAL = Config(label="original", universe="sp500", sector_pairs=True, n_pairs=5,
                  weighting="active", engine="original")
PRIMARY = Config()

# (label, overrides) -- one-at-a-time departures from PRIMARY (R8)
GRID_SPEC = [
    ("ma_14_70", dict(fast=14, slow=70)),
    ("ma_70_280", dict(fast=70, slow=280)),
    ("entry_0.75", dict(entry=0.75)),
    ("entry_1.25", dict(entry=1.25)),
    ("entry_1.50", dict(entry=1.50)),
    ("exit_0.00", dict(exit=0.0)),
    ("exit_0.50", dict(exit=0.50)),
    ("pairs_5", dict(n_pairs=5, cap=0.40)),
    ("pairs_10", dict(n_pairs=10, cap=0.20)),
    ("pairs_40", dict(n_pairs=40, cap=0.05)),
    ("w_active_eq8", dict(weighting="active")),
    ("w_rank_fixed", dict(weighting="rank")),
    ("w_equal", dict(weighting="equal")),
    ("hedged_spread", dict(signal="hedged")),
    ("complete_case", dict(min_coverage=1.0, ffill=False)),
    ("delay_1bar", dict(lag=1)),
    ("windows_60_21", dict(train_days=60, test_days=21)),
    ("universe_125", dict(top_k=125)),
    ("sp500_sector", dict(universe="sp500", sector_pairs=True)),
    ("engine_original", dict(engine="original")),
]


def grid() -> list[Config]:
    return [replace(PRIMARY, label=lab, **kw) for lab, kw in GRID_SPEC]


# fields that change the candidate statistics; configs that agree on all of
# them share one statistics pass per fold
STAT_FIELDS = ("universe", "top_k", "min_price", "min_vol", "min_coverage", "ffill",
               "sector_pairs", "signal", "fast", "slow", "train_days", "test_days")


def stat_key(c: Config) -> tuple:
    return tuple(getattr(c, f) for f in STAT_FIELDS)


DIAG_SAMPLE = 50000                  # candidate pairs per fold for the R1 diagnostics


# --------------------------------------------------------------------------- #
# selection criteria. column, higher_is_better (percentile ranks, m3)
# --------------------------------------------------------------------------- #
BASE = {"MCR": ("mcr", True), "Cointegration": ("coint_p", False),
        "Distance": ("distance", False), "Correlation": ("corr", True),
        "HalfLife": ("halflife", False)}
HYBRIDS = {"MCR+Cointegration": "Cointegration", "MCR+Distance": "Distance",
           "MCR+Correlation": "Correlation", "MCR+HalfLife": "HalfLife"}
CRITERIA = list(BASE) + list(HYBRIDS)
CHAMPION = "MCR+Cointegration"
BENCHMARKS = ["Cointegration", "Distance", "Correlation"]

# pre-registered Diebold-Mariano families
DM_FAMILIES = {
    "champion": [(CHAMPION, b) for b in BENCHMARKS],                 # as in the paper, Holm k = 3
    "hybrid_vs_component": [(h, c) for h, c in HYBRIDS.items()],
    "mcr_vs_classic": [("MCR", c) for c in ["Cointegration", "Distance", "Correlation", "HalfLife"]],
}
