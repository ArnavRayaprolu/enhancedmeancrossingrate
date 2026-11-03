"""Daily market panels for the multi-market EMCR study (data/alpaca/daily_full).

Signal = day d close; execution = day d+1 open (one overnight between the
information and the fill). D['px'][d] holds the NEXT day's open, so the lab's
'decision at sig[d], executed at px[d]' convention is respected and returns run
open-to-open.

Markets (each a liquidity band chosen on formation data inside the lab):
    us_large   common stocks, dollar-volume ranks 1-250
    us_mid     ranks 251-1000
    us_small   ranks 1001-2500, price >= $3
    etf        unleveraged, non-inverse ETFs and listed funds, ranks 1-300
Costs: tiered one-way half-spread by dollar volume (2-30 bps) + 0.5 bp fees;
borrow 0.5%/yr (large, ETF), 1% (mid), 3% (small).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "alpaca" / "daily_full"
CACHE = ROOT / "data" / "alpaca_daily_full.pkl"

MARKETS = {
    "us_large": dict(kind="stock", rank_lo=0, top_k=250, min_price=5.0, borrow=0.005),
    "us_mid": dict(kind="stock", rank_lo=250, top_k=750, min_price=5.0, borrow=0.01),
    "us_small": dict(kind="stock", rank_lo=1000, top_k=1500, min_price=3.0, borrow=0.03),
    "etf": dict(kind="etf", rank_lo=0, top_k=300, min_price=5.0, borrow=0.005),
}


def _panels():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    out = {}
    for f in ("open", "high", "low", "close", "volume"):
        parts = sorted(SRC.glob(f"{f}_*.parquet"))
        out[f] = pd.concat([pd.read_parquet(p) for p in parts], axis=1).sort_index().astype("float64")
    a = pd.read_csv(ROOT / "data" / "alpaca" / "assets.csv", keep_default_na=False, dtype={"symbol": str})
    a = a.sort_values("status").drop_duplicates("symbol").set_index("symbol")
    out["etfs"] = set((SRC / "etfs.txt").read_text().split())
    out["stock_ok"] = (a["instrument_ok"].astype(str) == "True")
    out["status"] = a["status"]
    pd.to_pickle(out, CACHE)
    return out


def market(name: str) -> tuple[dict, dict]:
    """(D, params) for the lab: D has sig, px, high, low, volume, instrument_ok."""
    P = _panels()
    m = MARKETS[name]
    cols = P["close"].columns
    if m["kind"] == "etf":
        ok = pd.Series([c in P["etfs"] for c in cols], index=cols)
    else:
        ok = P["stock_ok"].reindex(cols).fillna(False).astype(bool) & \
            pd.Series([c not in P["etfs"] for c in cols], index=cols)
    D = dict(market=name,
             sig=P["close"],
             px=P["open"].shift(-1),           # next day's open, aligned to the signal day
             high=P["high"], low=P["low"], volume=P["volume"],
             instrument_ok=ok, status=P["status"].reindex(cols))
    params = dict(rank_lo=m["rank_lo"], min_price=m["min_price"], cost_tiers=True,
                  prefilter=30000, roll_windows=(5, 20))
    return D, dict(params=params, top_k=m["top_k"], borrow=m["borrow"])
