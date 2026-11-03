"""Alpaca US equities, 2016 onward, 1-hour bars, split + dividend adjusted.

Source: Alpaca Market Data API (SIP feed, adjustment='all'), downloaded on
GitHub Actions by scripts/download_alpaca.py into data/alpaca/.

Preparation (R6, m5)
  * bars: Alpaca's native clock-hour bars (data/alpaca/hourly2/<year>/g<NN>.parquet,
    long format) that start 09:00 ... 15:00 ET, seven
    per day; bar t's close is the price at the end of its hour (the 15:00 bar
    closes at the 16:00 close). The 09:00 bar also contains 09:00-09:30
    pre-market trades. Dates on which fewer than half the usual number of
    names report are dropped.
  * adjustment: prices are adjusted for splits AND dividends by Alpaca, so the
    ratio, the signal and the returns are all computed from the same
    total-return price series (no inconsistency between signal prices and
    economic returns, m5).
  * instruments: common stocks and ADRs on NYSE/NASDAQ/AMEX; ETFs/ETNs, funds,
    SPAC shells, warrants, units, rights, preferreds and notes are removed by
    security name.
  * survivorship (R6): the asset list includes INACTIVE (delisted/merged)
    symbols; the hourly universe is the union over calendar years of each
    year's 260 most liquid common stocks on that year's own daily data (500
    symbols, of which ~30 are now delisted), and every fold picks its own
    universe from formation data only. Residual bias: Alpaca's list of
    inactive symbols is incomplete for early delistings, and a reused ticker
    maps two companies to one symbol.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "alpaca"
CACHE = ROOT / "data" / "alpaca_prepared.pkl"
FIELDS = ("open", "high", "low", "close", "volume")


def _panels() -> dict[str, pd.DataFrame]:
    """Wide (time x symbol) panels from the long hourly2/<year>/g<NN>.parquet
    files; only bars that START 09:00 ... 15:00 America/New_York are kept."""
    parts = sorted((DATA / "hourly2").glob("*/g*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    t = pd.DatetimeIndex(df["t"]).tz_localize("UTC")
    h = t.tz_convert("America/New_York").hour
    df = df[(h >= 9) & (h <= 15)]
    df = df.assign(t=t[(h >= 9) & (h <= 15)], symbol=df["symbol"].astype(str))
    df = df.drop_duplicates(["t", "symbol"], keep="last")
    return {f: df.pivot(index="t", columns="symbol", values=f).sort_index().astype("float32")
            for f in FIELDS}


def load(cache: bool = True) -> dict:
    if cache and CACHE.exists():
        return pd.read_pickle(CACHE)
    f = _panels()
    cols = f["close"].columns
    for k in f:
        f[k] = f[k].reindex(columns=cols)
    cnt = f["close"].notna().sum(axis=1)
    good = cnt >= 0.5 * cnt.rolling(140, min_periods=1).median()
    for k in f:
        f[k] = f[k][good.to_numpy()]
    assets = pd.read_csv(DATA / "assets.csv", keep_default_na=False, dtype={"symbol": str})
    assets = assets.drop_duplicates("symbol").set_index("symbol")
    ok = assets["instrument_ok"].astype(str) == "True"
    sp = pd.read_csv(Path(__file__).with_name("sp500_constituents.csv"))
    sp["ticker"] = sp["Symbol"].str.replace(".", "-", regex=False)
    data = dict(**f,
                names=assets["name"].reindex(cols).fillna(""),
                status=assets["status"].reindex(cols).fillna("unknown"),
                instrument_ok=ok.reindex(cols).fillna(False).astype(bool),
                sector=dict(zip(sp["ticker"], sp["GICS Sector"])),
                sp500_added=dict(zip(sp["ticker"], pd.to_datetime(sp["Date added"]))))
    if cache:
        pd.to_pickle(data, CACHE)
    return data


def trading_days(index: pd.DatetimeIndex) -> np.ndarray:
    return np.array(sorted(pd.unique(index.tz_convert("America/New_York").date)))


def make_folds(index: pd.DatetimeIndex, train_days: int, test_days: int) -> list[dict]:
    """Non-overlapping test windows of test_days, each preceded by train_days
    of formation; the next fold starts test_days later (rolling)."""
    d = trading_days(index)
    dates = index.tz_convert("America/New_York").date
    folds, s, i = [], 0, 0
    while s + train_days + test_days <= len(d):
        tr = (dates >= d[s]) & (dates <= d[s + train_days - 1])
        te = (dates >= d[s + train_days]) & (dates <= d[s + train_days + test_days - 1])
        folds.append(dict(fold=i, train_start=d[s], train_end=d[s + train_days - 1],
                          test_start=d[s + train_days], test_end=d[s + train_days + test_days - 1],
                          tr_rows=np.flatnonzero(tr), te_rows=np.flatnonzero(te)))
        s += test_days; i += 1
    return folds


def abdi_ranaldo(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Abdi and Ranaldo (2017) close-high-low effective spread estimator,
    averaged over the window: s^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})],
    c = log close, eta = mid-range (log high + log low)/2; a negative average
    is set to zero. Returns the proportional FULL spread per column."""
    c = np.log(close); eta = (np.log(high) + np.log(low)) / 2
    x = 4 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    s2 = np.nanmean(x, axis=0)
    return np.sqrt(np.clip(s2, 0, None))
