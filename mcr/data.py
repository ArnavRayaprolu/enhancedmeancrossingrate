"""Kaggle 'hourly-stock-prices' (leukipp), year 2021: all US-listed tickers.

Preparation (R6, m5):
  * regular session only: the seven bars stamped 09:30 ... 15:30 ET (each bar's
    close is the price at the end of its hour; the 15:30 bar closes at 16:00);
    pre-/post-market and the zero-volume 16:00 marker are dropped, and so are
    dates on which fewer than half of the usual number of tickers report
    (holiday artefacts).
  * the file's 'adjusted close' equals 'close': prices are adjusted neither for
    splits nor for dividends. Splits are detected and back-adjusted (see
    split_adjust); dividends cannot be recovered from the data and remain in
    the price series (documented limitation).
  * the instrument filter removes ETFs/ETNs, funds, SPAC shells, warrants,
    units, rights, preferreds and notes using the security name.
  * the file was assembled during 2021, so names that were delisted later in
    2021 are present until their last trade; the liquid universe is therefore
    chosen from what was tradable at each formation date (point in time).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "kaggle_hourly"
SESSION = [f"{h:02d}:30" for h in range(9, 16)]

EXCLUDE = re.compile(
    r"\bETF\b|\bETN\b|ETRACS|iShares|ProShares|Direxion|SPDR|Vanguard|Invesco|WisdomTree|"
    r"VanEck|Global X|First Trust|GraniteShares|MicroSectors|VelocityShares|iPath|"
    r"\bFund\b|Acquisition|Warrant|\bUnits?\b|\bRights?\b|Preferred|Depositary Shares, each|"
    r"\bNotes?\b|Debenture|Leveraged|\bIndex\b|Trust Units|Subordinated|\bBond\b|Treasury|"
    r"Capital Corp\.? ?I|SPAC|Blank Check", re.I)

SPLIT_FACTORS = np.array([1/20, 1/10, 1/8, 1/5, 1/4, 1/3, 1/2, 2, 3, 4, 5, 8, 10, 20])


def _load(field: str) -> pd.DataFrame:
    parts = sorted(DATA.glob(f"{field}_*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], axis=1).astype("float64")


def split_adjust(close: pd.DataFrame, volume: pd.DataFrame, log: list | None = None):
    """Back-adjust stock splits.

    A split is flagged at the first regular bar of a day when the overnight
    price ratio p_t / p_{t-1} lies within 3% of a standard split or reverse-split
    factor f in {1/20, 1/10, 1/8, 1/5, 1/4, 1/3, 1/2, 2, 3, 4, 5, 8, 10, 20}.
    3-for-2 and 2-for-3 factors are not used: overnight moves of -33%/+50% are
    common news events and cannot be told apart from splits in this file
    (share volumes are inconsistently adjusted, so they cannot confirm a split).
    A candidate is rejected when the median dollar volume of the 7 bars from
    the jump exceeds 3x the median of the previous 35 bars: news moves come
    with a burst of trading, splits do not.
    Prices before the split are multiplied by f and volumes divided by f.
    Spin-offs with a ratio near a split factor (e.g. DELL/VMware, Nov 2021) are
    adjusted the same way, which approximates the total return to holders.
    """
    px = close.copy(); vol = volume.copy()
    day = px.index.normalize()
    first = np.r_[True, day[1:] != day[:-1]]
    for t in px.columns:
        s = px[t]
        v = s.dropna()
        if len(v) < 80:
            continue
        r = (v / v.shift(1)).iloc[1:]
        firsts = pd.Series(first, index=px.index).reindex(r.index).fillna(False)
        cand = r[firsts.to_numpy() & ((r < 0.6) | (r > 1.7))]
        for ts, ratio in cand.items():
            j = int(np.argmin(np.abs(np.log(SPLIT_FACTORS) - np.log(ratio))))
            f = SPLIT_FACTORS[j]
            if abs(ratio / f - 1) > 0.03:
                continue
            # news jumps come with a burst of dollar volume; a split does not
            dv = (s * vol[t]).dropna()
            i = dv.index.get_indexer([ts])[0]
            if i < 35 or not dv.iloc[i:i + 7].median() <= 3 * dv.iloc[i - 35:i].median():
                continue
            px.loc[px.index < ts, t] *= f
            vol.loc[vol.index < ts, t] /= f
            if log is not None:
                log.append(dict(ticker=t, date=str(ts), factor=float(f), ratio=float(ratio)))
    return px, vol


def load(cache: bool = True) -> dict:
    """Regular-session OHLCV panels (UTC index), split adjusted, plus metadata."""
    out = ROOT / "data" / "kaggle_hourly_prepared.pkl"
    if cache and out.exists():
        return pd.read_pickle(out)
    meta = pd.read_csv(DATA / "meta.csv").set_index("ticker")
    f = {k: _load(k) for k in ("open", "high", "low", "close", "volume")}
    et = f["close"].index.tz_convert("America/New_York")
    keep = pd.Index(et.strftime("%H:%M")).isin(SESSION)
    for k in f:
        f[k] = f[k][keep]
    cnt = f["close"].notna().sum(axis=1)
    good = cnt >= 0.5 * cnt.median()
    for k in f:
        f[k] = f[k][good]
    splits: list = []
    close, vol = split_adjust(f["close"], f["volume"], splits)
    ratio = close / f["close"]
    for k in ("open", "high", "low"):
        f[k] = f[k] * ratio
    f["close"], f["volume"] = close, vol
    names = meta["name"].reindex(close.columns).fillna("")
    instrument_ok = ~names.str.contains(EXCLUDE)
    sp = pd.read_csv(Path(__file__).with_name("sp500_constituents.csv"))
    sp["ticker"] = sp["Symbol"].str.replace(".", "-", regex=False)
    data = dict(**f, names=names, instrument_ok=instrument_ok,
                splits=pd.DataFrame(splits),
                sector=dict(zip(sp["ticker"], sp["GICS Sector"])),
                sp500_added=dict(zip(sp["ticker"], pd.to_datetime(sp["Date added"]))))
    if cache:
        pd.to_pickle(data, out)
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
    c = log close, eta = mid-range (log high + log low)/2; negative estimates
    are set to zero. Returns the proportional FULL spread per column."""
    c = np.log(close); eta = (np.log(high) + np.log(low)) / 2
    x = 4 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    s2 = np.nanmean(x, axis=0)
    return np.sqrt(np.clip(s2, 0, None))
