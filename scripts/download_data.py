"""Download free cryptocurrency price data (bars of 1 hour or finer).

Run locally (``python scripts/download_data.py all``) or let the GitHub Actions
workflow ``.github/workflows/download-data.yml`` run it and commit the output.

Every source follows the same point-in-time-safe recipe:
  1. list every USD / USDT-quoted symbol the venue publishes (incl. delisted
     symbols where the venue's archive keeps them -- Binance does);
  2. download DAILY bars for all of them and compute a 90-day rolling median
     of dollar volume;
  3. keep every symbol that was EVER among the TOP most liquid on any day
     (a superset of any point-in-time top-K universe used in the backtest);
  4. download intraday bars (1h, or 15m) for those symbols.
The backtest itself chooses the universe in each formation window from
formation-window volume only, so no future information enters selection.

Output (wide parquet, UTC timestamps, float32, split by calendar year):
    data/<dataset>/close_<year>.parquet   last trade price of the bar
    data/<dataset>/dvol_<year>.parquet    quote (USD / USDT) volume of the bar
    data/<dataset>/meta.csv

Datasets
    binance_spot_1h    Binance spot  USDT pairs, 1h, 2017-08 ->     data.binance.vision
    binance_spot_15m   Binance spot  USDT pairs, 15m, 2021-01 ->    data.binance.vision
    binance_perp_1h    Binance USDT-M perpetuals, 1h, 2020-01 ->    data.binance.vision
    coinbase_1h        Coinbase USD pairs, 1h, 2017-01 ->           api.exchange.coinbase.com
    bitstamp_1h        Bitstamp USD pairs, 1h, 2017-01 ->           www.bitstamp.net
    okx_1h             OKX USDT pairs, 1h, 2019-01 ->               www.okx.com
    bybit_1h           Bybit spot USDT pairs, 1h, 2021-07 ->        api.bybit.com
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import re
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UA = {"User-Agent": "Mozilla/5.0 (academic research; pairs-trading study)"}
TOP = 50                       # keep symbols ever in the top-50 by 90d median volume
NOW = pd.Timestamp.now(tz="UTC").floor("D")

STABLE = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "PAX", "UST", "USTC", "SUSD",
          "EUR", "GBP", "AUD", "TRY", "BRL", "RUB", "AEUR", "EURI", "BIDR", "IDRT", "BKRW",
          "USDS", "USDSB", "XUSD", "USD1", "BFUSD", "RLUSD", "PYUSD", "USDE", "USDD", "GUSD",
          "USDT", "CAD", "JPY", "CHF", "SGD", "PAXG", "XAUT", "WBTC", "WBETH", "BETH", "STETH",
          "CBETH", "EURC", "LUSD", "FRAX", "FDUSD", "USDG", "USTC"}
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR|[235]L|[235]S)$")


def _ok_base(base: str) -> bool:
    return base.upper() not in STABLE and not LEVERAGED.search(base.upper())


# --------------------------------------------------------------------------- #
# HTTP with a per-host rate limit
# --------------------------------------------------------------------------- #
_session = requests.Session(); _session.headers.update(UA)
_locks: dict[str, tuple[threading.Lock, list]] = {}


def _get(url: str, params=None, tries: int = 6, min_interval: float = 0.0):
    host = url.split("/")[2]
    lock, last = _locks.setdefault(host, (threading.Lock(), [0.0]))
    for i in range(tries):
        if min_interval:
            with lock:
                wait = last[0] + min_interval - time.time()
                if wait > 0:
                    time.sleep(wait)
                last[0] = time.time()
        try:
            r = _session.get(url, params=params, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r
            if r.status_code in (403, 451):
                raise PermissionError(f"{host} refused ({r.status_code})")
        except requests.RequestException:
            pass
        time.sleep(min(60, 2 ** i))
    return None


def _parallel(fn, items, workers=16, label=""):
    out = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for n, f in enumerate(cf.as_completed(futs), 1):
            it = futs[f]
            try:
                out[it] = f.result()
            except PermissionError:
                raise
            except Exception as exc:
                print(f"  {label} {it}: {exc}", flush=True); out[it] = None
            if n % 50 == 0 or n == len(items):
                print(f"  {label} {n}/{len(items)}", flush=True)
    return out


def _save(name: str, frames: dict[str, pd.DataFrame], source: str) -> None:
    frames = {s: d for s, d in frames.items() if d is not None and len(d)}
    if not frames:
        raise RuntimeError(f"{name}: no data")
    close = pd.DataFrame({s: d["close"] for s, d in frames.items()}).sort_index()
    dvol = pd.DataFrame({s: d["qvol"] for s, d in frames.items()}).reindex(close.index)
    close.index = pd.to_datetime(close.index, utc=True); dvol.index = close.index
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.parquet"):
        f.unlink()
    for yr, c in close.groupby(close.index.year):
        c.astype("float32").to_parquet(out / f"close_{yr}.parquet", compression="zstd")
        dvol.loc[c.index].astype("float32").to_parquet(out / f"dvol_{yr}.parquet",
                                                        compression="zstd")
    pd.DataFrame({"ticker": close.columns, "group": "crypto", "source": source}).to_csv(
        out / "meta.csv", index=False)
    print(f"[{name}] saved {close.shape[0]} bars x {close.shape[1]} symbols "
          f"{close.index.min()} -> {close.index.max()}", flush=True)


def _first(daily: dict, sym: str, st: pd.Timestamp) -> pd.Timestamp:
    d = daily.get(sym)
    return max(st, d.index.min().floor("D")) if d is not None and len(d) else st


def _ever_top(daily_qvol: dict[str, pd.Series], top: int = TOP) -> list[str]:
    dv = pd.DataFrame({s: v for s, v in daily_qvol.items() if v is not None and len(v)})
    dv = dv.sort_index()
    med = dv.rolling(90, min_periods=20).median()
    rank = med.rank(axis=1, ascending=False)
    return sorted(rank.columns[(rank <= top).any()])


# --------------------------------------------------------------------------- #
# Binance bulk archive (spot and USDT-M futures), includes delisted symbols
# --------------------------------------------------------------------------- #
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BV = "https://data.binance.vision"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _s3_list(prefix: str):
    keys, prefixes, marker = [], [], ""
    while True:
        r = _get(f"{S3}?delimiter=/&prefix={prefix}&marker={marker}")
        if r is None:
            raise RuntimeError(f"cannot list {prefix}")
        root = ElementTree.fromstring(r.content)
        k = [e.find(f"{NS}Key").text for e in root.findall(f"{NS}Contents")]
        p = [e.find(f"{NS}Prefix").text for e in root.findall(f"{NS}CommonPrefixes")]
        keys += k; prefixes += p
        if root.find(f"{NS}IsTruncated").text != "true":
            return keys, prefixes
        nm = root.find(f"{NS}NextMarker")
        marker = nm.text if nm is not None else (k or p)[-1]


def _read_kline_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 4, 7])
    if not pd.api.types.is_numeric_dtype(df[0]):        # header row in some files
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
    t = df[0].astype("int64").to_numpy()
    t = np.where(t > 10 ** 14, t // 1000, t)              # 2025+ spot files use microseconds
    return pd.DataFrame({"close": df[4].astype(float).to_numpy(),
                         "qvol": df[7].astype(float).to_numpy()},
                        index=pd.to_datetime(t, unit="ms", utc=True))


def _binance_klines(market: str, sym: str, interval: str, start: str):
    keys, _ = _s3_list(f"data/{market}/monthly/klines/{sym}/{interval}/")
    keys = [k for k in keys if k.endswith(".zip") and k[-11:-4] >= start]
    parts = [_read_kline_zip(r.content) for k in keys if (r := _get(f"{BV}/{k}")) is not None]
    if not parts:
        return None
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated()]


def binance(market: str, interval: str, start: str, name: str, top: int = TOP):
    _, pref = _s3_list(f"data/{market}/monthly/klines/")
    syms = [p.rstrip("/").split("/")[-1] for p in pref]
    syms = sorted(s for s in syms if s.endswith("USDT") and _ok_base(s[:-4]))
    print(f"[{name}] {len(syms)} USDT symbols listed (incl. delisted)", flush=True)
    daily = _parallel(lambda s: _binance_klines(market, s, "1d", start), syms, 32, name)
    keep = _ever_top({s: d["qvol"] for s, d in daily.items() if d is not None}, top)
    print(f"[{name}] {len(keep)} symbols were ever top-{top}", flush=True)
    bars = _parallel(lambda s: _binance_klines(market, s, interval, start), keep, 16, name)
    _save(name, bars, f"binance-{market}")


# --------------------------------------------------------------------------- #
# REST sources
# --------------------------------------------------------------------------- #
def _page_back(fetch, start: pd.Timestamp, step: pd.Timedelta, per_call: int):
    """Generic backward/forward pager: fetch(t0, t1) -> DataFrame for [t0, t1)."""
    parts, t0 = [], start
    while t0 < NOW:
        t1 = min(t0 + step * per_call, NOW)
        df = fetch(t0, t1)
        if df is not None and len(df):
            parts.append(df)
        t0 = t1
    if not parts:
        return None
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated()]


def coinbase(start="2017-01-01", name="coinbase_1h", top=TOP):
    base = "https://api.exchange.coinbase.com"
    prods = _get(f"{base}/products").json()
    ids = sorted(p["id"] for p in prods if p["quote_currency"] == "USD"
                 and _ok_base(p["base_currency"]))
    print(f"[{name}] {len(ids)} USD products", flush=True)

    def candles(pid, gran, t0, t1):
        r = _get(f"{base}/products/{pid}/candles",
                 params=dict(granularity=gran, start=t0.isoformat(), end=t1.isoformat()),
                 min_interval=0.12)
        if r is None or not r.json():
            return None
        a = np.array(r.json(), dtype=float)            # time, low, high, open, close, volume
        return pd.DataFrame({"close": a[:, 4], "qvol": a[:, 4] * a[:, 5]},
                            index=pd.to_datetime(a[:, 0], unit="s", utc=True))

    st = pd.Timestamp(start, tz="UTC")
    daily = _parallel(lambda p: _page_back(lambda a, b: candles(p, 86400, a, b), st,
                                           pd.Timedelta("1D"), 300), ids, 4, name)
    keep = _ever_top({s: d["qvol"] for s, d in daily.items() if d is not None}, top)
    print(f"[{name}] {len(keep)} products were ever top-{top}", flush=True)
    bars = _parallel(lambda p: _page_back(lambda a, b: candles(p, 3600, a, b),
                                          _first(daily, p, st),
                                          pd.Timedelta("1h"), 300), keep, 4, name)
    _save(name, {k.replace("-", ""): v for k, v in bars.items()}, "coinbase")


def bitstamp(start="2017-01-01", name="bitstamp_1h", top=TOP):
    base = "https://www.bitstamp.net/api/v2"
    pairs = _get(f"{base}/trading-pairs-info/").json()
    ids = sorted(p["url_symbol"] for p in pairs if p["name"].endswith("/USD")
                 and _ok_base(p["name"].split("/")[0]))
    print(f"[{name}] {len(ids)} USD pairs", flush=True)

    def ohlc(pid, step, t0, t1):
        r = _get(f"{base}/ohlc/{pid}/", params=dict(step=step, limit=1000,
                                                     start=int(t0.timestamp()),
                                                     end=int(t1.timestamp())),
                 min_interval=0.15)
        if r is None:
            return None
        d = r.json().get("data", {}).get("ohlc", [])
        if not d:
            return None
        df = pd.DataFrame(d).astype(float)
        return pd.DataFrame({"close": df["close"].to_numpy(),
                             "qvol": (df["close"] * df["volume"]).to_numpy()},
                            index=pd.to_datetime(df["timestamp"], unit="s", utc=True))

    st = pd.Timestamp(start, tz="UTC")
    daily = _parallel(lambda p: _page_back(lambda a, b: ohlc(p, 86400, a, b), st,
                                           pd.Timedelta("1D"), 1000), ids, 4, name)
    keep = _ever_top({s: d["qvol"] for s, d in daily.items() if d is not None}, top)
    bars = _parallel(lambda p: _page_back(lambda a, b: ohlc(p, 3600, a, b),
                                          _first(daily, p, st),
                                          pd.Timedelta("1h"), 1000), keep, 4, name)
    _save(name, {k.upper(): v for k, v in bars.items()}, "bitstamp")


def okx(start="2019-01-01", name="okx_1h", top=TOP):
    base = "https://www.okx.com/api/v5"
    inst = _get(f"{base}/public/instruments", params=dict(instType="SPOT")).json()["data"]
    ids = sorted(i["instId"] for i in inst if i["quoteCcy"] == "USDT" and _ok_base(i["baseCcy"]))
    print(f"[{name}] {len(ids)} USDT pairs (listed now)", flush=True)

    def hist(iid, bar, t0, t1):
        r = _get(f"{base}/market/history-candles",
                 params=dict(instId=iid, bar=bar, after=int(t1.timestamp() * 1000),
                             before=int(t0.timestamp() * 1000) - 1, limit=100),
                 min_interval=0.11)
        if r is None or not r.json().get("data"):
            return None
        a = np.array(r.json()["data"], dtype=float)    # ts, o, h, l, c, vol, volCcy, volQuote
        return pd.DataFrame({"close": a[:, 4], "qvol": a[:, 7]},
                            index=pd.to_datetime(a[:, 0], unit="ms", utc=True))

    st = pd.Timestamp(start, tz="UTC")
    daily = _parallel(lambda p: _page_back(lambda a, b: hist(p, "1Dutc", a, b), st,
                                           pd.Timedelta("1D"), 100), ids, 4, name)
    keep = _ever_top({s: d["qvol"] for s, d in daily.items() if d is not None}, top)
    bars = _parallel(lambda p: _page_back(lambda a, b: hist(p, "1H", a, b),
                                          _first(daily, p, st),
                                          pd.Timedelta("1h"), 100), keep, 4, name)
    _save(name, {k.replace("-", ""): v for k, v in bars.items()}, "okx")


def bybit(start="2021-07-01", name="bybit_1h", top=TOP):
    base = "https://api.bybit.com/v5/market"
    lst = _get(f"{base}/instruments-info", params=dict(category="spot")).json()
    ids = sorted(i["symbol"] for i in lst["result"]["list"] if i["quoteCoin"] == "USDT"
                 and _ok_base(i["baseCoin"]))
    print(f"[{name}] {len(ids)} USDT pairs (listed now)", flush=True)

    def kl(sym, iv, t0, t1):
        r = _get(f"{base}/kline", params=dict(category="spot", symbol=sym, interval=iv,
                                               start=int(t0.timestamp() * 1000),
                                               end=int(t1.timestamp() * 1000) - 1, limit=1000),
                 min_interval=0.06)
        if r is None or not r.json().get("result", {}).get("list"):
            return None
        a = np.array(r.json()["result"]["list"], dtype=float)  # start,o,h,l,c,vol,turnover
        return pd.DataFrame({"close": a[:, 4], "qvol": a[:, 6]},
                            index=pd.to_datetime(a[:, 0], unit="ms", utc=True))

    st = pd.Timestamp(start, tz="UTC")
    daily = _parallel(lambda p: _page_back(lambda a, b: kl(p, "D", a, b), st,
                                           pd.Timedelta("1D"), 1000), ids, 4, name)
    keep = _ever_top({s: d["qvol"] for s, d in daily.items() if d is not None}, top)
    bars = _parallel(lambda p: _page_back(lambda a, b: kl(p, "60", a, b),
                                          _first(daily, p, st),
                                          pd.Timedelta("1h"), 1000), keep, 4, name)
    _save(name, bars, "bybit")


def _funding(sym: str, start: str):
    keys, _ = _s3_list(f"data/futures/um/monthly/fundingRate/{sym}/")
    keys = [k for k in keys if k.endswith(".zip") and k[-11:-4] >= start]
    parts = []
    for k in keys:
        r = _get(f"{BV}/{k}")
        if r is None:
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
        tcol = [c for c in df.columns if "time" in c.lower()][0]
        rcol = [c for c in df.columns if "rate" in c.lower()][0]
        parts.append(pd.Series(df[rcol].astype(float).to_numpy(),
                               index=pd.to_datetime(df[tcol].astype("int64"), unit="ms",
                                                    utc=True)))
    if not parts:
        return None
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated()]


def binance_funding(name="binance_perp_funding", start="2020-01"):
    """Funding rates of every USDT-M perpetual in data/binance_perp_1h, aligned to
    the hour in which they are settled (long positions pay a positive rate)."""
    syms = sorted(pd.read_csv(DATA / "binance_perp_1h" / "meta.csv")["ticker"])
    out = _parallel(lambda s: _funding(s, start), syms, 16, name)
    f = pd.DataFrame({s: v for s, v in out.items() if v is not None})
    f.index = f.index.floor("h")
    f = f.groupby(level=0).sum()
    d = DATA / name
    d.mkdir(parents=True, exist_ok=True)
    f.astype("float32").to_parquet(d / "funding.parquet", compression="zstd")
    print(f"[{name}] {f.shape}", flush=True)


# NIFTY 50 constituents (fallback if the official list cannot be fetched;
# approximate 2025-26 composition, includes recent additions/removals).
NIFTY50_FALLBACK = """ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE
BAJAJFINSV BEL BHARTIARTL CIPLA COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH HDFCBANK
HDFCLIFE HEROMOTOCO HINDALCO HINDUNILVR ICICIBANK INDUSINDBK INFY ITC JIOFIN JSWSTEEL
KOTAKBANK LT M&M MARUTI NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE SHRIRAMFIN SBIN
SUNPHARMA TATACONSUM TMPV TATASTEEL TCS TECHM TITAN TRENT ULTRACEMCO WIPRO INDIGO
MAXHEALTH""".split()


def nifty50_1h(name="nifty50_1h"):
    """NIFTY 50 constituents, hourly, last 730 days (Yahoo Finance, '.NS' tickers)."""
    meta = None
    try:
        r = _session.get("https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
                         timeout=30)
        if r.status_code == 200 and "Symbol" in r.text:
            m = pd.read_csv(io.StringIO(r.text))
            meta = pd.DataFrame({"ticker": m["Symbol"].str.strip(),
                                 "group": m["Industry"].str.strip(),
                                 "name": m["Company Name"].str.strip(), "date_added": ""})
            print(f"[{name}] official list: {len(meta)} constituents", flush=True)
    except Exception as exc:
        print(f"[{name}] official list unavailable ({exc}); using fallback", flush=True)
    if meta is None:
        meta = pd.DataFrame({"ticker": NIFTY50_FALLBACK, "group": "all",
                             "name": NIFTY50_FALLBACK, "date_added": ""})
    import yfinance as yf
    yt = [t + ".NS" for t in meta["ticker"]]
    df = yf.download(yt, period="730d", interval="1h", auto_adjust=True, progress=False,
                     threads=True, group_by="column")
    close = df["Close"].dropna(how="all", axis=1)
    vol = df["Volume"].reindex(columns=close.columns)
    close.columns = [c.replace(".NS", "") for c in close.columns]
    vol.columns = close.columns
    frames = {c: pd.DataFrame({"close": close[c], "qvol": close[c] * vol[c]})
              for c in close.columns}
    _save(name, frames, "yahoo-nse")
    m = meta[meta["ticker"].isin(close.columns)]
    m.to_csv(DATA / name / "meta.csv", index=False)


TASKS = {
    "binance_spot_1h": lambda: binance("spot", "1h", "2017-08", "binance_spot_1h"),
    "binance_spot_15m": lambda: binance("spot", "15m", "2021-01", "binance_spot_15m", top=40),
    "binance_perp_1h": lambda: binance("futures/um", "1h", "2020-01", "binance_perp_1h"),
    "binance_perp_15m": lambda: binance("futures/um", "15m", "2020-01", "binance_perp_15m",
                                        top=40),
    "coinbase_1h": coinbase,
    "bitstamp_1h": bitstamp,
    "okx_1h": okx,
    "bybit_1h": bybit,
    "binance_perp_funding": binance_funding,
    "nifty50_1h": nifty50_1h,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", choices=list(TASKS) + ["all"])
    args = ap.parse_args()
    todo = list(TASKS) if "all" in args.datasets else args.datasets
    failed = []
    for name in todo:
        t0 = time.time()
        try:
            TASKS[name]()
        except Exception as exc:
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            failed.append(name)
        print(f"[{name}] {time.time() - t0:.0f}s", flush=True)
    sys.exit(1 if len(failed) == len(todo) else 0)
