"""Download every dataset used in the revised study from free public sources.

Run locally (``python scripts/download_data.py all``) or let the GitHub Actions
workflow ``.github/workflows/download-data.yml`` run it and commit the output.

Output layout (wide parquet files, UTC timestamps, float32):

    data/<dataset>/close.parquet     close price (split/dividend adjusted for equities)
    data/<dataset>/dvol.parquet      dollar (quote) volume per bar
    data/<dataset>/meta.csv          ticker metadata (sector / group, date added ...)

Datasets
--------
sp500_1h   S&P 500 constituents, hourly, last 730 days            (Yahoo Finance)
sp500_1d   S&P 500 constituents, daily, 2005-01 onwards           (Yahoo Finance)
etf_1d     liquid sector / country / asset-class ETFs, daily      (Yahoo Finance)
binance_1h Binance spot USDT pairs, hourly, 2020-01 onwards,
           INCLUDING pairs that were later delisted                (data.binance.vision)

The crypto universe is chosen point-in-time inside the backtest (top coins by
trailing dollar volume in each formation window), so the download keeps every
symbol that was ever among the most liquid ``--crypto-top`` coins.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import re
import sys
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UA = {"User-Agent": "Mozilla/5.0 (research download; EMCR pairs study)"}

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _save(name: str, close: pd.DataFrame, dvol: pd.DataFrame, meta: pd.DataFrame) -> None:
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    close = close.sort_index().astype("float32")
    dvol = dvol.reindex_like(close).astype("float32")
    close.index = pd.to_datetime(close.index, utc=True)
    dvol.index = close.index
    close.to_parquet(out / "close.parquet", compression="zstd")
    dvol.to_parquet(out / "dvol.parquet", compression="zstd")
    meta.to_csv(out / "meta.csv", index=False)
    print(f"[{name}] saved {close.shape[0]} bars x {close.shape[1]} series "
          f"{close.index.min()} -> {close.index.max()}", flush=True)


def _get(url: str, tries: int = 5, **kw) -> requests.Response | None:
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60, **kw)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    return None


# --------------------------------------------------------------------------- #
# Yahoo Finance (equities and ETFs)
# --------------------------------------------------------------------------- #

ETFS = {
    # US sector SPDRs
    "XLB": "US sector", "XLE": "US sector", "XLF": "US sector", "XLI": "US sector",
    "XLK": "US sector", "XLP": "US sector", "XLU": "US sector", "XLV": "US sector",
    "XLY": "US sector",
    # US style / size / broad
    "SPY": "US broad", "QQQ": "US broad", "DIA": "US broad", "IWM": "US broad",
    "MDY": "US broad", "IVV": "US broad", "VTI": "US broad", "IWD": "US broad",
    "IWF": "US broad", "IWN": "US broad", "IWO": "US broad", "RSP": "US broad",
    # US industries
    "SMH": "US industry", "IBB": "US industry", "KBE": "US industry", "KRE": "US industry",
    "XHB": "US industry", "XRT": "US industry", "XME": "US industry", "XOP": "US industry",
    "OIH": "US industry", "ITB": "US industry", "IYR": "US industry", "VNQ": "US industry",
    "IYT": "US industry", "ITA": "US industry", "IHI": "US industry",
    # single-country / regional equity
    "EWA": "Intl equity", "EWC": "Intl equity", "EWD": "Intl equity", "EWG": "Intl equity",
    "EWH": "Intl equity", "EWI": "Intl equity", "EWJ": "Intl equity", "EWK": "Intl equity",
    "EWL": "Intl equity", "EWN": "Intl equity", "EWO": "Intl equity", "EWP": "Intl equity",
    "EWQ": "Intl equity", "EWS": "Intl equity", "EWT": "Intl equity", "EWU": "Intl equity",
    "EWW": "Intl equity", "EWY": "Intl equity", "EWZ": "Intl equity", "EZA": "Intl equity",
    "EFA": "Intl equity", "EEM": "Intl equity", "VGK": "Intl equity", "FXI": "Intl equity",
    "ILF": "Intl equity",
    # fixed income
    "TLT": "Fixed income", "IEF": "Fixed income", "SHY": "Fixed income", "LQD": "Fixed income",
    "HYG": "Fixed income", "TIP": "Fixed income", "AGG": "Fixed income", "EMB": "Fixed income",
    "MBB": "Fixed income",
    # commodities / currency
    "GLD": "Commodity", "SLV": "Commodity", "USO": "Commodity", "DBC": "Commodity",
    "DBA": "Commodity", "GDX": "Commodity", "UUP": "Commodity", "FXE": "Commodity",
    "FXY": "Commodity",
}


def _yf_batch(tickers, **kw):
    import yfinance as yf
    for attempt in range(5):
        try:
            df = yf.download(tickers, auto_adjust=True, progress=False, threads=True,
                             group_by="column", **kw)
            if df is not None and len(df):
                return df
        except Exception as exc:  # rate limits etc.
            print(f"  yfinance error ({exc}); retry {attempt + 1}", flush=True)
        time.sleep(10 * (attempt + 1))
    return None


def _yahoo(tickers, batch=40, **kw):
    closes, vols = [], []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        df = _yf_batch(chunk, **kw)
        if df is None:
            print(f"  batch {i // batch} failed: {chunk[:3]}...", flush=True)
            continue
        c = df["Close"]; v = df["Volume"]
        if isinstance(c, pd.Series):
            c = c.to_frame(chunk[0]); v = v.to_frame(chunk[0])
        closes.append(c); vols.append(v)
        print(f"  yahoo {min(i + batch, len(tickers))}/{len(tickers)}", flush=True)
        time.sleep(2)
    close = pd.concat(closes, axis=1)
    vol = pd.concat(vols, axis=1)
    close = close.loc[:, ~close.columns.duplicated()].dropna(how="all", axis=1)
    vol = vol.loc[:, ~vol.columns.duplicated()].reindex(columns=close.columns)
    return close, close * vol


def sp500_meta() -> pd.DataFrame:
    m = pd.read_csv(DATA / "sp500_constituents.csv")
    m = m.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "group",
                          "Date added": "date_added"})
    m["ticker"] = m["ticker"].str.replace(".", "-", regex=False).str.strip()
    return m[["ticker", "name", "group", "date_added"]].drop_duplicates("ticker")


def download_sp500_1h():
    meta = sp500_meta()
    close, dvol = _yahoo(meta["ticker"].tolist(), period="730d", interval="1h")
    _save("sp500_1h", close, dvol, meta[meta["ticker"].isin(close.columns)])


def download_sp500_1d():
    meta = sp500_meta()
    close, dvol = _yahoo(meta["ticker"].tolist(), start="2005-01-01", interval="1d")
    _save("sp500_1d", close, dvol, meta[meta["ticker"].isin(close.columns)])


def download_etf_1d():
    meta = pd.DataFrame({"ticker": list(ETFS), "group": list(ETFS.values())})
    meta["name"] = meta["ticker"]; meta["date_added"] = ""
    close, dvol = _yahoo(meta["ticker"].tolist(), start="2005-01-01", interval="1d")
    _save("etf_1d", close, dvol, meta[meta["ticker"].isin(close.columns)])


# --------------------------------------------------------------------------- #
# Binance spot (data.binance.vision bulk archive -- includes delisted symbols)
# --------------------------------------------------------------------------- #

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BV = "https://data.binance.vision"
STABLE = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "PAX", "UST", "USTC", "SUSD",
          "EUR", "GBP", "AUD", "TRY", "BRL", "RUB", "AEUR", "EURI", "BIDR", "IDRT", "BKRW",
          "USDS", "USDSB", "XUSD", "USD1", "BFUSD", "RLUSD", "PAXG", "WBTC", "WBETH", "BETH"}
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR)USDT$")


def _s3_list(prefix: str, delimiter: str = "/"):
    """List keys / common prefixes under an S3 prefix (handles pagination)."""
    keys, prefixes, marker = [], [], ""
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        r = _get(f"{S3}?delimiter={delimiter}&prefix={prefix}&marker={marker}")
        if r is None:
            raise RuntimeError(f"cannot list {prefix}")
        root = ElementTree.fromstring(r.content)
        keys += [e.find(f"{ns}Key").text for e in root.findall(f"{ns}Contents")]
        prefixes += [e.find(f"{ns}Prefix").text for e in root.findall(f"{ns}CommonPrefixes")]
        if root.find(f"{ns}IsTruncated").text != "true":
            return keys, prefixes
        marker = (keys or prefixes)[-1]


def _usdt_symbols():
    _, pref = _s3_list("data/spot/monthly/klines/")
    syms = [p.rstrip("/").split("/")[-1] for p in pref]
    out = []
    for s in syms:
        if not s.endswith("USDT") or LEVERAGED.search(s):
            continue
        if s[:-4] in STABLE:
            continue
        out.append(s)
    return sorted(out)


def _read_kline_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 4, 7])
    if not pd.api.types.is_numeric_dtype(df[0]):        # header row in some files
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
    t = df[0].astype("int64")
    unit = np.where(t > 10 ** 14, "us", "ms")             # 2025+ spot files are in microseconds
    ts = pd.to_datetime(np.where(unit == "us", t // 1000, t), unit="ms", utc=True)
    return pd.DataFrame({"close": df[4].astype(float).values,
                         "qvol": df[7].astype(float).values}, index=ts)


def _symbol_klines(sym: str, interval: str, start: str) -> pd.DataFrame | None:
    keys, _ = _s3_list(f"data/spot/monthly/klines/{sym}/{interval}/")
    keys = [k for k in keys if k.endswith(".zip")
            and k.split("-")[-2] + "-" + k.split("-")[-1][:2] >= start]
    parts = []
    for k in keys:
        r = _get(f"{BV}/{k}")
        if r is not None:
            parts.append(_read_kline_zip(r.content))
    if not parts:
        return None
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated()]


def _parallel(fn, items, workers=16):
    out = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for n, f in enumerate(cf.as_completed(futs), 1):
            it = futs[f]
            try:
                out[it] = f.result()
            except Exception as exc:
                print(f"  {it}: {exc}", flush=True); out[it] = None
            if n % 50 == 0:
                print(f"  {n}/{len(items)}", flush=True)
    return out


def download_binance_1h(start="2020-01", top=60):
    syms = _usdt_symbols()
    print(f"[binance] {len(syms)} USDT spot symbols (incl. delisted)", flush=True)
    daily = _parallel(lambda s: _symbol_klines(s, "1d", start), syms)
    dv = pd.DataFrame({s: d["qvol"] for s, d in daily.items() if d is not None}).sort_index()
    # every symbol that was ever among the `top` most liquid by 90-day median volume
    med = dv.rolling(90, min_periods=30).median()
    rank = med.rank(axis=1, ascending=False)
    keep = sorted(rank.columns[(rank <= top).any()])
    print(f"[binance] {len(keep)} symbols were ever top-{top}; downloading hourly", flush=True)
    hourly = _parallel(lambda s: _symbol_klines(s, "1h", start), keep)
    close = pd.DataFrame({s: d["close"] for s, d in hourly.items() if d is not None})
    qvol = pd.DataFrame({s: d["qvol"] for s, d in hourly.items() if d is not None})
    meta = pd.DataFrame({"ticker": close.columns, "group": "crypto", "name": close.columns,
                         "date_added": ""})
    _save("binance_1h", close, qvol, meta)


TASKS = {"sp500_1h": download_sp500_1h, "sp500_1d": download_sp500_1d,
         "etf_1d": download_etf_1d, "binance_1h": download_binance_1h}

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
            print(f"[{name}] FAILED: {exc}", flush=True); failed.append(name)
        print(f"[{name}] {time.time() - t0:.0f}s", flush=True)
    sys.exit(1 if len(failed) == len(todo) else 0)
