"""Download the Kaggle dataset leukipp/hourly-stock-prices and convert it to
hourly price panels (one column per ticker) under data/kaggle_hourly/.

    python scripts/download_kaggle.py

Runs on a GitHub Actions runner (kaggle.com is not reachable from the
research container). Public datasets download without credentials; if the
runner has KAGGLE_USERNAME / KAGGLE_KEY set they are used.

Output
    data/kaggle_hourly/manifest.txt        file tree, sizes, column names, heads
    data/kaggle_hourly/<field>_<k>.parquet open/high/low/close/volume panels,
                                           float32, split so no file exceeds ~80 MB
    data/kaggle_hourly/meta.csv            per-ticker source file, first/last bar, rows
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "kaggle_hourly"
FIELDS = ["open", "high", "low", "close", "adjusted_close", "volume"]
TIME_COLS = ["datetime", "date", "time", "timestamp", "date_time", "index", "unnamed: 0"]
TICKER_COLS = ["ticker", "symbol", "name", "stock"]
MAX_COLS_PER_FILE = 400


def log(msg: str, fh) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")


def read_any(p: Path) -> pd.DataFrame | None:
    s = p.suffix.lower()
    if s in (".csv", ".txt"):
        return pd.read_csv(p)
    if s == ".parquet":
        return pd.read_parquet(p)
    if s in (".json",):
        return None
    return None


def normalise(df: pd.DataFrame, stem: str) -> dict[str, pd.DataFrame]:
    """Return {ticker: frame indexed by UTC timestamp with columns in FIELDS}."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj_close": "close"})
    tcol = next((c for c in TIME_COLS if c in df.columns), None)
    if tcol is None:
        return {}
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.assign(_ts=ts).dropna(subset=["_ts"])
    kcol = next((c for c in TICKER_COLS if c in df.columns and df[c].dtype == object), None)
    groups = df.groupby(kcol) if kcol else [(stem, df)]
    out = {}
    for tk, g in groups:
        cols = [c for c in FIELDS if c in g.columns]
        if "close" not in cols:
            continue
        x = g.set_index("_ts")[cols].apply(pd.to_numeric, errors="coerce")
        x = x[~x.index.duplicated(keep="last")].sort_index()
        x.attrs["name"] = str(g["name"].iloc[0]) if "name" in g.columns else ""
        out[str(tk).upper().strip()] = x
    return out


def main() -> None:
    import kagglehub
    path = Path(kagglehub.dataset_download("leukipp/hourly-stock-prices"))
    print("Path to dataset files:", path, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    with open(OUT / "manifest.txt", "w") as fh:
        total = sum(p.stat().st_size for p in files)
        log(f"dataset root: {path}\nfiles: {len(files)}  total size: {total/1e6:.1f} MB", fh)
        for p in files[:300]:
            log(f"  {p.relative_to(path)}  {p.stat().st_size/1e3:.0f} kB", fh)
        if len(files) > 300:
            log(f"  ... {len(files) - 300} more files", fh)
        shown = 0
        for p in files:
            if p.suffix.lower() in (".csv", ".parquet", ".txt") and shown < 4:
                d = read_any(p)
                if d is not None:
                    log(f"\n== {p.relative_to(path)}  shape={d.shape}\ncolumns={list(d.columns)}\n"
                        f"{d.head(5).to_string()}\n...\n{d.tail(3).to_string()}", fh)
                    shown += 1
            if p.suffix.lower() in (".md", ".json") and p.stat().st_size < 50_000:
                log(f"\n== {p.relative_to(path)}\n{p.read_text(errors='ignore')[:4000]}", fh)

        frames: dict[str, pd.DataFrame] = {}
        meta = []
        for p in files:
            try:
                d = read_any(p)
            except Exception as e:  # noqa: BLE001
                log(f"skip {p.name}: {e}", fh)
                continue
            if d is None or d.empty:
                continue
            # folder name is often the interval or exchange; keep the file stem as ticker
            for tk, x in normalise(d, p.stem).items():
                if tk in frames:
                    nm = x.attrs.get("name", "")
                    x = pd.concat([frames[tk], x])
                    x.attrs["name"] = nm
                    x = x[~x.index.duplicated(keep="last")].sort_index()
                frames[tk] = x
                meta.append(dict(ticker=tk, file=str(p.relative_to(path)),
                                 name=x.attrs.get("name", "")))
        log(f"\ntickers parsed: {len(frames)}", fh)
        if not frames:
            sys.exit("no price data recognised; see manifest.txt")
        md = pd.DataFrame(meta).drop_duplicates("ticker").set_index("ticker")
        md["first"] = [frames[t].index.min() for t in md.index]
        md["last"] = [frames[t].index.max() for t in md.index]
        md["rows"] = [len(frames[t]) for t in md.index]
        md.to_csv(OUT / "meta.csv")
        log(md.describe(include="all").to_string(), fh)
        tickers = sorted(frames)
        for f in FIELDS:
            if not any(f in frames[t].columns for t in tickers):
                continue
            panel = pd.concat({t: frames[t][f] for t in tickers if f in frames[t].columns},
                              axis=1).sort_index().astype(np.float32)
            for k in range(0, panel.shape[1], MAX_COLS_PER_FILE):
                part = panel.iloc[:, k:k + MAX_COLS_PER_FILE]
                fn = OUT / f"{f}_{k // MAX_COLS_PER_FILE:03d}.parquet"
                part.to_parquet(fn, compression="zstd")
                log(f"wrote {fn.name} {part.shape} {fn.stat().st_size/1e6:.1f} MB", fh)


if __name__ == "__main__":
    main()
