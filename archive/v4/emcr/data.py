"""Load the standardised wide parquet datasets written by scripts/download_data.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load(ds: C.Dataset) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (close, dollar_volume, meta) for a dataset, resampled if required."""
    folder = DATA / ds.source
    read = lambda stem: pd.concat([pd.read_parquet(f) for f in
                                   sorted(folder.glob(f"{stem}*.parquet"))]).astype(float)
    close, dvol = read("close"), read("dvol")
    close = close[~close.index.duplicated()]; dvol = dvol[~dvol.index.duplicated()]
    meta = pd.read_csv(folder / "meta.csv", dtype=str).fillna("")
    close.index = pd.to_datetime(close.index, utc=True)
    dvol.index = close.index
    close = close.sort_index(); dvol = dvol.reindex(close.index)
    if ds.resample:
        # bar labelled by its left edge, close = last observed price in the bar
        close = close.resample(ds.resample, label="left", closed="left").last()
        dvol = dvol.resample(ds.resample, label="left", closed="left").sum(min_count=1)
        keep = close.notna().any(axis=1)
        close, dvol = close[keep], dvol[keep]
    close = close.where(close > 0)
    meta = meta.set_index("ticker").reindex(close.columns)
    meta["group"] = meta["group"].replace("", "all").fillna("all")
    if "date_added" not in meta:
        meta["date_added"] = ""
    return close, dvol, meta


def make_folds(n_bars: int, train: int = C.TRAIN_BARS, test: int = C.TEST_BARS):
    """Non-overlapping walk-forward folds on bar indices.
    Fold k trains on [s, s+train) and trades on [s+train, s+train+test), s = k*test."""
    folds, s, k = [], 0, 0
    while s + train + test <= n_bars:
        folds.append(dict(fold=k, start=s, train_end=s + train, end=s + train + test))
        s += test; k += 1
    return folds


def eligible(close: pd.DataFrame, dvol: pd.DataFrame, meta: pd.DataFrame,
             ds: C.Dataset, f: dict, complete_case: bool = False) -> list[str]:
    """Securities eligible in fold f, decided with formation-window data only.

    * price observed at the first and the last formation bar, and at least
      MIN_COVERAGE of formation bars observed (100% if complete_case);
    * equities: index member at the start of the formation window
      (date added <= formation start), i.e. no look-ahead from later additions;
    * crypto: the top-K by median formation-window dollar volume.
    """
    tr = close.iloc[f["start"]:f["train_end"]]
    cov = tr.notna().mean()
    need = 1.0 if complete_case else C.MIN_COVERAGE
    ok = (cov >= need) & tr.iloc[0].notna() & tr.iloc[-1].notna()
    if "date_added" in meta and (meta["date_added"] != "").any():
        added = pd.to_datetime(meta["date_added"], errors="coerce", utc=True)
        ok &= ~(added > tr.index[0]).reindex(ok.index).fillna(False)
    names = ok[ok].index
    if ds.top_k:
        med = dvol.iloc[f["start"]:f["train_end"]][names].median()
        names = med.sort_values(ascending=False).index[:ds.top_k]
    return sorted(names)


def candidate_pairs(names: list[str], meta: pd.DataFrame, rule: str):
    """Indices (ia, ib) of candidate pairs, ia < ib, within group or across all."""
    grp = meta.loc[names, "group"].to_numpy() if rule == "sector" else np.zeros(len(names))
    ia, ib = np.triu_indices(len(names), k=1)
    keep = grp[ia] == grp[ib]
    return ia[keep], ib[keep]
