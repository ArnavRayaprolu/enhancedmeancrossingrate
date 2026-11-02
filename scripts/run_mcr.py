"""Run the frozen protocol (mcr/config.py) on the Alpaca hourly data.

    python scripts/run_mcr.py run [--workers 4] [--folds N]   # backtests -> results/raw/
    python scripts/run_mcr.py report                          # tables -> results/, REPORT.md

Configurations that share their candidate statistics run together (one
statistics pass per fold). The ORIGINAL notebook configuration, PRIMARY and
the one-at-a-time GRID are all reported.
"""
from __future__ import annotations

import sys
import time
from collections import OrderedDict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcr import config as C, data as D, report as R, walkforward as W  # noqa: E402

RAW = R.RES / "raw"


def groups() -> "OrderedDict[tuple, list]":
    g: OrderedDict = OrderedDict()
    for c in [C.PRIMARY] + C.grid() + [C.ORIGINAL]:
        g.setdefault(C.stat_key(c), []).append(c)
    return g


def run(workers: int = 4, folds: int | None = None):
    RAW.mkdir(parents=True, exist_ok=True)
    data = D.load()
    print(f"data: {data['close'].shape[0]} bars x {data['close'].shape[1]} names, "
          f"{data['close'].index[0]} .. {data['close'].index[-1]}", flush=True)
    for i, (key, cfgs) in enumerate(groups().items()):
        tag = cfgs[0].label
        if (RAW / f"bars_{tag}.parquet").exists():
            print(f"skip {tag} (done)"); continue
        t0 = time.time()
        diag = cfgs[0].label == C.PRIMARY.label
        out = W.run_group(cfgs, data, diagnostics=diag, workers=workers, folds_limit=folds)
        out["bars"].to_parquet(RAW / f"bars_{tag}.parquet")
        out["picks"].to_parquet(RAW / f"picks_{tag}.parquet")
        out["folds"].assign(group=tag).to_csv(RAW / f"folds_{tag}.csv", index=False)
        if diag:
            out["diag"].to_csv(RAW / "diag_primary.csv", index=False)
            out["diag_sample"].to_parquet(RAW / "diag_sample_primary.parquet")
        print(f"== group {tag} ({[c.label for c in cfgs]}) done in {time.time() - t0:.0f}s",
              flush=True)


def report():
    bars = pd.concat([pd.read_parquet(p) for p in sorted(RAW.glob("bars_*.parquet"))])
    picks = pd.concat([pd.read_parquet(p) for p in sorted(RAW.glob("picks_*.parquet"))])
    folds = pd.concat([pd.read_csv(p) for p in sorted(RAW.glob("folds_*.csv"))])
    diag = pd.read_csv(RAW / "diag_primary.csv") if (RAW / "diag_primary.csv").exists() else None
    from mcr import report_md
    report_md.write(bars, picks, folds, diag)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "run":
        w = int(a[a.index("--workers") + 1]) if "--workers" in a else 4
        n = int(a[a.index("--folds") + 1]) if "--folds" in a else None
        run(w, n)
    elif a[0] == "report":
        report()
