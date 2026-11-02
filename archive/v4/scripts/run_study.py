"""Run the frozen protocol on one or more datasets and write results/<dataset>/*.csv.

    python scripts/run_study.py binance_1h perp_1h coinbase_1h ...
    python scripts/run_study.py all          # every dataset whose data folder exists
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emcr import backtest, config as C, report  # noqa: E402
from emcr.data import DATA  # noqa: E402

CACHE = report.RESULTS / "cache"


def available() -> list[str]:
    return [k for k, d in C.DATASETS.items()
            if d.role != "test" and any((DATA / d.source).glob("close*.parquet"))]


def light_report(res: dict, out: Path) -> None:
    """Performance and pre-registered DM tests for robustness-only runs."""
    import pandas as pd
    from emcr import inference as I
    out.mkdir(parents=True, exist_ok=True)
    ds = C.DATASETS[res["dataset"]]
    k = ds.cost_bps / 1e4
    nets, rows = {}, []
    for (key, m), df in res["detail"].items():
        n = report.net_of(df, k); nets[(key, m)] = n
        rows.append(dict(config=key, criterion=m, **I.metrics(n, ds.bars_per_year, df["exposure"]),
                         **I.breakeven(df["gross"].to_numpy(), df["turn"].to_numpy(),
                                       df["borrow"].to_numpy()),
                         turnover_pa=float(df["turn"].mean() * ds.bars_per_year)))
    pd.DataFrame(rows).to_csv(out / "performance.csv", index=False)
    dm = []
    for key in {k_ for k_, _ in nets}:
        for a, b in C.PRIMARY_TESTS:
            if (key, a) in nets and (key, b) in nets:
                x, y = nets[(key, a)], nets[(key, b)].reindex(nets[(key, a)].index)
                t = I.diebold_mariano(x.to_numpy(), y.to_numpy(), "nw")
                dm.append(dict(config=key, A=a, B=b, dm_nw=t["dm"], p_nw=t["p"],
                               mean_diff_pa=float((x - y).mean() * ds.bars_per_year)))
    pd.DataFrame(dm).to_csv(out / "dm.csv", index=False)
    res["folds"].to_csv(out / "folds.csv", index=False)


def main(names: list[str], reps: int = 2000, complete_case: bool = False,
         delayed: bool = False, lean: bool = False) -> None:
    """complete_case: robustness run in which a coin must be observed on EVERY
    formation bar (no forward filling in formation); detailed configs only."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.time()
        tag = name + ("_cc" if complete_case else "") + ("_lag1" if delayed else "")
        print(f"=== {tag}", flush=True)
        cfgs = [C.DELAYED] if delayed else (C.detailed() if (complete_case or lean) else None)
        res = backtest.run(name, configs=cfgs, ic=not (complete_case or delayed),
                           complete_case=complete_case, detail_extra=cfgs if delayed else None)
        if len(res["folds"]) == 0:
            print(f"=== {tag} skipped: history shorter than one walk-forward window "
                  f"({C.TRAIN_BARS + C.TEST_BARS} bars)", flush=True)
            continue
        if delayed:
            light_report(res, report.RESULTS / tag)
            print(f"=== {tag} done in {time.time() - t0:.0f}s", flush=True)
            continue
        with open(CACHE / f"{tag}.pkl", "wb") as fh:
            pickle.dump(res, fh)
        report.analyse(res, out=report.RESULTS / tag, reps=reps)
        print(f"=== {tag} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    cc = "--complete-case" in args
    dl = "--delayed" in args
    ln = "--lean" in args          # headline configurations only (memory-lean, long series)
    args = [a for a in args if a not in ("--complete-case", "--delayed", "--lean")] or ["all"]
    main(available() if args == ["all"] else args, complete_case=cc, delayed=dl, lean=ln)
