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


def main(names: list[str], reps: int = 2000, complete_case: bool = False) -> None:
    """complete_case: robustness run in which a coin must be observed on EVERY
    formation bar (no forward filling in formation); detailed configs only."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.time()
        tag = name + ("_cc" if complete_case else "")
        print(f"=== {tag}", flush=True)
        res = backtest.run(name, configs=C.detailed() if complete_case else None,
                           ic=not complete_case, complete_case=complete_case)
        if len(res["folds"]) == 0:
            print(f"=== {tag} skipped: history shorter than one walk-forward window "
                  f"({C.TRAIN_BARS + C.TEST_BARS} bars)", flush=True)
            continue
        with open(CACHE / f"{tag}.pkl", "wb") as fh:
            pickle.dump(res, fh)
        report.analyse(res, out=report.RESULTS / tag, reps=reps)
        print(f"=== {tag} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    cc = "--complete-case" in args
    args = [a for a in args if a != "--complete-case"] or ["all"]
    main(available() if args == ["all"] else args, complete_case=cc)
