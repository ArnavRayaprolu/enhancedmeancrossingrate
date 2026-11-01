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


def main(names: list[str], reps: int = 2000) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.time()
        print(f"=== {name}", flush=True)
        res = backtest.run(name)
        with open(CACHE / f"{name}.pkl", "wb") as fh:
            pickle.dump(res, fh)
        report.analyse(res, reps=reps)
        print(f"=== {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    main(available() if args == ["all"] else args)
