"""Raw price-ratio strategy (no z-score) with a magnitude-filtered crossing criterion.

    python research/ratio_scan.py dev       # 2020-2023 practice data (logged as trials)
    python research/ratio_scan.py posthoc   # 2024-2026, POST-HOC (exam already used)
"""
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from emcr.explore import Spec, evaluate  # noqa: E402

DS = "perp_1h"
CRIT = ["MCRmag", "MCR", "Cointegration", "HalfLife", "Distance", "Correlation"]
MEANS = [("RATIO_FIX", 140), ("RATIO_MA", 24), ("RATIO_MA", 140)]
DELTAS = [0.005, 0.01, 0.02, 0.04]
OUT = Path("results/ratio_scan")


def run(args):
    period, sig, w, c = args
    rows = []
    for dlt in DELTAS:
        sp = Spec(criterion=c, signal=sig, window=w, entry=dlt, exit=0.0, n_pairs=5,
                  max_uses=1, top_k=100, weighting="equal", params={"delta": dlt})
        m = evaluate(DS, sp, period=period, log=(period == "dev"), note="ratio scan")
        m.pop("net")
        rows.append(dict(period=period, mean=f"{sig}{'' if sig == 'RATIO_FIX' else w}",
                         delta=dlt, criterion=c, sharpe=m["sharpe"],
                         total_return=m["total_return"], max_dd=m["max_dd"],
                         gross_pa=m["gross_pa"], turnover_pa=m["turnover_pa"]))
    return rows


if __name__ == "__main__":
    period = sys.argv[1]
    jobs = [(period, s, w, c) for (s, w), c in itertools.product(MEANS, CRIT)]
    rows = []
    with Pool(3) as p:
        for r in p.imap_unordered(run, jobs):
            rows += r
            print(r[0]["mean"], r[0]["criterion"], [round(x["sharpe"], 2) for x in r], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"{period}.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.pivot_table(index=["mean", "delta"], columns="criterion", values="sharpe")
          .round(2).to_string())
