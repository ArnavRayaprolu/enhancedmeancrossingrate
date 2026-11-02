"""Multi-horizon ensembles: equal capital in every member sub-strategy.

    python research/ensemble.py it3

An ensemble is defined structurally (all horizons x all bands for a criterion and
portfolio size), never by picking members on results.  Each member is a
self-financing long/short book with unit capital and its own costs, so the
ensemble return is the average of member returns (no cost netting assumed).
"""
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from emcr import config as C, inference as I  # noqa: E402

VENUES = ["binance_1h", "coinbase_1h", "okx_1h", "perp_1h"]


def ensembles(keys):
    out = {}
    for c in sorted({k.split("|")[0] for k in keys}):
        for n in sorted({re.search(r"\|n(\d+)", k).group(1) for k in keys if "|n" in k}):
            out[f"{c} all-horizon n{n}"] = [k for k in keys if k.startswith(c + "|")
                                            and k.endswith(f"|n{n}")]
    crits = sorted({k.split("|")[0] for k in keys})
    if len(crits) > 1:
        for n in sorted({re.search(r"\|n(\d+)", k).group(1) for k in keys if "|n" in k}):
            out[f"{'+'.join(crits)} all-horizon n{n}"] = [k for k in keys if k.endswith(f"|n{n}")]
    return out


def main(batch):
    nets = pd.read_pickle(f"results/iterations/nets_{batch}.pkl")
    keys = sorted({k for k, _ in nets})
    rows = []
    for name, mem in ensembles(keys).items():
        r = {"ensemble": name, "members": len(mem)}
        for v in VENUES:
            if not all((m, v) in nets for m in mem):
                continue
            df = pd.concat([nets[(m, v)] for m in mem], axis=1).fillna(0.0)
            e = df.mean(axis=1)
            bpy = C.DATASETS[v].bars_per_year
            mt = I.metrics(e, bpy)
            r[v] = mt["sharpe"]
            r[v + " ret"] = mt["total_return"]
            r[v + " mdd"] = mt["max_drawdown"]
        rows.append(r)
    d = pd.DataFrame(rows).set_index("ensemble")
    d["mean"] = d[[v for v in VENUES if v in d]].mean(axis=1)
    d["min"] = d[[v for v in VENUES if v in d]].min(axis=1)
    pd.set_option("display.width", 250)
    print(d.sort_values("mean", ascending=False).round(2).to_string())
    d.to_csv(f"results/iterations/ensembles_{batch}.csv")


if __name__ == "__main__":
    main(sys.argv[1])
