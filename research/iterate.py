"""Multi-venue walk-forward evaluation of candidate designs (all data, all folds).

    python research/iterate.py <batch>

Each window is built once and every design in the batch is traded on it, then
the window is discarded (memory-lean).  Each design is scored on every venue;
results are appended to results/iterations/log.csv (every variant tried is kept).
"""
import itertools
import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from emcr import config as C  # noqa: E402
from emcr.data import DATA, load, make_folds  # noqa: E402
from emcr.explore import Spec, build_fold, load_funding, run_fold  # noqa: E402

VENUES = ["perp_1h", "binance_1h", "coinbase_1h", "okx_1h"]
OUT = Path("results/iterations")


def evaluate_many(ds_name: str, specs: dict, top_k: int = 100) -> dict:
    ds = replace(C.DATASETS[ds_name], top_k=top_k)
    close, dvol, meta = load(ds)
    perp = ds.source.startswith("binance_perp")
    fund = load_funding(close.index) if perp else None
    kappa = ds.cost_bps / 1e4
    borrow_bar = 0.0 if perp else ds.borrow_pa / ds.bars_per_year
    acc = {k: ([], [], []) for k in specs}
    for f in make_folds(len(close)):
        fd = build_fold(close, dvol, meta, ds, f, C.TRAIN_BARS, fund)
        if fd is None:
            continue
        idx = fd["idx"][fd["train"]:]
        for k, sp in specs.items():
            n, g, u = run_fold(fd, sp, kappa, borrow_bar, perp)
            acc[k][0].append(pd.Series(n, index=idx))
            acc[k][1].append(g.sum()); acc[k][2].append(u.sum())
    out = {}
    bpy = ds.bars_per_year
    for k, (nets, gs, us) in acc.items():
        net = pd.concat(nets).sort_index()
        out[k] = dict(sharpe=float(net.mean() / net.std() * np.sqrt(bpy)),
                      total_return=float((1 + net).prod() - 1),
                      max_dd=float(((1 + net).cumprod() / (1 + net).cumprod().cummax() - 1).min()),
                      gross_pa=float(sum(gs) / len(net) * bpy),
                      turnover_pa=float(sum(us) / len(net) * bpy),
                      first=str(net.index[0])[:10], last=str(net.index[-1])[:10],
                      net=net)
    return out


def _job(args):
    ds, specs = args
    t = time.time()
    res = evaluate_many(ds, specs)
    for v in res.values():
        v["seconds"] = time.time() - t
    return ds, res


def run_batch(name: str, specs: dict, venues=VENUES, workers=3):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, nets = [], {}
    with Pool(min(workers, len(venues))) as p:
        for ds, res in p.imap_unordered(_job, [(v, specs) for v in venues]):
            for k, m in res.items():
                nets[(k, ds)] = m.pop("net")
                rows.append(dict(batch=name, design=k, venue=ds, spec=specs[k].label(), **m))
                print(f"{name:8s} {k:34s} {ds:12s} sharpe={m['sharpe']:+.2f} "
                      f"ret={m['total_return']:+.2f} turn={m['turnover_pa']:.0f} "
                      f"gross={m['gross_pa']:+.2f}", flush=True)
    df = pd.DataFrame(rows)
    lp = OUT / "log.csv"
    df.to_csv(lp, mode="a", header=not lp.exists(), index=False)
    piv = df.pivot_table(index="design", columns="venue", values="sharpe")
    piv["mean"] = piv.mean(1); piv["min"] = piv[venues].min(1)
    print(piv.sort_values("mean", ascending=False).round(2).to_string(), flush=True)
    pd.to_pickle(nets, OUT / f"nets_{name}.pkl")
    return df


# --------------------------------------------------------------------------- #
# batches
# --------------------------------------------------------------------------- #
def batch_it1():
    """Baseline z-score rule, criteria x weighting rules (fixed, textbook params)."""
    base = dict(signal="LEVEL", window=24, entry=2.5, exit=0.0, n_pairs=10, max_uses=1,
                top_k=100)
    specs = {}
    for c, w in itertools.product(["MCR", "ECR", "MCRms", "HalfLife", "Cointegration",
                                   "Distance"], ["active", "rank", "rankvol"]):
        kw = dict(base, criterion=c, weighting="rank" if w == "rankvol" else w)
        prm = {"vol_scale": True} if w == "rankvol" else {}
        specs[f"{c}|{w}"] = Spec(**kw, params=prm)
    return specs




def batch_it2():
    """Fixed rank weights + cost-aware entry, band width, exit level, horizon."""
    specs = {}
    for c, w, e, x, md in itertools.product(["MCR", "HalfLife", "Cointegration"], [24, 72],
                                            [2.5, 3.0], [0.0, -0.5], [0, 3]):
        specs[f"{c}|w{w}|e{e}|x{x}|md{md}"] = Spec(
            criterion=c, signal="LEVEL", window=w, entry=e, exit=x, n_pairs=10, max_uses=1,
            top_k=100, weighting="rank", params={"min_dev_cost": md} if md else {})
    return specs


if __name__ == "__main__":
    name = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    run_batch(name, globals()[f"batch_{name}"](), workers=workers)
