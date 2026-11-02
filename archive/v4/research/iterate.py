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


def evaluate_many(ds_name: str, specs: dict, train: int = C.TRAIN_BARS,
                  test: int = C.TEST_BARS) -> dict:
    """Stream over walk-forward windows; one universe per distinct spec.top_k."""
    base = C.DATASETS[ds_name]
    close, dvol, meta = load(base)
    perp = base.source.startswith("binance_perp")
    fund = load_funding(close.index) if perp else None
    kappa = base.cost_bps / 1e4
    borrow_bar = 0.0 if perp else base.borrow_pa / base.bars_per_year
    ks = sorted({sp.top_k for sp in specs.values()})
    acc = {k: ([], [], []) for k in specs}
    for f in make_folds(len(close), train, test):
        for tk in ks:
            fd = build_fold(close, dvol, meta, replace(base, top_k=tk), f, train, fund)
            if fd is None:
                continue
            idx = fd["idx"][fd["train"]:]
            for k, sp in specs.items():
                if sp.top_k != tk:
                    continue
                n, g, u = run_fold(fd, sp, kappa, borrow_bar, perp)
                acc[k][0].append(pd.Series(n, index=idx))
                acc[k][1].append(g.sum()); acc[k][2].append(u.sum())
    ds = base
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


SCALE = {"perp_15m": 4, "binance_15m": 4, "perp_30m": 2, "binance_30m": 2}


def _job(args):
    ds, specs = args
    t = time.time()
    m = SCALE.get(ds, 1) if specs and next(iter(specs.values())).params.get("hourly_design") else 1
    res = evaluate_many(ds, specs, C.TRAIN_BARS * m, C.TEST_BARS * m)
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
    piv["mean"] = piv.mean(axis=1); piv["min"] = piv[venues].min(axis=1)
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


def batch_it3():
    """Members for multi-horizon ensembles: 5 horizons x 2 bands x 2 criteria x 2 sizes."""
    specs = {}
    for c, w, e, n in itertools.product(["MCR", "HalfLife"], [12, 24, 48, 72, 168],
                                        [2.5, 3.0], [10, 20]):
        specs[f"{c}|w{w}|e{e}|n{n}"] = Spec(
            criterion=c, signal="LEVEL", window=w, entry=e, exit=0.0, n_pairs=n, max_uses=1,
            top_k=100, weighting="rank")
    return specs


def batch_it4():
    """Short horizons: 6-16 bars, bands 2.0/2.5, 5 or 10 pairs; MCR vs half-life vs coint."""
    specs = {}
    for c, w, e, n in itertools.product(["MCR", "HalfLife", "Cointegration"], [6, 8, 12, 16],
                                        [2.0, 2.5], [5, 10]):
        specs[f"{c}|w{w}|e{e}|n{n}"] = Spec(
            criterion=c, signal="LEVEL", window=w, entry=e, exit=0.0, n_pairs=n, max_uses=1,
            top_k=100, weighting="rank")
    return specs


def batch_bounce():
    """Bid-ask-bounce check: leading designs with a 0- or 1-bar execution delay and a
    top-100 or top-50 (more liquid) universe; cointegration as benchmark."""
    specs = {}
    for c, (w, e, n), lag, k in itertools.product(
            ["MCR", "HalfLife", "Cointegration"],
            [(12, 2.0, 5), (12, 2.5, 10), (6, 2.0, 10)], [0, 1], [100, 50]):
        specs[f"{c}|w{w}|e{e}|n{n}|lag{lag}|k{k}"] = Spec(
            criterion=c, signal="LEVEL", window=w, entry=e, exit=0.0, n_pairs=n, max_uses=1,
            top_k=k, weighting="rank", params={"lag": lag} if lag else {})
    return specs


def batch_exec15():
    """Hourly design executed on 15-minute prices: all windows x4 in 15m bars
    (12h mean = 48 bars, 630/315-hour walk-forward), execution delay 0 / 1 / 4
    bars (0 / 15 / 60 minutes)."""
    specs = {}
    for c, (wh, e, n), lag in itertools.product(
            ["MCR", "HalfLife", "Cointegration", "Distance"],
            [(12, 2.0, 5), (12, 2.5, 10), (24, 2.5, 10)], [0, 1, 4]):
        specs[f"{c}|w{wh}h|e{e}|n{n}|delay{lag * 15}m"] = Spec(
            criterion=c, signal="LEVEL", window=wh * 4, entry=e, exit=0.0, n_pairs=n,
            max_uses=1, top_k=100, weighting="rank",
            params={"lag": lag, "hourly_design": True})
    return specs


def batch_slow():
    """Slower equilibria (3-14 day means on hourly bars) with 0 / 1-hour execution delay."""
    specs = {}
    for c, w, e, lag in itertools.product(["MCR", "HalfLife", "Cointegration", "Distance"],
                                          [72, 168, 336], [2.0, 2.5], [0, 1]):
        specs[f"{c}|w{w}|e{e}|n10|delay{lag}h"] = Spec(
            criterion=c, signal="LEVEL", window=w, entry=e, exit=0.0, n_pairs=10, max_uses=1,
            top_k=100, weighting="rank", params={"lag": lag} if lag else {})
    return specs


HOURLY = ["crypto_1h", "sp500_1h", "nifty50_1h", "etf_1h"]


def batch_it5():
    """Innovation round (all with a 1-bar execution delay): signals x engines x criteria."""
    sigs = {"ratio140": dict(signal="LEVEL", window=140, entry=2.0, exit=0.5),
            "ratio48": dict(signal="LEVEL", window=48, entry=2.0, exit=0.5),
            "maratio": dict(signal="MAD", window=140, entry=1.0, exit=0.25)}
    engines = {"threshold": {}, "confirm": {"confirm": True}, "agree": {"agree": 1.0},
               "costgate": {"min_dev_cost": 3},
               "confirm+costgate": {"confirm": True, "min_dev_cost": 3},
               "confirm+agree+stop": {"confirm": True, "agree": 1.0, "timestop": True}}
    crits = {"EMCR": "MCR", "EMCRc": "EMCRc", "EMCRamp": "EMCRamp",
             "EMCRamp+Coint+HalfLife": "EMCRamp+Coint+HalfLife",
             "EMCR+Coint+HalfLife": "EMCR+Coint+HalfLife",
             "EMCRc+Coint+HalfLife": "EMCRc+Coint+HalfLife", "Cointegration": "Cointegration",
             "HalfLife": "HalfLife", "Distance": "Distance", "Correlation": "Correlation"}
    specs = {}
    for (sn, sk), (en, ek), (cn, cc) in itertools.product(sigs.items(), engines.items(),
                                                          crits.items()):
        prm = {"lag": 1, **{k: v for k, v in ek.items() if k != "timestop"}}
        specs[f"{sn}|{en}|{cn}"] = Spec(criterion=cc, n_pairs=20, max_uses=1, top_k=100,
                                       weighting="rank", params=prm,
                                       max_hold=sk["window"] if ek.get("timestop") else None,
                                       **sk)
    return specs


if __name__ == "__main__":
    name = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    venues = sys.argv[3].split(",") if len(sys.argv) > 3 else (HOURLY if name == "it5" else VENUES)
    tag = name if len(sys.argv) <= 3 else f"{name}_{'_'.join(venues)}"
    run_batch(tag, globals()[f"batch_{name}"](), venues=venues, workers=workers)
