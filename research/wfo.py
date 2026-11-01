"""Walk-forward optimisation (WFO) with Bayesian optimisation.

    python research/wfo.py <dataset> [<dataset> ...]

Design (no hand-picked parameters)
----------------------------------
The data are cut into consecutive walk-forward windows (630-bar formation +
315-bar trading).  Before trading window k, a Bayesian optimiser (Optuna TPE,
Bergstra et al. 2011) chooses the strategy parameters theta by maximising the
Sharpe ratio the strategy WOULD HAVE earned over the previous (up to) CALIB
windows k-CALIB..k-1 -- data that were fully observed before window k starts.
Window k is then traded once with that theta.  Re-optimisation every REOPT
windows (previous best is kept in between).  The first MIN_CALIB windows are
burn-in (no trading) for every criterion alike.

Strategy (raw price ratio, no z-score)
    d_t = r_t / mean(r) - 1,  r_t = P_A / P_B
    criterion MCRmag: count crossings of the mean, a crossing counting only if the
              ratio first moved at least delta away from the mean; rate per bar
    entry  |d_t| >= delta (long the ratio below its mean, short above)
    exit   when the ratio crosses back through its mean
theta = (which mean, delta, number of pairs, capital rule); the same theta space and
optimiser are given to every competing criterion.
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, ".")
from emcr import config as C  # noqa: E402
from emcr.data import DATA, load, make_folds  # noqa: E402
from emcr.explore import Spec, build_fold, load_funding, run_fold  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

CRITERIA = ["MCRmag", "MCR", "Cointegration", "Distance", "Correlation", "HalfLife"]
SPACE = dict(mean=["FIX", 12, 24, 48, 96, 168, 336],
             delta=[0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06],
             n_pairs=[3, 5, 8, 10], weighting=["equal", "active"])
CALIB, MIN_CALIB, REOPT, TRIALS, TOP_K = 8, 3, 3, 30, 100
SEED = C.SEED
OUT = Path("results/wfo")


def to_spec(th: dict, criterion: str) -> Spec:
    fix = th["mean"] == "FIX"
    return Spec(criterion=criterion, signal="RATIO_FIX" if fix else "RATIO_MA",
                window=140 if fix else int(th["mean"]), entry=th["delta"], exit=0.0,
                n_pairs=th["n_pairs"], max_uses=1, top_k=TOP_K, weighting=th["weighting"],
                params={"delta": th["delta"]})


def sharpe(x: np.ndarray, bpy: float) -> float:
    s = x.std()
    return float(x.mean() / s * np.sqrt(bpy)) if s > 0 else -10.0


def run(ds_name: str) -> None:
    t0 = time.time()
    ds = replace(C.DATASETS[ds_name], top_k=TOP_K)
    close, dvol, meta = load(ds)
    fund = load_funding(close.index) if ds.source.startswith("binance_perp") and \
        (DATA / "binance_perp_funding" / "funding.parquet").exists() else None
    use_fund = fund is not None
    kappa = ds.cost_bps / 1e4
    borrow_bar = 0.0 if use_fund else ds.borrow_pa / ds.bars_per_year
    bpy = ds.bars_per_year

    past: deque = deque(maxlen=CALIB)           # previous folds (with their caches)
    best = {c: None for c in CRITERIA}
    nets = {c: [] for c in CRITERIA}
    grs = {c: [] for c in CRITERIA}
    tns = {c: [] for c in CRITERIA}
    chosen = []
    for k, f in enumerate(make_folds(len(close))):
        fd = build_fold(close, dvol, meta, ds, f, C.TRAIN_BARS, fund)
        if fd is None:
            continue
        te_idx = fd["idx"][fd["train"]:]
        if len(past) >= MIN_CALIB:
            for c in CRITERIA:
                if best[c] is None or k % REOPT == 0:
                    def objective(trial, c=c):
                        th = {p: trial.suggest_categorical(p, v) for p, v in SPACE.items()}
                        sp = to_spec(th, c)
                        x = np.concatenate([run_fold(g, sp, kappa, borrow_bar, use_fund)[0]
                                            for g in past])
                        return sharpe(x, bpy)
                    study = optuna.create_study(
                        direction="maximize",
                        sampler=optuna.samplers.TPESampler(seed=SEED + k, n_startup_trials=10))
                    if best[c] is not None:
                        study.enqueue_trial(best[c])
                    study.optimize(objective, n_trials=TRIALS)
                    best[c] = study.best_params
                    chosen.append(dict(fold=fd["fold"], test_start=te_idx[0], criterion=c,
                                       calib_sharpe=study.best_value, **study.best_params))
                n, g, u = run_fold(fd, to_spec(best[c], c), kappa, borrow_bar, use_fund)
                nets[c].append(pd.Series(n, index=te_idx))
                grs[c].append(pd.Series(g, index=te_idx))
                tns[c].append(pd.Series(u, index=te_idx))
        past.append(fd)
        print(f"[{ds_name}] fold {fd['fold']:>4} {str(te_idx[0])[:10]} "
              f"coins={len(fd['names'])} {time.time() - t0:6.0f}s", flush=True)

    out = OUT / ds_name
    out.mkdir(parents=True, exist_ok=True)
    for nm, dct in (("net", nets), ("gross", grs), ("turn", tns)):
        pd.DataFrame({c: pd.concat(v) for c, v in dct.items() if v}).to_parquet(
            out / f"{nm}.parquet")
    pd.DataFrame(chosen).to_csv(out / "params.csv", index=False)
    json.dump(dict(dataset=ds_name, calib=CALIB, min_calib=MIN_CALIB, reopt=REOPT,
                   trials=TRIALS, top_k=TOP_K, cost_bps=ds.cost_bps, funding=use_fund,
                   borrow_pa=0.0 if use_fund else ds.borrow_pa, space=SPACE,
                   seconds=time.time() - t0), open(out / "design.json", "w"), indent=1,
              default=str)
    print(f"[{ds_name}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:]
    with Pool(min(3, len(names))) as p:
        list(p.imap_unordered(run, names))
