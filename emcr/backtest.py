"""Walk-forward backtest of every criterion under every configuration."""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config as C
from .criteria import criterion_scores, pair_statistics, select
from .data import candidate_pairs, eligible, load, make_folds
from .engine import portfolio, positions_for_window, run_states
from .signals import leg_weights, signal_crossing_rate

MAXN = max(C.GRID_N)


def detailed_configs() -> list[C.Config]:
    """Base configuration of every signal under both capital rules."""
    return [C.Config(signal=s, weighting=w) for s in C.SIGNALS for w in C.GRID_WEIGHTING]


def _period_key(index: pd.DatetimeIndex, bars_per_year: float) -> np.ndarray:
    """Calendar-day buckets for intraday data (used for SPA / DSR / bootstrap)."""
    if bars_per_year > 400:
        return index.floor("D").to_numpy()
    return index.to_numpy()


def _trades(pos: np.ndarray, W: np.ndarray, unit: np.ndarray, idx, names, fold) -> list[dict]:
    out = []
    T, N = pos.shape
    for j in range(N):
        p = pos[:, j]; t = 0
        while t < T:
            if p[t] == 0:
                t += 1; continue
            s = t
            while t < T and p[t] == p[s]:
                t += 1
            seg = slice(s, t)
            out.append(dict(fold=fold, pair=names[j], side=int(p[s]), entry=idx[s],
                            exit=idx[t - 1], bars=t - s, w_entry=float(abs(W[s, j])),
                            w_max=float(np.abs(W[seg, j]).max()),
                            pnl_unit=float((p[seg] * unit[seg, j]).sum()),
                            forced_close=bool(t == T)))
    return out


def run(ds_name: str, configs: list[C.Config] | None = None, ic: bool = True,
        complete_case: bool = False, verbose: bool = True) -> dict:
    ds = C.DATASETS[ds_name]
    close, dvol, meta = load(ds)
    configs = configs if configs is not None else C.grid()
    detail_keys = {c.key for c in detailed_configs()}
    folds = make_folds(len(close))
    bpy = ds.bars_per_year
    kappa = ds.cost_bps / 1e4
    borrow_bar = ds.borrow_pa / bpy

    variants = sorted({(c.signal, c.fast if c.signal == "MAD" else 0, c.slow) for c in configs})
    thr = defaultdict(set)
    for c in configs:
        thr[(c.signal, c.fast if c.signal == "MAD" else 0, c.slow)].add((c.entry, c.exit))

    detail = defaultdict(list); grid = defaultdict(list); trades = []; picks = []
    ic_rows = []; fold_info = []
    t0 = time.time()
    for f in folds:
        names = eligible(close, dvol, meta, ds, f, complete_case)
        if len(names) < 4:
            continue
        blk = close.iloc[f["start"]:f["end"]][names].ffill()
        px = blk.to_numpy()
        idx = blk.index
        te = slice(f["train_end"] - f["start"], f["end"] - f["start"])
        test_idx = idx[te]
        ia, ib = candidate_pairs(names, meta, ds.pairs)
        stats = pair_statistics(px, ia, ib, te.start,
                                [(s, max(fa, 1), sl) for (s, fa, sl) in variants],
                                {(s, max(fa, 1), sl): sorted(v) for (s, fa, sl), v in thr.items()})
        rets = np.vstack([np.zeros((1, px.shape[1])), px[1:] / px[:-1] - 1.0])
        fold_info.append(dict(fold=f["fold"], train_start=idx[0], test_start=test_idx[0],
                              test_end=test_idx[-1], n_names=len(names), n_pairs=len(ia)))

        # ---------------- portfolios ----------------
        sel_cache = {}
        for c in configs:
            v = (c.signal, max(c.fast, 1) if c.signal == "MAD" else 1, c.slow)
            vs = stats["variants"][v]
            sk = (v, c.entry, c.exit)
            if sk not in sel_cache:
                sc = criterion_scores(stats, v, c.entry, c.exit)
                sel_cache[sk] = {m: (sc[m], select(sc[m], ia, ib, MAXN)) for m in C.CRITERIA}
            for m in C.CRITERIA:
                score, chosen_all = sel_cache[sk][m]
                chosen = chosen_all[:c.n_pairs]
                if len(chosen) == 0:
                    continue
                z = vs["z"][:, chosen]
                pos = positions_for_window(z, te.start, te.stop, c.entry, c.exit)
                beta = stats["beta"][chosen]
                a, b = leg_weights(c.signal, beta, len(chosen))
                ra, rb = rets[te][:, ia[chosen]], rets[te][:, ib[chosen]]
                res = portfolio(pos, ra, rb, a, b, score[chosen], c.weighting, borrow_bar)
                key = (c.key, m)
                if c.key in detail_keys:
                    df = pd.DataFrame(dict(gross=res["gross"], turn=res["turn"],
                                           borrow=res["borrow"], exposure=res["exposure"],
                                           n_active=(pos != 0).sum(1),
                                           hhi=np.where(res["exposure"] > 0,
                                                        (res["W"] ** 2).sum(1)
                                                        / np.maximum(res["exposure"], 1e-12) ** 2,
                                                        np.nan),
                                           max_w=np.abs(res["W"]).max(1),
                                           fold=f["fold"]), index=test_idx)
                    detail[key].append(df)
                    pn = [f"{names[ia[j]]}/{names[ib[j]]}" for j in chosen]
                    unit = a[None, :] * ra + b[None, :] * rb
                    for tr_ in _trades(pos, res["W"], unit, test_idx, pn, f["fold"]):
                        tr_.update(config=c.key, criterion=m); trades.append(tr_)
                    picks.append(dict(fold=f["fold"], config=c.key, criterion=m,
                                      pairs="; ".join(pn)))
                net = res["gross"] - kappa * res["turn"] - res["borrow"]
                grid[key].append(pd.Series(net, index=test_idx))

        # ---------------- pair-level (IC) records ----------------
        if ic:
            for sig in C.SIGNALS:
                c = C.Config(signal=sig)
                v = (sig, c.fast if sig == "MAD" else 1, c.slow)
                vs = stats["variants"][v]
                z = vs["z"]
                pos = positions_for_window(z, te.start, te.stop, c.entry, c.exit)
                a, b = leg_weights(sig, stats["beta"], len(ia))
                unit = a[None, :] * rets[te][:, ia] + b[None, :] * rets[te][:, ib]
                dpos = np.abs(np.diff(np.vstack([np.zeros((1, len(ia))), pos,
                                                 np.zeros((1, len(ia)))]), axis=0)).sum(0)
                short_share = np.where(pos > 0, np.abs(b)[None, :], np.abs(a)[None, :])
                borrow = (np.abs(pos) * short_share).sum(0) * borrow_bar
                gross = (pos * unit).sum(0)
                _, n_closed = run_states(z[te.start:te.stop - 1], c.entry, c.exit)
                zr = z[:te.start]
                zc = zr - np.nanmean(zr, axis=0)
                ar1 = np.nansum(zc[1:] * zc[:-1], axis=0) / np.nansum(zc * zc, axis=0)
                ic_rows.append(pd.DataFrame(dict(
                    fold=f["fold"], signal=sig, a=np.array(names)[ia], b=np.array(names)[ib],
                    scr=vs["scr"], ccr=vs["ccr"][(c.entry, c.exit)], halflife=vs["halflife"],
                    ar1=ar1, coint_p=stats["coint_p"], distance=stats["distance"],
                    corr=stats["corr"],
                    oos_gross=gross, oos_net=gross - kappa * dpos - borrow,
                    oos_trades=n_closed + (pos[-1] != 0), oos_in_market=(pos != 0).mean(0),
                    oos_scr=signal_crossing_rate(z[te]))))
        if verbose:
            print(f"  [{ds_name}] fold {f['fold']:>3}  names={len(names):>4} "
                  f"pairs={len(ia):>6}  {time.time() - t0:6.0f}s", flush=True)

    out = dict(dataset=ds_name, folds=pd.DataFrame(fold_info),
               detail={k: pd.concat(v) for k, v in detail.items()},
               grid={k: pd.concat(v) for k, v in grid.items()},
               trades=pd.DataFrame(trades), picks=pd.DataFrame(picks),
               ic=pd.concat(ic_rows, ignore_index=True) if ic_rows else None)
    return out
