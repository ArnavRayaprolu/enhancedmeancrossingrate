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


def _trades(pos, W, unit, idx, names, fold) -> list[dict]:
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
        complete_case: bool = False, verbose: bool = True, detail_extra=None) -> dict:
    """Run the walk-forward for one dataset.

    Returns bar-level results for the `C.detailed()` configurations, net
    returns for every configuration in `configs` (default: the full grid),
    trade ledgers, the selected pairs, and pair-level out-of-sample records
    for every candidate pair (for the information-coefficient analysis).
    """
    ds = C.DATASETS[ds_name]
    close, dvol, meta = load(ds)
    configs = configs if configs is not None else C.grid()
    detail_keys = {c.key for c in C.detailed() + list(detail_extra or [])}
    ic_configs = [C.PRIMARY]
    folds = make_folds(len(close))
    kappa = ds.cost_bps / 1e4
    borrow_bar = ds.borrow_pa / ds.bars_per_year
    fund = None
    if ds.source.startswith("binance_perp"):
        from .explore import load_funding
        try:
            fund = load_funding(close.index)
        except FileNotFoundError:
            fund = None

    variants = sorted({c.variant for c in configs + ic_configs})
    thr = defaultdict(set)
    for c in configs + ic_configs:
        thr[c.variant].add((c.entry, c.exit))

    detail = defaultdict(list); grid = defaultdict(list); trades = []; picks = []
    ic_rows = []; fold_info = []
    t0 = time.time()
    for f in folds:
        names = eligible(close, dvol, meta, ds, f, complete_case)
        if len(names) < 4:
            continue
        blk = close.iloc[f["start"]:f["end"]][names].ffill()
        px = blk.to_numpy(); idx = blk.index
        fblk = None if fund is None else \
            fund.iloc[f["start"]:f["end"]].reindex(columns=names).fillna(0.0).to_numpy()
        te = slice(f["train_end"] - f["start"], f["end"] - f["start"])
        test_idx = idx[te]
        ia, ib = candidate_pairs(names, meta, ds.pairs)
        stats = pair_statistics(px, ia, ib, te.start, variants,
                                {v: sorted(t) for v, t in thr.items()})
        rets = np.vstack([np.zeros((1, px.shape[1])), px[1:] / px[:-1] - 1.0])
        fold_info.append(dict(fold=f["fold"], train_start=idx[0], test_start=test_idx[0],
                              test_end=test_idx[-1], n_names=len(names), n_pairs=len(ia)))
        anames = np.array(names)

        # ---------------- portfolios ----------------
        sel_cache = {}
        for c in configs:
            vs = stats["variants"][c.variant]
            sk = (c.variant, c.entry, c.exit)
            if sk not in sel_cache:
                sc = criterion_scores(stats, c.variant, c.entry, c.exit)
                sel = {m: (sc[m], select(sc[m], ia, ib, MAXN)) for m in C.CRITERIA}
                # positions depend only on (signal, thresholds): one pass for the
                # union of every criterion's candidates
                union = np.unique(np.concatenate([v[1] for v in sel.values()]))
                pos_u = positions_for_window(vs["z"][:, union], te.start, te.stop,
                                             c.entry, c.exit)
                col = {j: i for i, j in enumerate(union)}
                sel_cache[sk] = (sel, pos_u, col)
            sel, pos_u, col = sel_cache[sk]
            for m in C.CRITERIA:
                score, chosen_all = sel[m]
                chosen = chosen_all[:c.n_pairs]
                if len(chosen) == 0:
                    continue
                pos = pos_u[:, [col[j] for j in chosen]]
                if c.lag:
                    # execution delay: decisions at the close of bar t are executed at
                    # the close of bar t + lag
                    pos = np.vstack([np.zeros((c.lag, pos.shape[1])), pos[:-c.lag]])
                a, b = leg_weights(c.signal, stats["beta"][chosen], len(chosen))
                ra, rb = rets[te][:, ia[chosen]], rets[te][:, ib[chosen]]
                fa = fb = None
                if fblk is not None:
                    fa, fb = fblk[te][:, ia[chosen]], fblk[te][:, ib[chosen]]
                res = portfolio(pos, ra, rb, a, b, score[chosen], c.weighting, borrow_bar,
                                fa, fb)
                key = (c.key, m)
                if c.key in detail_keys:
                    expo = res["exposure"]
                    detail[key].append(pd.DataFrame(dict(
                        gross=res["gross"], turn=res["turn"], borrow=res["borrow"],
                        exposure=expo, n_active=(pos != 0).sum(1),
                        hhi=np.where(expo > 0, (res["W"] ** 2).sum(1)
                                     / np.maximum(expo, 1e-12) ** 2, np.nan),
                        max_w=np.abs(res["W"]).max(1), fold=f["fold"]), index=test_idx))
                    pn = [f"{names[ia[j]]}/{names[ib[j]]}" for j in chosen]
                    unit = a[None, :] * ra + b[None, :] * rb
                    for tr_ in _trades(pos, res["W"], unit, test_idx, pn, f["fold"]):
                        tr_.update(config=c.key, criterion=m); trades.append(tr_)
                    picks.append(dict(fold=f["fold"], config=c.key, criterion=m,
                                      pairs="; ".join(pn)))
                net = res["gross"] - kappa * res["turn"] - res["borrow"]
                grid[key].append(pd.Series(net.astype("float32"), index=test_idx))

        # ---------------- pair-level records (every candidate pair) -----------
        if ic:
            for c in ic_configs:
                vs = stats["variants"][c.variant]
                z = vs["z"]
                pos = positions_for_window(z, te.start, te.stop, c.entry, c.exit)
                a, b = leg_weights(c.signal, stats["beta"], len(ia))
                unit = a[None, :] * rets[te][:, ia] + b[None, :] * rets[te][:, ib]
                P = len(ia)
                dpos = np.abs(np.diff(np.vstack([np.zeros((1, P)), pos, np.zeros((1, P))]),
                                      axis=0)).sum(0)
                short = np.where(pos > 0, np.abs(b)[None, :], np.abs(a)[None, :])
                borrow = (np.abs(pos) * short).sum(0) * borrow_bar
                gross = (pos * unit).sum(0)
                _, n_closed = run_states(z[te.start:te.stop - 1], c.entry, c.exit)
                zr = z[:te.start]
                zc = zr - np.nanmean(zr, axis=0)
                ar1 = np.nansum(zc[1:] * zc[:-1], axis=0) / np.nansum(zc * zc, axis=0)
                ic_rows.append(pd.DataFrame(dict(
                    fold=f["fold"], signal=c.signal, a=anames[ia], b=anames[ib],
                    mcr=vs["mcr"], ccr=vs["ccr"][(c.entry, c.exit)], halflife=vs["halflife"],
                    ar1=ar1, coint_p=stats["coint_p"], distance=stats["distance"],
                    corr=stats["corr"], oos_gross=gross,
                    oos_net=gross - kappa * dpos - borrow,
                    oos_trades=n_closed + (pos[-1] != 0), oos_in_market=(pos != 0).mean(0),
                    oos_mcr=signal_crossing_rate(z[te]))))
        if verbose:
            print(f"  [{ds_name}] fold {f['fold']:>3}  names={len(names):>4} "
                  f"pairs={len(ia):>6}  {time.time() - t0:6.0f}s", flush=True)

    return dict(dataset=ds_name, folds=pd.DataFrame(fold_info),
                detail={k: pd.concat(v) for k, v in detail.items()},
                grid={k: pd.concat(v) for k, v in grid.items()},
                trades=pd.DataFrame(trades), picks=pd.DataFrame(picks),
                ic=pd.concat(ic_rows, ignore_index=True) if ic_rows else None)
