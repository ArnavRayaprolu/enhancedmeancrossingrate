"""Walk-forward backtest.

Algorithm for one fold (R4 pseudocode)
--------------------------------------
 1. U <- names that, on the FORMATION window only, pass the instrument filter,
    trade on >= min_coverage of bars, have median price >= min_price and
    annualised volatility >= min_vol; keep the top_k by median bar dollar
    volume. (sp500 variant: current index members added on or before the
    formation start, all of them, same-sector pairs only.)
 2. Prices of U are forward-filled inside the fold (complete-case variant:
    names must trade on every formation bar, so nothing is filled there).
 3. Every candidate pair (all C(|U|, 2) combinations) gets its formation
    statistics (selection.fold_statistics); pairs with a missing MCR or
    cointegration statistic are dropped, so every criterion ranks the same
    candidate set (m4).
 4. For every criterion: percentile ranks -> score Phi -> greedy selection of
    n_pairs with no security used more than max_uses times.
 5. For the selected pairs, z is computed on formation + test bars with
    trailing windows; the position rules start flat at the last formation bar,
    execution is delayed by `lag` bars, the capital rule sets pair weights and
    the book is simulated with drift, trading costs, borrow and liquidation at
    the last test bar (engine.run_book).
Configurations that share the statistics inputs (config.stat_key) share
steps 1-3. Folds are independent and run in parallel processes.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import engine as E
from . import selection as SEL
from . import signals as S

_DATA: dict = {}


def fold_universe(data: dict, rows_tr: np.ndarray, cfg: C.Config, train_start) -> list[str]:
    close = data["close"].iloc[rows_tr]
    vol = data["volume"].iloc[rows_tr]
    cov = close.notna().mean()
    ok = data["instrument_ok"].reindex(close.columns).fillna(False).astype(bool)
    ok &= cov >= cfg.min_coverage - 1e-12
    ok &= close.median() >= cfg.min_price
    lr = np.log(close.ffill()).diff()
    ok &= (lr.std() * np.sqrt(C.BARS_PER_YEAR)) >= cfg.min_vol
    if cfg.universe == "sp500":
        added = data["sp500_added"]
        start = pd.Timestamp(train_start)
        ok &= pd.Series([t in added and added[t] <= start for t in close.columns],
                        index=close.columns)
        return sorted(close.columns[ok.to_numpy()])
    dv = (close * vol).median()
    dv = dv[ok.to_numpy()].sort_values(ascending=False)
    return sorted(dv.index[:cfg.top_k])


def cost_vectors(data, rows_tr, names, scen: C.Costs):
    """One-way cost per dollar traded for each name (R5), formation data only."""
    k = len(names)
    if scen.flat_bps > 0:
        return np.full(k, scen.flat_bps / 1e4), None
    c = data["close"].iloc[rows_tr][names].ffill().bfill().to_numpy(float)
    h = data["high"].iloc[rows_tr][names].ffill().bfill().to_numpy(float)
    lo = data["low"].iloc[rows_tr][names].ffill().bfill().to_numpy(float)
    half = np.clip(D.abdi_ranaldo(c, h, lo) / 2, 0.5e-4, 50e-4)
    cost = scen.spread_mult * half + scen.fixed_bps / 1e4
    imp = None
    if scen.impact_coef > 0:
        v = data["volume"].iloc[rows_tr][names].to_numpy(float)
        dv = np.nanmedian(c * v, axis=0)
        sig = np.nanstd(np.diff(np.log(c), axis=0), axis=0)
        imp = (sig, np.maximum(dv, 1e4))
    return cost, imp


def _trade(cfg, m, sc, stats, px, ntr, K, test_ret, costs, names, ts, fold, data):
    ch = SEL.select(sc[m], stats["a"].to_numpy(), stats["b"].to_numpy(), cfg.n_pairs, cfg.max_uses)
    a = stats["a"].to_numpy()[ch]; b = stats["b"].to_numpy()[ch]
    phi = sc[m][ch]
    beta = stats["beta"].to_numpy()[ch]
    pa, pb = px[:, a], px[:, b]
    if cfg.signal == "hedged":
        z = S.ma_signal(S.hedged_series(pa, pb, beta), cfg.fast, cfg.slow)
        hedge_a = 1 / (1 + np.clip(beta, 0.2, 5.0))
    else:
        z = S.ma_signal(S.ratio_series(pa, pb), cfg.fast, cfg.slow)
        hedge_a = np.full(len(ch), 0.5)
    dec = E.states(z, cfg.entry, cfg.exit, cfg.engine, start=ntr - 1)[:K]
    pos = np.zeros_like(dec)
    if cfg.lag:
        pos[cfg.lag:] = dec[:K - cfg.lag]
    else:
        pos = dec
    w = E.pair_weights(pos, phi, cfg.weighting, cfg.cap)
    ra, rb = test_ret[:, a], test_ret[:, b]
    rec = dict(time=ts, fold=fold, config=cfg.label, criterion=m)
    for cname, scen in C.COSTS.items():
        cv, imp = costs[cname]
        impact = None
        if imp is not None:
            impact = (imp[0][a], imp[0][b], imp[1][a], imp[1][b], scen.impact_coef, scen.aum)
        res = E.run_book(pos, w, ra, rb, hedge_a, cv[a], cv[b],
                         scen.borrow_pa / C.BARS_PER_YEAR, impact)
        rec[f"net_{cname}"] = res["net"]
        if cname == C.HEADLINE_COST:
            rec.update(gross=res["gross"], cost=res["cost"], borrow=res["borrow"],
                       turn=res["turn"], exposure=res["exposure"])
    bars = pd.DataFrame(rec)
    ppnl = (pos * w * (hedge_a * ra - (1 - hedge_a) * rb)).sum(0)
    prev = np.vstack([np.zeros((1, len(ch))), pos[:-1]])
    ntrades = ((pos != 0) & (pos != prev)).sum(0)
    picks = []
    for j in range(len(ch)):
        ta, tb = names[a[j]], names[b[j]]
        row = stats.iloc[ch[j]]
        picks.append(dict(fold=fold, config=cfg.label, criterion=m, a=ta, b=tb,
                          sector_a=data["sector"].get(ta, "n/a"),
                          sector_b=data["sector"].get(tb, "n/a"),
                          phi=float(phi[j]), mcr=row.mcr, coint_p=row.coint_p,
                          halflife=row.halflife, distance=row["distance"], corr=row["corr"],
                          avg_weight=float(w[:, j].mean()),
                          time_in_trade=float((pos[:, j] != 0).mean()),
                          trades=int(ntrades[j]), gross_pnl=float(ppnl[j]),
                          cost_a_bps=float(costs[C.HEADLINE_COST][0][a[j]] * 1e4),
                          cost_b_bps=float(costs[C.HEADLINE_COST][0][b[j]] * 1e4)))
    return bars, picks


def _run_fold(args):
    f, cfgs, criteria, diag = args
    data = _DATA
    base = cfgs[0]
    t0 = time.time()
    rows = np.r_[f["tr_rows"], f["te_rows"]]
    ntr, K = len(f["tr_rows"]), len(f["te_rows"])
    names = fold_universe(data, f["tr_rows"], base, f["train_start"])
    if len(names) < 4:
        return None
    raw = data["close"].iloc[rows][names]
    px_df = raw.astype("float64").ffill()
    if base.ffill:
        px_df.iloc[:ntr] = px_df.iloc[:ntr].bfill()
    px = px_df.to_numpy()
    ia, ib = SEL.pair_index(names, data["sector"], base.sector_pairs)
    stats = SEL.fold_statistics(px, ntr, ia, ib, base)
    stats = stats[np.isfinite(stats["mcr"]) & np.isfinite(stats["coint_p"])].reset_index(drop=True)
    sc = SEL.scores(stats)
    test_ret = px[ntr:] / px[ntr - 1:-1] - 1
    costs = {k: cost_vectors(data, f["tr_rows"], names, v) for k, v in C.COSTS.items()}
    ts = data["close"].index[f["te_rows"]]
    bars, picks = [], []
    for cfg in cfgs:
        for m in criteria:
            b_, p_ = _trade(cfg, m, sc, stats, px, ntr, K, test_ret, costs, names, ts, f["fold"], data)
            bars.append(b_); picks += p_
    info = dict(fold=f["fold"], train_start=f["train_start"], train_end=f["train_end"],
                test_start=f["test_start"], test_end=f["test_end"], n_names=len(names),
                n_pairs=len(stats), test_bars=K,
                inactive_names=int((data["status"].reindex(names) == "inactive").sum()),
                median_cost_bps=float(np.median(costs[C.HEADLINE_COST][0]) * 1e4),
                seconds=time.time() - t0)
    out = dict(bars=pd.concat(bars, ignore_index=True), picks=pd.DataFrame(picks), info=info)
    if diag:
        smp = stats.sample(min(C.DIAG_SAMPLE, len(stats)), random_state=int(f["fold"]))
        dg = SEL.diagnostics(px, ntr, smp.reset_index(drop=True), base,
                             costs[C.HEADLINE_COST][0])
        out["diag"] = _ic(dg, f["fold"])
        out["diag_sample"] = dg.sample(min(2000, len(dg)), random_state=0).assign(fold=f["fold"])
    return out


def run_group(cfgs: list[C.Config], data: dict, criteria=None, diagnostics=False,
              workers: int = 4, verbose=True, folds_limit: int | None = None) -> dict:
    """Run configurations that share config.stat_key (one statistics pass per fold)."""
    global _DATA
    assert len({C.stat_key(c) for c in cfgs}) == 1
    criteria = criteria or C.CRITERIA
    _DATA = data
    folds = D.make_folds(data["close"].index, cfgs[0].train_days, cfgs[0].test_days)
    if folds_limit:
        folds = folds[:folds_limit]
    jobs = [(f, cfgs, criteria, diagnostics) for f in folds]
    res = []
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, maxtasksperchild=4) as pool:
        for r in pool.imap(_run_fold, jobs):
            if r is None:
                continue
            res.append(r)
            if verbose:
                i = r["info"]
                print(f"[{cfgs[0].label}+{len(cfgs) - 1}] fold {i['fold']} {i['test_start']}: "
                      f"names={i['n_names']} pairs={i['n_pairs']} ({i['seconds']:.0f}s) "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    out = dict(bars=pd.concat([r["bars"] for r in res], ignore_index=True),
               picks=pd.concat([r["picks"] for r in res], ignore_index=True),
               folds=pd.DataFrame([r["info"] for r in res]))
    if diagnostics:
        out["diag"] = pd.concat([r["diag"] for r in res], ignore_index=True)
        out["diag_sample"] = pd.concat([r["diag_sample"] for r in res], ignore_index=True)
    return out


FEATURES = {  # name: orientation (+1 = a higher value is expected to be better)
    "mcr": 1, "ar1_z": -1, "halflife": -1, "xr_coint": 1, "xr_ratio": 1, "cycles": 1,
    "coint_p": -1, "distance": -1, "corr": 1,
}
OUTCOMES = ["oos_net", "oos_trades", "oos_mcr"]


def _ic(st: pd.DataFrame, fold: int) -> pd.DataFrame:
    """Fold-level Spearman correlations (i) between MCR and the alternative
    mean-reversion measures and (ii) between every formation statistic and
    the pair's own out-of-sample outcomes (R1)."""
    rows = []
    for f1 in FEATURES:
        x = st[f1].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        for f2 in list(FEATURES) + OUTCOMES:
            if f2 == f1:
                continue
            y = st[f2].replace([np.inf, -np.inf], np.nan).to_numpy(float)
            rows.append(dict(fold=fold, x=f1, y=f2, rho=S.spearman(x, y)))
    return pd.DataFrame(rows)
