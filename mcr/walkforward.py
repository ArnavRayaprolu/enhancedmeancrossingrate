"""Walk-forward backtest: for every fold, build the point-in-time universe from
formation data only, score ALL candidate pairs, let each criterion select its
book from the same candidates, and trade the test window with the common engine.

Algorithm (one fold), R4 pseudocode
-----------------------------------
 1. U <- names that, on the FORMATION window only, pass the instrument filter,
    have a trade on >= min_coverage of bars, median price >= min_price and
    annualised volatility >= min_vol; keep the top_k by median bar dollar
    volume (sp500 variant: index members added on or before the formation
    start, all of them, same-sector pairs only).
 2. Prices of U are forward-filled inside the fold (complete-case variant: no
    fill in formation, names must trade on every formation bar).
 3. For every candidate pair (all C(|U|, 2) combinations) compute the
    formation statistics (selection.fold_statistics).
 4. For every criterion: percentile ranks -> scores Phi -> greedy selection of
    n_pairs with no security used more than max_uses times.
 5. For the selected pairs compute z on formation + test bars (trailing
    windows only), run the position rules from the last formation bar, delay
    execution by `lag` bars, apply the capital rule, and simulate the book
    with drift, trading costs, borrow and final liquidation (engine.run_book).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import engine as E
from . import selection as SEL
from . import signals as S


def fold_universe(data: dict, rows_tr: np.ndarray, cfg: C.Config, train_start) -> list[str]:
    close = data["close"].iloc[rows_tr]
    vol = data["volume"].iloc[rows_tr]
    cov = close.notna().mean()
    ok = data["instrument_ok"] & (cov >= cfg.min_coverage - 1e-12)
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
    """One-way cost per dollar traded for each name (R5)."""
    k = len(names)
    if scen.flat_bps > 0:
        return np.full(k, scen.flat_bps / 1e4), None
    c = data["close"].iloc[rows_tr][names].ffill().bfill().to_numpy()
    h = data["high"].iloc[rows_tr][names].ffill().bfill().to_numpy()
    lo = data["low"].iloc[rows_tr][names].ffill().bfill().to_numpy()
    half = np.clip(D.abdi_ranaldo(c, h, lo) / 2, 0.5e-4, 50e-4)
    cost = scen.spread_mult * half + scen.fixed_bps / 1e4
    imp = None
    if scen.impact_coef > 0:
        v = data["volume"].iloc[rows_tr][names].to_numpy()
        dv = np.nanmedian(c * v, axis=0)
        sig = np.nanstd(np.diff(np.log(c), axis=0), axis=0)
        imp = (sig, np.maximum(dv, 1e4))
    return cost, imp


def run(cfg: C.Config, data: dict, criteria=None, diagnostics: bool = False,
        verbose: bool = True) -> dict:
    criteria = criteria or C.CRITERIA
    idx = data["close"].index
    folds = D.make_folds(idx, cfg.train_days, cfg.test_days)
    bars, picks, diag, foldinfo, ics = [], [], [], [], []
    t0 = time.time()
    for f in folds:
        rows = np.r_[f["tr_rows"], f["te_rows"]]
        ntr, K = len(f["tr_rows"]), len(f["te_rows"])
        names = fold_universe(data, f["tr_rows"], cfg, f["train_start"])
        raw = data["close"].iloc[rows][names]
        if cfg.ffill:
            px_df = raw.ffill()
            px_df.iloc[:ntr] = px_df.iloc[:ntr].bfill()
        else:
            px_df = raw.ffill()                    # only test-window gaps can remain
        px = px_df.to_numpy()
        ia, ib = SEL.pair_index(names, data["sector"], cfg.sector_pairs)
        test_ret = px[ntr:] / px[ntr - 1:-1] - 1
        costs = {k: cost_vectors(data, f["tr_rows"], names, v) for k, v in C.COSTS.items()}
        stats = SEL.fold_statistics(px, ntr, ia, ib, cfg, diagnostics=diagnostics,
                                    test_ret=test_ret if diagnostics else None,
                                    cost=costs[C.HEADLINE_COST][0])
        valid = np.isfinite(stats["mcr"]) & np.isfinite(stats["coint_p"])
        stats = stats[valid].reset_index(drop=True)
        sc = SEL.scores(stats)
        foldinfo.append(dict(fold=f["fold"], train_start=f["train_start"], train_end=f["train_end"],
                             test_start=f["test_start"], test_end=f["test_end"],
                             n_names=len(names), n_pairs=len(stats), test_bars=K))
        if diagnostics:
            diag.append(_diagnostics(stats, f["fold"]))
        ts = idx[f["te_rows"]]
        for m in criteria:
            ch = SEL.select(sc[m], stats["a"].to_numpy(), stats["b"].to_numpy(),
                            cfg.n_pairs, cfg.max_uses)
            a = stats["a"].to_numpy()[ch]; b = stats["b"].to_numpy()[ch]
            phi = sc[m][ch]
            beta = stats["beta"].to_numpy()[ch]
            pa, pb = px[:, a], px[:, b]
            if cfg.signal == "hedged":
                z = S.ma_signal(S.hedged_series(pa, pb, beta), cfg.fast, cfg.slow)
                bpos = np.clip(beta, 0.2, 5.0)
                hedge_a = 1 / (1 + bpos)
            else:
                z = S.ma_signal(S.ratio_series(pa, pb), cfg.fast, cfg.slow)
                hedge_a = np.full(len(ch), 0.5)
            dec = E.states(z, cfg.entry, cfg.exit, cfg.engine, start=ntr - 1)[:K]
            pos = np.zeros_like(dec)
            pos[cfg.lag:] = dec[:K - cfg.lag] if cfg.lag else dec
            w = E.pair_weights(pos, phi, cfg.weighting, cfg.cap)
            ra, rb = test_ret[:, a], test_ret[:, b]
            rec = dict(time=ts, fold=f["fold"], criterion=m)
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
            bars.append(pd.DataFrame(rec))
            # per-pair record for concentration / composition (m4, m7)
            ppnl = (pos * w * (hedge_a * ra - (1 - hedge_a) * rb)).sum(0)
            prev = np.vstack([np.zeros((1, len(ch))), pos[:-1]])
            ntrades = ((pos != 0) & (pos != prev)).sum(0)
            for j in range(len(ch)):
                ta, tb = names[a[j]], names[b[j]]
                picks.append(dict(fold=f["fold"], criterion=m, a=ta, b=tb,
                                  sector_a=data["sector"].get(ta, "n/a"),
                                  sector_b=data["sector"].get(tb, "n/a"),
                                  phi=phi[j], mcr=stats["mcr"].iat[ch[j]],
                                  coint_p=stats["coint_p"].iat[ch[j]],
                                  halflife=stats["halflife"].iat[ch[j]],
                                  distance=stats["distance"].iat[ch[j]],
                                  corr=stats["corr"].iat[ch[j]],
                                  avg_weight=float(w[:, j].mean()),
                                  time_in_trade=float((pos[:, j] != 0).mean()),
                                  trades=int(ntrades[j]), gross_pnl=float(ppnl[j])))
        if verbose:
            print(f"[{cfg.label}] fold {f['fold']}: {f['test_start']}..{f['test_end']} "
                  f"names={len(names)} pairs={len(stats)} [{time.time() - t0:.0f}s]", flush=True)
    out = dict(config=cfg, bars=pd.concat(bars, ignore_index=True), picks=pd.DataFrame(picks),
               folds=pd.DataFrame(foldinfo))
    if diagnostics:
        out["diag"] = pd.concat([d[0] for d in diag], ignore_index=True)
        out["diag_sample"] = pd.concat([d[1] for d in diag], ignore_index=True)
    return out


FEATURES = {  # name: orientation (+1 = a higher value is expected to be better)
    "mcr": 1, "ar1_z": -1, "halflife": -1, "xr_coint": 1, "xr_ratio": 1, "cycles": 1,
    "coint_p": -1, "distance": -1, "corr": 1,
}
OUTCOMES = ["oos_net", "oos_trades", "oos_mcr"]


def _diagnostics(st: pd.DataFrame, fold: int):
    """R1: fold-level Spearman correlations (i) between MCR and the alternative
    mean-reversion measures and (ii) between every formation statistic and the
    pair's own out-of-sample outcomes, over all candidate pairs."""
    rows = []
    for f1 in FEATURES:
        x = st[f1].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        for f2 in list(FEATURES) + OUTCOMES:
            if f2 == f1:
                continue
            y = st[f2].replace([np.inf, -np.inf], np.nan).to_numpy(float)
            rows.append(dict(fold=fold, x=f1, y=f2, rho=S.spearman(x, y)))
    sample = st.sample(min(5000, len(st)), random_state=fold).assign(fold=fold)
    return pd.DataFrame(rows), sample
