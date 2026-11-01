"""Turn a backtest result into the tables reported in the paper (CSV)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import inference as I

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def net_of(df: pd.DataFrame, kappa: float) -> pd.Series:
    return df["gross"] - kappa * df["turn"] - df["borrow"]


def per_period(s: pd.Series, bars_per_year: float) -> pd.Series:
    """Sum bar returns into calendar days for intraday data (SPA, DSR, CIs)."""
    if bars_per_year > 400:
        return s.groupby(s.index.floor("D")).sum()
    return s


def periods_per_year(bars_per_year: float) -> float:
    if bars_per_year > 400:                       # intraday -> calendar days
        return 365.0 if bars_per_year > 3000 else 252.0
    return bars_per_year


def analyse(res: dict, out: Path | None = None, reps: int = 2000) -> dict:
    ds = C.DATASETS[res["dataset"]]
    out = out or RESULTS / res["dataset"]
    out.mkdir(parents=True, exist_ok=True)
    k = ds.cost_bps / 1e4
    bpy = ds.bars_per_year
    ppy = periods_per_year(bpy)
    T = {}

    # ---------------- 1. performance of every detailed configuration -----------
    rows = []
    for (key, m), df in res["detail"].items():
        n = net_of(df, k)
        row = dict(config=key, criterion=m, **I.metrics(n, bpy, df["exposure"]),
                   **I.breakeven(df["gross"].to_numpy(), df["turn"].to_numpy(),
                                 df["borrow"].to_numpy()),
                   turnover_pa=float(df["turn"].mean() * bpy))
        rows.append(row)
    T["performance"] = pd.DataFrame(rows)

    # ---------------- 2. cost grid --------------------------------------------
    rows = []
    for (key, m), df in res["detail"].items():
        for c in C.COST_GRID_BPS:
            n = net_of(df, c / 1e4)
            rows.append(dict(config=key, criterion=m, cost_bps=c,
                             total_return=float((1 + n).prod() - 1),
                             sharpe=float(n.mean() / n.std() * np.sqrt(bpy))))
    T["cost_grid"] = pd.DataFrame(rows)

    # ---------------- 3. Diebold-Mariano (primary + secondary) -----------------
    rows = []
    for cfg in [C.Config(signal=s, weighting=w) for s in C.SIGNALS for w in C.GRID_WEIGHTING]:
        for a in C.PROPOSED:
            for b in C.BENCHMARKS:
                if (cfg.key, a) not in res["detail"] or (cfg.key, b) not in res["detail"]:
                    continue
                na = net_of(res["detail"][(cfg.key, a)], k)
                nb = net_of(res["detail"][(cfg.key, b)], k).reindex(na.index)
                d = (na - nb).to_numpy()
                r = dict(config=cfg.key, A=a, B=b,
                         primary=(cfg == C.PRIMARY) and (a, b) in C.PRIMARY_TESTS)
                for lag in ("nw", "nw2", "andrews"):
                    t = I.diebold_mariano(na.to_numpy(), nb.to_numpy(), lag)
                    r[f"dm_{lag}"] = t["dm"]; r[f"p_{lag}"] = t["p"]; r[f"lag_{lag}"] = t["lag"]
                r.update(I.bootstrap_mean_pvalue(d, reps=reps))
                r["mean_diff_pa"] = float(np.nanmean(d) * bpy)
                rows.append(r)
    dm = pd.DataFrame(rows)
    for key, g in dm.groupby("config"):
        dm.loc[g.index, "holm_within_config"] = I.holm(g["p_nw"].to_numpy())
    T["dm"] = dm

    # ---------------- 4. fold-level comparison --------------------------------
    rows = []
    prim = {m: net_of(res["detail"][(C.PRIMARY.key, m)], k) for m in C.CRITERIA
            if (C.PRIMARY.key, m) in res["detail"]}
    folds = res["detail"][(C.PRIMARY.key, "Cointegration")]["fold"]
    for m, n in prim.items():
        fr = n.groupby(folds.reindex(n.index)).apply(lambda x: (1 + x).prod() - 1)
        rows.append(pd.Series(fr, name=m))
    fold_ret = pd.concat(rows, axis=1)
    T["fold_returns"] = fold_ret.reset_index()
    sign = []
    for a, b in C.PRIMARY_TESTS:
        w = int((fold_ret[a] > fold_ret[b]).sum()); n_ = len(fold_ret)
        sign.append(dict(A=a, B=b, folds=n_, wins=w,
                         p_sign=float(stats.binomtest(w, n_, 0.5).pvalue)))
    T["fold_sign"] = pd.DataFrame(sign)

    # ---------------- 5. bootstrap CIs and PSR for the primary config ----------
    rows = []
    for m, n in prim.items():
        dly = per_period(n, bpy)
        rows.append(dict(criterion=m, **I.bootstrap_ci(dly.to_numpy(), ppy, reps=reps)))
    T["ci"] = pd.DataFrame(rows)

    # ---------------- 6. concentration and exposure ---------------------------
    rows = []
    tr = res["trades"]
    for (key, m), df in res["detail"].items():
        act = df["exposure"] > 0
        t = tr[(tr.config == key) & (tr.criterion == m)] if len(tr) else tr
        rows.append(dict(config=key, criterion=m,
                         time_in_market=float(act.mean()),
                         avg_exposure=float(df["exposure"].mean()),
                         avg_active_pairs=float(df.loc[act, "n_active"].mean()),
                         avg_hhi=float(df.loc[act, "hhi"].mean()),
                         share_single_pair=float((df.loc[act, "max_w"] > 0.999).mean()),
                         max_pair_weight=float(df["max_w"].max()),
                         trades=int(len(t)),
                         avg_bars_held=float(t["bars"].mean()) if len(t) else np.nan,
                         median_entry_weight=float(t["w_entry"].median()) if len(t) else np.nan,
                         forced_closes=int(t["forced_close"].sum()) if len(t) else 0,
                         long_trades=int((t["side"] > 0).sum()) if len(t) else 0))
    T["concentration"] = pd.DataFrame(rows)

    # ---------------- 7. full sensitivity grid --------------------------------
    rows = []
    daily = {}
    for (key, m), n in res["grid"].items():
        n = n.sort_index()
        d = per_period(n, bpy)
        daily[(key, m)] = d
        sd = n.std()
        rows.append(dict(config=key, criterion=m,
                         sharpe=float(n.mean() / sd * np.sqrt(bpy)) if sd > 0 else np.nan,
                         total_return=float((1 + n).prod() - 1),
                         sr_period=float(d.mean() / d.std()) if d.std() > 0 else 0.0))
    grid = pd.DataFrame(rows)
    parts = grid["config"].str.split("|", expand=True)
    grid["signal"], grid["ma"], grid["entry"] = parts[0], parts[1], parts[2]
    grid["exit"], grid["n"], grid["weighting"] = parts[3], parts[4], parts[5]
    T["grid"] = grid

    # hybrid minus cointegration, configuration by configuration
    piv = grid.pivot(index="config", columns="criterion", values="sharpe")
    rows = []
    for a in C.PROPOSED:
        for b in C.BENCHMARKS:
            diff = (piv[a] - piv[b]).dropna()
            rows.append(dict(A=a, B=b, configs=len(diff), share_A_better=float((diff > 0).mean()),
                             median_sharpe_diff=float(diff.median()),
                             q25=float(diff.quantile(0.25)), q75=float(diff.quantile(0.75))))
    T["grid_pairwise"] = pd.DataFrame(rows)

    # ---------------- 8. Reality Check / SPA over the searched space ----------
    rows = []
    D = pd.DataFrame({(key, m): daily[(key, m)] for (key, m) in daily})
    for scope in ("all",) + C.SIGNALS:
        cols = []
        for (key, m) in daily:
            if m not in C.PROPOSED or (scope != "all" and not key.startswith(scope + "|")):
                continue
            if (key, "Cointegration") in daily:
                cols.append(((key, m), (key, "Cointegration")))
        if not cols:
            continue
        diffs = pd.DataFrame({f"{a[0]}::{a[1]}": D[a] - D[b] for a, b in cols})
        rows.append(dict(scope=scope, benchmark="Cointegration (same config)",
                         **I.spa_test(diffs, reps=reps)))
    T["spa"] = pd.DataFrame(rows)

    # ---------------- 9. Deflated Sharpe ratio --------------------------------
    srs = grid["sr_period"].to_numpy()
    rows = []
    fam = grid[grid.criterion.isin(C.PROPOSED)]
    best = fam.loc[fam["sr_period"].idxmax()]
    for label, key, m in (("best proposed config in grid", best.config, best.criterion),
                          ("primary SCR+Coint", C.PRIMARY.key, "SCR+Coint"),
                          ("primary CCR+Coint", C.PRIMARY.key, "CCR+Coint")):
        if (key, m) in daily:
            rows.append(dict(label=label, config=key, criterion=m,
                             **I.deflated_sharpe(daily[(key, m)].to_numpy(), srs)))
    T["dsr"] = pd.DataFrame(rows)

    # ---------------- 10. pair-level analysis ---------------------------------
    if res.get("ic") is not None:
        ic = res["ic"]
        T["ic"] = pd.concat([I.information_coefficients(ic, o).assign(outcome=o)
                             for o in ("oos_net", "oos_trades", "oos_scr")])
        rows = []
        for (sig, fold), g in ic.groupby(["signal", "fold"]):
            x = g[["scr", "ccr", "halflife", "ar1", "coint_p", "oos_trades", "oos_net"]]
            x = x.replace([np.inf, -np.inf], np.nan)
            rho = x.rank().corr()
            for i in rho.index:
                for j in rho.columns:
                    rows.append(dict(signal=sig, fold=fold, x=i, y=j, rho=rho.loc[i, j]))
        T["feature_corr"] = (pd.DataFrame(rows).groupby(["signal", "x", "y"])["rho"]
                             .mean().unstack("y").reset_index())
        g = ic[np.isfinite(ic["ar1"]) & np.isfinite(ic["scr"])]
        implied = np.arccos(np.clip(g["ar1"], -1, 1)) / np.pi
        T["arcsine"] = (pd.DataFrame(dict(signal=g["signal"], err=(g["scr"] - implied).abs(),
                                          scr=g["scr"], implied=implied))
                        .groupby("signal").agg(mean_abs_err=("err", "mean"),
                                               median_scr=("scr", "median"),
                                               median_implied=("implied", "median"))
                        .reset_index())

    T["folds"] = res["folds"]
    if len(res["picks"]):
        T["picks"] = res["picks"]
    for name, df in T.items():
        df.to_csv(out / f"{name}.csv", index=False)
    return T
