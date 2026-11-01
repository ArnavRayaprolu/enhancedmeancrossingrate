"""Combine results/<dataset>/*.csv into the paper's tables (LaTeX) and figures.

    python scripts/summarise.py            # every dataset with results
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emcr import config as C, figures as F, inference as I  # noqa: E402
from emcr.report import RESULTS, net_of  # noqa: E402

PAPER = RESULTS.parents[0] / "paper"
TAB = PAPER / "tables"; FIG = PAPER / "figures"
ORDER = [k for k in C.DATASETS if C.DATASETS[k].role != "test"]


def _read(ds, name):
    p = RESULTS / ds / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _fmt(x, d=2, pct=False, sign=False):
    if x is None or not np.isfinite(x):
        return "--"
    if pct:
        x = 100 * x
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def _tex(df: pd.DataFrame, path: Path, colfmt: str):
    lines = [r"\begin{tabular}{" + colfmt + "}", r"\toprule",
             " & ".join(df.columns) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(v) for v in r.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(datasets=None):
    ds_list = [d for d in (datasets or ORDER) if (RESULTS / d / "performance.csv").exists()]
    if not ds_list:
        print("no results"); return
    lab = lambda d: C.DATASETS[d].label.replace("&", r"\&")
    P = C.PRIMARY.key

    # ---- Table: data overview -------------------------------------------------
    rows = []
    for d in ds_list:
        fo = _read(d, "folds"); ds = C.DATASETS[d]
        rows.append({"Dataset": lab(d),
                     "OOS period": f"{str(fo.test_start.iloc[0])[:7]} -- {str(fo.test_end.iloc[-1])[:7]}",
                     "Folds": len(fo), "OOS bars": f"{len(fo) * C.TEST_BARS:,}",
                     "Coins": f"{fo.n_names.mean():.0f}", "Pairs": f"{fo.n_pairs.mean():,.0f}",
                     "Cost (bps)": f"{ds.cost_bps:g}"})
    _tex(pd.DataFrame(rows), TAB / "data.tex", "lrrrrrr")

    # ---- Table/figure: pair-level IC -----------------------------------------
    ics = []
    for d in ds_list:
        ic = _read(d, "ic")
        if ic is not None:
            ics.append(ic.assign(dataset=d))
    if ics:
        ic = pd.concat(ics)
        feats = ["mcr", "ccr", "halflife", "coint_p", "distance", "corr"]
        fl = {"mcr": "MCR", "ccr": "CCR", "halflife": "Half-life", "coint_p": "Coint.",
              "distance": "Distance", "corr": "Correl."}
        sub = ic[(ic.outcome == "oos_net") & (ic.signal == "LEVEL")]
        rows = []
        for d in ds_list:
            r = {"Dataset": lab(d)}
            for f in feats:
                g = sub[(sub.dataset == d) & (sub.feature == f)]
                if len(g):
                    star = "^{*}" if g.p.iloc[0] < 0.05 else ""
                    r[fl[f]] = f"${100 * g['mean'].iloc[0]:+.1f}{star}$ ({g.t.iloc[0]:.1f})"
            rows.append(r)
        _tex(pd.DataFrame(rows), TAB / "ic.tex", "l" + "r" * len(feats))
        # MCR predicts OOS trades and OOS MCR (persistence)
        rows = []
        for d in ds_list:
            r = {"Dataset": lab(d)}
            for o, name in (("oos_trades", "OOS trades"), ("oos_mcr", "OOS MCR")):
                for f in ("mcr", "ccr", "halflife"):
                    g = ic[(ic.dataset == d) & (ic.outcome == o) & (ic.signal == "LEVEL")
                           & (ic.feature == f)]
                    if len(g):
                        r[f"{fl[f]} $\\to$ {name}"] = f"{g['mean'].iloc[0]:+.2f}"
            rows.append(r)
        _tex(pd.DataFrame(rows), TAB / "ic_persistence.tex", "l" + "r" * 6)
        F.ic_dots(ic, FIG / "fig_ic.pdf")

    # ---- Figure + table: primary performance across datasets -----------------
    sh = {}
    for d in ds_list:
        pf = _read(d, "performance")
        sh[d] = pf[pf.config == P].set_index("criterion")["sharpe"]
    shp = pd.DataFrame(sh).T
    F.sharpe_heatmap(shp, FIG / "fig_sharpe_heatmap.pdf")
    shp.to_csv(RESULTS / "summary_sharpe_primary.csv")

    main_ds = ds_list[0]
    for d in ds_list:
        pf = _read(d, "performance"); pf = pf[pf.config == P].set_index("criterion")
        ci = _read(d, "ci").set_index("criterion")
        rows = []
        for m in C.CRITERIA:
            r = pf.loc[m]
            rows.append({"Criterion": m, "Ret.\\%": _fmt(r.total_return, 1, True, True),
                         "CAGR\\%": _fmt(r.cagr, 1, True, True), "Vol\\%": _fmt(r.ann_vol, 1, True),
                         "Sharpe": _fmt(r.sharpe, 2, sign=True),
                         "95\\% CI": f"[{_fmt(ci.loc[m, 'sharpe_lo'], 2)}, {_fmt(ci.loc[m, 'sharpe_hi'], 2)}]",
                         "Sortino": _fmt(r.sortino, 2, sign=True),
                         "MDD\\%": _fmt(r.max_drawdown, 1, True),
                         "BE (bps)": _fmt(r.breakeven_arith_bps, 1),
                         "BE$_c$ (bps)": _fmt(r.breakeven_comp_bps, 1)})
        _tex(pd.DataFrame(rows), TAB / f"perf_{d}.tex", "lrrrrrrrrr")

    # ---- Table: primary DM tests ---------------------------------------------
    rows = []
    allp = []
    for d in ds_list:
        dm = _read(d, "dm")
        g = dm[(dm.config == P) & (dm.A == "MCR") & dm.B.isin(C.CLASSICS)].copy()
        g["holm3"] = I.holm(g["p_nw"].to_numpy())
        for _, r in g.iterrows():
            rows.append(dict(ds=d, B=r.B, dm=r.dm_nw, p=r.p_nw, p_and=r.p_andrews,
                             p_boot=r.p_boot, holm3=r.holm3, diff=r.mean_diff_pa))
    dmt = pd.DataFrame(rows)
    dmt["holm_all"] = I.holm(dmt["p"].to_numpy())
    dmt.to_csv(RESULTS / "summary_dm_primary.csv", index=False)
    out = dmt.assign(Dataset=dmt.ds.map(lab), Benchmark=dmt.B,
                     **{"$\\Delta$ p.a.\\%": dmt["diff"].map(lambda x: _fmt(x, 1, True, True)),
                        "DM": dmt.dm.map(lambda x: _fmt(x, 2, sign=True)),
                        "$p$": dmt.p.map(lambda x: _fmt(x, 3)),
                        "$p_{\\mathrm{And}}$": dmt.p_and.map(lambda x: _fmt(x, 3)),
                        "$p_{\\mathrm{boot}}$": dmt.p_boot.map(lambda x: _fmt(x, 3)),
                        "Holm$_{3}$": dmt.holm3.map(lambda x: _fmt(x, 3)),
                        "Holm$_{\\mathrm{all}}$": dmt.holm_all.map(lambda x: _fmt(x, 3))})
    out = out.drop(columns=["ds", "B", "dm", "p", "p_and", "p_boot", "holm3", "diff", "holm_all"])
    _tex(out, TAB / "dm.tex", "llrrrrrrr")

    # ---- Table: data snooping (SPA / RC / DSR) -------------------------------
    rows = []
    for d in ds_list:
        spa = _read(d, "spa"); dsr = _read(d, "dsr")
        s_all = spa[spa.scope == "all"].iloc[0]
        s_lvl = spa[spa.scope == "LEVEL"].iloc[0] if (spa.scope == "LEVEL").any() else s_all
        best = dsr[dsr.label.str.startswith("best")].iloc[0]
        prim = dsr[dsr.label == "primary MCR"].iloc[0]
        rows.append({"Dataset": lab(d), "K": int(s_all.K),
                     "SPA $p$": _fmt(s_all.spa_p, 3), "RC $p$": _fmt(s_all.rc_p, 3),
                     "PSR (MCR)": _fmt(prim.psr, 3), "DSR (best)": _fmt(best.dsr, 3),
                     "Best config": best.config.replace("|", " ") + " / " + best.criterion})
    _tex(pd.DataFrame(rows), TAB / "snooping.tex", "lrrrrrl")

    # ---- Table: robustness grid ----------------------------------------------
    rows = []; gd = {}
    for d in ds_list:
        gp = _read(d, "grid_pairwise"); gr = _read(d, "grid")
        r = {"Dataset": lab(d)}
        for b in C.CLASSICS:
            x = gp[(gp.A == "MCR") & (gp.B == b)].iloc[0]
            r[f"vs {b}"] = f"{100 * x.share_A_better:.0f}\\% ({_fmt(x.median_sharpe_diff, 2, sign=True)})"
        lvl = gr[gr.signal == "LEVEL"]
        r["MCR $>0$"] = f"{100 * (lvl[lvl.criterion == 'MCR'].sharpe > 0).mean():.0f}\\%"
        rows.append(r)
        piv = gr.pivot(index="config", columns="criterion", values="sharpe")
        gd[d] = {b: (piv["MCR"] - piv[b]).to_numpy() for b in C.CLASSICS}
    _tex(pd.DataFrame(rows), TAB / "grid.tex", "lrrrr")
    F.grid_differences(gd, FIG / "fig_grid.pdf")

    # ---- Table: signal / weighting variants ----------------------------------
    rows = []
    for d in ds_list:
        pf = _read(d, "performance").set_index(["config", "criterion"])
        for cfg, name in ((C.PRIMARY, "Ratio, rank-weighted (primary)"),
                          (C.Config(weighting="equal"), "Ratio, equal weight"),
                          (C.HEDGED_BASE, "Cointegrating spread ($\\hat\\beta$)"),
                          (C.MAD_ORIGINAL, "MA-difference (original)")):
            r = {"Dataset": lab(d), "Specification": name}
            for m in ("MCR", "CCR", "MCR+Coint") + C.CLASSICS:
                r[m] = _fmt(pf.loc[(cfg.key, m), "sharpe"], 2, sign=True) \
                    if (cfg.key, m) in pf.index else "--"
            rows.append(r)
    _tex(pd.DataFrame(rows), TAB / "variants.tex", "ll" + "r" * 6)

    # ---- Table: concentration (primary config) -------------------------------
    rows = []
    for d in ds_list:
        cc = _read(d, "concentration"); cc = cc[cc.config == P].set_index("criterion")
        for m in ("MCR", "Cointegration"):
            x = cc.loc[m]
            rows.append({"Dataset": lab(d), "Criterion": m,
                         "In market": _fmt(x.time_in_market, 0, True) + "\\%",
                         "Exposure": _fmt(x.avg_exposure, 2),
                         "Active pairs": _fmt(x.avg_active_pairs, 2),
                         "HHI": _fmt(x.avg_hhi, 2),
                         "Single pair": _fmt(x.share_single_pair, 0, True) + "\\%",
                         "Trades": int(x.trades), "Bars held": _fmt(x.avg_bars_held, 1),
                         "Forced": int(x.forced_closes)})
    _tex(pd.DataFrame(rows), TAB / "concentration.tex", "llrrrrrrrr")

    # ---- Table: what MCR measures (feature correlations, arcsine law) --------
    rows = []
    for d in ds_list:
        fc = _read(d, "feature_corr"); ar = _read(d, "arcsine")
        if fc is None:
            continue
        fc = fc[fc.signal == "LEVEL"].set_index("x")
        a = ar.set_index("signal").loc["LEVEL"]
        rows.append({"Dataset": lab(d),
                     "MCR--CCR": _fmt(fc.loc["mcr", "ccr"], 2, sign=True),
                     "MCR--half-life": _fmt(fc.loc["mcr", "halflife"], 2, sign=True),
                     "MCR--AR(1)": _fmt(fc.loc["mcr", "ar1"], 2, sign=True),
                     "MCR--coint. $p$": _fmt(fc.loc["mcr", "coint_p"], 2, sign=True),
                     "MCR--OOS trades": _fmt(fc.loc["mcr", "oos_trades"], 2, sign=True),
                     "Arcsine $|\\Delta|$": _fmt(a.mean_abs_err, 4)})
    _tex(pd.DataFrame(rows), TAB / "features.tex", "lrrrrrr")

    # ---- Figures from bar-level results --------------------------------------
    series, cost = {}, []
    for d in ds_list:
        cg = _read(d, "cost_grid")
        cost.append(cg[cg.config == P].assign(dataset=d))
        cache = RESULTS / "cache" / f"{d}.pkl"
        if cache.exists():
            res = pickle.load(open(cache, "rb"))
            k = C.DATASETS[d].cost_bps / 1e4
            series[d] = {m: net_of(res["detail"][(P, m)], k) for m in C.CRITERIA
                         if (P, m) in res["detail"]}
            if d == main_ds:
                F.walkforward(res["folds"].iloc[::max(1, len(res["folds"]) // 40)],
                              FIG / "fig_walkforward.pdf",
                              f"{C.DATASETS[d].label}: walk-forward folds (every "
                              f"{max(1, len(res['folds']) // 40)}th shown)")
    F.cost_curves(pd.concat(cost), FIG / "fig_cost.pdf")
    if series:
        F.equity_panels(series, FIG / "fig_equity.pdf")
    print("tables ->", TAB, "\nfigures ->", FIG)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
