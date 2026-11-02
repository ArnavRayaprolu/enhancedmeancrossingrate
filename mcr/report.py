"""Tables for every reviewer comment, from the walk-forward output.

    analyse(bars, picks, folds, diag) -> writes results/*.csv and REPORT.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import inference as I

RES = Path(__file__).resolve().parents[1] / "results"
BPY = C.BARS_PER_YEAR
COSTS = list(C.COSTS)


def _series(bars: pd.DataFrame, config: str, crit: str, col: str) -> pd.Series:
    x = bars[(bars.config == config) & (bars.criterion == crit)]
    return x.set_index("time")[col].sort_index()


# --------------------------------------------------------------------------- #
def performance(bars: pd.DataFrame) -> pd.DataFrame:
    """m6 conventions (inference.metrics): every out-of-sample bar is included;
    a bar with no open position contributes a zero return (and counts in the
    volatility); Sharpe = mean/sd x sqrt(1764), zero risk-free rate because the
    book is dollar neutral and self-financing; Sortino uses downside
    semideviation over all bars; max drawdown is on the compounded curve."""
    rows = []
    for (cfg, m), g in bars.groupby(["config", "criterion"]):
        g = g.sort_values("time")
        be = I.breakeven(g["gross"].to_numpy(), g["turn"].to_numpy(), g["borrow"].to_numpy())
        for c in COSTS:
            r = g.set_index("time")[f"net_{c}"]
            met = I.metrics(r, BPY, g.set_index("time")["exposure"])
            rows.append(dict(config=cfg, criterion=m, costs=c, **met,
                             turnover_pa=float(g["turn"].mean() * BPY),
                             gross_return_pa=float(g["gross"].mean() * BPY),
                             cost_pa=float(g["cost"].mean() * BPY) if c == C.HEADLINE_COST else np.nan,
                             **(be if c == C.HEADLINE_COST else {})))
    return pd.DataFrame(rows)


def dm_tests(bars: pd.DataFrame, config: str, cost: str, reps: int = 999) -> pd.DataFrame:
    """R3. d_t = n^A_t - n^B_t (net return per bar, A minus B). Loss = -n.
    H0: E[d_t] = 0, H1: E[d_t] != 0 (two-sided; DM > 0 favours A). Long-run
    variance: Bartlett kernel with floor(4 (T/100)^(2/9)) lags (the paper's rule),
    2x and 4x that, the Andrews (1991) plug-in, and one trading day (7 lags);
    also the same test on DAILY sums of d (intraday dependence removed) and a
    stationary-bootstrap p-value with Politis-White block length under the
    null (d recentred). Harvey-Leybourne-Newbold correction, t(T-1)."""
    rows = []
    for fam, tests in C.DM_FAMILIES.items():
        for a, b in tests:
            x = _series(bars, config, a, f"net_{cost}"); y = _series(bars, config, b, f"net_{cost}")
            x, y = x.align(y, join="inner")
            if len(x) < 50:
                continue
            row = dict(config=config, costs=cost, family=fam, A=a, B=b, T=len(x),
                       mean_diff_pa=float((x - y).mean() * BPY))
            for lag in ("nw", "nw2", "nw4", "andrews", 7):
                t = I.diebold_mariano(x.to_numpy(), y.to_numpy(), lag)
                row[f"dm_{lag}"] = t["dm"]; row[f"p_{lag}"] = t["p"]; row[f"lag_{lag}"] = t["lag"]
            dx = (x - y).groupby(x.index.tz_convert("America/New_York").date).sum()
            t = I.diebold_mariano(dx.to_numpy(), np.zeros(len(dx)), "nw")
            row.update(dm_daily=t["dm"], p_daily=t["p"])
            row.update(I.bootstrap_mean_pvalue((x - y).to_numpy(), reps=reps))
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["holm_family"] = df.groupby("family")["p_nw"].transform(lambda p: I.holm(p.to_numpy()))
        df["holm_all"] = I.holm(df["p_nw"].to_numpy())
    return df


def data_snooping(bars: pd.DataFrame, cost: str, reps: int = 999) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R2/R9. Every (configuration, criterion) strategy run in the study is a
    trial. SPA / Reality Check: H0 no MCR-family strategy beats the PRIMARY
    Cointegration book. Deflated Sharpe: the PRIMARY champion and the best
    trial, deflated by the number and dispersion of all trials."""
    piv = bars.pivot_table(index="time", columns=["config", "criterion"], values=f"net_{cost}")
    prim = piv[C.PRIMARY.label]
    bench = prim["Cointegration"]
    common = piv.dropna(how="any", axis=1)
    mcr_cols = [c for c in piv.columns if c[1].startswith("MCR")]
    rows = []
    for scope, cols in (("primary_config", [(C.PRIMARY.label, m) for m in C.CRITERIA
                                             if m.startswith("MCR")]),
                        ("all_configs", mcr_cols)):
        X = piv[cols].sub(bench, axis=0).dropna(how="any")
        if X.shape[0] < 100:
            continue
        r = I.spa_test(X, reps=reps)
        rows.append(dict(scope=scope, benchmark="primary Cointegration", costs=cost, **r))
    spa = pd.DataFrame(rows)
    sr_bar = piv.mean() / piv.std()
    out = []
    for label, col in (("primary champion", (C.PRIMARY.label, C.CHAMPION)),
                       ("best trial", sr_bar.idxmax())):
        r = piv[col].dropna().to_numpy()
        d = I.deflated_sharpe(r, sr_bar.dropna().to_numpy())
        out.append(dict(label=label, config=col[0], criterion=col[1], costs=cost,
                        sharpe_ann=d["sr"] * np.sqrt(BPY), sr0_ann=d["sr0"] * np.sqrt(BPY),
                        dsr=d["dsr"], psr=d["psr"], n_trials=d["N"], T=d["T"]))
    return spa, pd.DataFrame(out)


def confidence(bars: pd.DataFrame, config: str, cost: str, reps: int = 999) -> pd.DataFrame:
    rows = []
    for m in C.CRITERIA:
        r = _series(bars, config, m, f"net_{cost}")
        if len(r):
            rows.append(dict(config=config, criterion=m, costs=cost,
                             **I.bootstrap_ci(r.to_numpy(), BPY, reps=reps)))
    return pd.DataFrame(rows)


def subperiods(bars: pd.DataFrame, config: str, cost: str) -> pd.DataFrame:
    """R6: Sharpe by calendar year and by market regime."""
    x = bars[bars.config == config].copy()
    x["year"] = x["time"].dt.year
    x["regime"] = pd.cut(x["time"].dt.tz_convert(None),
                         [pd.Timestamp("2015-01-01"), pd.Timestamp("2020-01-01"),
                          pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01"),
                          pd.Timestamp("2023-01-01"), pd.Timestamp("2030-01-01")],
                         labels=["2016-19 expansion", "2020 covid", "2021 rally",
                                 "2022 bear", "2023-26"])
    def sr(s):
        return s.mean() / s.std() * np.sqrt(BPY) if s.std() > 0 else np.nan
    a = x.groupby(["criterion", "year"])[f"net_{cost}"].apply(sr).unstack()
    b = x.groupby(["criterion", "regime"], observed=True)[f"net_{cost}"].apply(sr).unstack()
    return pd.concat([a, b], axis=1)


def fold_table(bars: pd.DataFrame, config: str, cost: str) -> pd.DataFrame:
    x = bars[bars.config == config]
    return x.groupby(["fold", "criterion"])[f"net_{cost}"].apply(
        lambda s: float(np.prod(1 + s) - 1)).unstack()


def concentration(picks: pd.DataFrame, bars: pd.DataFrame, config: str) -> pd.DataFrame:
    """m7: how concentrated the books are, by pair, security, sector and trade."""
    p = picks[picks.config == config]
    rows = []
    for m, g in p.groupby("criterion"):
        per_fold = []
        for f, h in g.groupby("fold"):
            w = h["avg_weight"].to_numpy()
            sh = w / w.sum() if w.sum() > 0 else w
            sec = pd.concat([h[["sector_a", "avg_weight"]].rename(columns={"sector_a": "s"}),
                             h[["sector_b", "avg_weight"]].rename(columns={"sector_b": "s"})])
            secw = sec.groupby("s")["avg_weight"].sum()
            secw = secw / secw.sum() if secw.sum() > 0 else secw
            pnl = h["gross_pnl"].abs()
            per_fold.append(dict(max_pair_share=sh.max() if len(sh) else np.nan,
                                 hhi_pairs=(sh ** 2).sum(), effective_pairs=1 / (sh ** 2).sum()
                                 if (sh ** 2).sum() > 0 else np.nan,
                                 max_sector_share=secw.max() if len(secw) else np.nan,
                                 top_pair_pnl_share=pnl.max() / pnl.sum() if pnl.sum() > 0 else np.nan,
                                 trades=h["trades"].sum(), pairs=len(h),
                                 same_sector=(h.sector_a == h.sector_b).mean()))
        d = pd.DataFrame(per_fold).mean()
        x = bars[(bars.config == config) & (bars.criterion == m)]
        rows.append(dict(criterion=m, **d.to_dict(),
                         avg_exposure=float(x["exposure"].mean()),
                         time_invested=float((x["exposure"] > 0).mean())))
    return pd.DataFrame(rows)


def composition(picks: pd.DataFrame, config: str) -> pd.DataFrame:
    """m4: what each criterion selects from the same candidate set."""
    p = picks[picks.config == config].copy()
    p["pair"] = p["a"] + "/" + p["b"]
    champ = p[p.criterion == C.CHAMPION].groupby("fold")["pair"].apply(set)
    rows = []
    for m, g in p.groupby("criterion"):
        sets = g.groupby("fold")["pair"].apply(set)
        jac = [len(sets[f] & champ.get(f, set())) / max(1, len(sets[f] | champ.get(f, set())))
               for f in sets.index]
        rows.append(dict(criterion=m, median_mcr=g.mcr.median(), median_coint_p=g.coint_p.median(),
                         median_halflife_bars=g.halflife.replace(np.inf, np.nan).median(),
                         median_corr=g["corr"].median(), median_cost_bps=pd.concat(
                             [g.cost_a_bps, g.cost_b_bps]).median(),
                         time_in_trade=g.time_in_trade.mean(), trades_per_pair=g.trades.mean(),
                         jaccard_with_champion=float(np.mean(jac))))
    return pd.DataFrame(rows)


def diagnostics_table(diag: pd.DataFrame) -> pd.DataFrame:
    """R1: fold-average Spearman correlation with a t-statistic over folds."""
    g = diag.groupby(["x", "y"])["rho"]
    out = pd.DataFrame({"mean_rho": g.mean(), "folds": g.count(),
                        "share_positive": g.apply(lambda s: (s > 0).mean())})
    out["t"] = out["mean_rho"] / (g.std() / np.sqrt(out["folds"]))
    out["p"] = 2 * (1 - stats.t.cdf(out["t"].abs(), df=out["folds"] - 1))
    return out.reset_index()


def grid_table(perf: pd.DataFrame, cost: str) -> pd.DataFrame:
    """R8: Sharpe of every criterion under every configuration (all reported)."""
    p = perf[perf.costs == cost].pivot(index="config", columns="criterion", values="sharpe")
    p = p[[c for c in C.CRITERIA if c in p]]
    p["champion_rank"] = p.rank(axis=1, ascending=False)[C.CHAMPION]
    p["champion_minus_coint"] = p[C.CHAMPION] - p["Cointegration"]
    p["mcr_minus_best_classic"] = p["MCR"] - p[["Cointegration", "Distance", "Correlation",
                                                "HalfLife"]].max(axis=1)
    return p
