"""Write results/*.csv and results/REPORT.md, organised by reviewer comment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import report as R

P, H = C.PRIMARY.label, C.HEADLINE_COST


def md(df: pd.DataFrame, digits: int = 3, index: bool = False) -> str:
    df = df.reset_index() if index else df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def write(bars, picks, folds, diag, reps: int = 999):
    out = R.RES
    out.mkdir(exist_ok=True)
    perf = R.performance(bars); perf.to_csv(out / "performance.csv", index=False)
    dm = pd.concat([R.dm_tests(bars, P, c, reps) for c in C.COSTS]
                   + [R.dm_tests(bars, cfg, H, reps) for cfg in bars.config.unique() if cfg != P],
                   ignore_index=True)
    dm.to_csv(out / "dm_tests.csv", index=False)
    spa, dsr = R.data_snooping(bars, H, reps)
    spa.to_csv(out / "spa_reality_check.csv", index=False); dsr.to_csv(out / "deflated_sharpe.csv", index=False)
    ci = R.confidence(bars, P, H, reps); ci.to_csv(out / "confidence_intervals.csv", index=False)
    sub = R.subperiods(bars, P, H); sub.to_csv(out / "subperiods.csv")
    ft = R.fold_table(bars, P, H); ft.to_csv(out / "fold_returns.csv")
    conc = R.concentration(picks, bars, P); conc.to_csv(out / "concentration.csv", index=False)
    comp = R.composition(picks, P); comp.to_csv(out / "composition.csv", index=False)
    grid = R.grid_table(perf, H); grid.to_csv(out / "grid_sharpe.csv")
    folds.to_csv(out / "folds.csv", index=False)
    if diag is not None:
        dg = R.diagnostics_table(diag); dg.to_csv(out / "mcr_diagnostics.csv", index=False)

    pp = perf[(perf.config == P)].set_index(["criterion", "costs"])
    head = perf[(perf.config == P) & (perf.costs == H)].set_index("criterion").reindex(C.CRITERIA)
    L = [f"# Revised MCR model: walk-forward results (protocol {C.PROTOCOL})\n",
         "Data: Alpaca US equities, 1-hour bars (09:00-15:00 ET), split and dividend adjusted, "
         f"{bars.time.min():%Y-%m-%d} to {bars.time.max():%Y-%m-%d} out of sample. "
         f"{folds[folds.group == P].shape[0]} walk-forward windows "
         f"({C.PRIMARY.train_days} formation days, {C.PRIMARY.test_days} trading days). "
         "All parameters were fixed in mcr/config.py before any result was seen.\n"]
    fp = folds[folds.group == P]
    L.append(f"Universe per window: top {C.PRIMARY.top_k} liquid common stocks chosen on formation data "
             f"(median {fp.n_names.median():.0f} names, {fp.n_pairs.median():,.0f} candidate pairs, "
             f"{fp.inactive_names.mean():.0f} names per window that are now delisted). "
             f"Median estimated one-way cost {fp.median_cost_bps.median():.1f} bps.\n")
    L.append("## Headline (primary configuration, base costs)\n")
    cols = ["total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown",
            "time_in_market", "avg_exposure", "turnover_pa", "breakeven_arith_bps",
            "breakeven_comp_bps"]
    L.append(md(head[cols], index=True))
    L.append("\n## R1 - What MCR measures\n")
    L.append("MCR is the share of formation bars on which z = (MA_35 - MA_140)/SD_140 of the price "
             "ratio changes sign, i.e. the fast and slow moving averages of the ratio cross. It is "
             "not the ratio crossing its own mean. Fold-average Spearman correlations over a random "
             f"sample of {C.DIAG_SAMPLE:,} candidate pairs per window:\n")
    if diag is not None:
        keyx = dg[dg.x == "mcr"].set_index("y")[["mean_rho", "t", "share_positive"]]
        L.append(md(keyx, index=True))
        L.append("\nEvery formation statistic vs the pair's own out-of-sample net return "
                 "(oriented so positive = the statistic picks better pairs):\n")
        o = dg[dg.y == "oos_net"].set_index("x")[["mean_rho", "t", "share_positive"]]
        sign = pd.Series({k: v for k, v in __import__("mcr.walkforward", fromlist=["F"]).FEATURES.items()})
        sg = sign.reindex(o.index)
        o["mean_rho"] *= sg; o["t"] *= sg
        o["share_positive"] = np.where(sg > 0, o["share_positive"], 1 - o["share_positive"])
        L.append(md(o.sort_values("t", ascending=False), index=True))
    L.append("\n## R2 / R9 - Specification search, data snooping and deflated Sharpe\n")
    L.append(f"Trials: {bars.groupby(['config', 'criterion']).ngroups} strategies "
             f"({bars.config.nunique()} configurations x {len(C.CRITERIA)} criteria), every one reported "
             "in results/performance.csv.\n")
    L.append(md(spa)); L.append(""); L.append(md(dsr))
    L.append("\nBootstrap 95% intervals (stationary bootstrap, Politis-White block):\n")
    L.append(md(ci))
    L.append("\n## R3 - Diebold-Mariano tests (primary configuration)\n")
    L.append("d_t = n^A_t - n^B_t (net return per bar); loss = -net return; H0: E[d] = 0 vs two-sided. "
             "Columns: HAC lags = floor(4(T/100)^(2/9)) (nw), 2x, 4x, Andrews (1991), one day (7); "
             "the test on daily sums of d; stationary-bootstrap p (null imposed); Holm within family "
             "and across all.\n")
    dmp = dm[(dm.config == P)]
    L.append(md(dmp[["costs", "family", "A", "B", "mean_diff_pa", "dm_nw", "p_nw", "p_nw2", "p_nw4",
                     "p_andrews", "p_7", "p_daily", "p_boot", "holm_family", "holm_all"]], 4))
    L.append("\n## R4 - Engine\n")
    L.append("Rules in mcr/engine.py (docstring = pseudocode). Effect of the rule changes: compare "
             "configs 'primary' and 'engine_original' below.\n")
    L.append("## R5 - Costs and break-even\n")
    tab = perf[(perf.config == P)].pivot(index="criterion", columns="costs",
                                         values=["sharpe", "total_return"]).reindex(C.CRITERIA)
    tab.columns = [f"{a}_{b}" for a, b in tab.columns]
    tab["breakeven_arith_bps"] = head["breakeven_arith_bps"]
    tab["breakeven_comp_bps"] = head["breakeven_comp_bps"]
    L.append(md(tab, index=True))
    L.append("\n'original' = 1.5 bps per dollar traded (the paper's 3 bps round trip); 'base' = each "
             "stock's Abdi-Ranaldo half-spread + 0.5 bp fees + 0.5%/yr borrow on the short leg; "
             "'stress' = 2x spread + square-root impact for a $10m book + 2%/yr borrow. Break-even "
             "is the one-way cost per dollar traded, by sum of returns and by compounded wealth.\n")
    L.append("## R6 - Sample, regimes, survivorship, missing data\n")
    L.append(md(sub, 2, index=True))
    L.append("\n## R7 / R8 - Ratio vs hedged spread; one-at-a-time robustness (Sharpe, base costs)\n")
    L.append(md(grid, 2, index=True))
    L.append("\nNote: the selected pairs are the top 20 of ~31,000 candidates, so their percentile "
             "scores are all within ~0.001 of 1. Rank weights (Eq. 8 and 'rank') are therefore "
             "almost equal weights; 'w_rank_fixed' and 'w_equal' coincide to two decimals.\n")
    L.append("\n## m4 / m7 - Composition and concentration of the books (primary)\n")
    L.append(md(comp)); L.append(""); L.append(md(conc))
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "REPORT.md")
