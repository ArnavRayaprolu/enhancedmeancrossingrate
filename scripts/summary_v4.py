"""Cross-universe summary for protocol v4 (results/<dataset>/*.csv -> results/summary_v4/).

    python scripts/summary_v4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emcr import config as C, figures as F, inference as I  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "summary_v4"
UNIVERSES = ["crypto", "sp500", "nifty50", "etf"]
INTERVALS = ["5m", "15m", "30m", "1h", "4h", "1d"]
MIN_FOLDS = 5          # datasets with fewer walk-forward windows are shown but not ranked


def datasets():
    out = []
    for u in UNIVERSES:
        for iv in INTERVALS:
            d = f"{u}_{iv}"
            if d in C.DATASETS and (RES / d / "performance.csv").exists():
                out.append(d)
        if u == "crypto" and (RES / "crypto_spot_1d" / "performance.csv").exists():
            out.append("crypto_spot_1d")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = C.PRIMARY.key
    ds_list = datasets()
    perf, dms, ics, folds, spa, dsr = [], [], [], [], [], []
    for d in ds_list:
        r = RES / d
        pf = pd.read_csv(r / "performance.csv"); pf["dataset"] = d
        perf.append(pf)
        dm = pd.read_csv(r / "dm.csv"); dm["dataset"] = d; dms.append(dm)
        if (r / "ic.csv").exists():
            ic = pd.read_csv(r / "ic.csv"); ic["dataset"] = d; ics.append(ic)
        fo = pd.read_csv(r / "folds.csv"); folds.append(dict(dataset=d, folds=len(fo),
                                                              first=str(fo.test_start.iloc[0])[:10],
                                                              last=str(fo.test_end.iloc[-1])[:10],
                                                              names=fo.n_names.mean(),
                                                              pairs=fo.n_pairs.mean()))
        if (r / "spa.csv").exists():
            s = pd.read_csv(r / "spa.csv"); s["dataset"] = d; spa.append(s)
        if (r / "dsr.csv").exists():
            s = pd.read_csv(r / "dsr.csv"); s["dataset"] = d; dsr.append(s)
    perf = pd.concat(perf); dms = pd.concat(dms)
    fo = pd.DataFrame(folds); fo["ranked"] = fo["folds"] >= MIN_FOLDS
    fo.to_csv(OUT / "data_overview.csv", index=False)
    ranked = set(fo.loc[fo.ranked, "dataset"])

    # 1. Sharpe matrix (primary configuration)
    prim = perf[perf.config == P]
    sh = prim.pivot(index="dataset", columns="criterion", values="sharpe").reindex(ds_list)
    sh = sh[[c for c in C.CRITERIA if c in sh]]
    sh.to_csv(OUT / "sharpe_primary.csv")
    F.sharpe_heatmap(sh, OUT / "fig_sharpe_heatmap.pdf",
                     "Net Sharpe ratio by universe/interval and selection method")
    F.sharpe_heatmap(sh, OUT / "fig_sharpe_heatmap.png",
                     "Net Sharpe ratio by universe/interval and selection method")
    for col in ("total_return", "max_drawdown", "turnover_pa", "breakeven_arith_bps",
                "breakeven_comp_bps", "cagr", "ann_vol", "sortino"):
        prim.pivot(index="dataset", columns="criterion", values=col).reindex(ds_list)[
            sh.columns].to_csv(OUT / f"{col}_primary.csv")

    # 2. ranking of selection methods across datasets
    shr = sh.loc[[d for d in sh.index if d in ranked]]
    rank = shr.rank(axis=1, ascending=False)
    summ = pd.DataFrame({"mean_sharpe": shr.mean(), "median_sharpe": shr.median(),
                         "share_positive": (shr > 0).mean(), "mean_rank": rank.mean(),
                         "times_best": (rank == 1).sum()}).sort_values("mean_rank")
    summ.to_csv(OUT / "method_ranking.csv")

    # 3. hybrid vs its own classical component, dataset by dataset
    rows = []
    for h, x in [(f"EMCR+{c}", c) for c in C.CLASSICS] + [(e, "Cointegration") for e in C.EXTENDED]:
        diff = (shr[h] - shr[x]).dropna()
        rows.append(dict(hybrid=h, component=x, datasets=len(diff), hybrid_better=int((diff > 0).sum()),
                         median_sharpe_gain=float(diff.median()), mean_sharpe_gain=float(diff.mean())))
    pd.DataFrame(rows).to_csv(OUT / "hybrid_vs_component.csv", index=False)

    # 4. pre-registered DM tests, primary configuration, Holm within dataset and overall
    keep = dms[(dms.config == P) & dms.dataset.isin(ranked)
               & dms.apply(lambda r: (r.A, r.B) in C.PRIMARY_TESTS, axis=1)].copy()
    keep["holm_within"] = keep.groupby("dataset")["p_nw"].transform(lambda p: I.holm(p.to_numpy()))
    keep["holm_all"] = I.holm(keep["p_nw"].to_numpy())
    keep[["dataset", "A", "B", "dm_nw", "p_nw", "p_nw2", "p_andrews", "p_boot", "lag_nw",
          "mean_diff_pa", "holm_within", "holm_all"]].to_csv(OUT / "dm_primary.csv", index=False)

    # 5. pair-level information coefficients (EMCR = 'mcr' feature)
    if ics:
        ic = pd.concat(ics)
        ic = ic[ic.outcome == "oos_net"]
        ic.pivot_table(index="dataset", columns="feature", values="mean").reindex(ds_list).to_csv(
            OUT / "ic_mean.csv")
        ic.pivot_table(index="dataset", columns="feature", values="t").reindex(ds_list).to_csv(
            OUT / "ic_t.csv")
    # 6. execution-delay robustness (primary configuration executed one bar later)
    lag_rows = {}
    for d in ds_list:
        f = RES / f"{d}_lag1" / "performance.csv"
        if f.exists():
            pl = pd.read_csv(f)
            lag_rows[d] = pl[pl.config == C.DELAYED.key].set_index("criterion")["sharpe"]
    if lag_rows:
        lag = pd.DataFrame(lag_rows).T.reindex(columns=sh.columns)
        lag.to_csv(OUT / "sharpe_primary_lag1.csv")
        cmp_ = pd.DataFrame({"no_delay_mean": sh.loc[lag.index].mean(axis=1),
                             "delay1_mean": lag.mean(axis=1),
                             "best_delay1": lag.idxmax(axis=1), "best_delay1_sharpe": lag.max(axis=1)})
        cmp_.to_csv(OUT / "delay_summary.csv")
        print("\nExecution delay (1 bar)\n", cmp_.round(2).to_string())
    if spa:
        pd.concat(spa).to_csv(OUT / "spa.csv", index=False)
    if dsr:
        pd.concat(dsr).to_csv(OUT / "dsr.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print("Walk-forward windows per dataset\n", fo[["dataset", "folds", "first", "last",
                                                     "ranked"]].to_string(index=False))
    print("Sharpe, primary configuration\n", sh.round(2).to_string())
    print("\nMethod ranking\n", summ.round(2).to_string())
    print("\nHybrid vs component\n", pd.DataFrame(rows).round(2).to_string(index=False))
    sig = keep[keep.holm_all < 0.05]
    print(f"\nPre-registered DM tests: {len(keep)}; significant after overall Holm: {len(sig)}")
    print(keep.sort_values("p_nw").head(15)[["dataset", "A", "B", "dm_nw", "p_nw", "holm_all"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
