"""ONE-SHOT hold-out exam of the frozen finalist (research/final_spec.py).

Trading windows starting on/after 2024-01-01 on every venue.  Writes
results/holdout/{summary,dm,yearly,ci,dsr}.csv and net return series.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "research")
from emcr import config as C, inference as I  # noqa: E402
from emcr.explore import Spec, evaluate  # noqa: E402
from final_spec import COMPARE, HOLDOUT_VENUES, RULE  # noqa: E402

OUT = Path("results/holdout")


def run_venue(ds):
    rows, nets = [], {}
    for c in COMPARE:
        for cost in (None, 6.0):
            if cost is not None and C.DATASETS[ds].cost_bps == cost:
                continue
            m = evaluate(ds, Spec(criterion=c, **RULE), cost_bps=cost, period="holdout",
                         log=False)
            net = m.pop("net")
            tag = "own" if cost is None else "6bps"
            nets[(ds, c, tag)] = net
            rows.append(dict(venue=ds, criterion=c, costs=tag, **{k: m[k] for k in (
                "cost_bps", "sharpe", "sharpe_daily", "total_return", "ann_return", "max_dd",
                "gross_pa", "turnover_pa", "n_bars", "first", "last")}))
    return rows, nets


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    venues = sys.argv[1:] or HOLDOUT_VENUES
    rows, nets = [], {}
    with Pool(3) as p:
        for r, n in p.imap_unordered(run_venue, venues):
            rows += r; nets.update(n)
    summ = pd.DataFrame(rows).sort_values(["venue", "costs", "sharpe"], ascending=[1, 1, 0])
    summ.to_csv(OUT / "summary.csv", index=False)

    dm, yearly, ci = [], [], []
    for (ds, c, tag), net in nets.items():
        bpy = C.DATASETS[ds].bars_per_year
        for y, r in net.groupby(net.index.year):
            yearly.append(dict(venue=ds, criterion=c, costs=tag, year=y,
                               sharpe=r.mean() / r.std() * np.sqrt(bpy) if r.std() > 0 else np.nan,
                               ret=float((1 + r).prod() - 1)))
        dly = net.groupby(net.index.floor("D")).sum()
        ci.append(dict(venue=ds, criterion=c, costs=tag,
                       **I.bootstrap_ci(dly.to_numpy(), 365, reps=2000)))
        if c != "MCR":
            a = nets[(ds, "MCR", tag)]; b = net.reindex(a.index)
            row = dict(venue=ds, costs=tag, A="MCR", B=c)
            for lag in ("nw", "andrews"):
                t = I.diebold_mariano(a.to_numpy(), b.to_numpy(), lag)
                row[f"dm_{lag}"] = t["dm"]; row[f"p_{lag}"] = t["p"]
            row.update(I.bootstrap_mean_pvalue((a - b).to_numpy(), reps=2000))
            row["diff_pa"] = float((a - b).mean() * bpy)
            dm.append(row)
    dm = pd.DataFrame(dm)
    for (v, tg), g in dm.groupby(["venue", "costs"]):
        dm.loc[g.index, "holm4"] = I.holm(g["p_nw"].to_numpy())
    dm.to_csv(OUT / "dm.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "yearly.csv", index=False)
    pd.DataFrame(ci).to_csv(OUT / "ci.csv", index=False)

    # deflated Sharpe ratio of the finalist given every development trial
    tr = pd.read_csv("results/dev_search/trials.csv")
    sr_trials = (tr["sharpe_daily"] / np.sqrt(365)).dropna().to_numpy()
    dsr = []
    for ds in venues:
        if (ds, "MCR", "own") in nets:
            n = nets[(ds, "MCR", "own")]
            d = n.groupby(n.index.floor("D")).sum().to_numpy()
            dsr.append(dict(venue=ds, **I.deflated_sharpe(d, sr_trials)))
    pd.DataFrame(dsr).to_csv(OUT / "dsr.csv", index=False)
    pd.DataFrame({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in nets.items()}).to_parquet(
        OUT / "net_returns.parquet")

    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))
    print(dm.round(4).to_string(index=False))
    print(pd.DataFrame(yearly).pivot_table(index=["venue", "criterion", "costs"], columns="year",
                                           values="sharpe").round(2).to_string())
    print(pd.DataFrame(ci).round(2).to_string(index=False))
    print(pd.DataFrame(dsr).round(3).to_string(index=False))
