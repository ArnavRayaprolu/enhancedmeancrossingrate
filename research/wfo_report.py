"""Metrics for the walk-forward Bayesian-optimisation runs (results/wfo/<dataset>/).

    python research/wfo_report.py            # every finished dataset
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from emcr import config as C, inference as I  # noqa: E402

ROOT = Path("results/wfo")
ORDER = ["perp_1h", "perp_30m", "perp_15m", "binance_1h", "binance_30m", "binance_15m",
         "coinbase_1h", "okx_1h", "nifty50_1h", "sp500_1h_all"]


def one(ds: str):
    d = ROOT / ds
    net, gross, turn = (pd.read_parquet(d / f"{x}.parquet") for x in ("net", "gross", "turn"))
    design = json.load(open(d / "design.json"))
    bpy = C.DATASETS[ds].bars_per_year
    ppy = 365.0 if "sp500" not in ds and "nifty" not in ds else 252.0
    rows, dm, yearly = [], [], []
    borrow = gross - net - design["cost_bps"] / 1e4 * turn     # financing component
    for c in net.columns:
        n = net[c].dropna()
        m = I.metrics(n, bpy)
        be = I.breakeven(gross[c].to_numpy(), turn[c].to_numpy(), borrow[c].to_numpy())
        dly = n.groupby(n.index.floor("D")).sum()
        ci = I.bootstrap_ci(dly.to_numpy(), ppy, reps=1000)
        rows.append(dict(dataset=ds, criterion=c, sharpe=m["sharpe"], sharpe_lo=ci["sharpe_lo"],
                         sharpe_hi=ci["sharpe_hi"], cagr=m["cagr"], total_return=m["total_return"],
                         ann_vol=m["ann_vol"], sortino=m["sortino"], max_dd=m["max_drawdown"],
                         calmar=m["cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] < 0 else np.nan,
                         turnover_pa=float(turn[c].mean() * bpy),
                         in_market=float((gross[c] != 0).mean()),
                         breakeven_bps=be["breakeven_arith_bps"],
                         breakeven_comp_bps=be["breakeven_comp_bps"],
                         first=str(n.index[0])[:10], last=str(n.index[-1])[:10], bars=len(n)))
        for y, r in n.groupby(n.index.year):
            yearly.append(dict(dataset=ds, criterion=c, year=y,
                               sharpe=r.mean() / r.std() * np.sqrt(bpy) if r.std() > 0 else np.nan))
        if c != "MCRmag":
            a = net["MCRmag"]; b = net[c].reindex(a.index)
            t = I.diebold_mariano(a.to_numpy(), b.to_numpy(), "nw")
            ta = I.diebold_mariano(a.to_numpy(), b.to_numpy(), "andrews")
            bt = I.bootstrap_mean_pvalue((a - b).to_numpy(), reps=1000)
            dm.append(dict(dataset=ds, A="MCRmag", B=c, dm=t["dm"], p=t["p"], p_andrews=ta["p"],
                           p_boot=bt["p_boot"], diff_pa=float((a - b).mean() * bpy)))
    dm = pd.DataFrame(dm)
    if len(dm):
        dm["holm"] = I.holm(dm["p"].to_numpy())
    prm = pd.read_csv(d / "params.csv")
    prm = prm[prm.criterion == "MCRmag"]
    choice = {k: prm[k].astype(str).value_counts(normalize=True).round(2).head(3).to_dict()
              for k in ("mean", "delta", "n_pairs", "weighting")}
    return pd.DataFrame(rows), dm, pd.DataFrame(yearly), choice


if __name__ == "__main__":
    ds_list = [d for d in ORDER if (ROOT / d / "net.parquet").exists()]
    R, D, Y, P = [], [], [], {}
    for ds in ds_list:
        r, dm, y, ch = one(ds)
        R.append(r); D.append(dm); Y.append(y); P[ds] = ch
    R, D, Y = pd.concat(R), pd.concat(D), pd.concat(Y)
    R.to_csv(ROOT / "metrics.csv", index=False)
    D.to_csv(ROOT / "dm.csv", index=False)
    Y.to_csv(ROOT / "yearly.csv", index=False)
    json.dump(P, open(ROOT / "chosen_params.json", "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(R[["dataset", "criterion", "sharpe", "sharpe_lo", "sharpe_hi", "cagr", "total_return",
             "max_dd", "turnover_pa", "breakeven_bps", "first", "last"]].round(3).to_string(index=False))
    print(D.round(4).to_string(index=False))
    print(Y.pivot_table(index=["dataset", "criterion"], columns="year", values="sharpe")
          .round(2).to_string())
    print(json.dumps(P, indent=1))
