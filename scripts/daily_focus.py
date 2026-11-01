"""Headline table: daily bars, where results survive a one-bar execution delay.

    python scripts/daily_focus.py   ->  results/summary_v4/daily_focus.csv / .md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emcr import config as C, inference as I  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "summary_v4"
DAILY = ["crypto_spot_1d", "sp500_1d", "nifty50_1d", "etf_1d"]
LABEL = {"crypto_spot_1d": "Crypto (2017-26)", "sp500_1d": "S&P 500 (2005-26)",
         "nifty50_1d": "NIFTY 50 (2008-26)", "etf_1d": "Index funds (2005-26)"}


def main():
    P, L = C.PRIMARY.key, C.DELAYED.key
    rows = {}
    for d in DAILY:
        a = pd.read_csv(RES / d / "performance.csv")
        a = a[a.config == P].set_index("criterion")
        b = pd.read_csv(RES / f"{d}_lag1" / "performance.csv")
        b = b[b.config == L].set_index("criterion")
        rows[(LABEL[d], "Sharpe")] = a["sharpe"]
        rows[(LABEL[d], "Sharpe, 1-day delay")] = b["sharpe"]
        rows[(LABEL[d], "Max DD")] = a["max_drawdown"]
    t = pd.DataFrame(rows).reindex(list(C.CRITERIA))
    sh = t.xs("Sharpe", axis=1, level=1)
    sl = t.xs("Sharpe, 1-day delay", axis=1, level=1)
    t[("All daily", "mean Sharpe")] = sh.mean(axis=1)
    t[("All daily", "mean Sharpe, delay")] = sl.mean(axis=1)
    t[("All daily", "positive (of 4, delay)")] = (sl > 0).sum(axis=1)
    t[("All daily", "mean rank, delay")] = sl.rank(ascending=False).mean(axis=1)
    t = t.sort_values(("All daily", "mean Sharpe, delay"), ascending=False)
    t.to_csv(OUT / "daily_focus.csv")
    # DM tests on daily data: every EMCR-family method vs every classic (primary config)
    dm = []
    for d in DAILY:
        x = pd.read_csv(RES / d / "dm.csv")
        x = x[(x.config == P)]
        x["dataset"] = LABEL[d]
        dm.append(x)
    dm = pd.concat(dm)
    dm["holm_all"] = I.holm(dm["p_nw"].to_numpy())
    dm.sort_values("p_nw")[["dataset", "A", "B", "dm_nw", "p_nw", "p_andrews", "p_boot",
                            "mean_diff_pa", "holm_all"]].to_csv(OUT / "daily_dm.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(t.round(2).to_string())
    print(dm.sort_values("p_nw").head(12)[["dataset", "A", "B", "dm_nw", "p_nw", "holm_all"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
