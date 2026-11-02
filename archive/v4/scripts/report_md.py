"""Write results/summary_v4/REPORT.md from the v4.2 result files.

    python scripts/summary_v4.py && python scripts/report_md.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emcr import config as C  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
S = RES / "summary_v4"


def md(df: pd.DataFrame, digits: int = 2) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def main():
    ov = pd.read_csv(S / "data_overview.csv")
    sh = pd.read_csv(S / "sharpe_primary.csv", index_col=0)
    rk = pd.read_csv(S / "method_ranking.csv", index_col=0)
    hv = pd.read_csv(S / "hybrid_vs_component.csv")
    dm = pd.read_csv(S / "dm_primary.csv")
    be = pd.read_csv(S / "breakeven_arith_bps_primary.csv", index_col=0)
    to = pd.read_csv(S / "turnover_pa_primary.csv", index_col=0)
    L = []
    L.append(f"# EMCR pair selection: results (protocol v{C.PROTOCOL_VERSION})\n")
    L.append("All numbers are out of sample (non-overlapping walk-forward windows of 630 "
             "formation + 315 trading bars), net of venue-specific costs. The trading rule is "
             "identical for every selection method, so differences measure pair selection only.\n")
    L.append("**Fixed strategy.** Price ratio r = P_A/P_B; z = (r - MA_140)/SD_140; enter at "
             "|z| >= 2, exit at |z| <= 0.5; top 20 pairs, no security reused; capital "
             "rank-weighted by selection score (fixed for the window); costs: crypto 6 bps per "
             "side + actual funding, S&P 500 3 bps + 0.5%/yr borrow, NIFTY 50 5 bps, ETFs "
             "1.5 bps + 0.3%/yr borrow.\n")
    L.append("## 1. Data and walk-forward windows\n")
    L.append(md(ov[["dataset", "folds", "first", "last", "names", "pairs", "ranked"]], 0))
    L.append(f"\nDatasets with fewer than 5 windows are shown but excluded from rankings and "
             f"test families.\n")
    L.append("## 2. Net Sharpe ratio by dataset and selection method\n")
    L.append(md(sh.reset_index()))
    L.append("\n![heatmap](fig_sharpe_heatmap.png)\n")
    L.append("## 3. Ranking of selection methods (datasets with >= 5 windows)\n")
    L.append(md(rk.reset_index()))
    L.append("\n## 4. Does adding EMCR improve each classical method?\n")
    L.append(md(hv))
    L.append("\n## 5. Pre-registered Diebold-Mariano tests (primary configuration)\n")
    L.append("d_t = n^A_t - n^B_t (net return of A minus B per bar); H0: E[d] = 0, two-sided; "
             "Newey-West HAC with Bartlett kernel, bandwidth floor(4(T/100)^(2/9)) (also 2x and "
             "Andrews 1991 shown), Harvey-Leybourne-Newbold correction, t(T-1) reference; "
             "stationary-bootstrap p-value with Politis-White block length; Holm within dataset "
             "and across all datasets.\n")
    sig = dm[dm.holm_all < 0.05]
    L.append(f"Tests: {len(dm)}; significant at 5% after Holm across all: {len(sig)}.\n")
    L.append(md(dm.sort_values("p_nw").head(25), 4))
    if (S / "ic_t.csv").exists():
        L.append("\n## 6. Pair-level evidence: does the formation statistic predict a pair's "
                 "out-of-sample net return? (t-statistic of the fold-average rank IC)\n")
        ic = pd.read_csv(S / "ic_t.csv", index_col=0)
        L.append(md(ic.reset_index(), 1))
        L.append("\n`mcr` = EMCR, `ar1` = lag-1 autocorrelation of z (oriented: lower is "
                 "better; equivalent to the arcsine-implied EMCR), others as named.\n")
    if (S / "spa.csv").exists():
        sp = pd.read_csv(S / "spa.csv")
        L.append("\n## 7. Data-snooping checks\n")
        L.append(md(sp[sp.scope == "all"][["dataset", "K", "T", "spa_p", "rc_p"]], 3))
    if (S / "dsr.csv").exists():
        ds = pd.read_csv(S / "dsr.csv")
        L.append("\nDeflated Sharpe ratio (Bailey and Lopez de Prado 2014) of the best proposed "
                 "configuration in each dataset's sensitivity grid:\n")
        L.append(md(ds[ds.label.str.startswith("best")][["dataset", "config", "criterion", "sr",
                                                        "sr0", "dsr", "psr", "N"]], 3))
    if (S / "delay_summary.csv").exists():
        L.append("\n## 7b. Execution-delay robustness (decision at bar t, trade at bar t+1)\n")
        L.append(md(pd.read_csv(S / "delay_summary.csv")))
        L.append("\nSharpe by method with a one-bar delay:\n")
        L.append(md(pd.read_csv(S / "sharpe_primary_lag1.csv").rename(
            columns={"Unnamed: 0": "dataset"})))
    L.append("\n## 8. Costs: break-even one-way cost (bps) and turnover (x capital per year)\n")
    L.append(md(be.reset_index(), 1))
    L.append("\n")
    L.append(md(to.reset_index(), 0))
    L.append("\n## 9. Robustness: other signals and capital rules (mean Sharpe over methods)\n")
    rows = []
    for d in ov.dataset:
        pf = pd.read_csv(RES / d / "performance.csv")
        g = pf.groupby("config")["sharpe"].mean()
        rows.append(g.rename(d))
    rb = pd.DataFrame(rows)
    L.append(md(rb.reset_index().rename(columns={"index": "dataset"})))
    (S / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", S / "REPORT.md")


if __name__ == "__main__":
    main()
