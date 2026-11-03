"""Walk-forward optimisation of the OU / Avellaneda-Lee pairs strategy with EMCR selection.

    python research/ou_wfo.py hourly      # 500-name hourly panel, 2016-2026
    python research/ou_wfo.py m5          # 152-name 5-minute panel, 2024-2026

1. Walk-forward from the very start of the data: select pairs on the formation
   window (hourly: 60 trading days, 5-minute: 10 days), trade them on the next
   window (hourly: 10 days, 5-minute: 5 days), slide both windows forward by the
   trading window and repeat. Candidates: every pair of the window's most liquid
   names (whole universe, no industry restriction).
2. Walk-forward optimisation: for every selection criterion separately, before
   each trading window the setting (engine Z2 / Z15 / Avellaneda-Lee x 5 / 10 / 20 pairs) with the
   best Sharpe ratio over the previous LOOKBACK trading windows is chosen and used
   for that window only (until LOOKBACK windows exist, the default entry 2.0 / 10
   pairs is used). The choice uses past trading windows only.
   Spread hedge (argument): ols | tls | mkt (see research/ou_lab.py).
3. Report: performance, year-by-year Sharpe, break-even cost, Diebold-Mariano tests
   of each EMCR criterion against each classical criterion, deflated Sharpe.
Trading rules, costs and the spread z are those of research/ou_lab.py.
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcr import inference as I  # noqa: E402
from research import ou_lab as O, ou_run as R  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "research" / "ou_wfo"

CLASSIC = ["Distance", "Correlation", "Cointegration", "HalfLife", "SH|Correlation"]
EMCR = ["EMCR", "EMCRscore", "EMCRband", "EMCRscore+Correlation", "SH|EMCRscore", "SH|EMCRband"]
CRITS = CLASSIC + EMCR
# trading engines: (label, zmode, entry, exit_long, exit_short)
ENGINES = {"Z2": ("spread", 2.0, 0.0, 0.0),
           "Z15": ("spread", 1.5, 0.0, 0.0),
           "AL": ("ou", 1.25, -0.5, 0.75)}          # Avellaneda-Lee s-score rule
ENTRIES = tuple(ENGINES)
NPAIRS = (5, 10, 20)
DEFAULT = ("Z2", 10)


def fold_daily(r_net, when):
    s = pd.Series(r_net, index=when)
    return s.groupby(pd.DatetimeIndex(s.index).tz_convert("America/New_York").date).sum()


def sharpe(x):
    x = np.asarray(x, float)
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 2 and x.std() > 0 else -np.inf


def main(dataset, hedge="ols"):
    D = R.load(dataset); bpd = D["bpd"]
    if dataset == "hourly":
        F, T, top_k, lookback = 60 * bpd, 10 * bpd, 250, 25          # ~1 year of trading windows
    else:
        F, T, top_k, lookback = 10 * bpd, 5 * bpd, 152, 25           # ~6 months of trading windows
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # ---- step 1: walk-forward returns for every (criterion, setting), per fold ----
    per = {}            # (crit, entry, n) -> list of daily Series, one per fold
    turn = {}           # (crit, entry, n) -> (gross sum, turnover sum)
    for entry in ENTRIES:
        zmode, ent, xl, xs = ENGINES[entry]
        base = O.OUDesign(F=F, T=T, entry=ent, zmode=zmode, exit_long=xl, exit_short=xs,
                          hedge=hedge, bars_per_year=252 * bpd)
        prep = R.prepare(D, base, None, top_k)
        print(f"{dataset} {hedge} engine {entry}: {len(prep)} folds, "
              f"{np.mean([len(p['stats']) for p in prep]):.0f} candidate pairs per fold "
              f"[{time.time() - t0:.0f}s]", flush=True)
        for c, n in itertools.product(CRITS, NPAIRS):
            des = replace(base, criterion=c, n_pairs=n, max_uses=1 if n <= 10 else 2)
            fl, g, u = [], 0.0, 0.0
            for fp in prep:
                r, _ = O.run_fold(fp, des)
                fl.append(fold_daily(r["net"], fp["when"]))
                g += r["gross"].sum(); u += r["turn"].sum()
            per[(c, entry, n)] = fl; turn[(c, entry, n)] = (g, u)
        print(f"  settings done [{time.time() - t0:.0f}s]", flush=True)
    n_folds = len(next(iter(per.values())))
    # ---- step 2: walk-forward optimisation ----
    wfo, chosen = {}, []
    for c in CRITS:
        parts = []
        for k in range(n_folds):
            if k < lookback:
                best = DEFAULT
            else:
                hist = {s: sharpe(pd.concat(per[(c,) + s][k - lookback:k]).to_numpy())
                        for s in itertools.product(ENTRIES, NPAIRS)}
                best = max(hist, key=hist.get)
            parts.append(per[(c,) + best][k])
            chosen.append(dict(criterion=c, fold=k, engine=best[0], n_pairs=best[1]))
        wfo[c] = pd.concat(parts).sort_index()
    W = pd.DataFrame(wfo)
    W.to_csv(OUT / f"wfo_daily_{dataset}_{hedge}.csv")
    pd.DataFrame(chosen).to_csv(OUT / f"wfo_choices_{dataset}_{hedge}.csv", index=False)
    # static (fixed setting) results for reference
    static = []
    for (c, e, n), fl in per.items():
        x = pd.concat(fl)
        g, u = turn[(c, e, n)]
        static.append(dict(criterion=c, engine=e, n_pairs=n, sharpe=sharpe(x), ann_ret=x.mean() * 252,
                           breakeven_bps=1e4 * g / u if u > 0 else np.nan))
    pd.DataFrame(static).to_csv(OUT / f"static_{dataset}_{hedge}.csv", index=False)
    # ---- step 3: report ----
    rows = []
    for c in CRITS:
        x = W[c].dropna()
        eq = (1 + x).cumprod()
        rows.append(dict(criterion=c, sharpe=sharpe(x), ann_ret=x.mean() * 252, ann_vol=x.std() * np.sqrt(252),
                         max_dd=float((eq / eq.cummax() - 1).min()), total=float(eq.iloc[-1] - 1)))
    perf = pd.DataFrame(rows).set_index("criterion")
    years = W.groupby(pd.DatetimeIndex(W.index).year).apply(lambda d: d.apply(sharpe)).T
    dm = []
    for a, b in itertools.product(EMCR, CLASSIC):
        x, y = W[a].align(W[b], join="inner")
        t = I.diebold_mariano(x.to_numpy(), y.to_numpy(), "nw")
        bt = I.bootstrap_mean_pvalue((x - y).to_numpy(), reps=999)
        dm.append(dict(A=a, B=b, mean_diff_pa=(x - y).mean() * 252, dm=t["dm"], p_nw=t["p"], p_boot=bt["p_boot"]))
    dm = pd.DataFrame(dm); dm["holm"] = I.holm(dm["p_nw"].to_numpy())
    trials = np.array([sharpe(pd.concat(fl)) for fl in per.values()]) / np.sqrt(252)
    best = perf["sharpe"].idxmax()
    dsr = I.deflated_sharpe(W[best].dropna().to_numpy(), trials[np.isfinite(trials)])
    perf.to_csv(OUT / f"wfo_perf_{dataset}_{hedge}.csv"); years.to_csv(OUT / f"wfo_years_{dataset}_{hedge}.csv")
    dm.to_csv(OUT / f"wfo_dm_{dataset}_{hedge}.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nWALK-FORWARD OPTIMISED PERFORMANCE\n", perf.round(3).to_string())
    print("\nSHARPE BY YEAR\n", years.round(2).to_string())
    print("\nDIEBOLD-MARIANO (EMCR vs classic)\n", dm.round(4).to_string(index=False))
    print(f"\nbest: {best}; deflated Sharpe over {len(trials)} settings: {dsr['dsr']:.3f}")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ols")
