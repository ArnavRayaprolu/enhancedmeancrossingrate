"""Robustness of finalist strategies on the DEVELOPMENT period.

    python research/robust.py

For each finalist: Sharpe by calendar year, Sharpe of one-step parameter
neighbours, and Sharpe on other venues' pre-2024 data.
"""
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from emcr.explore import Spec, evaluate  # noqa: E402

BASE = dict(window=24, top_k=100, weighting="equal")
FINALISTS = {
    "MCR e3 x-.5 n5": Spec(criterion="MCR", entry=3.0, exit=-0.5, n_pairs=5, max_uses=1, **BASE),
    "HL e2.5 x-.5 g48 n10u3": Spec(criterion="HalfLife", entry=2.5, exit=-0.5, gate=48,
                                   n_pairs=10, max_uses=3, **BASE),
}
STEPS = dict(entry=(-0.5, 0.5), exit=(-0.5, 0.5), window=(12, 48), n_pairs=(-2, 5))


def yearly(net: pd.Series) -> dict:
    out = {}
    for y, r in net.groupby(net.index.year):
        out[y] = r.mean() / r.std() * np.sqrt(24 * 365) if r.std() > 0 else np.nan
    return out


def neighbours(sp: Spec):
    for k, (lo, hi) in STEPS.items():
        for d in (lo, hi):
            v = getattr(sp, k)
            nv = d if k == "window" else v + d
            if k == "window" and nv == v:
                continue
            yield f"{k}={nv}", replace(sp, **{k: nv})


if __name__ == "__main__":
    which = sys.argv[1:] or list(FINALISTS)
    for name in which:
        sp = FINALISTS[name]
        m = evaluate("perp_1h", sp, note=f"robust base {name}")
        print(f"\n### {name}: Sharpe {m['sharpe']:+.2f}  ret {m['total_return']:+.2f}")
        print("  by year:", {y: round(v, 2) for y, v in yearly(m["net"]).items()})
        for lab, nsp in neighbours(sp):
            mm = evaluate("perp_1h", nsp, note=f"robust nbr {name}")
            print(f"  {lab:12s} Sharpe {mm['sharpe']:+.2f}  ret {mm['total_return']:+.2f}")
        for ds in ("binance_1h", "bitstamp_1h"):
            try:
                mm = evaluate(ds, sp, note=f"robust venue {name}")
                m6 = evaluate(ds, sp, cost_bps=6.0, note=f"robust venue6 {name}")
                print(f"  {ds:12s} Sharpe {mm['sharpe']:+.2f} (own cost) {m6['sharpe']:+.2f} (6bps) "
                      f"{mm['first']}..{mm['last']}")
            except Exception as exc:
                print(f"  {ds}: {exc}")
