"""Criterion comparison on other venues' DEVELOPMENT data with the finalist rule."""
import sys
from multiprocessing import Pool
sys.path.insert(0, ".")
from emcr.explore import Spec, evaluate  # noqa: E402

CRIT = ["MCR", "HalfLife", "Cointegration", "Distance", "Correlation"]
RULE = dict(window=24, entry=3.0, exit=-0.5, n_pairs=5, max_uses=1, top_k=100, weighting="equal")


def run(ds):
    out = []
    for c in CRIT:
        m = evaluate(ds, Spec(criterion=c, **RULE), note=f"venues {ds}")
        out.append((ds, c, m["sharpe"], m["total_return"], m["first"], m["last"]))
    return out


if __name__ == "__main__":
    with Pool(3) as p:
        for res in p.imap_unordered(run, sys.argv[1:] or ["binance_1h", "bitstamp_1h", "coinbase_1h"]):
            for ds, c, sh, r, a, b in res:
                print(f"{ds:12s} {c:14s} sharpe={sh:+.2f} ret={r:+.2f} {a}..{b}", flush=True)
