"""Parallel development-period grid: python scratch/grid.py <name> (specs defined below)."""
import itertools
import sys
from multiprocessing import Pool

sys.path.insert(0, ".")
from emcr.explore import Spec, evaluate  # noqa: E402

DS = "perp_1h"


def run_group(args):
    note, specs = args
    out = []
    for sp in specs:
        m = evaluate(DS, sp, note=note)
        m.pop("net")
        out.append((sp, m))
    return out


def specs_scan1():
    crit = ["Distance", "Correlation", "Cointegration", "HalfLife", "MCR", "CCR",
            "MCR+Coint", "CCR+Coint"]
    groups = {}
    for sig, w in itertools.product(["LEVEL", "STATIC"], [12, 24, 48, 140]):
        if sig == "STATIC" and w != 140:
            continue
        for c, n, e in itertools.product(crit, [5, 20, 40], [2.0, 2.5]):
            groups.setdefault((sig, w), []).append(
                Spec(criterion=c, signal=sig, window=w, entry=e, n_pairs=n, weighting="equal"))
    return [(f"scan1 {k}", v) for k, v in groups.items()]


def specs_scan2():
    groups = {}
    for k, u, n, c in itertools.product([40, 100], [1, 3], [5, 10, 20],
                                        ["MCR", "Cointegration", "HalfLife"]):
        groups.setdefault((k, c), []).append(
            Spec(criterion=c, window=24, entry=2.5, n_pairs=n, weighting="equal",
                 top_k=k, max_uses=u))
    return [(f"scan2 {k}", v) for k, v in groups.items()]


if __name__ == "__main__":
    name = sys.argv[1]
    jobs = globals()[f"specs_{name}"]()
    with Pool(4) as p:
        for res in p.imap_unordered(run_group, jobs):
            for sp, m in res:
                print(f"{sp.signal:6s} w={sp.window:<4} e={sp.entry} n={sp.n_pairs:<3} "
                      f"k={sp.top_k:<3} u={sp.max_uses} g={sp.gate} "
                      f"{sp.criterion:13s} sharpe={m['sharpe']:+.2f} gross={m['gross_pa']:+.2f} "
                      f"turn={m['turnover_pa']:.0f} ret={m['total_return']:+.2f}", flush=True)
