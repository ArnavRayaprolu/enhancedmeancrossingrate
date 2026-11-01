"""Pair-level information coefficients on the development period.

For each dev fold: compute formation statistics for every candidate pair and the
pair's out-of-sample net return when traded alone with a fixed rule; report the
fold-average Spearman correlation (IC) and its t-statistic.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")
from emcr.explore import Spec, _ccr, _zcache, dev_folds  # noqa: E402
from emcr.engine import positions_for_window  # noqa: E402
from emcr.signals import leg_weights, signal_crossing_rate  # noqa: E402

DS = sys.argv[1] if len(sys.argv) > 1 else "perp_1h"
RULES = {"LEVEL24": Spec(signal="LEVEL", window=24), "LEVEL140": Spec(signal="LEVEL", window=140),
         "STATIC": Spec(signal="STATIC")}
KAPPA = 6e-4


def hurst_vr(s, lags=(1, 4, 16, 64)):
    """Variance-ratio based Hurst estimate of each column (H < 0.5 = mean reverting)."""
    v = [np.nanvar(s[l:] - s[:-l], axis=0) for l in lags]
    x = np.log(lags); y = np.log(np.vstack(v))
    xm = x - x.mean()
    return ((xm[:, None] * (y - y.mean(0))).sum(0) / (xm ** 2).sum()) / 2


rows = []
for f in dev_folds(DS):
    T = f["train"]; te = slice(T, len(f["idx"]))
    feats = {}
    for w in (12, 24, 48, 140, 280):
        feats[f"mcr_roll{w}"] = _zcache(f, "LEVEL", w)["mcr"]
    st = _zcache(f, "STATIC", 140)
    feats["mcr_static"] = st["mcr"]
    feats["ccr_static"] = _ccr(f, Spec(signal="STATIC"))
    feats["ccr_roll140"] = _ccr(f, Spec(signal="LEVEL", window=140))
    feats["ccr_roll24"] = _ccr(f, Spec(signal="LEVEL", window=24))
    feats["mcr_ratio_static_roll24"] = st["mcr"] / feats["mcr_roll24"]
    feats["mcr_ratio_roll140_roll12"] = feats["mcr_roll140"] / feats["mcr_roll12"]
    lr = np.log(f["px"][:T, f["ia"]] / f["px"][:T, f["ib"]])
    feats["hurst"] = -hurst_vr(lr)
    feats["neg_coint_p"] = -f["coint_p"]
    feats["corr"] = f["corr"]
    feats["neg_distance"] = -f["distance"]
    feats["neg_halflife"] = -st["halflife"]
    feats["sd_logratio"] = st["sd_rel"]
    feats["neg_sd_logratio"] = -st["sd_rel"]
    out = {}
    for name, sp in RULES.items():
        z = _zcache(f, sp.signal, sp.window)["z"]
        pos = positions_for_window(z, te.start, te.stop, sp.entry, sp.exit)
        a, b = leg_weights(sp.signal, f["beta"], len(f["ia"]))
        r = f["rets"][te]
        unit = a[None, :] * r[:, f["ia"]] + b[None, :] * r[:, f["ib"]]
        P = len(f["ia"])
        dpos = np.abs(np.diff(np.vstack([np.zeros((1, P)), pos, np.zeros((1, P))]), axis=0)).sum(0)
        out[name] = (pos * unit).sum(0) - KAPPA * dpos
    for fn, x in feats.items():
        for on, y in out.items():
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 20:
                rows.append(dict(fold=f["fold"], feature=fn, rule=on,
                                 ic=stats.spearmanr(x[ok], y[ok])[0]))
df = pd.DataFrame(rows)
g = df.groupby(["rule", "feature"])["ic"]
res = pd.DataFrame({"mean": g.mean(), "t": g.mean() / g.std() * np.sqrt(g.count()),
                    "pos": g.apply(lambda s: (s > 0).mean())}).round(3)
pd.set_option("display.width", 200)
print(res.unstack("rule").sort_values(("t", "STATIC"), ascending=False).to_string())
