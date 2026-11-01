"""Formation-window pair statistics, percentile ranks and pair selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnonp

from . import config as C
from .engine import completed_cycle_rate
from .signals import half_life, hedge_ratio, signal_crossing_rate, spread, zscore


# --------------------------------------------------------------------------- #
# Engle-Granger cointegration, vectorised
# --------------------------------------------------------------------------- #
def engle_granger_pvalue(log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """Identical to statsmodels.tsa.stattools.coint(log_a, log_b, trend='c',
    maxlag=1, autolag=None) column by column:
      step 1  log_a = c + beta log_b + e                (OLS)
      step 2  d e_t = g e_{t-1} + phi d e_{t-1} + u_t   (no constant)
      p-value of the t-statistic on g from MacKinnon (2010), N = 2, 'c'.
    """
    beta = hedge_ratio(log_a, log_b)
    e = (log_a - log_a.mean(0)) - beta * (log_b - log_b.mean(0))
    de = np.diff(e, axis=0)
    y = de[1:]; x1 = e[1:-1]; x2 = de[:-1]
    s11 = (x1 * x1).sum(0); s22 = (x2 * x2).sum(0); s12 = (x1 * x2).sum(0)
    s1y = (x1 * y).sum(0); s2y = (x2 * y).sum(0)
    det = s11 * s22 - s12 * s12
    g = (s22 * s1y - s12 * s2y) / det
    phi = (s11 * s2y - s12 * s1y) / det
    resid = y - g * x1 - phi * x2
    n = y.shape[0]
    sigma2 = (resid * resid).sum(0) / (n - 2)
    se_g = np.sqrt(sigma2 * s22 / det)
    tstat = g / se_g
    return np.array([mackinnonp(t, regression="c", N=2) if np.isfinite(t) else 1.0
                     for t in tstat])


# --------------------------------------------------------------------------- #
# all statistics for one fold
# --------------------------------------------------------------------------- #
def pair_statistics(px: np.ndarray, ia: np.ndarray, ib: np.ndarray, train_end: int,
                    variants: list[tuple[str, int, int]],
                    thresholds: dict[tuple[str, int, int], list[tuple[float, float]]]
                    ) -> dict:
    """Compute every formation-window statistic for the candidate pairs.

    px        (T, K) forward-filled prices of the fold (train + test)
    ia, ib    indices of leg A / leg B for each candidate pair (A < B)
    train_end first test bar; statistics use bars [0, train_end) only
    variants  (signal, fast, slow) combinations whose MCR/CCR are needed
    thresholds per variant, the (entry, exit) pairs for which CCR is needed

    Returns a dict with signal-independent statistics, and per variant the
    z-score over the WHOLE fold (for trading) plus MCR / CCR / half-life
    on the formation window.
    """
    tr = px[:train_end]
    la, lb = np.log(tr[:, ia]), np.log(tr[:, ib])
    lr = np.diff(np.log(tr), axis=0)
    lr = (lr - lr.mean(0)) / lr.std(0)
    corr = (lr[:, ia] * lr[:, ib]).mean(0)
    na, nb = tr[:, ia] / tr[0, ia], tr[:, ib] / tr[0, ib]
    out = dict(
        distance=((na - nb) ** 2).sum(0),
        corr=corr,
        coint_p=engle_granger_pvalue(la, lb),
        beta=hedge_ratio(la, lb),
        variants={},
    )
    pa_all, pb_all = px[:, ia], px[:, ib]
    for v in variants:
        sig, fast, slow = v
        s_all = spread(pa_all, pb_all, sig, out["beta"])
        z = zscore(s_all, sig, fast, slow)
        ztr = z[:train_end]
        d = dict(z=z, mcr=signal_crossing_rate(ztr), halflife=half_life(s_all[:train_end]),
                 ccr={})
        for (e, x) in thresholds.get(v, []):
            d["ccr"][(e, x)] = completed_cycle_rate(ztr, e, x)
        out["variants"][v] = d
    return out


# --------------------------------------------------------------------------- #
# ranks and selection
# --------------------------------------------------------------------------- #
def pct_rank(x: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Percentile rank in (0, 1], oriented so that larger = preferred.
    Ties receive their average rank; missing / non-finite values get the
    lowest possible score (0)."""
    s = pd.Series(np.where(np.isfinite(x), x, np.nan))
    r = s.rank(ascending=higher_is_better, method="average", pct=True)
    return r.fillna(0.0).to_numpy()


def criterion_scores(stats: dict, variant: tuple, entry: float, exit_: float) -> dict:
    """Percentile score Phi for every criterion (higher = better)."""
    v = stats["variants"][variant]
    rk = {
        "Distance": pct_rank(stats["distance"], False),
        "Correlation": pct_rank(stats["corr"], True),
        "Cointegration": pct_rank(stats["coint_p"], False),
        "HalfLife": pct_rank(v["halflife"], False),
        "MCR": pct_rank(v["mcr"], True),
        "CCR": pct_rank(v["ccr"][(entry, exit_)], True),
    }
    for h, (f1, f2) in C.HYBRIDS.items():
        rk[h] = 0.5 * rk[f1] + 0.5 * rk[f2]
    return rk


def select(score: np.ndarray, ia: np.ndarray, ib: np.ndarray, n: int,
           max_uses: int = 1) -> np.ndarray:
    """Greedy top-n selection; each security may appear in at most `max_uses`
    selected pairs (1 = no security used twice).  Ties in the score are broken
    by candidate order (deterministic); pairs with score 0 are never selected."""
    order = np.lexsort((np.arange(len(score)), -score))
    used: dict = {}
    chosen = []
    for j in order:
        if score[j] <= 0:
            break
        if used.get(ia[j], 0) >= max_uses or used.get(ib[j], 0) >= max_uses:
            continue
        chosen.append(j)
        used[ia[j]] = used.get(ia[j], 0) + 1; used[ib[j]] = used.get(ib[j], 0) + 1
        if len(chosen) == n:
            break
    return np.array(chosen, dtype=int)
