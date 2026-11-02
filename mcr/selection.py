"""Formation-window statistics for ALL candidate pairs, percentile ranks and
pair selection (R1 diagnostics, m3, m4).

Every criterion ranks the SAME candidate set in a fold (m4): the pairs of the
fold universe (all combinations, or same-sector combinations for the sp500
variant) whose statistics are finite. Selection is greedy down the ranking
and is run independently for each criterion; a security may appear in at most
`max_uses` selected pairs of that criterion's book.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import engine as E
from . import signals as S

CHUNK = 20000


def pair_index(tickers: list[str], sectors: dict | None, same_sector: bool):
    k = len(tickers)
    ia, ib = np.triu_indices(k, 1)
    if same_sector and sectors is not None:
        sec = np.array([sectors.get(t, "?") for t in tickers])
        keep = sec[ia] == sec[ib]
        ia, ib = ia[keep], ib[keep]
    return ia.astype(np.int32), ib.astype(np.int32)


def signal_matrix(pa, pb, cfg: C.Config, beta=None):
    if cfg.signal == "hedged":
        s = S.hedged_series(pa, pb, beta)
    else:
        s = S.ratio_series(pa, pb)
    return S.ma_signal(s, cfg.fast, cfg.slow)


def fold_statistics(px: np.ndarray, tr_end: int, ia: np.ndarray, ib: np.ndarray,
                    cfg: C.Config, diagnostics: bool = False, test_ret=None,
                    cost=None) -> pd.DataFrame:
    """Statistics on the formation rows [0, tr_end) of px (bars x names).

    distance  sum of squared differences of prices normalised to 1 (SSD)
    corr      correlation of bar log returns
    coint_p   Engle-Granger p-value (log prices, constant, 1 lag)
    beta      cointegrating slope (used by the hedged signal)
    halflife  AR(1) half-life of the log price ratio, bars
    mcr       zero-crossing rate of the traded signal z on formation bars
    With diagnostics=True (R1) also:
    ar1_z     lag-1 autocorrelation of z
    xr_coint  crossing rate of the cointegrating residual about its mean
    xr_ratio  crossing rate of the log ratio about its formation mean
    cycles    completed engine round trips on formation z, per 1000 bars
    and, if test_ret/cost are given, the pair's own out-of-sample outcome when
    traded alone: oos_net (sum of net returns, unit weight), oos_trades,
    oos_mcr (crossing rate of z in the test window).
    """
    tr = px[:tr_end]
    lr = np.diff(np.log(tr), axis=0)
    lr = (lr - lr.mean(0)) / np.where(lr.std(0) > 0, lr.std(0), np.nan)
    norm = tr / tr[0]
    out = []
    for s0 in range(0, len(ia), CHUNK):
        a, b = ia[s0:s0 + CHUNK], ib[s0:s0 + CHUNK]
        la, lb = np.log(tr[:, a]), np.log(tr[:, b])
        p, beta, e = S.engle_granger(la, lb)
        lratio = la - lb
        d = dict(a=a, b=b,
                 distance=((norm[:, a] - norm[:, b]) ** 2).sum(0),
                 corr=(lr[:, a] * lr[:, b]).mean(0),
                 coint_p=p, beta=beta,
                 halflife=S.half_life(lratio))
        if cfg.signal == "hedged":
            z = S.ma_signal(S.hedged_series(px[:, a], px[:, b], beta), cfg.fast, cfg.slow)
        else:
            z = S.ma_signal(S.ratio_series(px[:, a], px[:, b]), cfg.fast, cfg.slow)
        ztr = z[:tr_end]
        d["mcr"] = S.crossing_rate(ztr)
        if diagnostics:
            d["ar1_z"] = S.lag1_autocorr(ztr)
            d["xr_coint"] = S.crossing_rate(e - e.mean(0))
            d["xr_ratio"] = S.crossing_rate(lratio - lratio.mean(0))
            nz = np.isfinite(ztr).sum(0)
            d["cycles"] = 1000 * E.completed_cycles(ztr, cfg.entry, cfg.exit) / np.maximum(nz, 1)
            if test_ret is not None:
                st = E.states(z, cfg.entry, cfg.exit, "revised", start=tr_end - 1)[:-1]
                ra, rb = test_ret[:, a], test_ret[:, b]
                g = (st * (ra - rb) / 2).sum(0)
                dpos = np.abs(np.diff(np.vstack([np.zeros((1, st.shape[1])), st,
                                                 np.zeros((1, st.shape[1]))]), axis=0)).sum(0)
                c = (cost[a] + cost[b]) / 2 * dpos            # |dpos| x (1/2 per leg) x 2 legs
                d["oos_net"] = g - c
                prev = np.vstack([np.zeros((1, st.shape[1])), st[:-1]])
                d["oos_trades"] = ((prev != 0) & (st != prev)).sum(0) + (st[-1] != 0)
                d["oos_mcr"] = S.crossing_rate(z[tr_end:])
        out.append(pd.DataFrame(d))
    return pd.concat(out, ignore_index=True)


def pct_rank(x: pd.Series, higher_is_better: bool) -> pd.Series:
    """Percentile rank in (0, 1]: rank / number of finite values, with ties given
    their average rank and orientation such that LARGER = PREFERRED.  Missing or
    non-finite statistics get 0 and are never selected."""
    v = x.replace([np.inf, -np.inf], np.nan)
    if not higher_is_better:
        v = -v
    return v.rank(method="average", pct=True).fillna(0.0)


def scores(st: pd.DataFrame) -> dict[str, np.ndarray]:
    """Score Phi of every candidate under every criterion (m3, Eq. 7)."""
    rk = {m: pct_rank(st[col], hib).to_numpy() for m, (col, hib) in C.BASE.items()}
    for h, comp in C.HYBRIDS.items():
        both = (rk["MCR"] > 0) & (rk[comp] > 0)
        rk[h] = np.where(both, 0.5 * rk["MCR"] + 0.5 * rk[comp], 0.0)
    return rk


def select(score: np.ndarray, ia: np.ndarray, ib: np.ndarray, n: int,
           max_uses: int = 1) -> np.ndarray:
    """Top-n by score, walking down the ranking and skipping a pair if either
    security already appears max_uses times.  Exact ties in the score are
    broken by candidate order (alphabetical by ticker pair), so selection is
    deterministic.  Returns row indices into the candidate table."""
    order = np.lexsort((np.arange(len(score)), -score))
    used: dict[int, int] = {}
    chosen = []
    for j in order:
        if score[j] <= 0:
            break
        a, b = int(ia[j]), int(ib[j])
        if used.get(a, 0) >= max_uses or used.get(b, 0) >= max_uses:
            continue
        chosen.append(j)
        used[a] = used.get(a, 0) + 1; used[b] = used.get(b, 0) + 1
        if len(chosen) == n:
            break
    return np.array(chosen, dtype=int)
