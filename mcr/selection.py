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
                    cfg: C.Config) -> pd.DataFrame:
    """Selection statistics of every candidate pair on the formation rows
    [0, tr_end) of px (bars x names); nothing after tr_end is read.

    distance  sum of squared differences of prices normalised to 1 (SSD)
    corr      correlation of bar log returns
    coint_p   Engle-Granger p-value (log prices, constant, 1 lag)
    beta      cointegrating slope (used by the hedged signal)
    halflife  AR(1) half-life of the log price ratio, bars
    mcr       zero-crossing rate of the traded signal z on formation bars
              (z needs `slow` bars of warm-up, so it is measured on the last
              tr_end - slow + 1 formation bars)
    """
    tr = px[:tr_end]
    lr = np.diff(np.log(tr), axis=0)
    sd = lr.std(0)
    lr = (lr - lr.mean(0)) / np.where(sd > 0, sd, np.nan)
    norm = tr / tr[0]
    out = []
    for s0 in range(0, len(ia), CHUNK):
        a, b = ia[s0:s0 + CHUNK], ib[s0:s0 + CHUNK]
        la, lb = np.log(tr[:, a]), np.log(tr[:, b])
        p, beta, _ = S.engle_granger(la, lb)
        d = dict(a=a, b=b,
                 distance=((norm[:, a] - norm[:, b]) ** 2).sum(0),
                 corr=(lr[:, a] * lr[:, b]).mean(0),
                 coint_p=p, beta=beta,
                 halflife=S.half_life(la - lb))
        if cfg.signal == "hedged":
            z = S.ma_signal(S.hedged_series(tr[:, a], tr[:, b], beta), cfg.fast, cfg.slow)
        else:
            z = S.ma_signal(S.ratio_series(tr[:, a], tr[:, b]), cfg.fast, cfg.slow)
        d["mcr"] = S.crossing_rate(z)
        out.append(pd.DataFrame(d))
    return pd.concat(out, ignore_index=True)


def diagnostics(px: np.ndarray, tr_end: int, st: pd.DataFrame, cfg: C.Config,
                cost: np.ndarray) -> pd.DataFrame:
    """R1: alternative mean-reversion measures and each pair's own
    out-of-sample outcome, for the candidate pairs in `st` (a random sample).

    ar1_z     lag-1 autocorrelation of z (formation)
    xr_coint  crossing rate of the Engle-Granger residual about its mean
    xr_ratio  crossing rate of the log price ratio about its formation mean
              (the literal 'mean crossing' reading of the name)
    cycles    completed engine round trips on formation z, per 1000 bars
    oos_net   net return of the pair traded alone at unit weight over the
              test window (revised engine, headline costs)
    oos_trades completed round trips in the test window (incl. liquidation)
    oos_mcr   crossing rate of z in the test window
    """
    a = st["a"].to_numpy(); b = st["b"].to_numpy()
    out = st.copy()
    tr = px[:tr_end]
    la, lb = np.log(tr[:, a]), np.log(tr[:, b])
    _, _, e = S.engle_granger(la, lb)
    lratio = la - lb
    z = S.ma_signal(S.ratio_series(px[:, a], px[:, b]), cfg.fast, cfg.slow)
    ztr = z[:tr_end]
    out["ar1_z"] = S.lag1_autocorr(ztr)
    out["xr_coint"] = S.crossing_rate(e - e.mean(0))
    out["xr_ratio"] = S.crossing_rate(lratio - lratio.mean(0))
    nz = np.isfinite(ztr).sum(0)
    out["cycles"] = 1000 * E.completed_cycles(ztr, cfg.entry, cfg.exit) / np.maximum(nz, 1)
    ret = px[tr_end:] / px[tr_end - 1:-1] - 1
    st_ = E.states(z, cfg.entry, cfg.exit, "revised", start=tr_end - 1)[:len(ret)]
    g = (st_ * (ret[:, a] - ret[:, b]) / 2).sum(0)
    dpos = np.abs(np.diff(np.vstack([np.zeros((1, st_.shape[1])), st_,
                                     np.zeros((1, st_.shape[1]))]), axis=0)).sum(0)
    out["oos_net"] = g - (cost[a] + cost[b]) / 2 * dpos   # |dpos| x 1/2 per leg x 2 legs
    prev = np.vstack([np.zeros((1, st_.shape[1])), st_[:-1]])
    out["oos_trades"] = ((prev != 0) & (st_ != prev)).sum(0) + (st_[-1] != 0)
    out["oos_mcr"] = S.crossing_rate(z[tr_end:])
    return out


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
