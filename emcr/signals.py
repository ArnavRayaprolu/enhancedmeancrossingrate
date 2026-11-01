"""Trading signals and crossing statistics, vectorised over many pairs.

All arrays are (T, P): T bars, P pairs.  Every rolling window is trailing and
ends at bar t, so z[t] uses prices up to and including t only.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# rolling moments (trailing, NaN for the first w-1 bars)
# --------------------------------------------------------------------------- #
def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    x0 = x - x[:1]                                   # centre for numerical stability
    c = np.cumsum(x0, axis=0)
    out = np.full_like(x, np.nan, dtype=float)
    out[w - 1:] = c[w - 1:]
    out[w:] -= c[:-w]
    return out / w + x[:1]


def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    """Sample standard deviation (ddof = 1) over a trailing window of w bars."""
    x0 = x - x[:1]
    c1 = np.cumsum(x0, axis=0)
    c2 = np.cumsum(x0 * x0, axis=0)
    s1 = np.full_like(x, np.nan, dtype=float); s2 = np.full_like(x, np.nan, dtype=float)
    s1[w - 1:] = c1[w - 1:]; s2[w - 1:] = c2[w - 1:]
    s1[w:] -= c1[:-w]; s2[w:] -= c2[:-w]
    var = (s2 - s1 * s1 / w) / (w - 1)
    var[var <= 1e-18 * np.maximum(1.0, np.abs(s2) / w)] = np.nan
    return np.sqrt(var)


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #
def hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """OLS slope of log P_A on log P_B (with constant), per column."""
    xa = log_a - log_a.mean(axis=0); xb = log_b - log_b.mean(axis=0)
    return (xa * xb).sum(axis=0) / (xb * xb).sum(axis=0)


def spread(pa: np.ndarray, pb: np.ndarray, signal: str, beta: np.ndarray | None = None):
    """The series whose deviations are traded."""
    if signal in ("MAD", "LEVEL"):
        return pa / pb
    if signal == "HEDGED":
        return np.log(pa) - beta * np.log(pb)
    raise ValueError(signal)


def zscore(s: np.ndarray, signal: str, fast: int, slow: int) -> np.ndarray:
    """MAD:    (MA_fast(s) - MA_slow(s)) / SD_slow(s)
       LEVEL / HEDGED: (s - MA_slow(s)) / SD_slow(s)"""
    sd = rolling_std(s, slow)
    if signal == "MAD":
        return (rolling_mean(s, fast) - rolling_mean(s, slow)) / sd
    return (s - rolling_mean(s, slow)) / sd


def leg_weights(signal: str, beta: np.ndarray | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Dollar exposure of leg A and leg B for a +1 (long-spread) unit position.
    |a| + |b| = 1, i.e. one unit of pair capital is fully invested in the two legs.
    Ratio signals: equal dollars, a = +1/2, b = -1/2 (dollar neutral).
    Hedged signal: a = 1/(1+|beta|), b = -beta/(1+|beta|) (beta-weighted)."""
    if signal == "HEDGED":
        d = 1.0 + np.abs(beta)
        return 1.0 / d, -beta / d
    return np.full(n, 0.5), np.full(n, -0.5)


# --------------------------------------------------------------------------- #
# crossing statistics
# --------------------------------------------------------------------------- #
def signal_crossing_rate(z: np.ndarray) -> np.ndarray:
    """SCR: share of consecutive valid observations at which z changes sign.

    Valid = finite and non-zero. NaNs and exact zeros are dropped and the
    remaining observations are treated as consecutive, so a move from +0.3
    through an exact 0 to -0.2 counts as one crossing, and two consecutive
    zeros count as none.  SCR = (#sign changes) / (#valid - 1).
    """
    s = np.sign(z)
    s[~np.isfinite(z)] = 0.0
    T, P = s.shape
    out = np.full(P, np.nan)
    # carry the last non-zero sign forward so zeros/NaNs are skipped
    last = np.zeros(P); changes = np.zeros(P); valid = np.zeros(P)
    for t in range(T):
        st = s[t]
        nz = st != 0
        changes += nz & (last != 0) & (st != last)
        valid += nz
        last = np.where(nz, st, last)
    ok = valid >= 2
    out[ok] = changes[ok] / (valid[ok] - 1)
    return out


def lag1_autocorr(z: np.ndarray) -> np.ndarray:
    """Lag-one autocorrelation of each column over its finite observations."""
    out = np.full(z.shape[1], np.nan)
    for j in range(z.shape[1]):
        v = z[:, j]; v = v[np.isfinite(v)]
        if len(v) > 3:
            v = v - v.mean(); den = (v * v).sum()
            if den > 0:
                out[j] = (v[1:] * v[:-1]).sum() / den
    return out


def half_life(s: np.ndarray) -> np.ndarray:
    """AR(1) half-life of the spread level: ds_t = a + b s_{t-1} + e_t,
    HL = -ln 2 / ln(1 + b).  b >= 0 (no mean reversion) gives +inf."""
    y = np.diff(s, axis=0); x = s[:-1]
    xm = x - x.mean(axis=0); ym = y - y.mean(axis=0)
    b = (xm * ym).sum(axis=0) / (xm * xm).sum(axis=0)
    hl = np.full(s.shape[1], np.inf)
    ok = (b < 0) & (b > -1)
    hl[ok] = -np.log(2.0) / np.log1p(b[ok])
    hl[b <= -1] = 0.0
    return hl
