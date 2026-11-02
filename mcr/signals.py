"""Signals and pair statistics, vectorised over pairs (columns).

All arrays are (T, n): T bars, n pairs. Windows are trailing, so row t uses
rows <= t only.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.tsa.adfvalues import mackinnonp


# --------------------------------------------------------------------------- #
# rolling moments
# --------------------------------------------------------------------------- #
def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    out = np.full_like(x, np.nan, dtype=float)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    """Sample standard deviation (ddof = 1), as pandas .rolling(w).std()."""
    m = rolling_mean(x, w)
    m2 = rolling_mean(x * x, w)
    var = (m2 - m * m) * w / (w - 1)
    return np.sqrt(np.clip(var, 0.0, None))


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #
def ratio_series(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """r_t = P_A / P_B, rescaled by its first value (z is scale invariant; the
    rescaling only protects the rolling variance from cancellation)."""
    r = pa / pb
    return r / r[0]


def hedged_series(pa: np.ndarray, pb: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Cointegrating spread s_t = log P_A - beta log P_B, beta from the
    formation-window OLS of the Engle-Granger first step (R7)."""
    return np.log(pa) - beta * np.log(pb)


def ma_signal(s: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """z_t = (MA_fast(s)_t - MA_slow(s)_t) / SD_slow(s)_t  (Eq. 2 of the paper).
    A zero crossing of z means the fast and slow moving averages of s cross;
    it is NOT the ratio crossing its own mean (R1, m1)."""
    sd = rolling_std(s, slow)
    sd[sd <= 1e-12] = np.nan
    return (rolling_mean(s, fast) - rolling_mean(s, slow)) / sd


# --------------------------------------------------------------------------- #
# crossing statistics (m2: explicit treatment of zeros and missing values)
# --------------------------------------------------------------------------- #
def crossing_rate(x: np.ndarray) -> np.ndarray:
    """Zero-crossing rate per column.

    Missing values are dropped. Exact zeros carry no sign: the sign of the last
    non-zero observation is carried forward through them, so +,0,- counts one
    crossing and +,0,+ counts none; consecutive zeros are a single gap. The
    count is divided by (number of non-missing observations - 1), i.e. the
    number of consecutive-observation transitions in which a crossing could
    occur. A column with fewer than two non-missing values gets 0.
    """
    s = np.sign(x)
    s[s == 0] = np.nan                                    # zeros / NaN -> no sign
    # forward-fill the last known sign down each column
    idx = np.where(np.isfinite(s), np.arange(len(s))[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    filled = s[idx, np.arange(s.shape[1])]
    changes = (filled[1:] * filled[:-1] < 0).sum(0)
    n = np.isfinite(x).sum(0)
    return np.where(n >= 2, changes / np.maximum(n - 1, 1), 0.0)


def lag1_autocorr(x: np.ndarray) -> np.ndarray:
    xc = x - np.nanmean(x, axis=0)
    num = np.nansum(xc[1:] * xc[:-1], axis=0)
    den = np.nansum(xc * xc, axis=0)
    return np.where(den > 0, num / den, np.nan)


def half_life(s: np.ndarray) -> np.ndarray:
    """Half-life (bars) of an AR(1) fitted by OLS: ds_t = a + b s_{t-1} + e.
    HL = -ln 2 / ln(1 + b); +inf when b >= 0 (no mean reversion)."""
    y = np.diff(s, axis=0)
    x = s[:-1] - s[:-1].mean(0)
    b = (x * (y - y.mean(0))).sum(0) / (x * x).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = -np.log(2) / np.log1p(b)
    return np.where((b < 0) & (b > -1), hl, np.inf)


# --------------------------------------------------------------------------- #
# Engle-Granger cointegration, vectorised (identical to statsmodels
# coint(log_a, log_b, trend='c', maxlag=1, autolag=None) column by column)
# --------------------------------------------------------------------------- #
_TGRID = np.linspace(-30.0, 10.0, 40001)
_PGRID = np.array([mackinnonp(t, regression="c", N=2) for t in _TGRID])


def ols_beta(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    xc = x - x.mean(0)
    return (xc * (y - y.mean(0))).sum(0) / (xc * xc).sum(0)


def engle_granger(log_a: np.ndarray, log_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (p-value, beta, residual e).  Step 1: log_a = c + beta log_b + e.
    Step 2: de_t = g e_{t-1} + phi de_{t-1} + u_t (no constant); the t-statistic
    on g is mapped to a p-value with MacKinnon (2010), N = 2, constant."""
    beta = ols_beta(log_a, log_b)
    e = (log_a - log_a.mean(0)) - beta * (log_b - log_b.mean(0))
    de = np.diff(e, axis=0)
    y = de[1:]; x1 = e[1:-1]; x2 = de[:-1]
    s11 = (x1 * x1).sum(0); s22 = (x2 * x2).sum(0); s12 = (x1 * x2).sum(0)
    s1y = (x1 * y).sum(0); s2y = (x2 * y).sum(0)
    det = s11 * s22 - s12 * s12
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (s22 * s1y - s12 * s2y) / det
        phi = (s11 * s2y - s12 * s1y) / det
        resid = y - g * x1 - phi * x2
        sigma2 = (resid * resid).sum(0) / (y.shape[0] - 2)
        t = g / np.sqrt(sigma2 * s22 / det)
    p = np.interp(np.where(np.isfinite(t), t, 10.0), _TGRID, _PGRID)
    return p, beta, e


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    return float(stats.spearmanr(x[ok], y[ok])[0])
