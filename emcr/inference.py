"""Performance metrics and inference.

Diebold-Mariano comparison of two trading strategies (precise definition)
------------------------------------------------------------------------
Let n^A_t, n^B_t be the net returns of strategies A and B over the same
out-of-sample bars, and take the loss to be the negative net return,
L_t = -n_t.  The loss differential is d_t = L^B_t - L^A_t = n^A_t - n^B_t.
    H0: E[d_t] = 0  (equal expected net return)
    H1: E[d_t] != 0 (two-sided; a positive statistic favours A)
DM = mean(d) / sqrt(LRV / T), LRV the Bartlett-kernel (Newey-West) long-run
variance; the Harvey-Leybourne-Newbold factor is applied and p-values use
t(T-1).  The inference therefore concerns the difference in MEAN net return
per bar, not Sharpe ratios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch.bootstrap import SPA, StationaryBootstrap, optimal_block_length
from scipy import stats
from scipy.optimize import brentq

EULER = 0.5772156649015329


# --------------------------------------------------------------------------- #
# performance metrics (conventions stated in the paper)
# --------------------------------------------------------------------------- #
def metrics(r: pd.Series, bars_per_year: float, exposure: pd.Series | None = None) -> dict:
    """All bars are included, flat bars contribute a zero return.
    Sharpe  = mean / sd * sqrt(bars per year)   (zero risk-free rate: the book
              is dollar neutral and self-financing)
    Sortino = mean / sqrt(mean(min(r,0)^2)) * sqrt(bars per year)
    MDD     = largest peak-to-trough fall of the compounded equity curve."""
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / bars_per_year
    sd = r.std(); dd = np.sqrt((np.minimum(r, 0) ** 2).mean())
    mdd = float((eq / eq.cummax() - 1).min())
    out = dict(total_return=float(eq.iloc[-1] - 1),
               cagr=float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0,
               ann_vol=float(sd * np.sqrt(bars_per_year)),
               sharpe=float(r.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else np.nan,
               sortino=float(r.mean() / dd * np.sqrt(bars_per_year)) if dd > 0 else np.nan,
               max_drawdown=mdd,
               skew=float(stats.skew(r)), kurtosis=float(stats.kurtosis(r, fisher=False)),
               n_bars=len(r))
    if exposure is not None:
        out["time_in_market"] = float((exposure.reindex(r.index) > 0).mean())
        out["avg_exposure"] = float(exposure.reindex(r.index).mean())
    return out


def breakeven(gross: np.ndarray, turn: np.ndarray, borrow: np.ndarray) -> dict:
    """One-way cost (per $ traded, per leg) at which the strategy breaks even.
    arithmetic: sum of net returns = 0  -> closed form
    compounded: terminal wealth = 1     -> root of prod(1 + g - b - k u) = 1"""
    g = gross - borrow
    ar = g.sum() / turn.sum() if turn.sum() > 0 else np.nan
    def f(k):
        x = g - k * turn
        return np.log1p(x).sum() if (x > -1).all() else -np.inf

    comp = np.nan
    if f(0.0) > 0:
        hi = max(ar, 1e-7)
        while f(hi) > 0 and hi < 1.0:
            hi *= 2
        if f(hi) <= 0:
            comp = brentq(lambda k: f(k) if np.isfinite(f(k)) else -1e9, 0.0, hi)
    return dict(breakeven_arith_bps=1e4 * ar if ar > 0 else np.nan,
                breakeven_comp_bps=1e4 * comp)


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #
def nw_lag(T: int) -> int:
    return max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))


def andrews_lag(d: np.ndarray) -> int:
    """Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel."""
    x = d - d.mean()
    rho = np.clip((x[1:] @ x[:-1]) / (x @ x), -0.97, 0.97)
    a1 = 4 * rho ** 2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return max(1, int(np.ceil(1.1447 * (a1 * len(d)) ** (1 / 3))))


def long_run_variance(d: np.ndarray, lag: int) -> float:
    x = d - d.mean(); T = len(x)
    v = x @ x / T
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / T
    return float(v)


def diebold_mariano(a: np.ndarray, b: np.ndarray, lag: int | str = "nw") -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]; T = len(d)
    L = {"nw": nw_lag(T), "nw2": 2 * nw_lag(T), "andrews": andrews_lag(d)}.get(lag, lag)
    lrv = long_run_variance(d, int(L))
    dm = d.mean() / np.sqrt(lrv / T) if lrv > 0 else np.nan
    hln = dm * np.sqrt((T - 1) / T)                       # HLN factor with h = 1
    p = 2 * (1 - stats.t.cdf(abs(hln), df=T - 1))
    return dict(dm=float(hln), p=float(p), lag=int(L), T=T, mean_diff=float(d.mean()))


def bootstrap_mean_pvalue(d: np.ndarray, reps: int = 2000, seed: int = 0) -> dict:
    """Two-sided stationary-bootstrap p-value for H0: E[d] = 0.  The series is
    re-centred at zero (imposing the null) and the expected block length is the
    Politis-White (2004) / Patton et al. (2009) automatic choice."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    b = float(optimal_block_length(d)["stationary"].iloc[0])
    b = max(1.0, b)
    bs = StationaryBootstrap(b, d - d.mean(), seed=seed)
    means = np.array([x[0][0].mean() for x in bs.bootstrap(reps)])
    return dict(p_boot=float((np.abs(means) >= abs(d.mean())).mean()), block=b)


def holm(p) -> np.ndarray:
    p = np.asarray(p, float); k = len(p); out = np.empty(k); run = 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min((k - i) * p[j], 1.0)); out[j] = run
    return out


# --------------------------------------------------------------------------- #
# data-snooping: White's Reality Check and Hansen's SPA
# --------------------------------------------------------------------------- #
def spa_test(diffs: pd.DataFrame, reps: int = 2000, seed: int = 0) -> dict:
    """diffs: (T, K) performance differentials d_k,t = n_k,t - n_bench(k),t.
    H0: max_k E[d_k] <= 0  (no searched strategy beats its benchmark).
    Returns Hansen's consistent SPA p-value and White's Reality Check
    (the 'upper' p-value of the same bootstrap)."""
    X = diffs.dropna(axis=0, how="any").to_numpy()
    b = float(np.median(optimal_block_length(X[:, :min(20, X.shape[1])])["stationary"]))
    spa = SPA(np.zeros(len(X)), -X, block_size=max(1, int(round(b))), reps=reps,
              bootstrap="stationary", studentize=True, seed=seed)
    spa.compute()
    pv = spa.pvalues
    return dict(spa_p=float(pv["consistent"]), rc_p=float(pv["upper"]),
                spa_lower_p=float(pv["lower"]), K=X.shape[1], T=X.shape[0], block=b)


def deflated_sharpe(r: np.ndarray, sr_trials: np.ndarray) -> dict:
    """Bailey and Lopez de Prado (2014).  r: per-period returns of the selected
    strategy; sr_trials: per-period Sharpe ratios of ALL trials considered."""
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    T = len(r); sr = r.mean() / r.std(ddof=1)
    N = len(sr_trials); V = np.var(sr_trials, ddof=1)
    sr0 = np.sqrt(V) * ((1 - EULER) * stats.norm.ppf(1 - 1 / N)
                        + EULER * stats.norm.ppf(1 - 1 / (N * np.e)))
    g3 = stats.skew(r); g4 = stats.kurtosis(r, fisher=False)
    den = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / den)
    psr = stats.norm.cdf(sr * np.sqrt(T - 1) / den)
    return dict(sr=float(sr), sr0=float(sr0), dsr=float(dsr), psr=float(psr), N=N, T=T)


def bootstrap_ci(r: np.ndarray, periods_per_year: float, reps: int = 2000,
                 seed: int = 0, level: float = 0.95) -> dict:
    """Stationary-bootstrap percentile intervals for the annualised Sharpe
    ratio and the total compounded return."""
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    b = max(1.0, float(optimal_block_length(r)["stationary"].iloc[0]))
    bs = StationaryBootstrap(b, r, seed=seed)
    sh, tr = [], []
    for x in bs.bootstrap(reps):
        y = x[0][0]
        s = y.std()
        sh.append(y.mean() / s * np.sqrt(periods_per_year) if s > 0 else 0.0)
        tr.append(np.prod(1 + y) - 1)
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    return dict(sharpe_lo=float(np.quantile(sh, q[0])), sharpe_hi=float(np.quantile(sh, q[1])),
                ret_lo=float(np.quantile(tr, q[0])), ret_hi=float(np.quantile(tr, q[1])),
                block=b)


# --------------------------------------------------------------------------- #
# pair-level information coefficients
# --------------------------------------------------------------------------- #
IC_FEATURES = {"mcr": True, "ccr": True, "halflife": False, "ar1": False,
               "coint_p": False, "distance": False, "corr": True}


def information_coefficients(ic: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Spearman rank correlation, fold by fold, between each formation-window
    statistic (oriented so that positive = the criterion works) and an
    out-of-sample outcome over ALL candidate pairs.  Folds have disjoint test
    windows, so the fold-level ICs are treated as independent draws."""
    rows = []
    for (sig, fold), g in ic.groupby(["signal", "fold"]):
        y = g[outcome]
        for feat, hib in IC_FEATURES.items():
            x = g[feat].replace([np.inf, -np.inf], np.nan)
            ok = x.notna() & y.notna()
            if ok.sum() < 10:
                continue
            rho = stats.spearmanr(x[ok], y[ok])[0] * (1 if hib else -1)
            rows.append(dict(signal=sig, fold=fold, feature=feat, ic=rho))
    df = pd.DataFrame(rows)
    out = df.groupby(["signal", "feature"])["ic"].agg(["mean", "std", "count"])
    out["t"] = out["mean"] / (out["std"] / np.sqrt(out["count"]))
    out["share_pos"] = df.groupby(["signal", "feature"])["ic"].apply(lambda s: (s > 0).mean())
    out["p"] = 2 * (1 - stats.t.cdf(out["t"].abs(), df=out["count"] - 1))
    return out.reset_index()
