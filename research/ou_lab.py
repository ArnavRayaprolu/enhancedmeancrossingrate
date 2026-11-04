"""Cointegration / Ornstein-Uhlenbeck pairs lab with EMCR selection (any bar frequency).

Spread and z-score
    hedge ratio beta  : OLS of ln Y on ln X over a trailing window of W bars
    spread S_t        : ln Y_t - beta ln X_t
    z_t               : (S_t - mean of S) / std of S over the trailing W bars
    OU fit (AR(1))    : S_{t+1} = a + b S_t + e  ->  kappa = -ln b (per bar),
                        half-life HL = ln 2 / kappa, theta = a / (1-b),
                        sigma_eq = sd(e) / sqrt(1 - b^2)  (Avellaneda-Lee s-score
                        (S - theta) / sigma_eq is available as zmode='ou')
Trading rules (Avellaneda-Lee / Gatev style, decided at the close of bar t,
executed at the next execution price)
    z <= -entry : long the spread  (buy Y, sell beta-weighted X)
    z >= +entry : short the spread
    exit        : z crosses 0
    time stop   : position open for more than hl_mult x HL bars
    hard stop   : |z| >= stop; after a stop the pair is blocked until z crosses 0
EMCR family (all computed on the formation z of the spread)
    EMCR         zero crossings of z per year
    EMCRtrade    completed reversions from |z| >= entry back to 0 per year
    EMCRsuccess  share of |z| >= entry excursions that reach 0 within hl_mult x HL
                 bars without touching the stop
    EMCRscore    EMCRtrade x EMCRsuccess  (successful reversions per year)
    EMCRspeed    mean bars from |z| >= entry back to 0 (lower = better)
    EMCRreg      coefficient of variation of the gaps between crossings (lower = better)
    EMCRexcess   observed crossing rate / OU-implied rate arccos(b)/pi
Classic: Distance (SSD of normalised prices), Correlation, Cointegration (EG p),
HalfLife (OU), Kappa (OU speed).
Selection as in the original paper: percentile ranks (higher = better), hybrids
'A+B' = 50/50 average of the two percentile ranks, top-N pairs with no security
used more than max_uses times; capital rank-weighted across active pairs (Eq. 8).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcr import engine as E, signals as S  # noqa: E402


@dataclass
class OUDesign:
    label: str = ""
    F: int = 252                 # formation bars (also the trailing window W)
    T: int = 63                  # trading bars per fold
    entry: float = 2.0
    stop: float = 3.5
    hl_mult: float = 2.5
    zmode: str = "spread"        # spread: (S - mean)/std ; ou: Avellaneda-Lee s-score
    hedge: str = "ols"           # ols: beta from OLS of ln Y on ln X ; tls: total least squares
                                 # (symmetric in Y and X) ; mkt: both legs first stripped of the
                                 # equal-weight market factor, then OLS on the residual prices
    exit_long: float = 0.0       # close a long spread when z >= exit_long (A-L: -0.5)
    exit_short: float = 0.0      # close a short spread when z <= exit_short (A-L: +0.75)
    n_pairs: int = 20
    max_uses: int = 2
    criterion: str = "EMCRscore"
    cost: float = 2.5e-4         # per dollar traded per leg (half-spread + fees), flat model
    ticks: float = 0.0           # > 0: tick-based model, half-spread = ticks x $0.01 / (2 x price)
    fees_bps: float = 0.3        #      plus commission + regulatory fees per dollar traded
    borrow_pa: float = 0.005
    bars_per_year: float = 252.0
    weighting: str = "active"    # active: the paper's Eq. 8, w_i = Phi_i 1{active} / sum_j Phi_j 1{active}
                                 # equal: 1/N committed per selected pair
    params: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# formation statistics
# --------------------------------------------------------------------------- #
def ou_fit(s: np.ndarray):
    """AR(1) on each column: returns b, kappa, theta, sigma_eq, half-life (bars)."""
    x = s[:-1]; y = s[1:]
    xm, ym = x.mean(0), y.mean(0)
    b = ((x - xm) * (y - ym)).sum(0) / ((x - xm) ** 2).sum(0)
    a = ym - b * xm
    e = y - a - b * x
    ok = (b > 0) & (b < 1)
    kappa = np.where(ok, -np.log(np.where(ok, b, 0.5)), 0.0)
    theta = np.where(ok, a / np.where(ok, 1 - b, 1), s.mean(0))
    seq = np.where(ok, e.std(0) / np.sqrt(np.maximum(1 - b * b, 1e-12)), s.std(0))
    hl = np.where(kappa > 0, np.log(2) / np.where(kappa > 0, kappa, 1), np.inf)
    return b, kappa, theta, seq, hl


def excursion_stats(z: np.ndarray, entry: float, stop: float, maxbars: np.ndarray):
    """Walk each column of z: count excursions beyond +-entry, how many came back
    to 0 within maxbars without touching +-stop, and the bars they took."""
    T, n = z.shape
    state = np.zeros(n)                 # +1 above entry, -1 below, 0 none
    start = np.zeros(n); dead = np.zeros(n, bool)
    exc = np.zeros(n); succ = np.zeros(n); comp = np.zeros(n); dur = np.zeros(n)
    for t in range(T):
        zt = z[t]
        back = ((state == 1) & (zt <= 0)) | ((state == -1) & (zt >= 0))
        took = t - start
        ok = back & ~dead
        comp += back
        succ += ok & (took <= maxbars)
        dur += np.where(ok, took, 0.0)
        state = np.where(back, 0, state)
        dead = np.where(back, False, dead | ((state != 0) & (np.abs(zt) >= stop)))
        newp = (state == 0) & (zt >= entry); newn = (state == 0) & (zt <= -entry)
        exc += newp | newn
        start = np.where(newp | newn, t, start)
        state = np.where(newp, 1, np.where(newn, -1, state))
    return exc, succ, comp, np.where(comp > 0, dur / np.maximum(comp, 1), np.nan)


def crossing_gaps_cv(z: np.ndarray):
    sgn = np.sign(z)
    cv = np.full(z.shape[1], np.nan)
    for j in range(z.shape[1]):
        s = sgn[:, j]; s = s[s != 0]
        idx = np.flatnonzero(s[1:] != s[:-1])
        if len(idx) >= 3:
            g = np.diff(idx); cv[j] = g.std() / g.mean()
    return cv


def hedge_ratio(y, x, mode):
    if mode == "tls":
        # total least squares on demeaned series: slope of the first principal axis
        yc, xc = y - y.mean(0), x - x.mean(0)
        sxx, syy, sxy = (xc * xc).mean(0), (yc * yc).mean(0), (xc * yc).mean(0)
        return (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * np.where(np.abs(sxy) > 1e-12, sxy, 1e-12))
    xc = x - x.mean(0)
    return (xc * (y - y.mean(0))).sum(0) / (xc * xc).sum(0)


def market_strip(lp: np.ndarray, fit_rows: int | None = None) -> np.ndarray:
    """Remove each name's exposure to the equal-weight market (cross-sectional mean
    log return, known at each bar). Market betas are estimated on the first
    fit_rows rows only (the formation window), so nothing after it is used."""
    r = np.diff(lp, axis=0); m = r.mean(1, keepdims=True)
    k = r.shape[0] if fit_rows is None else max(fit_rows - 1, 2)
    b = (r[:k] * m[:k]).sum(0) / (m[:k] * m[:k]).sum()
    res = r - m * b
    return np.vstack([lp[:1], lp[:1] + np.cumsum(res, axis=0)])


def band_crossings(z: np.ndarray, band: float = 0.5) -> np.ndarray:
    """Crossings of the equilibrium counted with hysteresis: a crossing is a move
    from above +band to below -band or the reverse (small jiggles around 0 are ignored)."""
    state = np.zeros(z.shape[1]); cnt = np.zeros(z.shape[1])
    for t in range(z.shape[0]):
        zt = z[t]
        up = zt >= band; dn = zt <= -band
        cnt += ((state == -1) & up) | ((state == 1) & dn)
        state = np.where(up, 1, np.where(dn, -1, state))
    return cnt


def pair_stats(P: np.ndarray, ia, ib, des: OUDesign) -> pd.DataFrame:
    """Formation statistics for candidate pairs; P = formation prices (F x names)."""
    lp = np.log(P)
    lps = market_strip(lp) if des.hedge == "mkt" else lp
    years = P.shape[0] / des.bars_per_year
    r = np.diff(lp, axis=0); r = (r - r.mean(0)) / np.where(r.std(0) > 0, r.std(0), 1)
    norm = P / P[0]
    out = []
    for s0 in range(0, len(ia), 20000):
        a, b = ia[s0:s0 + 20000], ib[s0:s0 + 20000]
        y, x = lps[:, a], lps[:, b]
        p, beta, _ = S.engle_granger(y, x)
        if des.hedge == "tls":
            beta = hedge_ratio(y, x, "tls")
        spread = y - beta * x
        bb, kap, th, seq, hl = ou_fit(spread)
        if des.zmode == "ou":
            z = (spread - th) / seq
        else:
            z = (spread - spread.mean(0)) / spread.std(0)
        exc, succ, comp, dur = excursion_stats(z, des.entry, des.stop, des.hl_mult * hl)
        zc = S.crossing_rate(z) * (z.shape[0] - 1)
        implied = np.arccos(np.clip(bb, -1, 1)) / np.pi * (z.shape[0] - 1)
        d = dict(a=a, b=b, beta=beta,
                 Distance=((norm[:, a] - norm[:, b]) ** 2).sum(0),
                 Correlation=(r[:, a] * r[:, b]).mean(0),
                 Cointegration=p, HalfLife=hl, Kappa=kap,
                 EMCR=zc / years, EMCRtrade=comp / years,
                 EMCRsuccess=np.where(exc > 0, succ / np.maximum(exc, 1), 0.0),
                 EMCRspeed=dur, EMCRexcess=np.where(implied > 0, zc / implied, np.nan))
        d["EMCRscore"] = d["EMCRtrade"] * d["EMCRsuccess"]
        d["EMCRband"] = band_crossings(z, 0.5) / years                 # hysteresis crossings per year
        d["EMCRhl"] = zc * np.where(np.isfinite(hl), hl, 0) / z.shape[0]  # crossings per half-life
        out.append(pd.DataFrame(d))
    st = pd.concat(out, ignore_index=True)
    st["EMCRreg"] = np.nan
    return st


ORIENT = {"Distance": False, "Correlation": True, "Cointegration": False, "HalfLife": False,
          "Kappa": True, "EMCR": True, "EMCRtrade": True, "EMCRsuccess": True,
          "EMCRscore": True, "EMCRspeed": False, "EMCRexcess": True, "EMCRreg": False,
          "EMCRband": True, "EMCRhl": False}


def pct(x, hib):
    v = pd.Series(np.asarray(x, float)).replace([np.inf, -np.inf], np.nan)
    return (v if hib else -v).rank(pct=True, method="average").fillna(0).to_numpy()


def score(st, crit, q=0.5):
    if "|" in crit:
        f, r = crit.split("|")
        fs = score(st, f, q); rs = score(st, r, q)
        return np.where(fs >= np.quantile(fs[fs > 0], 1 - q) if (fs > 0).any() else 1, rs, 0.0)
    return np.mean([pct(st[c], ORIENT[c]) for c in crit.split("+")], axis=0)


def select(sc, a, b, n, max_uses):
    order = np.lexsort((np.arange(len(sc)), -sc)); used = {}; ch = []
    for j in order:
        if sc[j] <= 0:
            break
        if used.get(a[j], 0) >= max_uses or used.get(b[j], 0) >= max_uses:
            continue
        ch.append(j); used[a[j]] = used.get(a[j], 0) + 1; used[b[j]] = used.get(b[j], 0) + 1
        if len(ch) == n:
            break
    return np.array(ch, dtype=int)


# --------------------------------------------------------------------------- #
# trading
# --------------------------------------------------------------------------- #
def rolling_z(ly, lx, W, zmode, hedge="ols"):
    """Trailing-window beta, spread and z for every bar (rows >= W-1)."""
    T, n = ly.shape
    cs = lambda v: np.cumsum(np.vstack([np.zeros((1, n)), v]), axis=0)
    sx, sy, sxx, sxy = cs(lx), cs(ly), cs(lx * lx), cs(lx * ly)
    win = lambda c, t: c[t + 1] - c[t + 1 - W]
    z = np.full((T, n), np.nan); beta = np.full((T, n), np.nan)
    for t in range(W - 1, T):
        mx, my = win(sx, t) / W, win(sy, t) / W
        vx = win(sxx, t) / W - mx * mx; cxy = win(sxy, t) / W - mx * my
        bt = cxy / np.maximum(vx, 1e-12)
        if hedge == "tls":
            bt = hedge_ratio(ly[t - W + 1:t + 1], lx[t - W + 1:t + 1], "tls")
        sp = ly[t - W + 1:t + 1] - bt * lx[t - W + 1:t + 1]
        if zmode == "ou":
            _, _, th, seq, _ = ou_fit(sp)
            z[t] = (sp[-1] - th) / seq
        else:
            z[t] = (sp[-1] - sp.mean(0)) / np.maximum(sp.std(0), 1e-12)
        beta[t] = bt
    return z, beta


def rules(z, entry, stop, maxbars, exit_long=0.0, exit_short=0.0):
    """Position path (decided each bar) under the entry / zero-exit / time-stop /
    hard-stop rules. z: (K, n). Returns positions in {-1, 0, 1}."""
    K, n = z.shape
    st = np.zeros(n); held = np.zeros(n); blocked = np.zeros(n, bool)
    out = np.zeros((K, n))
    for t in range(K):
        zt = np.nan_to_num(z[t])
        crossed = ((st > 0) & (zt >= exit_long)) | ((st < 0) & (zt <= exit_short))
        st = np.where(crossed, 0, st)
        unblock = blocked & (np.abs(zt) <= 0.25)
        blocked = blocked & ~unblock
        kill = (st != 0) & ((np.abs(zt) >= stop) | (held >= maxbars))
        blocked = blocked | kill
        st = np.where(kill, 0, st)
        enter_l = (st == 0) & ~blocked & (zt <= -entry) & (zt > -stop)
        enter_s = (st == 0) & ~blocked & (zt >= entry) & (zt < stop)
        new = enter_l | enter_s
        st = np.where(enter_l, 1, np.where(enter_s, -1, st))
        held = np.where(new, 0, np.where(st != 0, held + 1, 0))
        out[t] = st
    return out


def run_fold(fp, des: OUDesign):
    st = fp["stats"]
    sc = score(st, des.criterion, des.params.get("q", 0.5))
    a_all, b_all = st["a"].to_numpy(), st["b"].to_numpy()
    ch = select(sc, a_all, b_all, des.n_pairs, des.max_uses)
    if len(ch) == 0:
        K = fp["ex"].shape[0] - 1
        z0 = np.zeros(K)
        return dict(net=z0, gross=z0, turn=z0, cost=z0, borrow=z0, exposure=z0), ch
    a, b = a_all[ch], b_all[ch]
    hl = st["HalfLife"].to_numpy()[ch]
    W = des.F
    sig = fp["sig"]                                   # (F + K, names): formation + test signal prices
    lsig = np.log(sig)
    if des.hedge == "mkt":
        lsig = market_strip(lsig, fit_rows=W)  # betas from formation rows only
    z, beta = rolling_z(lsig[:, a], lsig[:, b], W, des.zmode, des.hedge)
    zt = z[W - 1:]                                     # row 0 = last formation bar
    maxbars = np.where(np.isfinite(hl), des.hl_mult * hl, 1e9)
    dec = rules(zt, des.entry, des.stop, maxbars, des.exit_long, des.exit_short)
    pos = dec[:-1]                                     # executed at next execution price
    ex = fp["ex"]                                      # execution prices: row 0 = last formation bar
    ra = ex[1:, a] / ex[:-1, a] - 1; rb = ex[1:, b] / ex[:-1, b] - 1
    bad = lambda r: np.abs(np.log1p(r)) >= np.log(3)
    ra = np.where(bad(ra), 0.0, ra); rb = np.where(bad(rb), 0.0, rb)
    bt = np.clip(np.abs(st["beta"].to_numpy()[ch]), 0.2, 5.0)
    hedge_a = 1 / (1 + bt)
    n = len(ch)
    if des.weighting == "active":
        w = E.pair_weights(pos, sc[ch], "active", 1.0)      # Eq. 8 with the criterion scores Phi
    else:
        w = np.where(pos != 0, 1.0 / n, 0.0)
    if des.ticks > 0:
        p0 = fp["sig"][des.F - 1]                       # price at the end of formation
        cvec = des.ticks * 0.01 / (2 * p0) + des.fees_bps / 1e4
        ca, cb = cvec[a], cvec[b]
    else:
        ca, cb = np.full(n, des.cost), np.full(n, des.cost)
    res = E.run_book(pos, w, ra, rb, hedge_a, ca, cb, des.borrow_pa / des.bars_per_year)
    return res, ch
