"""EMCR research lab: daily walk-forward on the Alpaca 500-symbol universe.

Daily bars are built from the hourly data:
    sig   = price at 15:00 ET (close of the 14:00 bar)  -> the signal
    px    = 16:00 close (close of the 15:00 bar)         -> execution and returns
so every decision is taken one hour before the order is filled at the close
(no look-ahead, no bid-ask bounce between the signal and the fill).

DEV folds (test windows starting before 2023-01-01) are used for all design
choices; HOLDOUT folds (2023-01-01 onward) are touched once, at the end.
Every evaluated design is appended to research/trials.csv.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcr import data as MD, engine as E, signals as S  # noqa: E402

HOLDOUT = pd.Timestamp("2023-01-01").date()
TRIALS = ROOT / "research" / "trials_live.csv"   # ignored; snapshot to trials.csv when committing
CACHE = ROOT / "data" / "alpaca_daily_lab.pkl"
BPY = 252


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def daily_data(cache=True) -> dict:
    if cache and CACHE.exists():
        return pd.read_pickle(CACHE)
    d = MD.load()
    c, o, h, l, v = (d[k] for k in ("close", "open", "high", "low", "volume"))
    et = c.index.tz_convert("America/New_York")
    day = pd.Index(et.date, name="date"); hr = et.hour
    def pick(frame, hour):
        x = frame[hr == hour]; x.index = day[hr == hour]; return x[~x.index.duplicated()]
    sig = pick(c, 14)          # close of 14:00 bar = 15:00 price
    px = pick(c, 15)           # close of 15:00 bar = 16:00 close
    op = pick(o, 9)
    hi = h.groupby(day).max(); lo = l.groupby(day).min(); vol = v.groupby(day).sum()
    idx = px.index.intersection(sig.index)
    out = dict(sig=sig.reindex(idx).astype("float64"), px=px.reindex(idx).astype("float64"),
               open=op.reindex(idx).astype("float64"), high=hi.reindex(idx).astype("float64"),
               low=lo.reindex(idx).astype("float64"), volume=vol.reindex(idx).astype("float64"),
               instrument_ok=d["instrument_ok"], status=d["status"], sector=d["sector"])
    if cache:
        pd.to_pickle(out, CACHE)
    return out


def folds(dates, F=252, T=63):
    out, s, i = [], 0, 0
    while s + F + T <= len(dates):
        out.append(dict(fold=i, f0=s, f1=s + F, t1=s + F + T,
                        test_start=dates[s + F], test_end=dates[s + F + T - 1]))
        s += T; i += 1
    return out


# --------------------------------------------------------------------------- #
# design
# --------------------------------------------------------------------------- #
@dataclass
class Design:
    label: str = ""
    F: int = 252                 # formation days
    T: int = 63                  # trading days
    top_k: int = 250             # universe per fold (formation dollar volume)
    signal: str = "static"       # static: (lr - mu_F)/sd_F ; rolling: (lr - MA_w)/SD_w
    window: int = 20             # rolling window (days) for signal='rolling'
    entry: float = 2.0
    exit: float = 0.0
    stop: float = 0.0            # |z| stop (0 = none)
    max_hold: int = 0            # time stop in days (0 = none)
    n_pairs: int = 20
    max_uses: int = 1
    weighting: str = "equal"     # equal (committed capital 1/N) | capped
    criterion: str = "Distance"
    params: dict = field(default_factory=dict)
    cost_scale: float = 1.0      # x estimated half-spread
    fixed_bps: float = 0.5
    borrow_pa: float = 0.005


# --------------------------------------------------------------------------- #
# per-fold candidate statistics
# --------------------------------------------------------------------------- #
def amp_cycles(x: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Completed reversions: an excursion beyond +-thr (per column) followed by
    a crossing of zero. Returns the count per column."""
    T, n = x.shape
    armed = np.zeros(n, dtype=np.int8)      # +1 above, -1 below, 0 not armed
    cnt = np.zeros(n)
    for t in range(T):
        xt = x[t]
        cnt += ((armed == 1) & (xt <= 0)) | ((armed == -1) & (xt >= 0))
        armed = np.where(((armed == 1) & (xt <= 0)) | ((armed == -1) & (xt >= 0)), 0, armed)
        armed = np.where(xt >= thr, 1, np.where(xt <= -thr, -1, armed))
    return cnt


def rule_pnl(z: np.ndarray, dlr: np.ndarray, entry: float, exit_: float, cost_rt: np.ndarray,
             full: bool = False):
    """Formation backtest of a rule on z (states decided at t earn dlr[t+1]/2 per
    unit gross); cost_rt = cost per unit gross per round trip.
    full=True also returns per-trade statistics: number of profitable completed
    reversions (net of costs), number of losing ones, and the daily Sharpe."""
    st = E.states(z, entry, exit_, "revised")
    daily = st[:-1] * dlr[1:] / 2
    dpos = np.abs(np.diff(np.vstack([np.zeros((1, st.shape[1])), st, np.zeros((1, st.shape[1]))]),
                          axis=0))
    tc = cost_rt / 2 * dpos                       # cost when the position changes
    net = daily - tc[1:-1]
    pnl = daily.sum(0) - (cost_rt / 2 * dpos.sum(0))
    if not full:
        return pnl, (st != 0).mean(0)
    n = st.shape[1]
    cur = np.zeros(n); wins = np.zeros(n); losses = np.zeros(n)
    prev = np.zeros(n)
    for t in range(st.shape[0]):
        s_t = st[t]
        closing = (prev != 0) & (s_t != prev)
        wins += closing & (cur > 0); losses += closing & (cur <= 0)
        cur = np.where(closing, 0.0, cur)
        opening = (s_t != 0) & (s_t != prev)
        cur = cur - np.where(opening, cost_rt, 0.0)     # charge the round trip at entry
        if t + 1 < st.shape[0]:
            cur = cur + np.where(s_t != 0, s_t * dlr[t + 1] / 2, 0.0)
        prev = s_t
    open_end = prev != 0
    wins += open_end & (cur > 0); losses += open_end & (cur <= 0)
    sd = net.std(0)
    sh = np.where(sd > 0, net.mean(0) / np.where(sd > 0, sd, 1) * np.sqrt(BPY), 0.0)
    return pnl, (st != 0).mean(0), wins, losses, sh


def kalman_pair(ya: np.ndarray, xb: np.ndarray, delta: float = 1e-4, theta0=None, ve=None):
    """Kalman filter for y_t = alpha_t + beta_t x_t + e_t with random-walk
    coefficients (process noise delta/(1-delta) I), vectorised over pairs.
    Returns one-step prediction errors e_t, their variances S_t, and the final
    state (theta, P). Initial state: formation OLS (theta0), P = 1e-3 I."""
    T, n = ya.shape
    th = np.zeros((2, n)) if theta0 is None else theta0.copy()
    P = np.zeros((2, 2, n)); P[0, 0] = P[1, 1] = 1e-3
    vw = delta / (1 - delta)
    if ve is None:
        ve = np.full(n, 1e-4)
    E = np.empty((T, n)); SS = np.empty((T, n))
    for t in range(T):
        P[0, 0] += vw; P[1, 1] += vw
        h0 = 1.0; h1 = xb[t]
        yhat = th[0] + th[1] * h1
        e = ya[t] - yhat
        Ph0 = P[0, 0] * h0 + P[0, 1] * h1; Ph1 = P[1, 0] * h0 + P[1, 1] * h1
        Sv = h0 * Ph0 + h1 * Ph1 + ve
        K0 = Ph0 / Sv; K1 = Ph1 / Sv
        th[0] += K0 * e; th[1] += K1 * e
        P00 = P[0, 0] - K0 * Ph0; P01 = P[0, 1] - K0 * Ph1
        P10 = P[1, 0] - K1 * Ph0; P11 = P[1, 1] - K1 * Ph1
        P[0, 0], P[0, 1], P[1, 0], P[1, 1] = P00, P01, P10, P11
        E[t] = e; SS[t] = Sv
    return E, SS, th, P


def vwma(x: np.ndarray, w: np.ndarray, win: int) -> np.ndarray:
    """Volume-weighted trailing mean of x with weights w (rows <= t)."""
    cx = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x * w]), axis=0)
    cw = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), w]), axis=0)
    out = np.full_like(x, np.nan, dtype=float)
    num = cx[win:] - cx[:-win]; den = cw[win:] - cw[:-win]
    out[win - 1:] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out


def fold_stats(px_f, cost, ia, ib, des: Design, vol_f=None):
    """All candidate statistics on formation closes px_f (F x names)."""
    la = np.log(px_f); n = px_f.shape[0]
    r = np.diff(la, axis=0); r = (r - r.mean(0)) / r.std(0)
    norm = px_f / px_f[0]
    out = []
    for s0 in range(0, len(ia), 20000):
        a, b = ia[s0:s0 + 20000], ib[s0:s0 + 20000]
        lr = la[:, a] - la[:, b]
        mu, sd = lr.mean(0), lr.std(0)
        x = lr - mu
        p, beta, _ = S.engle_granger(la[:, a], la[:, b])
        c_rt = 2 * (cost[a] + cost[b])                # round trip, per unit gross (both legs in+out)
        d = dict(a=a, b=b, mu=mu, sd=sd, beta=beta,
                 distance=((norm[:, a] - norm[:, b]) ** 2).sum(0),
                 corr=(r[:, a] * r[:, b]).mean(0),
                 coint_p=p, halflife=S.half_life(lr),
                 zc=S.crossing_rate(x) * (n - 1),
                 cyc1=amp_cycles(x, 1.0 * sd), cyc2=amp_cycles(x, 2.0 * sd))
        # cost-aware cycles: excursion must exceed m x the round-trip cost in log-ratio
        # terms (a reversion of size D earns ~D/2 per unit gross, cost ~c_rt)
        m = des.params.get("cost_mult", 3.0)
        d["cycc"] = amp_cycles(x, np.maximum(1.0 * sd, m * 2 * c_rt))
        half = n // 2
        d["cyc1_min"] = np.minimum(amp_cycles(x[:half], 1.0 * sd), amp_cycles(x[half:], 1.0 * sd))
        z = x / sd
        pnl, tin = rule_pnl(z, np.vstack([np.zeros((1, len(a))), np.diff(lr, axis=0)]),
                            des.entry, des.exit, c_rt)
        d["fpnl"] = pnl; d["ftin"] = tin
        d["amp"] = sd                                   # spread volatility (log ratio)
        # variance ratio VR(5): < 1 = mean reverting at the weekly horizon
        d1 = np.diff(lr, axis=0); d5 = lr[5:] - lr[:-5]
        d["vr5"] = d5.var(0) / (5 * d1.var(0))
        # Kalman EMCR: completed reversions of the standardised prediction error
        if des.params.get("kalman", True):
            res_ols = (la[:, a] - la[:, a].mean(0)) - beta * (la[:, b] - la[:, b].mean(0))
            th0 = np.vstack([la[:, a].mean(0) - beta * la[:, b].mean(0), beta])
            E, SS, _, _ = kalman_pair(la[:, a], la[:, b], des.params.get("kdelta", 1e-7), th0,
                                      ve=np.maximum(res_ols.var(0), 1e-8))
            zk = E / np.sqrt(SS)
            zk = zk[20:]                                   # burn-in
            d["kzc"] = S.crossing_rate(zk)
            d["kcyc"] = amp_cycles(zk, np.full(len(a), 1.0))
            d["kar1"] = S.lag1_autocorr(zk)
        # moving-average EMCR (the paper's MCR on daily data) and its volume-weighted version
        f_, s_ = des.params.get("ma_fast", 5), des.params.get("ma_slow", 20)
        sdl = S.rolling_std(lr, s_); sdl[sdl <= 1e-12] = np.nan
        zma = (S.rolling_mean(lr, f_) - S.rolling_mean(lr, s_)) / sdl
        d["mazc"] = S.crossing_rate(zma)
        d["macyc"] = amp_cycles(np.nan_to_num(zma), np.full(len(a), 1.0))
        if vol_f is not None:
            wv = np.sqrt(np.maximum(vol_f[:, a] * px_f[:, a], 0) * np.maximum(vol_f[:, b] * px_f[:, b], 0)) + 1.0
            zvw = (vwma(lr, wv, f_) - vwma(lr, wv, s_)) / sdl
            d["vwzc"] = S.crossing_rate(zvw)
            d["vwcyc"] = amp_cycles(np.nan_to_num(zvw), np.full(len(a), 1.0))
        for w in des.params.get("roll_windows", (5, 10, 20)):
          with np.errstate(divide="ignore", invalid="ignore"):
              m_ = S.rolling_mean(lr, w); sd_ = S.rolling_std(lr, w)
              zr = np.empty_like(lr); zr[:] = np.nan
              zr[w:] = (lr[w:] - m_[w - 1:-1]) / sd_[w - 1:-1]     # vs previous w closes
              zr = zr[w:]
              d[f"zc{w}"] = S.crossing_rate(zr)
              d[f"cyc{w}"] = amp_cycles(np.nan_to_num(zr), np.full(len(a), 1.5))
              dl = np.vstack([np.zeros((1, len(a))), np.diff(lr[w:], axis=0)])
              pn, _, wn, ls, shp = rule_pnl(np.nan_to_num(zr), dl, des.params.get("roll_entry", 1.5),
                                            0.0, c_rt, full=True)
              d[f"fpnl{w}"] = pn
              d[f"win{w}"] = wn                         # profitable completed reversions
              d[f"wnet{w}"] = wn - ls                   # profitable minus losing reversions
              d[f"fsh{w}"] = shp                        # formation Sharpe of the rule
              d[f"rev{w}"] = -S.lag1_autocorr(np.diff(lr, axis=0)[w - 1:])  # daily spread-return reversal
        out.append(pd.DataFrame(d))
    return pd.concat(out, ignore_index=True)


def pct(x, hib=True):
    v = pd.Series(np.asarray(x, float)).replace([np.inf, -np.inf], np.nan)
    if not hib:
        v = -v
    return v.rank(pct=True, method="average").fillna(0).to_numpy()


CRIT_COLS = {  # base statistics: column, higher-is-better
    "Distance": ("distance", False), "Correlation": ("corr", True),
    "Cointegration": ("coint_p", False), "HalfLife": ("halflife", False),
    "EMCR": ("zc", True), "EMCR1": ("cyc1", True), "EMCR2": ("cyc2", True),
    "EMCRc": ("cycc", True), "EMCRstable": ("cyc1_min", True), "FormPnL": ("fpnl", True),
    "VR": ("vr5", False),
    "EMCRk": ("kzc", True), "EMCRkcyc": ("kcyc", True), "KalmanAR": ("kar1", False),
    "EMCRma": ("mazc", True), "EMCRmacyc": ("macyc", True),
    "EMCRvw": ("vwzc", True), "EMCRvwcyc": ("vwcyc", True),
}
for _w in (3, 5, 10, 20):
    CRIT_COLS[f"EMCR{_w}d"] = (f"zc{_w}", True)
    CRIT_COLS[f"EMCRcyc{_w}d"] = (f"cyc{_w}", True)
    CRIT_COLS[f"FormPnL{_w}d"] = (f"fpnl{_w}", True)
    CRIT_COLS[f"Rev{_w}d"] = (f"rev{_w}", True)
    CRIT_COLS[f"EMCRwin{_w}d"] = (f"win{_w}", True)
    CRIT_COLS[f"EMCRnet{_w}d"] = (f"wnet{_w}", True)
    CRIT_COLS[f"EMCRsh{_w}d"] = (f"fsh{_w}", True)


def score(st: pd.DataFrame, crit: str, params: dict) -> np.ndarray:
    """Criterion score; 'A+B+...' = average percentile rank; 'A|B' = filter on
    A's top q (params['q'], default 0.2) then rank by B."""
    if "|" in crit:
        f, r = crit.split("|")
        q = params.get("q", 0.2)
        fs = score(st, f, params)
        rs = score(st, r, params)
        return np.where(fs >= 1 - q, rs, 0.0)
    parts = crit.split("+")
    rk = [pct(st[CRIT_COLS[p][0]], CRIT_COLS[p][1]) for p in parts]
    s = np.mean(rk, axis=0)
    mn = params.get("min_zc", 0)
    if mn:
        s = np.where(st["zc"] >= mn, s, 0.0)
    return s


def select(sc, a, b, n, max_uses):
    order = np.lexsort((np.arange(len(sc)), -sc))
    used, ch = {}, []
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
# fold preparation (cached per (F, T, top_k))
# --------------------------------------------------------------------------- #
_PREP: dict = {}
BADPRINTS = [0]          # count of test-window one-day moves >= 3x set to zero


def prepare(D, des: Design):
    key = (D.get("market", "large_hourly"), des.params.get("rank_lo", 0), des.params.get("cost_tiers", False),
           des.params.get("prefilter", 0), des.params.get("min_price", 5.0),
           des.F, des.T, des.top_k, des.entry, des.exit, des.signal, des.params.get("half_bps", 2.0),
           des.params.get("same"), des.params.get("kdelta", 1e-7), des.params.get("ma_fast", 5),
           des.params.get("ma_slow", 20),
           des.params.get("cost_mult", 3.0), tuple(des.params.get("roll_windows", (5, 10, 20))))
    if key in _PREP:
        return _PREP[key]
    dates = D["px"].index
    res = []
    for f in folds(dates, des.F, des.T):
        rows_f = slice(f["f0"], f["f1"]); rows_t = slice(f["f1"] - 1, f["t1"])
        pxf = D["px"].iloc[rows_f]
        cov = pxf.notna().mean()
        ok = D["instrument_ok"].reindex(pxf.columns).fillna(False).astype(bool) & (cov >= 0.95) \
            & (pxf.median() >= des.params.get("min_price", 5.0))
        dv = (pxf * D["volume"].iloc[rows_f]).median()[ok.to_numpy()].sort_values(ascending=False)
        # data-quality screen: no one-day move of 3x or more (up or down) inside the
        # formation window (corporate-action / bad-print artefacts, e.g. AQB Jan 2017)
        lr_f = np.log(pxf.ffill()).diff().abs()
        clean = (lr_f.max() < np.log(3)).reindex(dv.index).fillna(False).to_numpy()
        dv = dv[clean]
        lo = des.params.get("rank_lo", 0)
        names = sorted(dv.index[lo:lo + des.top_k])
        if len(names) < 20:
            continue
        P = D["px"][names].iloc[:f["t1"]].ffill()
        pf = P.iloc[rows_f].bfill().to_numpy()
        pt = P.iloc[rows_t].to_numpy()                         # last formation day + test days
        st_sig = D["sig"][names].iloc[:f["t1"]].ffill().iloc[rows_t].to_numpy()
        st_sig = np.where(np.isfinite(st_sig), st_sig, pt)
        hf = D["high"][names].iloc[rows_f].ffill().bfill().to_numpy()
        lf = D["low"][names].iloc[rows_f].ffill().bfill().to_numpy()
        # flat half-spread for every name (daily Abdi-Ranaldo estimates for large caps
        # are dominated by volatility noise: median 9 bps, 2020 ~28 bps); 2 bps is the
        # typical effective half-spread of the 250 most liquid US stocks after 2015
        half = np.full(len(names), des.params.get("half_bps", 2.0) / 1e4)
        if des.params.get("cost_tiers", False):
            # one-way half-spread by formation median daily dollar volume (effective
            # spreads of US stocks/ETFs after 2015, conservative)
            mdv = dv.reindex(names).to_numpy()
            half = np.select([mdv > 2e8, mdv > 5e7, mdv > 2e7, mdv > 5e6],
                             [2e-4, 4e-4, 8e-4, 15e-4], 30e-4)
        ia, ib = np.triu_indices(len(names), 1)
        pre = des.params.get("prefilter", 0)
        if pre and len(ia) > pre:
            # keep the `pre` most correlated pairs (daily log returns, formation only)
            r = np.diff(np.log(pf), axis=0)
            r = (r - r.mean(0)) / np.where(r.std(0) > 0, r.std(0), 1)
            cm = (r.T @ r) / r.shape[0]
            cv = cm[ia, ib]
            top = np.argpartition(-cv, pre)[:pre]
            ia, ib = ia[top], ib[top]
            o = np.lexsort((ib, ia)); ia, ib = ia[o], ib[o]
        same = des.params.get("same")
        if same:
            sp = pd.read_csv(ROOT / "mcr" / "sp500_constituents.csv")
            col = {"sector": "GICS Sector", "subind": "GICS Sub-Industry"}[same]
            grp = dict(zip(sp["Symbol"].str.replace(".", "-", regex=False), sp[col]))
            g = np.array([grp.get(t, f"_none_{i}") for i, t in enumerate(names)])
            keep = g[ia] == g[ib]
            ia, ib = ia[keep], ib[keep]
        des0 = Design(entry=des.entry, exit=des.exit, params=des.params)
        if des.signal == "rolling":
            des0.params = dict(des.params, roll_entry=des.entry)
        VV = D["volume"][names].iloc[:f["t1"]].fillna(0.0)
        vf = VV.iloc[rows_f].to_numpy(); vt = VV.iloc[rows_t].to_numpy()
        stt = fold_stats(pf, half + 0.5e-4, ia, ib, des0, vol_f=vf)
        stt = stt[np.isfinite(stt.coint_p) & (stt.sd > 0)].reset_index(drop=True)
        res.append(dict(fold=f, names=names, pf=pf, pt=pt, sig=st_sig, half=half, stats=stt, vf=vf, vt=vt,
                        dev=pd.Timestamp(f["test_start"]) < pd.Timestamp(HOLDOUT)))
    _PREP[key] = res
    return res


# --------------------------------------------------------------------------- #
# trading
# --------------------------------------------------------------------------- #
def z_test(fp, a, b, des, mu, sd):
    """z on test days from the 15:00 signal price; row 0 = last formation day."""
    lr_sig = np.log(fp["sig"][:, a]) - np.log(fp["sig"][:, b])
    if des.signal == "static":
        return (lr_sig - mu) / sd
    if des.signal == "kalman":
        # run the filter on formation closes, then each test day: z from today's
        # signal price with yesterday's state, then update with today's close
        la_f = np.log(fp["pf"][:, a]); lb_f = np.log(fp["pf"][:, b])
        beta = S.ols_beta(la_f, lb_f)
        th0 = np.vstack([la_f.mean(0) - beta * lb_f.mean(0), beta])
        res_ols = (la_f - la_f.mean(0)) - beta * (lb_f - lb_f.mean(0))
        ve = np.maximum(res_ols.var(0), 1e-8)
        _, _, th, P = kalman_pair(la_f[:-1], lb_f[:-1], des.params.get("kdelta", 1e-7), th0, ve)
        closes = np.vstack([fp["pf"][:-1], fp["pt"]])          # pt row 0 = last formation day
        F1 = fp["pf"].shape[0] - 1
        z = np.empty_like(lr_sig)
        vw = des.params.get("kdelta", 1e-7) / (1 - des.params.get("kdelta", 1e-7))
        for k in range(lr_sig.shape[0]):
            xs = np.log(fp["sig"][k, b]); ys = np.log(fp["sig"][k, a])
            Pp = P.copy(); Pp[0, 0] += vw; Pp[1, 1] += vw
            Sv = Pp[0, 0] + 2 * Pp[0, 1] * xs + Pp[1, 1] * xs * xs + ve
            z[k] = (ys - th[0] - th[1] * xs) / np.sqrt(Sv)
            _, _, th, P = _kstep(np.log(closes[F1 + k, a]), np.log(closes[F1 + k, b]), th, P, vw, ve)
        return z
    if des.signal == "vwma":
        f_, s_ = des.params.get("ma_fast", 5), des.params.get("ma_slow", 20)
        closes = np.vstack([fp["pf"], fp["pt"][1:]])
        vols = np.vstack([fp["vf"], fp["vt"][1:]])
        F = fp["pf"].shape[0]
        z = np.empty_like(lr_sig)
        for k in range(lr_sig.shape[0]):
            hist_p = closes[F - 1 + k - s_ + 1:F - 1 + k + 1].copy()
            hist_v = vols[F - 1 + k - s_:F - 1 + k].copy()     # volumes known before today's decision
            hist_p[-1] = fp["sig"][k]                            # today's signal price
            lr_h = np.log(hist_p[:, a]) - np.log(hist_p[:, b])
            w = np.sqrt(np.maximum(hist_v[:, a] * hist_p[:, a], 0) * np.maximum(hist_v[:, b] * hist_p[:, b], 0)) + 1.0
            fast = (lr_h[-f_:] * w[-f_:]).sum(0) / w[-f_:].sum(0)
            slow = (lr_h * w).sum(0) / w.sum(0)
            z[k] = (fast - slow) / lr_h.std(0, ddof=1)
        return z
    # rolling: mean/sd of the w daily closes up to the PREVIOUS day, applied to
    # today's 15:00 log ratio
    closes = np.vstack([fp["pf"], fp["pt"][1:]])
    lr_close = np.log(closes[:, a]) - np.log(closes[:, b])
    F = fp["pf"].shape[0]; w = des.window
    z = np.empty_like(lr_sig)
    for k in range(lr_sig.shape[0]):
        hist = lr_close[F - 1 + k - w:F - 1 + k]
        z[k] = (lr_sig[k] - hist.mean(0)) / hist.std(0)
    return z


def _kstep(y, x, th, P, vw, ve):
    """One Kalman update with observation (y, x); returns (None, None, th, P)."""
    P = P.copy(); P[0, 0] += vw; P[1, 1] += vw
    Ph0 = P[0, 0] + P[0, 1] * x; Ph1 = P[1, 0] + P[1, 1] * x
    Sv = Ph0 + x * Ph1 + ve
    e = y - th[0] - th[1] * x
    K0 = Ph0 / Sv; K1 = Ph1 / Sv
    th = np.vstack([th[0] + K0 * e, th[1] + K1 * e])
    P00 = P[0, 0] - K0 * Ph0; P01 = P[0, 1] - K0 * Ph1
    P10 = P[1, 0] - K1 * Ph0; P11 = P[1, 1] - K1 * Ph1
    P[0, 0], P[0, 1], P[1, 0], P[1, 1] = P00, P01, P10, P11
    return None, None, th, P


def states_ext(z, des):
    st = E.states(z, des.entry, des.exit, "revised")
    if des.stop > 0 or des.max_hold > 0:
        held = np.zeros(z.shape[1]); cur = np.zeros(z.shape[1]); blocked = np.zeros(z.shape[1], bool)
        for t in range(z.shape[0]):
            s = st[t].copy()
            new = (s != 0) & (s != cur)
            held = np.where(new, 0, np.where(s != 0, held + 1, 0))
            kill = np.zeros_like(new)
            if des.stop > 0:
                kill |= (s != 0) & (np.abs(z[t]) >= des.stop)
            if des.max_hold > 0:
                kill |= (s != 0) & (held >= des.max_hold)
            blocked = (blocked | kill) & ~((np.abs(z[t]) <= max(des.exit, 0.0) + 1e-9))
            s = np.where(blocked, 0, s)
            st[t] = s; cur = s
        # the underlying machine keeps its own state; blocked pairs re-arm after |z| <= exit
    return st


def run_fold(fp, des: Design):
    stt = fp["stats"]
    sc = score(stt, des.criterion, des.params)
    a_all, b_all = stt["a"].to_numpy(), stt["b"].to_numpy()
    ch = select(sc, a_all, b_all, des.n_pairs, des.max_uses)
    a, b = a_all[ch], b_all[ch]
    mu, sd = stt["mu"].to_numpy()[ch], stt["sd"].to_numpy()[ch]
    z = z_test(fp, a, b, des, mu, sd)                  # K+1 rows: last formation day + K test days
    dec = states_ext(z, des)                            # decision at 15:00 of each day
    pos = dec[:-1]                                      # executed at that day's close, earns next day
    K = pos.shape[0]
    pt = fp["pt"]
    ra = pt[1:, a] / pt[:-1, a] - 1; rb = pt[1:, b] / pt[:-1, b] - 1
    bad = lambda r: np.abs(np.log1p(r)) >= np.log(3)
    BADPRINTS[0] += int(bad(ra).sum() + bad(rb).sum())
    ra = np.where(bad(ra), 0.0, ra); rb = np.where(bad(rb), 0.0, rb)
    n = len(ch)
    if des.weighting == "equal":
        w = np.where(pos != 0, 1.0 / max(n, 1), 0.0)
    else:
        w = E.pair_weights(pos, np.ones(n), "capped", 2.0 / max(n, 1))
    cost = des.cost_scale * fp["half"] + des.fixed_bps / 1e4
    hedge_a = np.full(n, 0.5)
    if des.signal == "kalman":
        hedge_a = 1 / (1 + np.clip(stt["beta"].to_numpy()[ch], 0.2, 5.0))
    res = E.run_book(pos, w, ra, rb, hedge_a, cost[a], cost[b], des.borrow_pa / BPY)
    return res, ch


def evaluate(D, des: Design, part="dev", log=True):
    prep = prepare(D, des)
    nets, gross, turn, dts = [], [], [], []
    dates = D["px"].index
    for fp in prep:
        if (part == "dev") != fp["dev"] and part != "all":
            continue
        res, _ = run_fold(fp, des)
        nets.append(res["net"]); gross.append(res["gross"]); turn.append(res["turn"])
        dts.append(dates[fp["fold"]["f1"]:fp["fold"]["t1"]])
    net = np.concatenate(nets); g = np.concatenate(gross); u = np.concatenate(turn)
    sh = net.mean() / net.std() * np.sqrt(BPY) if net.std() > 0 else np.nan
    out = dict(label=des.label, part=part, criterion=des.criterion, sharpe=sh,
               ann_ret=net.mean() * BPY, ann_vol=net.std() * np.sqrt(BPY),
               gross_ret=g.mean() * BPY, turnover=u.mean() * BPY, total=np.prod(1 + net) - 1,
               days=len(net), design=str({k: v for k, v in asdict(des).items() if k != "label"}))
    if log:
        pd.DataFrame([out]).to_csv(TRIALS, mode="a", header=not TRIALS.exists(), index=False)
    return out, pd.Series(net, index=pd.Index(np.concatenate(dts), name="date"))


def pair_ic(D, des: Design, part="dev"):
    """Spearman IC of each statistic with every candidate pair's own test-window
    net return when traded alone with the design's rule (unit gross)."""
    prep = prepare(D, des)
    rows = []
    for fp in prep:
        if (part == "dev") != fp["dev"] and part != "all":
            continue
        stt = fp["stats"]
        idx = np.random.default_rng(fp["fold"]["fold"]).choice(len(stt), min(20000, len(stt)), False)
        sub = stt.iloc[idx]
        a, b = sub.a.to_numpy(), sub.b.to_numpy()
        z = z_test(fp, a, b, des, sub.mu.to_numpy(), sub.sd.to_numpy())
        dec = states_ext(z, des); pos = dec[:-1]
        pt = fp["pt"]
        ra = pt[1:, a] / pt[:-1, a] - 1; rb = pt[1:, b] / pt[:-1, b] - 1
        g = (pos * (ra - rb) / 2).sum(0)
        dpos = np.abs(np.diff(np.vstack([np.zeros((1, len(a))), pos, np.zeros((1, len(a)))]), axis=0)).sum(0)
        cost = des.cost_scale * fp["half"] + des.fixed_bps / 1e4
        y = g - (cost[a] + cost[b]) / 2 * dpos
        for c, (col, hib) in CRIT_COLS.items():
            if col not in sub:
                continue
            x = sub[col].replace([np.inf, -np.inf], np.nan).to_numpy(float)
            ok = np.isfinite(x)
            rho = stats.spearmanr(x[ok], y[ok])[0] * (1 if hib else -1)
            rows.append(dict(fold=fp["fold"]["fold"], criterion=c, ic=rho, mean_y=y.mean()))
    df = pd.DataFrame(rows)
    g = df.groupby("criterion")["ic"]
    return pd.DataFrame({"ic": g.mean(), "t": g.mean() / g.std() * np.sqrt(g.count()),
                         "pos": g.apply(lambda s: (s > 0).mean())}).sort_values("t", ascending=False), df
