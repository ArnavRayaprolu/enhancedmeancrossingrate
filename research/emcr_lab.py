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
TRIALS = ROOT / "research" / "trials.csv"
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


def fold_stats(px_f, cost, ia, ib, des: Design):
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


def prepare(D, des: Design):
    key = (des.F, des.T, des.top_k, des.entry, des.exit, des.signal,
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
            & (pxf.median() >= 5)
        dv = (pxf * D["volume"].iloc[rows_f]).median()[ok.to_numpy()].sort_values(ascending=False)
        names = sorted(dv.index[:des.top_k])
        P = D["px"][names].iloc[:f["t1"]].ffill()
        pf = P.iloc[rows_f].bfill().to_numpy()
        pt = P.iloc[rows_t].to_numpy()                         # last formation day + test days
        st_sig = D["sig"][names].iloc[:f["t1"]].ffill().iloc[rows_t].to_numpy()
        st_sig = np.where(np.isfinite(st_sig), st_sig, pt)
        hf = D["high"][names].iloc[rows_f].ffill().bfill().to_numpy()
        lf = D["low"][names].iloc[rows_f].ffill().bfill().to_numpy()
        half = np.clip(MD.abdi_ranaldo(pf, hf, lf) / 2, 0.5e-4, 50e-4)
        ia, ib = np.triu_indices(len(names), 1)
        des0 = Design(entry=des.entry, exit=des.exit, params=des.params)
        if des.signal == "rolling":
            des0.params = dict(des.params, roll_entry=des.entry)
        stt = fold_stats(pf, half + 0.5e-4, ia, ib, des0)
        stt = stt[np.isfinite(stt.coint_p) & (stt.sd > 0)].reset_index(drop=True)
        res.append(dict(fold=f, names=names, pf=pf, pt=pt, sig=st_sig, half=half, stats=stt,
                        dev=f["test_start"] < HOLDOUT))
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
    n = len(ch)
    if des.weighting == "equal":
        w = np.where(pos != 0, 1.0 / max(n, 1), 0.0)
    else:
        w = E.pair_weights(pos, np.ones(n), "capped", 2.0 / max(n, 1))
    cost = des.cost_scale * fp["half"] + des.fixed_bps / 1e4
    res = E.run_book(pos, w, ra, rb, np.full(n, 0.5), cost[a], cost[b], des.borrow_pa / BPY)
    return res, ch


def evaluate(D, des: Design, part="dev", log=True):
    prep = prepare(D, des)
    nets, gross, turn = [], [], []
    for fp in prep:
        if (part == "dev") != fp["dev"] and part != "all":
            continue
        res, _ = run_fold(fp, des)
        nets.append(res["net"]); gross.append(res["gross"]); turn.append(res["turn"])
    net = np.concatenate(nets); g = np.concatenate(gross); u = np.concatenate(turn)
    sh = net.mean() / net.std() * np.sqrt(BPY) if net.std() > 0 else np.nan
    out = dict(label=des.label, part=part, criterion=des.criterion, sharpe=sh,
               ann_ret=net.mean() * BPY, ann_vol=net.std() * np.sqrt(BPY),
               gross_ret=g.mean() * BPY, turnover=u.mean() * BPY, total=np.prod(1 + net) - 1,
               days=len(net), design=str({k: v for k, v in asdict(des).items() if k != "label"}))
    if log:
        pd.DataFrame([out]).to_csv(TRIALS, mode="a", header=not TRIALS.exists(), index=False)
    return out, net


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
            x = sub[col].replace([np.inf, -np.inf], np.nan).to_numpy(float)
            ok = np.isfinite(x)
            rho = stats.spearmanr(x[ok], y[ok])[0] * (1 if hib else -1)
            rows.append(dict(fold=fp["fold"]["fold"], criterion=c, ic=rho, mean_y=y.mean()))
    df = pd.DataFrame(rows)
    g = df.groupby("criterion")["ic"]
    return pd.DataFrame({"ic": g.mean(), "t": g.mean() / g.std() * np.sqrt(g.count()),
                         "pos": g.apply(lambda s: (s > 0).mean())}).sort_values("t", ascending=False), df
