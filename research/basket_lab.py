"""EMCR basket stat-arb: each stock against its industry peer basket.

For every fold (walk-forward, 2016-2026, no separate test period):
  * universe: the 250 most liquid names (formation dollar volume) that have a GICS
    sub-industry with >= 3 members in the universe (else the GICS sector).
  * basket of stock i = equal-weight average of the daily log returns of the
    other members of its group; beta_i from formation OLS of r_i on r_basket.
  * residual level s_i,t = cumulative sum of (r_i,t - beta_i r_basket,t).
  * selection statistics on formation (per stock):
        EMCR      zero crossings of the demeaned residual level
        EMCR1     completed reversions after a 1-sigma excursion
        EMCRk     completed reversions of the Kalman-standardised residual
        R2        basket fit (analogue of correlation)
        ADF       Engle-Granger style ADF p-value of s_i (analogue of cointegration)
        SSD       sum of squared residual deviations (analogue of distance)
        HalfLife  AR(1) half-life of s_i
  * trading: z_i = (s_i - mu_form)/sd_form from the 15:00 price, executed at the
    16:00 close; long stock / short basket when z <= -entry, reverse at +entry,
    exit when z crosses back to exit; equal risk per selected stock.
  * costs: 2.5 bps per dollar traded on the stock leg and on the basket leg
    (basket trades are charged in full, no netting), 0.5%/yr borrow on shorts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcr import engine as E, signals as S  # noqa: E402
from research import emcr_lab as L  # noqa: E402

BPY = 252


def groups(names):
    sp = pd.read_csv(ROOT / "mcr" / "sp500_constituents.csv")
    sp["t"] = sp["Symbol"].str.replace(".", "-", regex=False)
    sub = dict(zip(sp.t, sp["GICS Sub-Industry"])); sec = dict(zip(sp.t, sp["GICS Sector"]))
    g = {}
    cnt = pd.Series([sub.get(n) for n in names]).value_counts()
    for n in names:
        if n not in sec:
            continue
        g[n] = sub[n] if cnt.get(sub[n], 0) >= 3 else sec[n]
    return g


def adf_p(s):
    """ADF (constant, 1 lag) p-value per column via the vectorised EG machinery."""
    from statsmodels.tsa.adfvalues import mackinnonp
    e = s - s.mean(0)
    de = np.diff(e, axis=0); y = de[1:]; x1 = e[1:-1]; x2 = de[:-1]
    s11 = (x1 * x1).sum(0); s22 = (x2 * x2).sum(0); s12 = (x1 * x2).sum(0)
    s1y = (x1 * y).sum(0); s2y = (x2 * y).sum(0); det = s11 * s22 - s12 * s12
    g = (s22 * s1y - s12 * s2y) / det; phi = (s11 * s2y - s12 * s1y) / det
    res = y - g * x1 - phi * x2
    t = g / np.sqrt((res * res).sum(0) / (len(y) - 2) * s22 / det)
    return np.array([mackinnonp(v, regression="c", N=1) if np.isfinite(v) else 1.0 for v in t])


def prepare(D, F=252, T=63, top_k=250):
    out = []
    dates = D["px"].index
    for f in L.folds(dates, F, T):
        rows_f = slice(f["f0"], f["f1"]); rows_t = slice(f["f1"] - 1, f["t1"])
        pxf = D["px"].iloc[rows_f]
        ok = D["instrument_ok"].reindex(pxf.columns).fillna(False).astype(bool) & \
            (pxf.notna().mean() >= 0.95) & (pxf.median() >= 5)
        dv = (pxf * D["volume"].iloc[rows_f]).median()[ok.to_numpy()].sort_values(ascending=False)
        names = list(dv.index[:top_k])
        g = groups(names)
        names = sorted(n for n in names if n in g and sum(1 for m in names if g.get(m) == g[n]) >= 3)
        P = D["px"][names].iloc[:f["t1"]].ffill()
        pf = P.iloc[rows_f].bfill().to_numpy(); pt = P.iloc[rows_t].to_numpy()
        sig = D["sig"][names].iloc[:f["t1"]].ffill().iloc[rows_t].to_numpy()
        sig = np.where(np.isfinite(sig), sig, pt)
        G = np.array([g[n] for n in names])
        M = (G[:, None] == G[None, :]).astype(float); np.fill_diagonal(M, 0)
        M = M / M.sum(1, keepdims=True)                     # row i: weights of i's peers
        rf = np.diff(np.log(pf), axis=0)
        rb = rf @ M.T                                        # basket returns
        beta = (rf * rb).sum(0) / (rb * rb).sum(0)
        res = rf - beta * rb
        s = np.vstack([np.zeros((1, len(names))), np.cumsum(res, axis=0)])
        mu, sd = s.mean(0), s.std(0)
        x = s - mu
        r2 = 1 - res.var(0) / rf.var(0)
        th0 = np.vstack([np.zeros(len(names)), np.ones(len(names))])
        Ek, SSk, _, _ = L.kalman_pair(s, np.zeros_like(s), 1e-7, th0, np.maximum(np.diff(s, axis=0).var(0), 1e-8))
        zk = (Ek / np.sqrt(SSk))[20:]
        st = pd.DataFrame(dict(
            EMCR=S.crossing_rate(x), EMCR1=L.amp_cycles(x, sd), EMCR2=L.amp_cycles(x, 2 * sd),
            EMCRk=L.amp_cycles(zk, np.full(len(names), 1.0)),
            R2=r2, ADF=adf_p(s), SSD=sd,                   # residual spread size (distance analogue)
            HalfLife=S.half_life(s), mu=mu, sd=sd, beta=beta))
        out.append(dict(fold=f, names=names, pf=pf, pt=pt, sig=sig, M=M, stats=st,
                        s_end=s[-1], lp_end=np.log(pf[-1])))
    return out


ORIENT = {"EMCR": True, "EMCR1": True, "EMCR2": True, "EMCRk": True, "R2": True,
          "ADF": False, "SSD": False, "HalfLife": False}


def score(st, crit):
    if "|" in crit:
        f, r = crit.split("|")
        fs = score(st, f); rs = score(st, r)
        return np.where(fs >= np.quantile(fs, 0.5), rs, 0.0)
    parts = crit.split("+")
    return np.mean([L.pct(st[p], ORIENT[p]) for p in parts], axis=0)


def run_fold(fp, crit, n_sel, entry, exit_, cost=2.5e-4, borrow=0.005):
    st = fp["stats"]; sc = score(st, crit)
    sel = np.argsort(-sc)[:n_sel]
    sel = sel[sc[sel] > 0]
    pt, sig, M = fp["pt"], fp["sig"], fp["M"]
    # residual level on test days from the 15:00 signal price: s_t = s_end + sum of
    # residual returns from the last formation close to today's 15:00 price
    lr_close = np.log(pt); lr_sig = np.log(sig)
    rb_close = np.diff(lr_close, axis=0) @ M.T
    beta = st["beta"].to_numpy()
    res_close = np.diff(lr_close, axis=0) - beta * rb_close
    cum = np.vstack([np.zeros((1, pt.shape[1])), np.cumsum(res_close, axis=0)])
    # intraday leg: from yesterday's close to today's 15:00 price
    d_sig = lr_sig - np.vstack([lr_close[:1], lr_close[:-1]])
    d_sig[0] = 0.0
    rb_sig = d_sig @ M.T
    s_sig = fp["s_end"] + np.vstack([cum[:1], cum[:-1]]) + (d_sig - beta * rb_sig)
    s_sig[0] = fp["s_end"]
    z = (s_sig[:, sel] - st["mu"].to_numpy()[sel]) / st["sd"].to_numpy()[sel]
    dec = E.states(z, entry, exit_, "revised")
    pos = dec[:-1]
    K = pos.shape[0]
    ra = pt[1:, sel] / pt[:-1, sel] - 1
    rbk = (pt[1:] / pt[:-1] - 1) @ M.T
    rb = rbk[:, sel] * beta[sel]
    n = max(len(sel), 1)
    iv = 1 / np.maximum(st["sd"].to_numpy()[sel], 1e-4)
    w = np.where(pos != 0, (iv / iv.sum())[None, :], 0.0)
    # book: stock leg weight w/2 * pos, basket leg -w/2 * pos (beta-scaled returns)
    res = E.run_book(pos, w, ra, rb, np.full(len(sel), 0.5), np.full(len(sel), cost),
                     np.full(len(sel), cost), borrow / BPY)
    return res


def evaluate(prep, crit, n_sel=100, entry=2.0, exit_=0.0, dates=None):
    nets, gr, tu, dts = [], [], [], []
    for fp in prep:
        r = run_fold(fp, crit, n_sel, entry, exit_)
        nets.append(r["net"]); gr.append(r["gross"]); tu.append(r["turn"])
        if dates is not None:
            dts.append(dates[fp["fold"]["f1"]:fp["fold"]["t1"]])
    net = np.concatenate(nets)
    out = dict(crit=crit, n=n_sel, entry=entry, exit=exit_,
               sharpe=net.mean() / net.std() * np.sqrt(BPY), ret=net.mean() * BPY,
               vol=net.std() * np.sqrt(BPY), gross=np.concatenate(gr).mean() * BPY,
               turn=np.concatenate(tu).mean() * BPY)
    s = pd.Series(net, index=np.concatenate(dts)) if dates is not None else None
    return out, s
