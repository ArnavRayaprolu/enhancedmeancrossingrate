"""Strategy / criterion search on the DEVELOPMENT period only.

Every trading window used here ends before HOLDOUT_START; trying to touch a
later window raises.  Every evaluated variant is appended to
results/dev_search/log.csv (the trial count feeds the deflated Sharpe ratio
and the reality-check tests reported in the paper).
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .criteria import engle_granger_pvalue, pct_rank, select
from .data import DATA, candidate_pairs, eligible, load, make_folds
from .engine import completed_cycle_rate, portfolio, positions_for_window, run_states
from .signals import half_life, hedge_ratio, leg_weights, rolling_mean, rolling_std, \
    signal_crossing_rate, spread, zscore

HOLDOUT_START = pd.Timestamp("2024-01-01", tz="UTC")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dev_search"
CACHE = ROOT / "results" / "cache"


# --------------------------------------------------------------------------- #
# development data, prepared once per (dataset, formation, trading length)
# --------------------------------------------------------------------------- #
_MEM: dict = {}


def dev_folds(ds_name: str, train: int = C.TRAIN_BARS, test: int = C.TEST_BARS,
              start: str | None = None, top_k: int | None = None,
              period: str = "dev") -> list[dict]:
    """period="dev": trading windows ending before HOLDOUT_START (the search data).
    period="holdout": trading windows starting on/after HOLDOUT_START (final exam)."""
    assert period in ("dev", "holdout")
    tag = "dev" if period == "dev" else "holdout"
    key = CACHE / f"{tag}_{ds_name}_{train}_{test}_{start}_{top_k}.pkl"
    if key in _MEM:
        return _MEM[key]
    if key.exists():
        _MEM[key] = pickle.load(open(key, "rb"))
        return _MEM[key]
    ds = C.DATASETS[ds_name]
    if top_k is not None:
        from dataclasses import replace
        ds = replace(ds, top_k=top_k)
    close, dvol, meta = load(ds)
    if start:
        close, dvol = close[close.index >= start], dvol[dvol.index >= start]
    fund = None
    fp = DATA / "binance_perp_funding" / "funding.parquet"
    if ds.source == "binance_perp_1h" and fp.exists():
        fund = pd.read_parquet(fp).astype(float).reindex(close.index).fillna(0.0)
    folds = []
    for f in make_folds(len(close), train, test):
        test_end = close.index[f["end"] - 1]
        test_start = close.index[f["train_end"]]
        if period == "dev" and test_end >= HOLDOUT_START:
            break
        if period == "holdout" and test_start < HOLDOUT_START:
            continue
        names = eligible(close, dvol, meta, ds, f)
        if len(names) < 4:
            continue
        blk = close.iloc[f["start"]:f["end"]][names].ffill()
        px = blk.to_numpy()
        ia, ib = candidate_pairs(names, meta, ds.pairs)
        tr = px[:train]
        la, lb = np.log(tr[:, ia]), np.log(tr[:, ib])
        lr = np.diff(np.log(tr), axis=0); lr = (lr - lr.mean(0)) / lr.std(0)
        na, nb = tr[:, ia] / tr[0, ia], tr[:, ib] / tr[0, ib]
        folds.append(dict(
            fold=f["fold"], idx=blk.index, train=train, names=names, px=px,
            rets=np.vstack([np.zeros((1, px.shape[1])), px[1:] / px[:-1] - 1.0]),
            fund=None if fund is None else
            fund.iloc[f["start"]:f["end"]].reindex(columns=names).fillna(0.0).to_numpy(),
            ia=ia, ib=ib, distance=((na - nb) ** 2).sum(0),
            corr=(lr[:, ia] * lr[:, ib]).mean(0), coint_p=engle_granger_pvalue(la, lb),
            beta=hedge_ratio(la, lb)))
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(folds, open(key, "wb"))
    _MEM[key] = folds
    return folds


# --------------------------------------------------------------------------- #
# a strategy = signal + criterion + trading rule + portfolio rule
# --------------------------------------------------------------------------- #
@dataclass
class Spec:
    criterion: str = "MCR"
    signal: str = "LEVEL"
    window: int = 140
    entry: float = 2.0
    exit: float = 0.0
    stop: float | None = None
    max_hold: int | None = None
    n_pairs: int = 5
    weighting: str = "active"
    coint_filter: float | None = None   # keep only pairs with EG p-value below this
    top_k: int = 40                      # universe: top-K coins by formation dollar volume
    max_uses: int = 1                    # max selected pairs per coin
    gate: int | None = None              # regime gate: lookback (bars) of the recent MCR
    params: dict = field(default_factory=dict)

    def label(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v not in (None, {}, [])}
        return json.dumps(d, sort_keys=True)


def _zcache(f: dict, signal: str, window: int, fast: int = 35):
    k = ("z", signal, window, fast)
    if k not in f:
        pa, pb = f["px"][:, f["ia"]], f["px"][:, f["ib"]]
        tr = slice(0, f["train"])
        if signal == "STATIC":
            # log price ratio around its FIXED formation-period mean (Gatev et al.)
            s = np.log(pa / pb)
            mu, sd = s[tr].mean(0), s[tr].std(0, ddof=1)
            z = (s - mu) / sd
            f[k] = dict(z=z.astype("float32"), mcr=signal_crossing_rate(z[tr]),
                        halflife=half_life(s[tr]), sd_rel=sd)
            return f[k]
        s = spread(pa, pb, signal, f["beta"])
        z = zscore(s, signal, fast, window)
        sd_rel = rolling_std(s, window) / np.abs(rolling_mean(s, window))
        f[k] = dict(z=z.astype("float32"), mcr=signal_crossing_rate(z[tr]),
                    halflife=half_life(s[tr]),
                    sd_rel=np.nanmean(sd_rel[tr], axis=0))
    return f[k]


def _ccr(f, sp: Spec):
    k = ("ccr", sp.signal, sp.window, sp.entry, sp.exit, sp.stop, sp.max_hold)
    if k not in f:
        z = _zcache(f, sp.signal, sp.window)["z"][:f["train"]]
        f[k] = completed_cycle_rate(z, sp.entry, sp.exit, stop=sp.stop, max_hold=sp.max_hold)
    return f[k]


def _form_backtest(f, sp: Spec, kappa: float):
    """In-sample (formation-window) net return per bar of each pair traded alone."""
    k = ("form", sp.signal, sp.window, sp.entry, sp.exit, sp.stop, sp.max_hold, kappa)
    if k not in f:
        zc = _zcache(f, sp.signal, sp.window)
        T = f["train"]
        w0 = 0 if sp.signal == "STATIC" else sp.window
        pos = positions_for_window(zc["z"], w0, T, sp.entry, sp.exit,
                                   stop=sp.stop, max_hold=sp.max_hold)
        a, b = leg_weights(sp.signal, f["beta"], len(f["ia"]))
        r = f["rets"][w0:T]
        unit = a[None, :] * r[:, f["ia"]] + b[None, :] * r[:, f["ib"]]
        dpos = np.abs(np.diff(np.vstack([pos, np.zeros((1, pos.shape[1]))]), axis=0,
                              prepend=0)).sum(0)
        f[k] = ((pos * unit).sum(0) - kappa * dpos) / max(T - w0, 1)
    return f[k]


def scores(f: dict, sp: Spec, kappa: float) -> np.ndarray:
    """Selection score (higher = better) of every candidate pair for criterion sp.criterion."""
    zc = _zcache(f, sp.signal, sp.window)
    c = sp.criterion
    rk = lambda x, hi=True: pct_rank(x, hi)
    if c == "Distance":
        s = rk(f["distance"], False)
    elif c == "Correlation":
        s = rk(f["corr"], True)
    elif c == "Cointegration":
        s = rk(f["coint_p"], False)
    elif c == "HalfLife":
        s = rk(zc["halflife"], False)
    elif c == "MCR":
        mw = sp.params.get("mcr_window", sp.window)
        s = rk(_zcache(f, sp.signal, mw)["mcr"])
    elif c == "CCR":
        s = rk(_ccr(f, sp))
    elif c == "MCR+Coint":
        s = 0.5 * rk(zc["mcr"]) + 0.5 * rk(f["coint_p"], False)
    elif c == "CCR+Coint":
        s = 0.5 * rk(_ccr(f, sp)) + 0.5 * rk(f["coint_p"], False)
    elif c == "FormationReturn":
        s = rk(_form_backtest(f, sp, kappa))
    elif c == "PCR":
        # Profitable Crossing Rate: completed cycles per bar x expected net gain per
        # cycle, gain ~ 1/2 * entry * relative sd of the ratio, minus round-trip cost
        g = 0.5 * sp.entry * zc["sd_rel"] - 2 * kappa
        s = rk(_ccr(f, sp) * g)
    elif c == "MCR2":
        # two-horizon mean-crossing rate: crossings of the ratio through its fast
        # moving mean (tempo) and through its long-run formation mean (equilibrium)
        st = _zcache(f, "STATIC", 0)
        s = 0.5 * rk(zc["mcr"]) + 0.5 * rk(st["mcr"])
    elif c == "MCRstatic":
        s = rk(_zcache(f, "STATIC", 0)["mcr"])
    elif c == "MCR+HalfLife":
        s = 0.5 * rk(zc["mcr"]) + 0.5 * rk(zc["halflife"], False)
    elif c == "MCRxAmp":
        s = rk(zc["mcr"] * zc["sd_rel"])
    else:
        raise ValueError(c)
    if sp.coint_filter is not None:
        s = np.where(f["coint_p"] < sp.coint_filter, s, 0.0)
    return s


def evaluate(ds_name: str, sp: Spec, cost_bps: float | None = None, funding: bool = True,
             train: int = C.TRAIN_BARS, test: int = C.TEST_BARS, start: str | None = None,
             log: bool = True, note: str = "", period: str = "dev") -> dict:
    ds = C.DATASETS[ds_name]
    kappa = (cost_bps if cost_bps is not None else ds.cost_bps) / 1e4
    use_fund = funding and ds.source == "binance_perp_1h"
    borrow_bar = 0.0 if use_fund else ds.borrow_pa / ds.bars_per_year
    nets, gross, turn = [], [], []
    for f in dev_folds(ds_name, train, test, start, None if sp.top_k == 40 else sp.top_k,
                       period):
        te = slice(f["train"], len(f["idx"]))
        if period == "dev":
            assert f["idx"][-1] < HOLDOUT_START
        else:
            assert f["idx"][te][0] >= HOLDOUT_START
        sc = scores(f, sp, kappa)
        chosen = select(sc, f["ia"], f["ib"], sp.n_pairs, sp.max_uses)
        if len(chosen) == 0:
            nets.append(pd.Series(0.0, index=f["idx"][te])); continue
        zc = _zcache(f, sp.signal, sp.window)
        z = zc["z"][:, chosen]
        allow = None
        if sp.gate:
            # open new trades only while the pair's recent crossing rate (last
            # `gate` bars) is at least its formation-period crossing rate
            sg = np.sign(z); sg[~np.isfinite(z)] = 0
            ch = np.vstack([np.zeros((1, z.shape[1])), (sg[1:] * sg[:-1] < 0)]).astype(float)
            rec = pd.DataFrame(ch).rolling(sp.gate, min_periods=sp.gate).mean().to_numpy()
            allow = rec >= zc["mcr"][chosen][None, :]
        pos = positions_for_window(z, te.start, te.stop, sp.entry, sp.exit,
                                   stop=sp.stop, max_hold=sp.max_hold, allow=allow)
        a, b = leg_weights(sp.signal, f["beta"][chosen], len(chosen))
        ia, ib = f["ia"][chosen], f["ib"][chosen]
        r = f["rets"][te]
        fa = fb = None
        if use_fund and f["fund"] is not None:
            fa, fb = f["fund"][te][:, ia], f["fund"][te][:, ib]
        res = portfolio(pos, r[:, ia], r[:, ib], a, b, sc[chosen], sp.weighting, borrow_bar,
                        fa, fb)
        n = res["gross"] - kappa * res["turn"] - res["borrow"]
        nets.append(pd.Series(n, index=f["idx"][te]))
        gross.append(res["gross"].sum()); turn.append(res["turn"].sum())
    net = pd.concat(nets).sort_index()
    bpy = ds.bars_per_year * (C.TEST_BARS / test if False else 1)
    dly = net.groupby(net.index.floor("D")).sum()
    m = dict(dataset=ds_name, spec=sp.label(), cost_bps=kappa * 1e4, train=train, test=test,
             start=start, sharpe=float(net.mean() / net.std() * np.sqrt(ds.bars_per_year)),
             sharpe_daily=float(dly.mean() / dly.std() * np.sqrt(365)),
             total_return=float((1 + net).prod() - 1),
             ann_return=float(net.mean() * ds.bars_per_year),
             max_dd=float(((1 + net).cumprod() / (1 + net).cumprod().cummax() - 1).min()),
             gross_pa=float(np.sum(gross) / len(net) * ds.bars_per_year),
             turnover_pa=float(np.sum(turn) / len(net) * ds.bars_per_year),
             n_bars=len(net), first=str(net.index[0])[:10], last=str(net.index[-1])[:10],
             note=note, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    m["period"] = period
    if log and period == "dev":
        OUT.mkdir(parents=True, exist_ok=True)
        lp = OUT / "log.csv"
        pd.DataFrame([m]).to_csv(lp, mode="a", header=not lp.exists(), index=False)
    m["net"] = net
    return m
