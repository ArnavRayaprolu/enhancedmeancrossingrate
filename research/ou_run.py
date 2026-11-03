"""Walk-forward runner for the OU / Avellaneda-Lee pairs lab on hourly or 5-minute bars.

    python research/ou_run.py hourly subind     # 500-name hourly panel, same GICS sub-industry pairs
    python research/ou_run.py hourly all        # all pairs of the fold's 250 most liquid names
    python research/ou_run.py m5 subind|all     # 152-name 5-minute panel (2024 onward)

Signal = close of bar t; execution = open of bar t+1 (so the decision and the fill
are separated by the bar boundary and bid-ask bounce in the closes cannot be
harvested). Every fold is out of sample (walk-forward from the start of the data).
"""
from __future__ import annotations

import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import ou_lab as O  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "research" / "ou_results"


def load(dataset: str):
    if dataset == "hourly":
        from mcr import data as MD
        d = MD.load()
        close, op, vol = d["close"], d["open"], d["volume"]
        ok = d["instrument_ok"]
        bpd = 7
    else:
        parts = sorted((ROOT / "data" / "alpaca" / "m5").glob("*/g*.parquet"))
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df["symbol"] = df["symbol"].astype(str)
        t = pd.DatetimeIndex(df["t"]).tz_localize("UTC")
        hm = t.tz_convert("America/New_York")
        keep = ((hm.hour * 60 + hm.minute) >= 570) & ((hm.hour * 60 + hm.minute) < 960)
        df = df[keep].assign(t=t[keep]).drop_duplicates(["t", "symbol"], keep="last")
        close = df.pivot(index="t", columns="symbol", values="close").sort_index().astype("float64")
        op = df.pivot(index="t", columns="symbol", values="open").reindex_like(close).astype("float64")
        vol = df.pivot(index="t", columns="symbol", values="volume").reindex_like(close).astype("float64")
        cnt = close.notna().sum(axis=1)
        good = cnt >= 0.5 * cnt.median()
        close, op, vol = close[good], op[good], vol[good]
        ok = pd.Series(True, index=close.columns)
        bpd = 78
    sp = pd.read_csv(ROOT / "mcr" / "sp500_constituents.csv")
    sp["t"] = sp["Symbol"].str.replace(".", "-", regex=False)
    sub = dict(zip(sp.t, sp["GICS Sub-Industry"]))
    return dict(close=close, ex=op.shift(-1), vol=vol, ok=ok, sub=sub, bpd=bpd)


def folds(n_bars, F, T):
    out, s = [], 0
    while s + F + T <= n_bars:
        out.append((s, s + F, s + F + T)); s += T
    return out


_PREP: dict = {}


def prepare(D, des: O.OUDesign, same: str | None, top_k: int):
    key = (id(D), des.F, des.T, des.entry, des.stop, des.hl_mult, des.zmode, same, top_k)
    if key in _PREP:
        return _PREP[key]
    C, X, V = D["close"], D["ex"], D["vol"]
    res = []
    for f0, f1, t1 in folds(len(C), des.F, des.T):
        cf = C.iloc[f0:f1]
        okc = D["ok"].reindex(cf.columns).fillna(False).astype(bool) & (cf.notna().mean() >= 0.95) & \
            (cf.median() >= 5)
        lr = np.log(cf.ffill()).diff().abs()
        okc &= (lr.max() < np.log(3)).reindex(cf.columns).fillna(False)
        dv = (cf * V.iloc[f0:f1]).median()[okc.to_numpy()].sort_values(ascending=False)
        names = sorted(dv.index[:top_k])
        if same:
            names = [n for n in names if n in D["sub"]]
        if len(names) < 4:
            continue
        Cs = C[names].iloc[:t1].ffill()
        sig = Cs.iloc[f0:t1].bfill().to_numpy()
        ex = X[names].iloc[:t1].ffill().iloc[f1 - 1:t1]
        ex = ex.fillna(Cs.iloc[f1 - 1:t1]).to_numpy()
        ia, ib = np.triu_indices(len(names), 1)
        if same:
            g = np.array([D["sub"][n] for n in names])
            k = g[ia] == g[ib]; ia, ib = ia[k], ib[k]
        if len(ia) == 0:
            continue
        st = O.pair_stats(sig[:des.F], ia, ib, des)
        st = st[np.isfinite(st["Cointegration"])].reset_index(drop=True)
        res.append(dict(f=(f0, f1, t1), names=names, sig=sig, ex=ex, stats=st,
                        when=C.index[f1:t1]))
    _PREP[key] = res
    return res


def evaluate(D, des, same, top_k):
    prep = prepare(D, des, same, top_k)
    nets, gross, turn, idx = [], [], [], []
    for fp in prep:
        r, ch = O.run_fold(fp, des)
        nets.append(r["net"]); gross.append(r["gross"]); turn.append(r["turn"]); idx.append(fp["when"])
    net = np.concatenate(nets); g = np.concatenate(gross); u = np.concatenate(turn)
    s = pd.Series(net, index=np.concatenate(idx))
    daily = s.groupby(pd.DatetimeIndex(s.index).tz_convert("America/New_York").date).sum()
    sh = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    return dict(sharpe=sh, ann_ret=daily.mean() * 252, ann_vol=daily.std() * np.sqrt(252),
                gross_ret=g.sum() / max(len(daily), 1) * 252, turnover=u.sum() / max(len(daily), 1) * 252,
                total=float(np.prod(1 + net) - 1), days=len(daily)), daily


CRITS = ["Distance", "Correlation", "Cointegration", "HalfLife", "Kappa",
         "EMCR", "EMCRtrade", "EMCRsuccess", "EMCRscore", "EMCRspeed", "EMCRexcess",
         "EMCRscore+Cointegration", "EMCRscore+Correlation", "EMCRscore+HalfLife",
         "Correlation|EMCRscore", "Cointegration|EMCRscore", "EMCRtrade+EMCRsuccess+Kappa"]


def main(dataset, same_arg):
    same = None if same_arg == "all" else same_arg
    D = load(dataset)
    bpd = D["bpd"]
    print(f"{dataset}: {D['close'].shape}, {D['close'].index[0]} .. {D['close'].index[-1]}", flush=True)
    OUT.mkdir(exist_ok=True)
    if dataset == "hourly":
        designs = [dict(F=60 * bpd, T=10 * bpd), dict(F=120 * bpd, T=20 * bpd)]
        top_k = 250
    else:
        designs = [dict(F=10 * bpd, T=5 * bpd), dict(F=20 * bpd, T=5 * bpd)]
        top_k = 152
    rows = []; t0 = time.time()
    tag = f"{dataset}_{same_arg}"
    for dz in designs:
        for entry in (2.0, 1.5):
            for zmode in ("spread", "ou"):
                base = O.OUDesign(F=dz["F"], T=dz["T"], entry=entry, zmode=zmode, bars_per_year=252 * bpd)
                for c in CRITS:
                    for n in (10, 20, 50):
                        des = replace(base, criterion=c, n_pairs=n, max_uses=2 if n <= 20 else 4)
                        o, daily = evaluate(D, des, same, top_k)
                        rows.append(dict(dataset=dataset, same=same_arg, F_days=dz["F"] // bpd, T_days=dz["T"] // bpd,
                                         entry=entry, zmode=zmode, n=n, crit=c, **o))
                        daily.to_frame("net").to_parquet(OUT / f"daily_{tag}_F{dz['F']//bpd}_e{entry}_{zmode}_{c.replace('|','_I_')}_n{n}.parquet")
                print(f"{tag} F{dz['F']//bpd}d T{dz['T']//bpd}d e{entry} {zmode} done [{time.time()-t0:.0f}s]", flush=True)
                pd.DataFrame(rows).to_csv(OUT / f"scan_{tag}.csv", index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
