"""
Requirements: pip install yfinance pandas numpy scipy statsmodels matplotlib

Data: Yahoo Finance 5-minute bars, last 60 calendar days (hard limit for 5m).
Around 4,500 bars/ticker.
Split: 70% formation, 30% trading (out of sample)

Trading engine details (identical across all methods)
  Entry: open when |z-score| crosses 2.2
  Exit: close when |z-score| crosses back through 0.75 (partial reversion)
  Stop-loss: close if |z-score| widens past (=3.5), then cooldown of 75 bars

Five selection methods:
1. Cointegration 
2. Correlation   
3. Distance       
4. MeanCross      (the proposed methodology)
5. Composite      Equal-weight rank combination of Cointegration + MeanCross

Enhanced MeanCross (novel contribution)
The raw zero-crossing rate MCR = #{sign changes} / (T-1) approximates κ/π
for a stationary OU process. This paper enhances it with:

  a) ADF stationarity pre-screen (p < 0.1)

  b) OU half-life 

  c) Spread R² quality filter

  d) Final ranking

Capital:
₹1,00,000 total
₹20,000/pair (5 pairs)
₹10,000/leg (dollar-neutral)
Shares = int(₹10,000 / price)
"""

import os
import time
import warnings
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
os.makedirs("results", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("results/run.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


CACHE_FILE     = Path("nifty100_5m.csv")
BARS_PER_DAY   = 75         
FORMATION_FRAC = 0.70       
TOP_N          = 5

Z_WINDOW  = 240   
ENTRY_Z   = 2.2   
EXIT_Z    = 0.75  
STOP_Z    = 3.5   
COOLDOWN  = 75    
MAX_HOLD  = 75    

ADF_P_MAX  = 0.10   
H_MAX      = 200    
R2_MIN     = 0.50  
MIN_LOT    = 3      

INITIAL_CAPITAL  = 100_000.0
LEG_CAPITAL      = 10_000.0
TC               = 0.0003
RISK_FREE_ANNUAL = 0.0679
BARS_PER_YEAR    = BARS_PER_DAY * 252

BETA_MIN = 0.05
BETA_MAX = 15.0   

TREND_WINDOW    = 150   
TREND_SLOPE_MAX = 0.008 

# NIFTY 100 TICKERS
NIFTY100 = [
    "HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","SBILIFE.NS","HDFCLIFE.NS","SBICARD.NS",
    "INDUSINDBK.NS","FEDERALBNK.NS","BANDHANBNK.NS","IDFCFIRSTB.NS","PFC.NS",
    "RECLTD.NS","CHOLAFIN.NS","MUTHOOTFIN.NS","LICHSGFIN.NS","ICICIGI.NS",
    "TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS",
    "LTIM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS",
    "RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS","HINDPETRO.NS",
    "GAIL.NS","PETRONET.NS","TATAPOWER.NS","NTPC.NS","POWERGRID.NS",
    "ADANIGREEN.NS","ADANIPORTS.NS","ADANIENT.NS",
    "MARUTI.NS","TMPV.NS","M&M.NS","BAJAJ-AUTO.NS","EICHERMOT.NS",
    "HEROMOTOCO.NS","TVSMOTOR.NS","ASHOKLEY.NS",
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","APOLLOHOSP.NS",
    "LUPIN.NS","AUROPHARMA.NS","TORNTPHARM.NS","ALKEM.NS",
    "HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","TATACONSUM.NS",
    "MARICO.NS","DABUR.NS","COLPAL.NS","GODREJCP.NS",
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","COALINDIA.NS",
    "NMDC.NS","NATIONALUM.NS","SAIL.NS",
    "ULTRACEMCO.NS","GRASIM.NS","AMBUJACEM.NS","ACC.NS",
    "LT.NS","SIEMENS.NS","ABB.NS","BHEL.NS","HAVELLS.NS",
    "POLYCAB.NS","CUMMINSIND.NS",
    "BHARTIARTL.NS","INDUSTOWER.NS",
    "DLF.NS","GODREJPROP.NS","OBEROIRLTY.NS","PHOENIXLTD.NS",
    "TITAN.NS","TRENT.NS","PAGEIND.NS","ASIANPAINT.NS","BERGER.NS",
    "PIDILITIND.NS","VOLTAS.NS",
    "NAUKRI.NS","ZOMATO.NS","PAYTM.NS","IRCTC.NS","CONCOR.NS",
    "INDIGO.NS","BEL.NS","HAL.NS","OFSS.NS","JIOFIN.NS",
]
NIFTY100 = list(dict.fromkeys(NIFTY100))


# DATA
def download_or_load() -> pd.DataFrame:
    """Download 5-min bars (last 60 days) from Yahoo Finance, or load cache."""
    if CACHE_FILE.exists():
        age_h = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
        if age_h < 23:
            log.info(f"Loading cached 5m data from {CACHE_FILE} ...")
            px = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
            log.info(f"  {px.shape[0]} bars × {px.shape[1]} tickers  "
                     f"({px.index[0]} → {px.index[-1]})")
            return px
        CACHE_FILE.unlink()

    log.info("Downloading 5m bars (last 60 days) from Yahoo Finance ...")
    log.info(f"  {len(NIFTY100)} tickers | interval=5m | period=60d")
    failed, frames = [], {}

    for i in range(0, len(NIFTY100), 10):
        batch = NIFTY100[i:i+10]
        log.info(f"  Batch {i//10+1}/{(len(NIFTY100)-1)//10+1}: "
                 f"{batch[0]} … {batch[-1]}")
        for ticker in batch:
            try:
                raw = yf.download(ticker, period="60d", interval="5m",
                                  auto_adjust=True, progress=False)
                if raw is None or len(raw) < 50:
                    failed.append(ticker); continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                close = raw["Close"].dropna()
                if close.index.tz is None:
                    close.index = close.index.tz_localize("UTC")
                close.index = close.index.tz_convert("Asia/Kolkata")
                t = close.index.time
                mkt = [(tt.hour == 9 and tt.minute >= 15) or
                       (10 <= tt.hour <= 14) or
                       (tt.hour == 15 and tt.minute <= 30) for tt in t]
                close = close[mkt]
                if len(close) < 50:
                    failed.append(ticker); continue
                close.index = close.index.tz_localize(None)
                frames[ticker] = close
            except Exception:
                failed.append(ticker)
            time.sleep(0.2)
        time.sleep(0.5)

    if not frames:
        raise RuntimeError("No data downloaded.")
    px = pd.DataFrame(frames).sort_index()
    px.ffill(inplace=True); px.bfill(inplace=True)
    px = px.loc[:, px.notna().mean() >= 0.80]
    px.to_csv(CACHE_FILE)
    log.info(f"  Saved {px.shape[0]} bars × {px.shape[1]} tickers "
             f"| failed: {len(failed)}")
    return px


# UTILITIES
def ols_beta_r2(y: np.ndarray, x: np.ndarray):
    if len(y) < 30:
        return 1.0, 0.0
    xc = add_constant(x)
    try:
        res = OLS(y, xc).fit()
        return float(res.params[1]), float(res.rsquared)
    except Exception:
        return 1.0, 0.0


def beta_ok(beta: float) -> bool:
    return BETA_MIN <= abs(beta) <= BETA_MAX


def log_beta_r2(p1, p2):
    return ols_beta_r2(p1, p2)


def ou_halflife(spread: np.ndarray) -> float:
    ds = np.diff(spread)
    sl = spread[:-1]
    xc = add_constant(sl)
    try:
        res  = OLS(ds, xc).fit()
        kappa = -float(res.params[1])
        return np.log(2) / kappa if kappa > 0 else np.inf
    except Exception:
        return np.inf


def rolling_zscore(spread: np.ndarray, window: int) -> np.ndarray:
    n = len(spread)
    z = np.full(n, np.nan)
    for i in range(window, n):
        w = spread[i - window:i]
        mu, sg = w.mean(), w.std(ddof=0)
        z[i] = (spread[i] - mu) / sg if sg > 1e-8 else 0.0
    return z


def zscore_trend_ok(spread: np.ndarray) -> bool:
    if len(spread) < TREND_WINDOW + Z_WINDOW:
        return True  
    tail_start = len(spread) - TREND_WINDOW - Z_WINDOW
    tail_spread = spread[tail_start:]
    z_full = rolling_zscore(tail_spread, Z_WINDOW)
    z_tail = z_full[Z_WINDOW:]  
    valid = z_tail[~np.isnan(z_tail)]
    if len(valid) < 20:
        return True
    t = np.arange(len(valid), dtype=float)
    tc = np.column_stack([np.ones_like(t), t])
    try:
        slope = float(np.linalg.lstsq(tc, valid, rcond=None)[0][1])
    except Exception:
        return True
    return abs(slope) <= TREND_SLOPE_MAX

# PAIR SELECTION
def select_cointegration(px: pd.DataFrame, top_n: int = TOP_N) -> list:
    rows = []
    for a, b in combinations(px.columns, 2):
        beta, _ = log_beta_r2(px[a].values, px[b].values)
        if not beta_ok(beta):
            continue
        spread = px[a].values - beta * px[b].values
        h = ou_halflife(spread)
        if h > H_MAX or not np.isfinite(h):
            continue
        _lp1 = np.log(np.maximum(px[a].values, 1e-9))
        _lp2 = np.log(np.maximum(px[b].values, 1e-9))
        try:
            _, pv, _ = coint(_lp1, _lp2,
                             trend="c", autolag="AIC")
        except Exception:
            continue
        if np.isfinite(pv):
            if not zscore_trend_ok(spread):
                continue
            rows.append((a, b, beta, float(pv), h))
    rows.sort(key=lambda x: x[3])
    log.info(f"  Cointegration: {len(rows)} candidates (h<{H_MAX}bars) → top {top_n}")
    for a, b, beta, pv, h in rows[:top_n]:
        log.info(f"    {a.replace('.NS',''):12s}/{b.replace('.NS',''):12s}  "
                 f"β={beta:+.4f}  p={pv:.5f}  h={h:.1f}bars")
    return [(a, b, beta) for a, b, beta, _, _ in rows[:top_n]]


def select_correlation(px: pd.DataFrame, top_n: int = TOP_N) -> list:
    """Rank by Pearson |ρ| of 5-min log-returns (descending)."""
    rets = px.pct_change().dropna(how="all").fillna(0)
    rows = []
    for a, b in combinations(px.columns, 2):
        beta, _ = log_beta_r2(px[a].values, px[b].values)
        if not beta_ok(beta):
            continue
        try:
            r, _ = pearsonr(rets[a].values, rets[b].values)
        except Exception:
            continue
        if np.isfinite(r):
            spread = px[a].values - beta * px[b].values
            if not zscore_trend_ok(spread):
                continue
            rows.append((a, b, beta, abs(float(r))))
    rows.sort(key=lambda x: x[3], reverse=True)
    log.info(f"  Correlation: {len(rows)} candidates → top {top_n}")
    for a, b, beta, rho in rows[:top_n]:
        log.info(f"    {a.replace('.NS',''):12s}/{b.replace('.NS',''):12s}  "
                 f"β={beta:+.4f}  |ρ|={rho:.4f}")
    return [(a, b, beta) for a, b, beta, _ in rows[:top_n]]


def select_distance(px: pd.DataFrame, top_n: int = TOP_N) -> list:
    """Rank by SSD of normalised cumulative log-returns (Gatev 2006)."""
    lp  = np.log(px.replace(0, np.nan)).ffill().bfill()
    cum = lp - lp.iloc[0]
    rows = []
    for a, b in combinations(px.columns, 2):
        beta, _ = log_beta_r2(px[a].values, px[b].values)
        if not beta_ok(beta):
            continue
        ssd = float(((cum[a] - cum[b]) ** 2).sum())
        spread = px[a].values - beta * px[b].values
        if not zscore_trend_ok(spread):
            continue
        rows.append((a, b, beta, ssd))
    rows.sort(key=lambda x: x[3])
    log.info(f"  Distance: {len(rows)} candidates → top {top_n}")
    for a, b, beta, ssd in rows[:top_n]:
        log.info(f"    {a.replace('.NS',''):12s}/{b.replace('.NS',''):12s}  "
                 f"β={beta:+.4f}  SSD={ssd:.4f}")
    return [(a, b, beta) for a, b, beta, _ in rows[:top_n]]


def select_meancross(px: pd.DataFrame, top_n: int = TOP_N) -> list:
    rows = []
    for a, b in combinations(px.columns, 2):
        beta, r2 = log_beta_r2(px[a].values, px[b].values)
        if not beta_ok(beta):
            continue
        if r2 < R2_MIN:
            continue
        spread = px[a].values - beta * px[b].values
        sg = spread.std()
        if sg < 1e-8:
            continue

        try:
            adf_p = adfuller(spread, autolag="AIC")[1]
        except Exception:
            continue
        if adf_p >= ADF_P_MAX:
            continue

        if not zscore_trend_ok(spread):
            continue

        h = ou_halflife(spread)
        if h > H_MAX or not np.isfinite(h):
            continue

        z   = (spread - spread.mean()) / sg
        mcr = float(np.sum(np.diff(np.sign(z)) != 0)) / max(len(z) - 1, 1)
        rows.append((a, b, beta, mcr, adf_p, h, r2))

    rows.sort(key=lambda x: x[3], reverse=True)

    log.info(f"  MeanCross (enhanced MCR ≈ κ/π, Rice 1944): "
             f"{len(rows)} passed all 3 screens → top {top_n}")
    log.info(f"    Screens: ADF p<{ADF_P_MAX}  OU h<{H_MAX}bars  R²≥{R2_MIN}")
    for a, b, beta, mcr, pv, h, r2 in rows[:top_n]:
        log.info(f"    {a.replace('.NS',''):12s}/{b.replace('.NS',''):12s}  "
                 f"β={beta:+.4f}  MCR={mcr:.5f}  "
                 f"ADF_p={pv:.4f}  h={h:.1f}bars  R²={r2:.3f}")

    if not rows:
        log.warning("  MeanCross: no pairs passed all screens — "
                    "falling back to Cointegration")
        return select_cointegration(px, top_n)

    return [(a, b, beta) for a, b, beta, *_ in rows[:top_n]]


def select_composite(px: pd.DataFrame, top_n: int = TOP_N) -> list:
    coint_rows = []
    for a, b in combinations(px.columns, 2):
        beta, _ = log_beta_r2(px[a].values, px[b].values)
        if not beta_ok(beta):
            continue
        _lp1 = np.log(np.maximum(px[a].values, 1e-9))
        _lp2 = np.log(np.maximum(px[b].values, 1e-9))
        try:
            _, pv, _ = coint(_lp1, _lp2, trend="c", autolag="AIC")
        except Exception:
            continue
        if np.isfinite(pv):
            spread_c = px[a].values - beta * px[b].values
            if not zscore_trend_ok(spread_c):
                continue
            coint_rows.append((a, b, float(pv)))

    mcr_rows = []
    for a, b in combinations(px.columns, 2):
        beta, r2 = log_beta_r2(px[a].values, px[b].values)
        if not beta_ok(beta) or r2 < R2_MIN:
            continue
        spread = px[a].values - beta * px[b].values
        sg = spread.std()
        if sg < 1e-8:
            continue
        try:
            adf_p = adfuller(spread, autolag="AIC")[1]
        except Exception:
            continue
        if adf_p >= ADF_P_MAX:
            continue
        if not zscore_trend_ok(spread):
            continue
        h = ou_halflife(spread)
        if h > H_MAX or not np.isfinite(h):
            continue
        z   = (spread - spread.mean()) / sg
        mcr = float(np.sum(np.diff(np.sign(z)) != 0)) / max(len(z) - 1, 1)
        mcr_rows.append((a, b, beta, mcr))

    mcr_dict  = {(r[0], r[1]): r[3] for r in mcr_rows}
    beta_dict = {(r[0], r[1]): r[2] for r in mcr_rows}

    combined = [
        {"a": a, "b": b, "beta": beta_dict[(a, b)],
         "coint_p": pv, "mcr": mcr_dict[(a, b)]}
        for a, b, pv in coint_rows if (a, b) in mcr_dict
    ]
    if not combined:
        log.warning("  Composite: no intersection — fallback to Cointegration")
        return select_cointegration(px, top_n)

    df = pd.DataFrame(combined)
    n  = len(df)
    df["coint_rank"] = (n - df["coint_p"].rank(ascending=True)) / max(n - 1, 1)
    df["mcr_rank"]   = (df["mcr"].rank(ascending=True) - 1)     / max(n - 1, 1)
    df["score"]      = 0.5 * df["coint_rank"] + 0.5 * df["mcr_rank"]
    df = df.sort_values("score", ascending=False).head(top_n)

    log.info(f"  Composite (50% Coint + 50% enhanced MCR): "
             f"{n} intersecting pairs → top {top_n}")
    for _, row in df.iterrows():
        log.info(f"    {row['a'].replace('.NS',''):12s}/{row['b'].replace('.NS',''):12s}  "
                 f"β={row['beta']:+.4f}  p={row['coint_p']:.5f}  "
                 f"MCR={row['mcr']:.5f}  score={row['score']:.4f}")
    return [(row["a"], row["b"], row["beta"]) for _, row in df.iterrows()]


# TRADING ENGINE — simple threshold 
def _make_trade(pos, sa, sb, epa, epb, cp1, cp2,
                ez, xz, hold, net, etype, ta, tb) -> dict:
    if pos == 1:
        l, s = ta, tb; lep, sep = epa, epb; lcp, scp = cp1, cp2; lq, sq = abs(sa), abs(sb)
    else:
        l, s = tb, ta; lep, sep = epb, epa; lcp, scp = cp2, cp1; lq, sq = abs(sb), abs(sa)
    return {
        "pnl": net, "hold": hold,
        "entry_z": round(ez, 3), "exit_z": round(xz, 3),
        "exit_type": etype, "direction": pos,
        "long_stock":      l,   "long_qty":        int(lq),
        "long_entry_px":   round(lep, 2),
        "long_entry_val":  round(lq * lep, 2),
        "long_pnl":        round(lq * (lcp - lep), 2),
        "short_stock":     s,   "short_qty":       int(sq),
        "short_entry_px":  round(sep, 2),
        "short_entry_val": round(sq * sep, 2),
        "short_pnl":       round(-sq * (scp - sep), 2),
    }


def backtest(p1: np.ndarray, p2: np.ndarray, beta: float,
             p1_warm: np.ndarray, p2_warm: np.ndarray,
             ta: str = "A", tb: str = "B") -> tuple:
    p1_full = np.concatenate([p1_warm, p1])
    p2_full = np.concatenate([p2_warm, p2])

    spread_full = p1_full - beta * p2_full
    z_full      = rolling_zscore(spread_full, Z_WINDOW)

    warm = len(p1_warm)
    z    = z_full[warm:]

    cash = float(LEG_CAPITAL * 2)
    n    = len(p1)
    eq   = np.full(n, np.nan)

    pos      = 0
    sa       = sb = epa = epb = ez = 0.0
    eb       = 0
    cooldown = 0
    trades   = []

    for i in range(n):
        eq[i] = cash + (sa*(p1[i]-epa) + sb*(p2[i]-epb)) if pos else cash

        zi = z[i]
        if np.isnan(zi):
            continue

        if pos and (i - eb) >= MAX_HOLD:
            pnl = sa*(p1[i]-epa) + sb*(p2[i]-epb)
            tc  = TC*(abs(sa)*p1[i] + abs(sb)*p2[i])
            net = pnl - tc
            cash += net; eq[i] = cash
            trades.append(_make_trade(pos,sa,sb,epa,epb,p1[i],p2[i],
                                      ez,zi,i-eb,net,"maxhold",ta,tb))
            pos = 0; sa = sb = 0.0
            continue

        if pos and abs(zi) >= STOP_Z:
            pnl = sa*(p1[i]-epa) + sb*(p2[i]-epb)
            tc  = TC*(abs(sa)*p1[i] + abs(sb)*p2[i])
            net = pnl - tc
            cash += net; eq[i] = cash
            trades.append(_make_trade(pos,sa,sb,epa,epb,p1[i],p2[i],
                                      ez,zi,i-eb,net,"stop",ta,tb))
            pos = 0; sa = sb = 0.0
            cooldown = COOLDOWN      
            continue

        exit_hit = (pos == 1 and zi >= EXIT_Z) or (pos == -1 and zi <= -EXIT_Z)
        if pos and exit_hit:
            pnl = sa*(p1[i]-epa) + sb*(p2[i]-epb)
            tc  = TC*(abs(sa)*p1[i] + abs(sb)*p2[i])
            net = pnl - tc
            cash += net; eq[i] = cash
            trades.append(_make_trade(pos,sa,sb,epa,epb,p1[i],p2[i],
                                      ez,zi,i-eb,net,"reversion",ta,tb))
            pos = 0; sa = sb = 0.0
            continue

        if not pos:
            if cooldown > 0:
                cooldown -= 1; continue
            if   zi >  ENTRY_Z: direction = -1
            elif zi < -ENTRY_Z: direction =  1
            else: continue

            qa = int(LEG_CAPITAL / max(p1[i], 1e-9))
            qb = int(LEG_CAPITAL / max(p2[i], 1e-9))
            if qa == 0 or qb == 0:
                continue

            sa =  direction * qa
            sb = -direction * qb
            tc  = TC*(qa*p1[i] + qb*p2[i])
            cash -= tc
            epa, epb = p1[i], p2[i]
            ez, eb, pos = zi, i, direction

    if pos:
        i   = n - 1
        pnl = sa*(p1[i]-epa) + sb*(p2[i]-epb)
        tc  = TC*(abs(sa)*p1[i] + abs(sb)*p2[i])
        net = pnl - tc
        cash += net; eq[i] = cash
        trades.append(_make_trade(pos,sa,sb,epa,epb,p1[i],p2[i],ez,
                                  float(z[i]) if not np.isnan(z[i]) else 0.0,
                                  i-eb,net,"end",ta,tb))

    eq = pd.Series(eq).ffill().bfill().values
    return eq, trades, z

# METRICS
def compute_metrics(eq: np.ndarray, trades: list) -> dict:
    eq = np.asarray(eq, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return dict(ret=0.,sharpe=0.,mdd=0.,wr=0.,pf=0.,
                    avg_hold=0.,n=0,n_rev=0,n_stop=0,n_mh=0,n_end=0)
    ret  = (eq[-1]/eq[0] - 1.0)*100.0
    br   = np.diff(eq)/np.where(eq[:-1]>0, eq[:-1], 1.0)
    act  = br[np.abs(br)>1e-9]
    if len(act) >= 5:
        rf   = RISK_FREE_ANNUAL / BARS_PER_YEAR
        sd   = float(np.std(act, ddof=1))
        shp  = float(np.mean(act-rf)/sd*np.sqrt(BARS_PER_YEAR)) if sd>1e-10 else 0.
    else:
        shp = 0.
    peak = np.maximum.accumulate(eq)
    mdd  = float(np.min((eq-peak)/np.where(peak>0,peak,1.)))*100.
    n_tr = len(trades); pnls = [t["pnl"] for t in trades]
    wr   = (sum(1 for p in pnls if p>0)/n_tr*100.) if n_tr else 0.
    gp   = sum(p for p in pnls if p>0)
    gl   = -sum(p for p in pnls if p<0)
    pf   = round(gp/gl,3) if gl>1e-8 else (999. if gp>0 else 0.)
    ah   = float(np.mean([t["hold"] for t in trades])) if n_tr else 0.
    return dict(
        ret=round(ret,3), sharpe=round(shp,3), mdd=round(mdd,3),
        wr=round(wr,2), pf=pf, avg_hold=round(ah,1), n=n_tr,
        n_rev=sum(1 for t in trades if t["exit_type"]=="reversion"),
        n_stop=sum(1 for t in trades if t["exit_type"]=="stop"),
        n_mh=sum(1 for t in trades if t["exit_type"]=="maxhold"),
        n_end=sum(1 for t in trades if t["exit_type"]=="end"),
    )


# RUN ONE METHOD
def run_method(name: str, pairs: list,
               px_trade: pd.DataFrame,
               px_form: pd.DataFrame,
               dates: pd.DatetimeIndex) -> dict:
    port_eq = None
    pair_results = []
    log.info(f"\n{'─'*62}\n  BACKTEST: {name}  "
             f"(₹{int(LEG_CAPITAL*2):,}/pair)\n{'─'*62}")

    for a, b, beta in pairs:
        if a not in px_trade.columns or b not in px_trade.columns:
            log.warning(f"    SKIP {a}/{b} — not in trading data"); continue

        p1_warm = px_form[a].values[-Z_WINDOW:] if a in px_form.columns else np.array([])
        p2_warm = px_form[b].values[-Z_WINDOW:] if b in px_form.columns else np.array([])
        eq, trades, z = backtest(px_trade[a].values, px_trade[b].values,
                                  beta, p1_warm, p2_warm, a, b)
        m = compute_metrics(eq, trades)
        log.info(
            f"    {a.replace('.NS',''):12s}/{b.replace('.NS',''):12s}  "
            f"β={beta:+.4f}  ret={m['ret']:+.2f}%  Sh={m['sharpe']:.2f}  "
            f"MDD={m['mdd']:.2f}%  WR={m['wr']:.1f}%  PF={m['pf']:.2f}  "
            f"T={m['n']} [rev={m['n_rev']} stop={m['n_stop']} mh={m['n_mh']} end={m['n_end']}]  "
            f"avgHold={m['avg_hold']:.1f}bars"
        )
        for ti, t in enumerate(trades, 1):
            log.info(
                f"      T{ti:03d}: LONG  {t['long_stock'].replace('.NS',''):12s} "
                f"{t['long_qty']:4d}sh @₹{t['long_entry_px']:8.2f} "
                f"(₹{t['long_entry_val']:7,.0f})  "
                f"SHORT {t['short_stock'].replace('.NS',''):12s} "
                f"{t['short_qty']:4d}sh @₹{t['short_entry_px']:8.2f} "
                f"(₹{t['short_entry_val']:7,.0f})  "
                f"z={t['entry_z']:+.2f}→{t['exit_z']:+.2f}  "
                f"hold={t['hold']}bars  PnL=₹{t['pnl']:+,.0f}  [{t['exit_type']}]"
            )

        pair_results.append({
            **m, "pair":f"{a}/{b}", "beta":round(beta,4),
            "equity":eq, "z":z, "trades":trades,
            "p1":px_trade[a].values, "p2":px_trade[b].values,
        })
        if port_eq is None:
            port_eq = eq.copy()
        else:
            ml = min(len(port_eq), len(eq))
            port_eq = port_eq[:ml] + eq[:ml]

    if port_eq is None:
        log.warning(f"  No results for {name}."); return {}

    all_trades = [t for pr in pair_results for t in pr["trades"]]
    pm = compute_metrics(port_eq, all_trades)
    log.info(
        f"\n  ▶ {name} PORTFOLIO  ret={pm['ret']:+.2f}%  "
        f"Sh={pm['sharpe']:.3f}  MDD={pm['mdd']:.2f}%  "
        f"WR={pm['wr']:.1f}%  PF={pm['pf']:.3f}  T={pm['n']}  "
        f"[rev={pm['n_rev']} stop={pm['n_stop']} mh={pm['n_mh']} end={pm['n_end']}]"
    )
    return {"name":name,"port_equity":port_eq,
            "port_metrics":pm,"pairs":pair_results}

# CHARTS
COLORS = {"Cointegration":"#2196F3","Correlation":"#F44336",
          "Distance":"#4CAF50","MeanCross":"#FF9800","Composite":"#9C27B0"}


def plot_equity(results: dict, dates: pd.DatetimeIndex):
    fig, ax = plt.subplots(figsize=(15, 6))
    for name, res in results.items():
        if not res: continue
        pm = res["port_metrics"]; n = min(len(res["port_equity"]), len(dates))
        ax.plot(dates[:n], res["port_equity"][:n],
                color=COLORS.get(name,"gray"), lw=1.5,
                label=f"{name}  ret={pm['ret']:+.2f}%  "
                      f"Sh={pm['sharpe']:.2f}  MDD={pm['mdd']:.2f}%")
    ax.axhline(INITIAL_CAPITAL, color="black", ls=":", lw=1.0, alpha=0.5)
    ax.set_title(
        f"Portfolio Equity — NIFTY 100 5-min  |  "
        f"Simple threshold engine (Gatev 2006)  |  Entry={ENTRY_Z}σ  "
        f"Exit={EXIT_Z}σ  Stop={STOP_Z}σ  Z={Z_WINDOW}bars",
        fontsize=10)
    ax.set_xlabel("Datetime"); ax.set_ylabel("Portfolio value (₹)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/equity_comparison.png", dpi=150); plt.close()
    log.info("  Saved equity_comparison.png")


def plot_summary(results: dict, trade_end: str):
    names  = [n for n,r in results.items() if r]
    colors = [COLORS.get(n,"gray") for n in names]
    x      = np.arange(len(names))
    def g(k): return [results[n]["port_metrics"][k] for n in names]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10)); axes = axes.flatten()

    def bar(ax, vals, title, ylabel, baseline=None):
        bs = ax.bar(x, vals, color=colors, edgecolor="white", alpha=0.85, width=0.5)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
        ax.set_title(title, fontsize=11); ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        if baseline is not None:
            ax.axhline(baseline, color="black", ls="--", lw=0.8, alpha=0.5)
        vm = max(abs(max(vals, default=0.01)), 0.01)
        for b_,v in zip(bs,vals):
            ax.text(b_.get_x()+b_.get_width()/2, b_.get_height()+vm*0.03,
                    f"{'+'if v>0 else''}{v:.2f}",
                    ha="center", fontsize=9, fontweight="bold")

    bar(axes[0], g("ret"),    "Total Return (%)",            "Return (%)",  0)
    bar(axes[1], g("sharpe"), "Sharpe Ratio (active bars)",  "Sharpe",      0)
    bar(axes[2], g("wr"),     "Win Rate (%)",                "WR (%)",      50)
    bar(axes[3], [min(v,10) for v in g("pf")],
                              "Profit Factor (cap 10)",       "PF",          1)
    bar(axes[4], g("mdd"),    "Max Drawdown (%)",             "MDD (%)")

    ax5 = axes[5]; w = 0.18
    ax5.bar(x-1.5*w,[results[n]["port_metrics"]["n_rev"]  for n in names],
            w, label="Reversion", color="#4CAF50", alpha=0.85)
    ax5.bar(x-0.5*w,[results[n]["port_metrics"]["n_stop"] for n in names],
            w, label="Stop-loss",  color="#F44336", alpha=0.85)
    ax5.bar(x+0.5*w,[results[n]["port_metrics"]["n_mh"]   for n in names],
            w, label="Max-hold",   color="#FF9800", alpha=0.85)
    ax5.bar(x+1.5*w,[results[n]["port_metrics"]["n_end"]  for n in names],
            w, label="End-of-period", color="#607D8B", alpha=0.85)
    ax5.set_xticks(x); ax5.set_xticklabels(names, fontsize=10)
    ax5.set_title("Exit Type Composition", fontsize=11)
    ax5.set_ylabel("Count"); ax5.legend(fontsize=9); ax5.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"5-Minute Pairs Trading — NIFTY 100  |  Trading ends {trade_end}\n"
        f"★ MeanCross (novel): enhanced MCR ≈ κ/π (Rice 1944)  "
        f"— ADF screen + OU half-life + R² filter\n"
        f"Engine: Entry={ENTRY_Z}σ  Exit={EXIT_Z}σ  Stop={STOP_Z}σ  "
        f"Z={Z_WINDOW}bars  ₹{int(LEG_CAPITAL*2):,}/pair  Top-{TOP_N} pairs/method",
        fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig("results/summary_bars.png", dpi=150, bbox_inches="tight")
    plt.close(); log.info("  Saved summary_bars.png")


def plot_heatmap(results: dict):
    names = [n for n,r in results.items() if r]
    all_pairs: list = []
    for n in names:
        for pr in results[n]["pairs"]:
            if pr["pair"] not in all_pairs: all_pairs.append(pr["pair"])
    mat = np.full((len(names), len(all_pairs)), np.nan)
    for i,n in enumerate(names):
        for pr in results[n]["pairs"]:
            mat[i, all_pairs.index(pr["pair"])] = pr["ret"]
    vmax = max(np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1., 1.)
    fig, ax = plt.subplots(
        figsize=(max(14, len(all_pairs)*1.4), max(5, len(names)*1.2)))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto",
                   norm=mc.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=10)
    ax.set_xticks(range(len(all_pairs)))
    ax.set_xticklabels([p.replace(".NS","") for p in all_pairs],
                       rotation=45, ha="right", fontsize=7)
    for i in range(len(names)):
        for j in range(len(all_pairs)):
            v = mat[i,j]
            if not np.isnan(v):
                ax.text(j,i,f"{v:+.1f}%",ha="center",va="center",
                        fontsize=6,fontweight="bold")
    plt.colorbar(im, ax=ax, label="Return (%)")
    ax.set_title("Individual Pair Returns by Selection Method (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/heatmap_pair_returns.png", dpi=150, bbox_inches="tight")
    plt.close(); log.info("  Saved heatmap_pair_returns.png")


def plot_detail(res: dict, dates: pd.DatetimeIndex):
    if not res: return
    name = res["name"]; pairs = res["pairs"]; nrows = len(pairs)
    if nrows == 0: return
    fig, axes = plt.subplots(nrows, 2, figsize=(18, 5*nrows), squeeze=False)

    for row, pr in enumerate(pairs):
        z = pr["z"]; eq = pr["equity"]
        nt = min(len(eq), len(dates)); ds = dates[:nt]
        a_s = pr["pair"].split("/")[0].replace(".NS","")
        b_s = pr["pair"].split("/")[1].replace(".NS","")

        ax = axes[row,0]
        ax.plot(ds, z[:nt], lw=0.7, color="indigo", alpha=0.85)
        for th,col,ls,lbl in [
            ( ENTRY_Z,"#F44336","--",f"Entry ±{ENTRY_Z}σ"),
            (-ENTRY_Z,"#4CAF50","--",""),
            ( EXIT_Z, "#FF9800",":",f"Exit ±{EXIT_Z}σ"),
            (-EXIT_Z, "#FF9800",":",""),(STOP_Z,"black","-.",""),
            (-STOP_Z, "black","-.","")]:
            ax.axhline(th, color=col, ls=ls, lw=0.9,
                       label=lbl if lbl else "_nolegend_")
        ax.axhline(0, color="gray", ls=":", lw=0.4); ax.set_ylim(-5,5)
        ax.set_title(
            f"[{name}] {a_s}/{b_s}  β={pr['beta']:+.4f}  "
            f"ret={pr['ret']:+.2f}%  Sh={pr['sharpe']:.2f}  "
            f"WR={pr['wr']:.1f}%  PF={pr['pf']:.2f}  "
            f"T={pr['n']} [rev={pr['n_rev']} stop={pr['n_stop']} end={pr['n_end']}]",
            fontsize=8)
        ax.set_ylabel("z-score"); ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.3); ax.set_xlabel("Datetime")

        alloc = LEG_CAPITAL*2; ax2 = axes[row,1]
        ax2.plot(ds, eq[:nt], color="#2196F3", lw=1.2)
        ax2.axhline(alloc, color="black", ls=":", lw=0.9,
                    label=f"Allocated ₹{int(alloc):,}")
        eq_a = np.array(eq[:nt])
        ax2.fill_between(ds, alloc, eq_a, where=eq_a>=alloc,
                         color="#4CAF50", alpha=0.15, label="Gain")
        ax2.fill_between(ds, alloc, eq_a, where=eq_a<alloc,
                         color="#F44336", alpha=0.15, label="Loss")
        bi = Z_WINDOW
        for t in pr["trades"]:
            l = t["long_stock"].replace(".NS","")
            s = t["short_stock"].replace(".NS","")
            lbl = f"L:{l} {t['long_qty']}sh\nS:{s} {t['short_qty']}sh"
            if bi < nt:
                ax2.annotate(lbl, xy=(ds[bi], float(eq[bi])),
                             xytext=(0,14), textcoords="offset points",
                             fontsize=4.5, ha="center", color="navy",
                             bbox=dict(boxstyle="round,pad=0.1", fc="lightyellow",
                                       alpha=0.8, ec="gray", lw=0.5),
                             arrowprops=dict(arrowstyle="-|>", color="gray",
                                            lw=0.4, mutation_scale=5))
            bi += max(t["hold"], 1) + 1
        ax2.set_ylabel("₹ (pair slice)"); ax2.set_xlabel("Datetime")
        ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    plt.suptitle(f"{name} — 5-Minute Pair Detail  "
                 f"(₹{int(LEG_CAPITAL):,}/leg, dollar-neutral)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fname = f"results/detail_{name.lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"  Saved {fname}")

# CSV OUTPUTS
def save_csvs(results: dict):
    port_rows, pair_rows, trade_rows = [], [], []
    for name, res in results.items():
        if not res: continue
        pm = res["port_metrics"]
        port_rows.append({
            "Method":name,"Return_%":pm["ret"],"Sharpe":pm["sharpe"],
            "Max_DD_%":pm["mdd"],"Win_Rate_%":pm["wr"],
            "Profit_Factor":pm["pf"],"N_Trades":pm["n"],
            "N_Rev":pm["n_rev"],"N_Stop":pm["n_stop"],"N_MaxHold":pm["n_mh"],"N_End":pm["n_end"],
        })
        for pr in res["pairs"]:
            pair_rows.append({
                "Method":name,"Pair":pr["pair"],"Beta":pr["beta"],
                "Return_%":pr["ret"],"Sharpe":pr["sharpe"],
                "Max_DD_%":pr["mdd"],"Win_Rate_%":pr["wr"],
                "Profit_Factor":pr["pf"],"N_Trades":pr["n"],
                "N_Rev":pr["n_rev"],"N_Stop":pr["n_stop"],"N_MaxHold":pr["n_mh"],"N_End":pr["n_end"],
                "Avg_Hold_Bars":pr["avg_hold"],
            })
            for ti, t in enumerate(pr["trades"], 1):
                trade_rows.append({
                    "Method":name,"Pair":pr["pair"],"Trade#":ti,
                    "LONG_Stock":t["long_stock"],"LONG_Qty":t["long_qty"],
                    "LONG_EntryPx":t["long_entry_px"],"LONG_EntryVal":t["long_entry_val"],
                    "LONG_PnL":t["long_pnl"],
                    "SHORT_Stock":t["short_stock"],"SHORT_Qty":t["short_qty"],
                    "SHORT_EntryPx":t["short_entry_px"],"SHORT_EntryVal":t["short_entry_val"],
                    "SHORT_PnL":t["short_pnl"],
                    "Total_PnL_Rs":round(t["pnl"],2),
                    "Entry_Z":t["entry_z"],"Exit_Z":t["exit_z"],
                    "Hold_Bars":t["hold"],"Exit_Type":t["exit_type"],
                })
    pd.DataFrame(port_rows).to_csv("results/portfolio_summary.csv",  index=False)
    pd.DataFrame(pair_rows).to_csv("results/pair_results.csv",       index=False)
    pd.DataFrame(trade_rows).to_csv("results/trade_detail.csv",      index=False)
    log.info("  Saved portfolio_summary.csv, pair_results.csv, trade_detail.csv")

# MAIN
def main():
    log.info("="*68)
    log.info("PAIRS TRADING — NIFTY 100  |  5-minute bars")
    log.info("Novel: Enhanced MeanCross (ADF + OU half-life + R² filters)")
    log.info("Engine: simple threshold — Gatev et al. (2006) style")
    log.info("="*68)

    px_all = download_or_load()

    n_total  = len(px_all)
    n_form   = int(n_total * FORMATION_FRAC)
    px_form  = px_all.iloc[:n_form].copy()
    px_trade = px_all.iloc[n_form:].copy()

    form_start  = px_form.index[0]
    form_end    = px_form.index[-1]
    trade_start = px_trade.index[0]
    trade_end   = px_trade.index[-1]

    log.info(f"  Formation : {form_start} → {form_end}  ({len(px_form)} bars)")
    log.info(f"  Trading   : {trade_start} → {trade_end}  ({len(px_trade)} bars)")
    log.info(f"  Capital   : ₹{INITIAL_CAPITAL:,.0f}  |  "
             f"₹{int(INITIAL_CAPITAL/TOP_N):,}/pair  |  ₹{int(LEG_CAPITAL):,}/leg")
    log.info(f"  Engine    : Entry={ENTRY_Z}σ  Exit={EXIT_Z}σ  "
             f"Stop={STOP_Z}σ  Z={Z_WINDOW}bars  Cooldown={COOLDOWN}bars")
    log.info(f"  MCR screens: ADF_p<{ADF_P_MAX}  h<{H_MAX}bars  R²≥{R2_MIN}")
    log.info(f"  Trend filter: tail={TREND_WINDOW}bars  slope_max={TREND_SLOPE_MAX}/bar  (all methods)")

    good = sorted(
        set(px_form.columns[ px_form.notna().mean()  >= 0.90]) &
        set(px_trade.columns[px_trade.notna().mean() >= 0.90])
    )
    px_form  = px_form[good]; px_trade = px_trade[good]
    dates    = pd.DatetimeIndex(px_trade.index)
    log.info(f"  Universe  : {len(good)} tickers (≥90% coverage both windows)")

    if len(px_form) < Z_WINDOW + 100:
        raise ValueError(f"Formation too short ({len(px_form)} bars).")

    log.info("\n"+"="*68+"\nPAIR SELECTION  (formation period)\n"+"="*68)
    log.info("\n  Cointegration:")
    coint_p = select_cointegration(px_form)
    log.info("\n  Correlation:")
    corr_p  = select_correlation(px_form)
    log.info("\n  Distance:")
    dist_p  = select_distance(px_form)
    log.info("\n  MeanCross (enhanced — novel):")
    mc_p    = select_meancross(px_form)
    log.info("\n  Composite (50% Coint + 50% enhanced MCR):")
    comp_p  = select_composite(px_form)

    log.info("\n"+"="*68+"\nBACKTESTING  (trading period)\n"+"="*68)
    results = {
        "Cointegration": run_method("Cointegration", coint_p, px_trade, px_form, dates),
        "Correlation":   run_method("Correlation",   corr_p,  px_trade, px_form, dates),
        "Distance":      run_method("Distance",       dist_p,  px_trade, px_form, dates),
        "MeanCross":     run_method("MeanCross",      mc_p,   px_trade, px_form, dates),
        "Composite":     run_method("Composite",      comp_p,  px_trade, px_form, dates),
    }

    log.info("\n"+"="*68+"\nFINAL RESULTS\n"+"="*68)
    log.info(f"{'Method':<16} {'Return%':>8} {'Sharpe':>7} {'MDD%':>7} "
             f"{'WR%':>6} {'PF':>6} {'Trades':>7}")
    log.info("-"*68)
    for name, res in results.items():
        if not res: continue
        pm = res["port_metrics"]
        log.info(f"{name:<16} {pm['ret']:>+8.2f} {pm['sharpe']:>7.3f} "
                 f"{pm['mdd']:>7.2f} {pm['wr']:>6.1f} {pm['pf']:>6.3f} "
                 f"{pm['n']:>7}")
    log.info("="*68)

    log.info("\nGenerating outputs ...")
    plot_equity(results, dates)
    plot_summary(results, str(trade_end))
    plot_heatmap(results)
    save_csvs(results)
    for name, res in results.items():
        plot_detail(res, dates)

    log.info("\nAll outputs saved to results/")
    log.info(f"  nifty100_5m.csv                  — 5m cache (refreshes daily)")
    log.info("  results/equity_comparison.png    — 5 methods overlaid")
    log.info("  results/summary_bars.png         — 6-panel metrics")
    log.info("  results/heatmap_pair_returns.png — pair-level heatmap")
    log.info("  results/detail_{method}.png      — z-score + equity per pair")
    log.info("  results/portfolio_summary.csv    — portfolio metrics")
    log.info("  results/pair_results.csv         — per-pair metrics")
    log.info("  results/trade_detail.csv         — every trade")
    log.info("  results/run.log                  — full execution log")


if __name__ == "__main__":
    main()