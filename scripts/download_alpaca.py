"""Download US equity bars from Alpaca Market Data (free plan, SIP feed,
split + dividend adjusted) for the walk-forward study.

    python scripts/download_alpaca.py assets            # active + inactive (delisted) assets
    python scripts/download_alpaca.py daily             # daily bars, every asset, 2016 ->
    python scripts/download_alpaca.py select            # point-in-time liquid superset
    python scripts/download_alpaca.py hourly K N        # chunk K of N of the superset (slow; superseded)
    python scripts/download_alpaca.py hourly2 [MIN]     # resumable regular-session download

Needs ALPACA_KEY_ID and ALPACA_SECRET_KEY in the environment (a free Alpaca
account; paper-trading keys work). Runs on GitHub Actions because the research
container cannot reach alpaca.markets.

Survivorship. The asset list includes INACTIVE (delisted, merged, renamed)
symbols, and the hourly download universe is the union, over calendar years,
of each year's 1,500 most liquid common stocks measured on that year's own
daily data. A stock that was liquid in 2017 and delisted in 2019 is therefore
downloaded; the model then picks each fold's universe from formation data only.

Output (data/alpaca/)
    assets.csv                         symbol, name, exchange, status, ...
    daily/{close,volume}_{part}.parquet
    universe.csv                       yearly liquidity ranks, superset flag
    hourly/c{K}/{open,high,low,close,volume}_{year}.parquet
        Alpaca's native 1-hour bars (clock hours). Kept: the seven bars that
        start 09:00 ... 15:00 ET. The 09:00 bar includes 09:00-09:30 pre-market
        trades together with the opening half hour; its close is the 10:00
        price. The 15:00 bar closes at the 16:00 close. Hours without a trade
        are missing.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "alpaca"
TRADE = "https://paper-api.alpaca.markets/v2"
DATA = "https://data.alpaca.markets/v2/stocks/bars"
START, END = "2016-01-01T00:00:00Z", None          # END None -> yesterday
TOP_PER_YEAR = 1500
EXCLUDE = re.compile(
    r"\bETF\b|\bETN\b|ETRACS|iShares|ProShares|Direxion|SPDR|Vanguard|Invesco|WisdomTree|"
    r"VanEck|Global X|First Trust|GraniteShares|MicroSectors|VelocityShares|iPath|"
    r"\bFund\b|Acquisition|Warrant|\bUnits?\b|\bRights?\b|Preferred|Depositary Shares, each|"
    r"\bNotes?\b|Debenture|Leveraged|\bIndex\b|Trust Units|Subordinated|\bBond\b|Treasury|"
    r"Blank Check|\b\d+(?:\.\d+)?%", re.I)

S = requests.Session()
S.headers.update({"APCA-API-KEY-ID": os.environ.get("ALPACA_KEY_ID", ""),
                  "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", "")})
_last = [0.0]
TICKER = re.compile(r"^[A-Z]{1,5}([.][A-Z]{1,2})?$")


class BadRequest(Exception):
    pass


def get(url, params=None):
    """GET with the free-plan limit (200 requests / minute) and retries."""
    for attempt in range(8):
        wait = 0.31 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = S.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt); continue
        if r.status_code == 429:
            reset = r.headers.get("X-RateLimit-Reset")
            wait = float(reset) - time.time() + 0.5 if reset else 5.0
            time.sleep(min(max(1.0, wait), 65.0)); continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt); continue
        if r.status_code == 400:
            raise BadRequest(r.text[:200])
        if r.status_code in (401, 403):
            sys.exit(f"Alpaca refused the request ({r.status_code}): set ALPACA_KEY_ID / "
                     f"ALPACA_SECRET_KEY repository secrets. {r.text[:200]}")
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"giving up on {url}")


def bars(symbols: list[str], timeframe: str, start: str, end: str | None):
    """{symbol: list of bar dicts} for a batch; a batch the API rejects is
    split in halves until the offending symbol is isolated and skipped."""
    try:
        return _bars(symbols, timeframe, start, end)
    except BadRequest as e:
        if len(symbols) == 1:
            print(f"  skip {symbols[0]}: {e}", flush=True)
            return {}
        h = len(symbols) // 2
        out = bars(symbols[:h], timeframe, start, end)
        out.update(bars(symbols[h:], timeframe, start, end))
        return out


def _bars(symbols: list[str], timeframe: str, start: str, end: str | None):
    params = dict(symbols=",".join(symbols), timeframe=timeframe, start=start,
                  limit=10000, adjustment="all", feed="sip", sort="asc")
    if end:
        params["end"] = end
    acc: dict[str, list] = {}
    while True:
        js = get(DATA, params)
        for sym, b in (js.get("bars") or {}).items():
            acc.setdefault(sym, []).extend(b)
        tok = js.get("next_page_token")
        if not tok:
            break
        params["page_token"] = tok
    return acc


def end_ts():
    return END or (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")


# --------------------------------------------------------------------------- #
def stage_assets():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for status in ("active", "inactive"):
        js = get(f"{TRADE}/assets", dict(status=status, asset_class="us_equity"))
        rows += js
    a = pd.DataFrame(rows)
    print(f"assets returned: {len(a)}; symbols listed more than once: "
          f"{int(a['symbol'].duplicated().sum())}", flush=True)
    a = a[a["exchange"].isin(["NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"])]
    a = a[a["symbol"].astype(str).str.match(TICKER)]
    a["instrument_ok"] = ~a["name"].fillna("").str.contains(EXCLUDE.pattern, flags=re.I, regex=True) & (a["exchange"] != "ARCA")
    a.to_csv(OUT / "assets.csv", index=False)
    print(a.groupby(["status", "instrument_ok"]).size(), flush=True)


def stage_daily():
    a = pd.read_csv(OUT / "assets.csv", keep_default_na=False, dtype={"symbol": str})
    a["instrument_ok"] = a["instrument_ok"].astype(str) == "True"
    syms = sorted(s for s in a.loc[a.instrument_ok, "symbol"].astype(str).unique() if TICKER.match(s))
    print(f"daily bars for {len(syms)} symbols", flush=True)
    close, vol = {}, {}
    for i in range(0, len(syms), 100):
        acc = bars(syms[i:i + 100], "1Day", START, end_ts())
        for s, b in acc.items():
            if not b:
                continue
            d = pd.DataFrame(b)
            ix = pd.to_datetime(d["t"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
            close[s] = pd.Series(d["c"].to_numpy(np.float32), index=ix)
            vol[s] = pd.Series(d["v"].to_numpy(np.float32), index=ix)
        print(f"  {i + 100}/{len(syms)} symbols, {len(close)} with data", flush=True)
    (OUT / "daily").mkdir(exist_ok=True)
    for name, dct in (("close", close), ("volume", vol)):
        df = pd.DataFrame(dct).sort_index()
        cols = sorted(df.columns)
        for k in range(0, len(cols), 2000):
            df[cols[k:k + 2000]].to_parquet(OUT / "daily" / f"{name}_{k // 2000:02d}.parquet",
                                            compression="zstd")


ETF_NAME = re.compile(r"\bETF\b|iShares|SPDR|Vanguard|Invesco|WisdomTree|VanEck|Global X|First Trust|"
                      r"Schwab|\bFund\b", re.I)
LEVERED = re.compile(r"\b[23]X\b|Ultra|Inverse|Short|Bear|Bull|Daily|Leveraged|-1X|1\.5X", re.I)


def stage_daily_full():
    """Daily OHLCV, 2016 onward, for every common stock (active and inactive) and
    every unleveraged, non-inverse ETF/fund; split + dividend adjusted.
    Output: daily_full/{open,high,low,close,volume}_{part}.parquet, etfs.txt."""
    a = pd.read_csv(OUT / "assets.csv", keep_default_na=False, dtype={"symbol": str})
    a = a[a["symbol"].astype(str).str.match(TICKER)]
    stock = a["instrument_ok"].astype(str) == "True"
    etf = a["name"].str.contains(ETF_NAME.pattern, flags=re.I, regex=True) & \
        ~a["name"].str.contains(LEVERED.pattern, flags=re.I, regex=True)
    syms = sorted(set(a.loc[stock | etf, "symbol"].astype(str)))
    out = OUT / "daily_full"; out.mkdir(exist_ok=True)
    (out / "etfs.txt").write_text("\n".join(sorted(set(a.loc[etf, "symbol"].astype(str)))) + "\n")
    print(f"daily_full: {len(syms)} symbols ({int(etf.sum())} ETFs/funds)", flush=True)
    fields = {k: {} for k in ("open", "high", "low", "close", "volume")}
    for i in range(0, len(syms), 100):
        acc = bars(syms[i:i + 100], "1Day", START, end_ts())
        for s_, b in acc.items():
            if not b:
                continue
            d = pd.DataFrame(b)
            ix = pd.to_datetime(d["t"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
            for f, col in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"), ("volume", "v")):
                fields[f][s_] = pd.Series(d[col].to_numpy(np.float32), index=ix)
        if i % 1000 == 0:
            print(f"  {i + 100}/{len(syms)}", flush=True)
    for f, dct in fields.items():
        df = pd.DataFrame(dct).sort_index()
        cols = sorted(df.columns)
        for k in range(0, len(cols), 2000):
            df[cols[k:k + 2000]].to_parquet(out / f"{f}_{k // 2000:02d}.parquet", compression="zstd")


def stage_select():
    c = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "daily").glob("close_*.parquet"))], axis=1)
    v = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "daily").glob("volume_*.parquet"))], axis=1)
    dv = (c * v)
    ranks = {}
    for y, g in c.groupby(c.index.year):
        cov = g.notna().mean()
        ok = (cov >= 0.5) & (g.median() >= 5.0)
        med = dv.loc[g.index].median()[ok]
        ranks[y] = med.rank(ascending=False)
    rk = pd.DataFrame(ranks)
    rk["best_rank"] = rk.min(axis=1)
    rk["superset"] = rk["best_rank"] <= TOP_PER_YEAR
    rk.index.name = "symbol"
    rk.to_csv(OUT / "universe.csv")
    a = pd.read_csv(OUT / "assets.csv", keep_default_na=False, dtype={"symbol": str})
    # a symbol can be listed twice (an inactive old issuer and an active new one)
    a = a.sort_values("status").drop_duplicates("symbol").set_index("symbol")
    sel = rk.index[rk.superset]
    print(f"superset: {len(sel)} symbols; of which inactive (delisted) now: "
          f"{int((a.reindex(sel)['status'] == 'inactive').sum())}", flush=True)


def stage_hourly(k: int, n: int):
    rk = pd.read_csv(OUT / "universe.csv", index_col=0, keep_default_na=False, na_values=[""],
                     dtype={"symbol": str})
    rk["superset"] = rk["superset"].astype(str) == "True"
    syms = sorted(rk.index[rk.superset].astype(str))
    syms = syms[k::n]
    out = OUT / "hourly" / (f"c{k}" if n == 3 else f"n{n}c{k}")
    if list(out.glob("close_*.parquet")):
        print(f"chunk {k} already downloaded; skipping", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    print(f"chunk {k}/{n}: {len(syms)} symbols", flush=True)
    fields = {f: {} for f in ("open", "high", "low", "close", "volume")}
    for i in range(0, len(syms), 50):
        acc = bars(syms[i:i + 50], "1Hour", START, end_ts())
        for s_, b in acc.items():
            if not b:
                continue
            d = pd.DataFrame(b)
            t = pd.to_datetime(d["t"], utc=True)
            et = t.dt.tz_convert("America/New_York")
            keep = ((et.dt.hour >= 9) & (et.dt.hour <= 15)).to_numpy()
            d = d[keep]
            ix = pd.DatetimeIndex(t[keep].dt.tz_localize(None))       # naive UTC
            for f, col in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"),
                           ("volume", "v")):
                fields[f][s_] = pd.Series(d[col].to_numpy(np.float32), index=ix)
        print(f"  {min(i + 50, len(syms))}/{len(syms)}", flush=True)
    for f, dct in fields.items():
        df = pd.DataFrame(dct).sort_index()
        for y, g in df.groupby(df.index.year):
            g.to_parquet(out / f"{f}_{y}.parquet", compression="zstd")


# --------------------------------------------------------------------------- #
# resumable regular-session download (hourly2)
# --------------------------------------------------------------------------- #
TOP_HOURLY = 260          # per-year liquidity cut that gives ~500 symbols in total
GROUP = 28                # symbols per request: ~8 bars x 28 ~ one ~240-bar page per day
H2 = OUT / "hourly2"


def hourly_symbols() -> list[str]:
    rk = pd.read_csv(OUT / "universe.csv", index_col=0, keep_default_na=False, na_values=[""],
                     dtype={"symbol": str})
    yrs = [c for c in rk.columns if str(c).isdigit()]
    best = rk[yrs].min(axis=1)
    return sorted(str(x) for x in best.index[best <= TOP_HOURLY])


def git_save(path: Path, msg: str) -> None:
    import subprocess
    run = lambda *a: subprocess.run(a, check=False, capture_output=True, text=True)
    run("git", "add", "-f", str(path))
    if run("git", "diff", "--cached", "--quiet").returncode == 0:
        return
    run("git", "commit", "-q", "-m", msg)
    br = os.environ.get("BRANCH", "")
    for i in range(6):
        if run("git", "pull", "-q", "--rebase", "origin", br).returncode == 0 and \
                run("git", "push", "-q", "origin", f"HEAD:{br}").returncode == 0:
            return
        time.sleep(5 * (i + 1))
    print("  WARNING: push failed; data kept locally", flush=True)


def stage_hourly2(minutes: float = 330.0):
    """One work unit = (calendar year, group of GROUP symbols). For every
    trading day of the year one request asks for the group's 1-hour bars
    starting 09:00 ... 15:00 America/New_York (the window follows daylight
    saving time), so 7 bars x 28 symbols fit in about one ~240-bar page and
    extended-hours bars are never fetched (the first units, fetched with a
    13:00-21:00 UTC window, carry a few extra bars that the loader drops). Each finished unit is written to hourly2/<year>/g<NN>.parquet and
    committed, so a run that hits the time limit loses at most six units and
    the next run continues where it stopped."""
    t_end = time.time() + 60 * minutes
    syms = hourly_symbols()
    groups = [syms[i:i + GROUP] for i in range(0, len(syms), GROUP)]
    c = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "daily").glob("close_*.parquet"))], axis=1)
    days = pd.DatetimeIndex(c.index[c.notna().sum(axis=1) > 100])
    print(f"hourly2: {len(syms)} symbols in {len(groups)} groups, {len(days)} trading days "
          f"{days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d}", flush=True)
    (H2 / "symbols.txt").parent.mkdir(parents=True, exist_ok=True)
    (H2 / "symbols.txt").write_text("\n".join(syms) + "\n")
    units = [(y, g) for y in sorted(set(days.year)) for g in range(len(groups))]
    todo = [(y, g) for y, g in units if not (H2 / str(y) / f"g{g:02d}.parquet").exists()]
    print(f"units: {len(units)} total, {len(todo)} to do", flush=True)
    n_new = 0
    for y, g in todo:
        if time.time() > t_end - 600:
            print("time budget reached; stopping cleanly", flush=True)
            break
        t0 = time.time(); rows = []
        for d in days[days.year == y]:
            ds = d.strftime("%Y-%m-%d")
            a = pd.Timestamp(f"{ds} 09:00", tz="America/New_York").tz_convert("UTC")
            b = pd.Timestamp(f"{ds} 15:01", tz="America/New_York").tz_convert("UTC")
            acc = bars(groups[g], "1Hour", a.strftime("%Y-%m-%dT%H:%M:%SZ"),
                       b.strftime("%Y-%m-%dT%H:%M:%SZ"))
            for s_, b in acc.items():
                for x in b:
                    rows.append((s_, x["t"], x["o"], x["h"], x["l"], x["c"], x["v"]))
        df = pd.DataFrame(rows, columns=["symbol", "t", "open", "high", "low", "close", "volume"])
        df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_localize(None)
        for k in ("open", "high", "low", "close", "volume"):
            df[k] = df[k].astype(np.float32)
        df["symbol"] = df["symbol"].astype("category")
        out = H2 / str(y) / f"g{g:02d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, compression="zstd", index=False)
        n_new += 1
        if n_new % 6 == 0:
            git_save(H2, f"data: alpaca hourly2 through {y} group {g:02d} (GitHub Actions)")
        done = sum((H2 / str(yy) / f"g{gg:02d}.parquet").exists() for yy, gg in units)
        print(f"  {y} g{g:02d}: {len(df):,} bars in {time.time() - t0:.0f}s  "
              f"[{done}/{len(units)} units]", flush=True)
    left = sum(not (H2 / str(yy) / f"g{gg:02d}.parquet").exists() for yy, gg in units)
    git_save(H2, f"data: alpaca hourly2 {'complete' if left == 0 else f'{left} units left'} "
                 f"(GitHub Actions)")


# --------------------------------------------------------------------------- #
# resumable regular-session 5-minute download (m5)
# --------------------------------------------------------------------------- #
M5 = OUT / "m5"
M5_START = "2024-01-01"


def stage_m5(minutes: float = 330.0):
    """5-minute bars, 09:30-16:00 ET, for the symbols in data/alpaca/m5_symbols.txt
    (liquid S&P 500 names with same-sub-industry peers), 2024 onward. One unit =
    (calendar month, group of 28 symbols); one request per trading day per group;
    units are written to m5/<YYYY-MM>/g<NN>.parquet and committed every 6 units."""
    t_end = time.time() + 60 * minutes
    syms = (OUT / "m5_symbols.txt").read_text().split()
    groups = [syms[i:i + GROUP] for i in range(0, len(syms), GROUP)]
    c = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "daily").glob("close_*.parquet"))], axis=1)
    days = pd.DatetimeIndex(c.index[c.notna().sum(axis=1) > 100])
    days = days[days >= pd.Timestamp(M5_START)]
    months = sorted(set(days.strftime("%Y-%m")))
    units = [(m, g) for m in months for g in range(len(groups))]
    todo = [(m, g) for m, g in units if not (M5 / m / f"g{g:02d}.parquet").exists()]
    print(f"m5: {len(syms)} symbols, {len(groups)} groups, {len(days)} days, "
          f"{len(units)} units, {len(todo)} to do", flush=True)
    n_new = 0
    for m, g in todo:
        if time.time() > t_end - 600:
            print("time budget reached; stopping cleanly", flush=True)
            break
        t0 = time.time(); rows = []
        for d in days[days.strftime("%Y-%m") == m]:
            ds = d.strftime("%Y-%m-%d")
            a = pd.Timestamp(f"{ds} 09:30", tz="America/New_York").tz_convert("UTC")
            b = pd.Timestamp(f"{ds} 15:56", tz="America/New_York").tz_convert("UTC")
            acc = bars(groups[g], "5Min", a.strftime("%Y-%m-%dT%H:%M:%SZ"), b.strftime("%Y-%m-%dT%H:%M:%SZ"))
            for s_, bl in acc.items():
                for x in bl:
                    rows.append((s_, x["t"], x["o"], x["h"], x["l"], x["c"], x["v"]))
        df = pd.DataFrame(rows, columns=["symbol", "t", "open", "high", "low", "close", "volume"])
        df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_localize(None)
        for k in ("open", "high", "low", "close", "volume"):
            df[k] = df[k].astype(np.float32)
        df["symbol"] = df["symbol"].astype("category")
        out = M5 / m / f"g{g:02d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, compression="zstd", index=False)
        n_new += 1
        if n_new % 6 == 0:
            git_save(M5, f"data: alpaca m5 through {m} group {g:02d} (GitHub Actions)")
        print(f"  {m} g{g:02d}: {len(df):,} bars in {time.time() - t0:.0f}s", flush=True)
    left = sum(not (M5 / mm / f"g{gg:02d}.parquet").exists() for mm, gg in units)
    git_save(M5, f"data: alpaca m5 {'complete' if left == 0 else f'{left} units left'} (GitHub Actions)")


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "assets":
        stage_assets()
    elif st == "daily":
        stage_daily()
    elif st == "select":
        stage_select()
    elif st == "m5":
        stage_m5(float(sys.argv[2]) if len(sys.argv) > 2 else 330.0)
    elif st == "daily_full":
        stage_daily_full()
    elif st == "hourly":
        stage_hourly(int(sys.argv[2]), int(sys.argv[3]))
    elif st == "hourly2":
        stage_hourly2(float(sys.argv[2]) if len(sys.argv) > 2 else 330.0)
    else:
        sys.exit(__doc__)
