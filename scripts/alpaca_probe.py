"""Time different Alpaca bar-request shapes to find the download bottleneck."""
import os
import time

import requests

S = requests.Session()
S.headers.update({"APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"],
                  "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]})
URL = "https://data.alpaca.markets/v2/stocks/bars"
SYMS = ("AAPL MSFT AMZN GOOGL META NVDA JPM XOM JNJ PG KO PEP WMT HD BAC CVX MRK ABBV PFE CSCO "
        "ORCL INTC T VZ DIS NFLX ADBE CRM TXN QCOM AMD IBM GE CAT MMM HON UNP UPS LMT RTX "
        "GS MS C WFC AXP BLK SPGI MCO ICE CME").split()


def probe(label, symbols, start, end, timeframe="1Hour", feed="sip", pages=3, limit=10000):
    p = dict(symbols=",".join(symbols), timeframe=timeframe, start=start, end=end,
             limit=limit, adjustment="all", feed=feed, sort="asc")
    tot = 0
    for i in range(pages):
        t = time.time()
        r = S.get(URL, params=p, timeout=120)
        dt = time.time() - t
        js = r.json() if r.status_code == 200 else {}
        n = sum(len(v) for v in (js.get("bars") or {}).values())
        tot += n
        print(f"{label:40s} page {i}: status={r.status_code} {dt:6.2f}s bars={n:6d} "
              f"bytes={len(r.content):8d} remaining={r.headers.get('X-RateLimit-Remaining')} "
              f"limit={r.headers.get('X-RateLimit-Limit')} {r.text[:120] if r.status_code != 200 else ''}",
              flush=True)
        tok = js.get("next_page_token")
        if not tok:
            break
        p["page_token"] = tok


probe("1 sym 1Hour 2019", ["AAPL"], "2019-01-01T00:00:00Z", "2020-01-01T00:00:00Z")
probe("1 sym 1Hour 2016-2026", ["AAPL"], "2016-01-01T00:00:00Z", "2026-10-31T00:00:00Z")
probe("10 sym 1Hour 2016-2026", SYMS[:10], "2016-01-01T00:00:00Z", "2026-10-31T00:00:00Z")
probe("50 sym 1Hour 2016-2026", SYMS, "2016-01-01T00:00:00Z", "2026-10-31T00:00:00Z")
probe("50 sym 1Hour 2019 only", SYMS, "2019-01-01T00:00:00Z", "2020-01-01T00:00:00Z")
probe("50 sym 1Hour 2019 iex", SYMS, "2019-01-01T00:00:00Z", "2020-01-01T00:00:00Z", feed="iex")
probe("50 sym 1Hour 2024 only", SYMS, "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z")
probe("50 sym 1Day 2016-2026", SYMS, "2016-01-01T00:00:00Z", "2026-10-31T00:00:00Z")
probe("1 sym 1Hour 2019 limit1000", ["AAPL"], "2019-01-01T00:00:00Z", "2020-01-01T00:00:00Z", limit=1000)
