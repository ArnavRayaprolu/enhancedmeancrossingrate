"""Build a dataset from a public GitHub mirror of the Binance bulk archive.

Used because the sandbox that produced the first results could not reach
Binance or Yahoo directly.  The mirror (github.com/KevinSherry2016/Crypto,
folder marketData/) holds the unmodified daily kline files published at
data.binance.vision for 39 USDT-margined perpetual futures, 2024-01 to 2026-09.

    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/KevinSherry2016/Crypto data/_mirror/perp39
    git -C data/_mirror/perp39 sparse-checkout set --no-cone /marketData/
    python scripts/build_mirror_data.py data/_mirror/perp39/marketData binance_perp_1h
"""
from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_data import _read_kline_zip, _save  # noqa: E402


def _one(p: str):
    try:
        return Path(p).name.split("-1h-")[0], _read_kline_zip(Path(p).read_bytes())
    except Exception as exc:  # corrupt file
        print(f"skip {p}: {exc}")
        return None


def main(src: str, name: str):
    files = sorted(str(p) for p in Path(src).glob("*-1h-*.zip"))
    parts = defaultdict(list)
    with ProcessPoolExecutor() as ex:
        for res in ex.map(_one, files, chunksize=256):
            if res is not None:
                parts[res[0]].append(res[1])
    close, qvol = {}, {}
    for sym, dfs in parts.items():
        df = pd.concat(dfs).sort_index()
        df = df[~df.index.duplicated()]
        close[sym], qvol[sym] = df["close"], df["qvol"]
    close = pd.DataFrame(close); qvol = pd.DataFrame(qvol)
    meta = pd.DataFrame({"ticker": close.columns, "group": "crypto", "name": close.columns,
                         "date_added": ""})
    _save(name, close, qvol, meta)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
