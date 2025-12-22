"""Backfill Kraken trade history into hourly OHLCV SQLite storage."""
from __future__ import annotations

import argparse
import io
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, List

import pandas as pd

from core.ohlcv_store import OHLCVStore

CHUNK_SIZE = 500_000
TIMEFRAME = "1h"
EXCHANGE = "kraken"


def _csv_sources(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() == ".csv")
    return [path]


def _read_csv_stream(file_obj: io.TextIOBase) -> Iterator[pd.DataFrame]:
    cols = ["price", "volume", "time", "side", "ordertype", "misc"]
    for chunk in pd.read_csv(
        file_obj,
        header=None,
        names=cols,
        usecols=[0, 1, 2],
        chunksize=CHUNK_SIZE,
        dtype=float,
    ):
        yield chunk.dropna(subset=["price", "volume", "time"])


def iter_trades(path: Path) -> Iterator[pd.DataFrame]:
    """Yield trade dataframes from a CSV, directory of CSVs, or Kraken ZIP."""

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if not name.lower().endswith(".csv"):
                    continue
                with zf.open(name) as raw:
                    with io.TextIOWrapper(raw, encoding="utf-8") as text_f:
                        yield from _read_csv_stream(text_f)
        return

    for csv_path in _csv_sources(path):
        with open(csv_path, "r", encoding="utf-8") as f:
            yield from _read_csv_stream(f)


def aggregate_hourly(chunks: Iterable[pd.DataFrame]) -> List[dict]:
    """Aggregate tick trades into 1h OHLCV bars."""

    bars: dict[int, dict] = defaultdict(
        lambda: {
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": 0.0,
            "vwap_sum": 0.0,
            "trades": 0,
            "first_ts": None,
            "last_ts": None,
        }
    )

    for chunk in chunks:
        if chunk.empty:
            continue
        chunk = chunk.sort_values("time")
        chunk["hour_ts"] = (chunk["time"] // 3600 * 3600).astype(int)

        for hour_ts, grp in chunk.groupby("hour_ts"):
            prices = grp["price"].to_numpy()
            volumes = grp["volume"].to_numpy()
            times = grp["time"].to_numpy()
            if len(prices) == 0:
                continue

            bar = bars[hour_ts]
            bar["high"] = prices.max() if bar["high"] is None else max(bar["high"], prices.max())
            bar["low"] = prices.min() if bar["low"] is None else min(bar["low"], prices.min())
            bar["volume"] += volumes.sum()
            bar["vwap_sum"] += (prices * volumes).sum()
            bar["trades"] += len(prices)

            first_ts = times[0]
            last_ts = times[-1]
            if bar["first_ts"] is None or first_ts < bar["first_ts"]:
                bar["first_ts"] = first_ts
                bar["open"] = prices[0]
            if bar["last_ts"] is None or last_ts > bar["last_ts"]:
                bar["last_ts"] = last_ts
                bar["close"] = prices[-1]

    out: List[dict] = []
    for ts in sorted(bars.keys()):
        bar = bars[ts]
        volume = bar["volume"]
        vwap = bar["vwap_sum"] / volume if volume > 0 else None
        out.append(
            {
                "ts": int(ts),
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": volume,
                "vwap": vwap,
                "trades": bar["trades"],
                "timeframe": TIMEFRAME,
                "pair": None,
            }
        )
    return out


def import_file(source: Path, pair: str, db_path: Path | None = None) -> int:
    store = OHLCVStore(path=str(db_path) if db_path else None)
    bars = aggregate_hourly(iter_trades(source))
    for bar in bars:
        bar["pair"] = pair
    return store.upsert_ohlcv(bars, exchange=EXCHANGE, timeframe=TIMEFRAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Kraken historical trades into SQLite OHLCV store")
    parser.add_argument("source", type=Path, help="Path to Kraken ZIP or CSV (or directory of CSVs)")
    parser.add_argument("--pair", default="XBTUSD", help="Target trading pair, e.g., XBTUSD")
    parser.add_argument("--db", type=Path, default=None, help="Optional path to SQLite DB (defaults to config DB_PATH)")
    args = parser.parse_args()

    rows = import_file(args.source, pair=args.pair, db_path=args.db)
    print(f"Imported {rows} rows for {args.pair} into {args.db or 'default DB'}")


if __name__ == "__main__":
    main()
