"""Data feed module for fetching and caching OHLC data from Kraken."""
import time
from datetime import datetime
from typing import Optional

import pandas as pd
from krakenex import API

from .cache import MarketCache

CACHE_MAX_AGE = 300  # seconds; refresh more often than the 1h bar to capture new closes
# Ensure a long enough history for training; 5000 1h bars ~= 7 months
MIN_CANDLES = 5000


class DataFeed:
    """Fetches OHLC data from Kraken with simple per-pair caching."""

    def __init__(self, api_key: str, api_secret: str, interval: int = 60):
        self.api = API(api_key, api_secret)
        self.cache = MarketCache()
        self.interval = interval  # minutes, Kraken-style
        self.last_time = {}

    def _should_use_cache(self, pair: str) -> bool:
        cached = self.cache.get(pair)
        if cached is None:
            return False
        last_fetch = self.cache.last_fetch.get(pair, 0)
        return (time.time() - last_fetch) < CACHE_MAX_AGE

    def fetch_ohlc(self, pair: str, interval: Optional[int] = None) -> pd.DataFrame:
        """Fetch OHLC data for a pair, enforcing at least MIN_CANDLES history."""
        interval = interval or self.interval
        if interval != 60:
            # enforce 1h timeframe
            interval = 60

        cached = self.cache.get(pair)
        if self._should_use_cache(pair) and cached is not None:
            return cached

        if cached is None:
            since = int(time.time()) - interval * 60 * MIN_CANDLES
        else:
            last_ts = int(cached["time"].iloc[-1].timestamp())
            since = last_ts

        resp = self.api.query_public("OHLC", {"pair": pair, "interval": interval, "since": since})

        key = next(k for k in resp["result"].keys() if k != "last")
        raw = resp["result"][key]

        df_new = pd.DataFrame(
            raw,
            columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
            dtype=float,
        )
        # Normalize to timezone-aware UTC timestamps to avoid mixing aware/naive datetimes
        df_new["time"] = pd.to_datetime(df_new["time"], unit="s", utc=True)

        if cached is not None:
            df = pd.concat([cached, df_new], ignore_index=True)
            df = df.drop_duplicates(subset="time", keep="last").sort_values("time").reset_index(drop=True)
        else:
            df = df_new

        if not df.empty:
            self.last_time[pair] = df["time"].iloc[-1]

        self.cache.set(pair, df)
        return df

    @staticmethod
    def current_bar_timestamp(df: pd.DataFrame) -> datetime:
        """Return the timestamp of the last *closed* bar (second-to-last row)."""
        if len(df) < 2:
            raise ValueError("Not enough data to determine current bar")
        return df["time"].iloc[-2]
