"""Data feed module for fetching and caching OHLC data from Kraken."""
import time
from datetime import datetime
from typing import Optional

import pandas as pd
from krakenex import API

from .cache import MarketCache

CACHE_MAX_AGE = 300  # seconds; refresh more often than the 1h bar to capture new closes
MIN_CANDLES = 200


class DataFeed:
    """Fetches OHLC data from Kraken with simple per-pair caching."""

    def __init__(self, api_key: str, api_secret: str, interval: int = 60):
        self.api = API(api_key, api_secret)
        self.cache = MarketCache()
        self.interval = interval  # minutes, Kraken-style

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

        if self._should_use_cache(pair):
            return self.cache.get(pair)

        since = int(time.time()) - interval * 60 * MIN_CANDLES
        resp = self.api.query_public("OHLC", {"pair": pair, "interval": interval, "since": since})

        key = next(k for k in resp["result"].keys() if k != "last")
        raw = resp["result"][key]

        df = pd.DataFrame(raw, columns=[
            "time", "open", "high", "low", "close", "vwap", "volume", "count"
        ], dtype=float)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        self.cache.set(pair, df)
        return df

    @staticmethod
    def current_bar_timestamp(df: pd.DataFrame) -> datetime:
        """Return the timestamp of the last *closed* bar (second-to-last row)."""
        if len(df) < 2:
            raise ValueError("Not enough data to determine current bar")
        return df["time"].iloc[-2]
