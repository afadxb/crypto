import time
import pandas as pd
from datetime import datetime
from krakenex import API
from .cache import MarketCache

CACHE_MAX_AGE = 500  # seconds; < 1h but enough to avoid refetching everything

class DataFeed:
    def __init__(self, api_key, api_secret):
        self.api = API(api_key, api_secret)
        self.cache = MarketCache()

    def fetch_ohlc(self, pair, interval=60):
        cached = self.cache.get(pair)
        if cached is not None and (time.time() - self.cache.last_fetch[pair]) < CACHE_MAX_AGE:
            return cached

        now = int(time.time())
        since = now - interval * 60 * 200
        resp = self.api.query_public("OHLC", {"pair": pair, "interval": interval, "since": since})

        key = list(resp["result"].keys())[0]
        raw = resp["result"][key]

        df = pd.DataFrame(raw, columns=[
            "time","open","high","low","close","vwap","volume","count"
        ], dtype=float)

        df["time"] = pd.to_datetime(df["time"], unit="s")
        self.cache.set(pair, df)
        return df

    def current_bar_timestamp(self, df):
        return df["time"].iloc[-2]  # closed bar
