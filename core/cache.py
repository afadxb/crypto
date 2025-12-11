import time
import pandas as pd

class MarketCache:
    def __init__(self):
        self.data = {}   # pair -> DataFrame
        self.last_fetch = {}  # pair -> timestamp

    def get(self, pair):
        return self.data.get(pair)

    def set(self, pair, df):
        self.data[pair] = df
        self.last_fetch[pair] = time.time()
