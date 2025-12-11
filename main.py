"""Main loop for Kraken trading bot."""
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from krakenex import API

import config as CFG
from core.alerts import send_alert
from core.data_feed import DataFeed
from core.db import DB
from core.execution import Execution
from core.strategy import analyze

load_dotenv()

api = API(os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET"))
db = DB(CFG.DB_PATH)
feed = DataFeed(os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET"), CFG.TRADING_PARAMS["ohlc_interval"])
execu = Execution(db, api, CFG)


def run():
    interval = CFG.TRADING_PARAMS["trading_interval"]
    while True:
        for pair in CFG.PAIRS:
            df = feed.fetch_ohlc(pair, interval=CFG.TRADING_PARAMS["ohlc_interval"])
            result = analyze(df, CFG.INDICATORS)

            sig = result["signal"]
            conf = result.get("confidence", 0)
            bar_time = result.get("bar_time")
            price = result.get("price")

            db.insert_decision(
                (
                    datetime.utcnow(),
                    pair,
                    sig,
                    conf,
                    price,
                    result.get("st_dir"),
                    result.get("ema_fast"),
                    result.get("ema_slow"),
                    result.get("atr"),
                    result.get("atr_pct"),
                    str(bar_time),
                )
            )

            if sig == "HOLD" or bar_time is None:
                continue

            if execu.already_ordered_this_bar(pair, bar_time):
                continue

            size = CFG.TRADING_PARAMS["position_size_usd"] / price if price else 0

            status = execu.place_limit(pair, sig, price, size, bar_time)
            if status == "SUBMITTED" or conf >= 0.8:
                send_alert(f"{pair} {sig}", f"Limit {sig} @ {price:.2f} (conf={conf:.2f})")

        time.sleep(interval)


if __name__ == "__main__":
    run()
