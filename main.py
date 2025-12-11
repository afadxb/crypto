import os
import time
from dotenv import load_dotenv
from datetime import datetime

from core.data_feed import DataFeed
from core.strategy import analyze
from core.execution import Execution
from core.db import DB
from core.alerts import send_alert
from krakenex import API
import config as CFG

load_dotenv()

api = API(os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET"))
db = DB(os.getenv("DB_PATH"))
feed = DataFeed(os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET"))
execu = Execution(db, api, CFG)

def run():
    while True:
        for pair in CFG.PAIRS:
            df = feed.fetch_ohlc(pair, interval=60)
            result = analyze(df, CFG.INDICATORS)

            sig = result["signal"]
            conf = result["confidence"]
            bar_time = result["bar_time"]
            price = result["price"]

            db.insert_decision((
                datetime.utcnow(), pair, sig, conf, price,
                result["st_dir"], result["ema_fast"], result["ema_slow"],
                result["atr"], result["atr_pct"], str(bar_time)
            ))

            if sig == "HOLD": 
                continue

            if execu.already_ordered_this_bar(pair, bar_time):
                continue

            size = CFG.position_size_usd / price

            status = execu.place_limit(pair, sig, price, size, bar_time)
            send_alert(f"{pair} {sig}", f"Limit {sig} @ {price} (conf={conf})")

        time.sleep(3600)

if __name__ == "__main__":
    run()
