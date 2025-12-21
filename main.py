"""Main loop for Kraken trading bot."""
import logging
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from krakenex import API

import pandas as pd

import config as CFG
from core.alerts import send_alert
from core.data_feed import DataFeed
from core.db import DB
from core.execution import Execution
from core.strategy import analyze

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

api = API(os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET"))
db = DB(CFG.DB_PATH)
feed = DataFeed(os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET"), CFG.TRADING_PARAMS["ohlc_interval"])
execu = Execution(db, api, CFG)
LOCAL_TZ = CFG.LOCAL_TZ


def load_last_processed_bar(db: DB):
    q = "SELECT pair, bar_id FROM decisions ORDER BY timestamp DESC"
    seen = {}
    for pair, bar_id in db.conn.execute(q):
        if pair not in seen:
            seen[pair] = bar_id
    return seen


def run():
    interval_minutes = CFG.TRADING_PARAMS["ohlc_interval"]
    bar_interval = pd.Timedelta(minutes=interval_minutes)

    last_processed_bar = load_last_processed_bar(db)
    logger.info("Starting trading loop for pairs: %s", ", ".join(CFG.PAIRS))
    while True:
        next_bar_closes = []
        for pair in CFG.PAIRS:
            df = feed.fetch_ohlc(pair, interval=CFG.TRADING_PARAMS["ohlc_interval"])
            if len(df) < 2:
                logger.warning("Not enough OHLC data for %s; received %d bars", pair, len(df))
                continue

            bar_time = df["time"].iloc[-2]
            bar_time_local = bar_time.tz_convert(LOCAL_TZ)
            bar_open_unix = int(bar_time.timestamp())
            bar_id = f"{pair}-{interval_minutes}-{bar_open_unix}"
            next_bar_closes.append(bar_time + bar_interval)

            logger.info("Processing bar %s (open=%s)", bar_id, bar_time)

            if last_processed_bar.get(pair) == bar_id:
                logger.debug("Already processed bar %s; skipping", bar_id)
                continue

            last_sig = db.last_signal(pair)
            result = analyze(df, CFG.INDICATORS, last_sig, pair=pair)

            sig = result["signal"]
            conf = result.get("confidence", 0)
            price = result.get("price")

            atr_pct_val = result.get("atr_pct")
            atr_str = f"{atr_pct_val:.4f}" if atr_pct_val is not None else "n/a"
            indicator_info = f"st={result.get('st_dir')} atr_pct={atr_str} ml_proba={result.get('ml_proba')} ml_gate={result.get('ml_gate')}"

            logger.info(
                "Signal for %s -> %s (confidence=%.2f, price=%s, %s)",
                pair,
                sig,
                conf,
                f"{price:.2f}" if price else "N/A",
                indicator_info,
            )

            expired = execu.expire_unfilled_orders(str(bar_id))
            for oid, expired_pair in expired:
                logger.warning("Order %s for %s expired", oid, expired_pair)

            db.insert_decision(
                (
                    datetime.now(tz=LOCAL_TZ),
                    pair,
                    sig,
                    conf,
                    price,
                    result.get("st_dir"),
                    result.get("ema_fast"),
                    result.get("ema_slow"),
                    result.get("atr"),
                    result.get("atr_pct"),
                    str(bar_time_local),
                    str(bar_id),
                    result.get("ml_proba"),
                    int(result.get("ml_gate")) if result.get("ml_gate") is not None else None,
                    result.get("ml_reason"),
                )
            )

            last_processed_bar[pair] = bar_id

            if sig == "HOLD" or bar_time is None:
                logger.info("No action for %s on bar %s", pair, bar_id)
                continue

            if execu.already_ordered_this_bar(bar_id):
                logger.info("Order already placed for %s; skipping new order", bar_id)
                continue

            size = CFG.TRADING_PARAMS["position_size_usd"] / price if price else 0

            slippage = CFG.TRADING_PARAMS.get("limit_slippage_pct", 0)
            limit_price = price * (1 - slippage) if sig.upper() == "BUY" else price * (1 + slippage)

            status = execu.place_limit(pair, sig, price, size, bar_time_local, bar_id)
            if status == "SUBMITTED":
                send_alert(
                    f"{pair} {sig} ORDER",
                    f"Limit {sig} @ {limit_price:.2f}, size={size:.4f}",
                )
                logger.info(
                    "%s order submitted: limit=%s size=%.4f", sig, f"{limit_price:.2f}", size
                )
            elif status == "ERROR":
                send_alert(f"{pair} ORDER ERROR", "See logs / DB for details", priority=1)
                logger.error("Error placing %s order for %s", sig, pair)

        if next_bar_closes:
            next_bar_close = min(next_bar_closes).tz_convert(LOCAL_TZ)
            sleep_seconds = (next_bar_close - pd.Timestamp.now(tz=LOCAL_TZ)).total_seconds()
            # Safety floor prevents non-positive sleep when clock skew makes next close appear past-due.
            time.sleep(max(1.0, sleep_seconds))
        else:
            logger.info(
                "No bars processed; sleeping %s seconds before retry", CFG.TRADING_PARAMS["trading_interval"]
            )
            time.sleep(CFG.TRADING_PARAMS["trading_interval"])


if __name__ == "__main__":
    run()
