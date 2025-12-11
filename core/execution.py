"""Order execution layer."""
from datetime import datetime
from typing import Any

from krakenex import API

from .db import DB


class Execution:
    def __init__(self, db: DB, api: API, cfg: Any):
        self.db = db
        self.api = api
        self.cfg = cfg

    def already_ordered_this_bar(self, pair: str, bar_time) -> bool:
        return self.db.last_order_for_bar(pair, str(bar_time))

    def _calc_size(self, price: float) -> float:
        usd_size = self.cfg.TRADING_PARAMS["position_size_usd"]
        return usd_size / price if price else 0

    def place_limit(self, pair: str, side: str, price: float, size: float = None, bar_time=None):
        order_size = size if size is not None else self._calc_size(price)

        order = self.api.query_private(
            "AddOrder",
            {
                "pair": pair,
                "type": side.lower(),
                "ordertype": "limit",
                "price": str(round(price, 2)),
                "volume": str(order_size),
            },
        )

        status = "SUBMITTED" if "error" not in order or not order["error"] else "ERROR"

        self.db.insert_order(
            (
                datetime.utcnow(),
                pair,
                side,
                price,
                order_size,
                status,
                str(bar_time),
            )
        )

        return status
