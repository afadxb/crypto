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
        self.local_tz = cfg.LOCAL_TZ

    def _latest_price(self, pair: str, fallback: float) -> float:
        row = self.db.conn.execute(
            "SELECT price FROM decisions WHERE pair=? ORDER BY timestamp DESC LIMIT 1",
            (pair,),
        ).fetchone()
        if row and row[0] is not None:
            return row[0]
        return fallback

    def already_ordered_this_bar(self, bar_id: str) -> bool:
        return self.db.last_order_for_bar(str(bar_id))

    def _calc_size(self, price: float) -> float:
        usd_size = self.cfg.TRADING_PARAMS["position_size_usd"]
        return usd_size / price if price else 0

    def current_total_exposure(self, price_lookup):
        total = 0.0
        q = "SELECT pair, side, size FROM positions WHERE side IS NOT NULL"
        for pair, side, size in self.db.conn.execute(q):
            if side != "LONG":
                continue
            p = price_lookup(pair)
            if p:
                total += max(size, 0) * p
        return total

    def _pair_exposure(self, pair: str, price_lookup):
        position = self.db.get_position(pair)
        if not position or position.get("side") != "LONG":
            return 0.0
        price = price_lookup(pair)
        size = max(position.get("size", 0), 0)
        return size * price if price else 0.0

    def expire_unfilled_orders(self, current_bar_id):
        q = "SELECT id, pair FROM orders WHERE status='SUBMITTED' AND bar_id != ?"
        rows = self.db.conn.execute(q, (current_bar_id,)).fetchall()
        for oid, pair in rows:
            # Kraken cancel would go here if txid were stored
            self.db.conn.execute("UPDATE orders SET status='EXPIRED' WHERE id=?", (oid,))
        if rows:
            self.db.conn.commit()
        return rows

    def place_limit(self, pair: str, side: str, price: float, size: float = None, bar_time=None, bar_id=None):
        side = side.upper()
        position = self.db.get_position(pair)
        has_long = position is not None and position.get("side") == "LONG"
        position_size = position.get("size", 0) if position else 0

        if side == "BUY" and not has_long:
            order_size = size if size is not None else self._calc_size(price)
        elif side == "BUY" and has_long:
            # Pyramiding/add to existing long
            order_size = size if size is not None else self._calc_size(price)
        elif side == "SELL":
            if not has_long or position_size <= 0:
                return "SKIPPED"
            calc_size = size if size is not None else self._calc_size(price)
            order_size = min(calc_size, position_size)
            if order_size <= 0:
                return "SKIPPED"
        else:
            return "SKIPPED"

        slippage = self.cfg.TRADING_PARAMS.get("limit_slippage_pct", 0)
        limit_price = price * (1 - slippage) if side == "BUY" else price * (1 + slippage)

        def price_lookup(p):
            if p == pair:
                return price
            return self._latest_price(p, price)

        pair_exposure = self._pair_exposure(pair, price_lookup)
        projected_pair = pair_exposure + abs(order_size * price)
        total_exposure = self.current_total_exposure(price_lookup) + abs(order_size * price)

        max_pair = self.cfg.TRADING_PARAMS.get("max_pair_exposure_usd")
        max_total = self.cfg.TRADING_PARAMS.get("max_total_exposure_usd")

        if (max_pair and projected_pair > max_pair) or (max_total and total_exposure > max_total):
            print(f"Exposure cap hit for {pair}; skipping order.")
            return "SKIPPED"

        order = self.api.query_private(
            "AddOrder",
            {
                "pair": pair,
                "type": side.lower(),
                "ordertype": "limit",
                "price": str(round(limit_price, 2)),
                "volume": str(order_size),
            },
        )

        status = "SUBMITTED" if "error" not in order or not order["error"] else "ERROR"

        self.db.insert_order(
            (
                datetime.now(tz=self.local_tz),
                pair,
                side,
                limit_price,
                order_size,
                status,
                str(bar_time),
                0,
                0,
                None,
                str(bar_id),
            )
        )

        return status
