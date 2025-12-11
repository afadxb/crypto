from datetime import datetime

class Execution:
    def __init__(self, db, api, cfg):
        self.db = db
        self.api = api
        self.cfg = cfg

    def already_ordered_this_bar(self, pair, bar_time):
        return self.db.last_order_for_bar(pair, str(bar_time))

    def place_limit(self, pair, side, price, size, bar_time):
        order = self.api.query_private("AddOrder", {
            "pair": pair,
            "type": side.lower(),
            "ordertype": "limit",
            "price": str(round(price, 2)),
            "volume": str(size),
        })

        status = "SUBMITTED" if "error" not in order or not order["error"] else "ERROR"

        self.db.insert_order((
            datetime.utcnow(), pair, side, price, size, status, str(bar_time)
        ))

        return status
