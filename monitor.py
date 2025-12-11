"""Simple CLI monitor for recent decisions and orders."""
from tabulate import tabulate

import config as CFG
from core.db import DB


def main(decisions: int = 10, orders: int = 10):
    db = DB(CFG.DB_PATH)

    print("=== Recent Decisions ===")
    dec_rows = db.fetch_last_decisions(decisions)
    dec_table = [
        [
            r[1],
            r[2],
            f"{r[3]:.0%}",
            f"{r[4]:.2f}",
            r[5],
            "ema9>ema21" if r[6] > r[7] else "ema9<ema21",
            f"{r[8]*100:.2f}%",
            r[9],
        ]
        for r in dec_rows
    ]
    print(tabulate(dec_table, headers=["Pair", "Signal", "Conf", "Price", "ST Dir", "EMA", "ATR%", "Bar Time"]))

    print("\n=== Recent Orders ===")
    ord_rows = db.fetch_last_orders(orders)
    ord_table = [
        [r[1], r[2], f"{r[3]:.2f}", f"{r[4]:.6f}", r[5], r[6]] for r in ord_rows
    ]
    print(tabulate(ord_table, headers=["Pair", "Side", "Price", "Size", "Status", "Bar Time"]))


if __name__ == "__main__":
    main()
