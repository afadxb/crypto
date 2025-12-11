"""SQLite integration for recording decisions and orders."""
import sqlite3
from typing import Optional, Tuple

import config as CFG


class DB:
    def __init__(self, path: Optional[str] = None):
        self.path = path or CFG.DB_PATH
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        c = self.conn.cursor()

        c.execute(
            """
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            pair TEXT,
            signal TEXT,
            confidence REAL,
            price REAL,
            st_dir TEXT,
            ema_fast REAL,
            ema_slow REAL,
            atr REAL,
            atr_pct REAL,
            bar_time TEXT
        );
        """
        )

        c.execute(
            """
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            pair TEXT,
            side TEXT,
            price REAL,
            size REAL,
            status TEXT,
            bar_time TEXT
        );
        """
        )

        c.execute(
            """
        CREATE TABLE IF NOT EXISTS balance_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            asset TEXT,
            balance REAL
        );
        """
        )

        self.conn.commit()

    def insert_decision(self, row: Tuple):
        query = """INSERT INTO decisions(
            timestamp, pair, signal, confidence, price, st_dir,
            ema_fast, ema_slow, atr, atr_pct, bar_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        self.conn.execute(query, row)
        self.conn.commit()

    def insert_order(self, row: Tuple):
        query = """INSERT INTO orders(
            timestamp, pair, side, price, size, status, bar_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?)"""
        self.conn.execute(query, row)
        self.conn.commit()

    def last_order_for_bar(self, pair: str, bar_time: str) -> bool:
        query = "SELECT id FROM orders WHERE pair=? AND bar_time=? LIMIT 1"
        result = self.conn.execute(query, (pair, bar_time)).fetchone()
        return result is not None

    def fetch_last_decisions(self, limit: int = 10):
        query = (
            "SELECT timestamp, pair, signal, confidence, price, st_dir, "
            "ema_fast, ema_slow, atr_pct, bar_time FROM decisions "
            "ORDER BY id DESC LIMIT ?"
        )
        return self.conn.execute(query, (limit,)).fetchall()

    def fetch_last_orders(self, limit: int = 10):
        query = (
            "SELECT timestamp, pair, side, price, size, status, bar_time "
            "FROM orders ORDER BY id DESC LIMIT ?"
        )
        return self.conn.execute(query, (limit,)).fetchall()
