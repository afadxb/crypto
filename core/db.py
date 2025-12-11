import sqlite3
import os
from datetime import datetime

class DB:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        c = self.conn.cursor()

        c.execute("""
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
        """)

        c.execute("""
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
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS balance_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            asset TEXT,
            balance REAL
        );
        """)

        self.conn.commit()

    def insert_decision(self, row):
        q = """INSERT INTO decisions(
            timestamp, pair, signal, confidence, price, st_dir,
            ema_fast, ema_slow, atr, atr_pct, bar_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        self.conn.execute(q, row)
        self.conn.commit()

    def insert_order(self, row):
        q = """INSERT INTO orders(
            timestamp, pair, side, price, size, status, bar_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?)"""
        self.conn.execute(q, row)
        self.conn.commit()

    def last_order_for_bar(self, pair, bar_time):
        q = "SELECT id FROM orders WHERE pair=? AND bar_time=? LIMIT 1"
        r = self.conn.execute(q, (pair, bar_time)).fetchone()
        return r is not None
    