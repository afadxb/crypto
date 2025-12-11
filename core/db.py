"""SQLite integration for recording decisions and orders."""
import sqlite3
from datetime import datetime
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
            bar_time TEXT,
            bar_id TEXT
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
            bar_time TEXT,
            filled_size REAL DEFAULT 0,
            filled_price REAL DEFAULT 0,
            filled_at TEXT,
            bar_id TEXT
        );
        """
        )

        c.execute(
            """
        CREATE TABLE IF NOT EXISTS positions (
            pair TEXT PRIMARY KEY,
            side TEXT,
            size REAL,
            avg_price REAL,
            updated_at TEXT
        );
        """
        )

        for alter in (
            "ALTER TABLE orders ADD COLUMN filled_size REAL DEFAULT 0",
            "ALTER TABLE orders ADD COLUMN filled_price REAL DEFAULT 0",
            "ALTER TABLE orders ADD COLUMN filled_at TEXT",
            "ALTER TABLE orders ADD COLUMN bar_id TEXT",
            "ALTER TABLE decisions ADD COLUMN bar_id TEXT",
        ):
            try:
                c.execute(alter)
            except sqlite3.OperationalError:
                # Column already exists
                pass

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

        c.execute("CREATE INDEX IF NOT EXISTS idx_decisions_pair_bar ON decisions(pair, bar_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_orders_pair_bar ON orders(pair, bar_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_decisions_time ON decisions(timestamp);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_orders_time ON orders(timestamp);")

        self.conn.commit()

    def insert_decision(self, row: Tuple):
        query = """INSERT INTO decisions(
            timestamp, pair, signal, confidence, price, st_dir,
            ema_fast, ema_slow, atr, atr_pct, bar_time, bar_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        self.conn.execute(query, row)
        self.conn.commit()

    def insert_order(self, row: Tuple):
        query = """INSERT INTO orders(
            timestamp, pair, side, price, size, status, bar_time, filled_size, filled_price, filled_at, bar_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        self.conn.execute(query, row)
        self.conn.commit()

    def last_order_for_bar(self, bar_id: str) -> bool:
        query = "SELECT id FROM orders WHERE bar_id=? LIMIT 1"
        result = self.conn.execute(query, (bar_id,)).fetchone()
        return result is not None

    def fetch_last_decisions(self, limit: int = 10):
        query = (
            "SELECT timestamp, pair, signal, confidence, price, st_dir, "
            "ema_fast, ema_slow, atr_pct, bar_time, bar_id FROM decisions "
            "ORDER BY id DESC LIMIT ?"
        )
        return self.conn.execute(query, (limit,)).fetchall()

    def fetch_last_orders(self, limit: int = 10):
        query = (
            "SELECT timestamp, pair, side, price, size, status, bar_time, bar_id "
            "FROM orders ORDER BY id DESC LIMIT ?"
        )
        return self.conn.execute(query, (limit,)).fetchall()

    def last_signal(self, pair: str) -> Optional[str]:
        q = "SELECT signal FROM decisions WHERE pair=? ORDER BY timestamp DESC LIMIT 1"
        row = self.conn.execute(q, (pair,)).fetchone()
        return row[0] if row else None

    def get_position(self, pair: str):
        q = "SELECT side, size, avg_price FROM positions WHERE pair=?"
        row = self.conn.execute(q, (pair,)).fetchone()
        if not row:
            return None
        return {"side": row[0], "size": row[1], "avg_price": row[2]}

    def upsert_position(self, pair: str, side: str, size: float, avg_price: float):
        ts = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO positions (pair, side, size, avg_price, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(pair) DO UPDATE SET side=excluded.side,
                                            size=excluded.size,
                                            avg_price=excluded.avg_price,
                                            updated_at=excluded.updated_at
            """,
            (pair, side, size, avg_price, ts),
        )
        self.conn.commit()
