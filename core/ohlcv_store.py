"""SQLite-backed OHLCV storage for historical and training data."""
from __future__ import annotations

import sqlite3
from typing import Iterable, List, Optional

import config as CFG


class OHLCVStore:
    """Persist and query OHLCV bars with idempotent upserts.

    Bars are keyed by (exchange, pair, timeframe, ts). ``ts`` is stored as the
    UTC epoch second for the start of the bar (e.g., 14:00:00Z for the hour
    covering 14:00-14:59:59).
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or CFG.DB_PATH
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._create_table()

    def _create_table(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                exchange TEXT NOT NULL,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                ts INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                vwap REAL,
                trades INTEGER,
                PRIMARY KEY (exchange, pair, timeframe, ts)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_pair_timeframe_ts ON ohlcv(pair, timeframe, ts);"
        )
        self.conn.commit()

    def upsert_ohlcv(
        self,
        bars: Iterable[dict],
        exchange: str = "kraken",
        pair: Optional[str] = None,
        timeframe: str = "1h",
    ) -> int:
        """Insert or update bars, returning the number of rows processed."""

        rows = [
            (
                exchange,
                bar.get("pair", pair),
                bar.get("timeframe", timeframe),
                int(bar["ts"]),
                bar.get("open"),
                bar.get("high"),
                bar.get("low"),
                bar.get("close"),
                bar.get("volume"),
                bar.get("vwap"),
                bar.get("trades"),
            )
            for bar in bars
        ]
        if not rows:
            return 0

        self.conn.executemany(
            """
            INSERT INTO ohlcv (
                exchange, pair, timeframe, ts, open, high, low, close, volume, vwap, trades
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(exchange, pair, timeframe, ts) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                vwap=excluded.vwap,
                trades=excluded.trades
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str = "1h",
        limit: Optional[int] = None,
        start_ts: Optional[int] = None,
        exchange: str = "kraken",
    ) -> List[dict]:
        """Fetch OHLCV rows in ascending order."""

        query = (
            "SELECT ts, open, high, low, close, volume, vwap, trades FROM ohlcv "
            "WHERE exchange=? AND pair=? AND timeframe=?"
        )
        params: list = [exchange, pair, timeframe]
        if start_ts is not None:
            query += " AND ts >= ?"
            params.append(int(start_ts))
        query += " ORDER BY ts ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))

        cur = self.conn.execute(query, params)
        columns = [col[0] for col in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def latest_ts(self, pair: str, timeframe: str = "1h", exchange: str = "kraken") -> Optional[int]:
        cur = self.conn.execute(
            "SELECT ts FROM ohlcv WHERE exchange=? AND pair=? AND timeframe=? ORDER BY ts DESC LIMIT 1",
            (exchange, pair, timeframe),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None
