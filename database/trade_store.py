"""Trade persistence module for the trading engine."""
import logging
import re
import threading
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Optional

from engine.models import Trade

logger = logging.getLogger(__name__)


class TradeStore:
    """Manages trade persistence using SQLite."""

    def __init__(self, db_path: str = "bot/data/trades.db"):
        self.db_path = db_path
        self._conn = None
        self._lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._counter = 0
        self._init_db()

    def _init_db(self):
        import sqlite3
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                strategy TEXT NOT NULL,
                direction INTEGER NOT NULL,
                entry REAL NOT NULL,
                sl REAL NOT NULL,
                tp REAL NOT NULL,
                lots REAL NOT NULL,
                open_time TEXT NOT NULL,
                close_time TEXT,
                pnl REAL,
                status TEXT DEFAULT 'open',
                position_id INTEGER,
                _trail_px REAL DEFAULT 0.0,
                _sl_dist REAL DEFAULT 0.0,
                exit_price REAL,
                exit_reason TEXT
            )
        """)
        # Migration: add trail columns to pre-existing DBs
        cols = [d[1] for d in self._conn.execute("PRAGMA table_info(trades)").fetchall()]
        for col in ("_trail_px", "_sl_dist"):
            if col not in cols:
                self._conn.execute(f"ALTER TABLE trades ADD COLUMN {col} REAL DEFAULT 0.0")
        # Migration: fail_reason persists WHY an entry never traded (docker logs
        # die with the container; the DB keeps the explanation)
        if "fail_reason" not in cols:
            self._conn.execute("ALTER TABLE trades ADD COLUMN fail_reason TEXT")
        # Migration: exit price/reason (parity + dashboard exit visibility)
        if "exit_price" not in cols:
            self._conn.execute("ALTER TABLE trades ADD COLUMN exit_price REAL")
        if "exit_reason" not in cols:
            self._conn.execute("ALTER TABLE trades ADD COLUMN exit_reason TEXT")
        self._conn.commit()

    def _connect(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _next_trade_id(self) -> int:
        with self._id_lock:
            self._counter += 1
            return self._counter

    def reseed_counter(self, rows: list):
        """Reseed from the rows passed — counts ALL statuses (failed rows' ids must
        not be reused, otherwise INSERT OR REPLACE would overwrite them)."""
        max_num = 0
        for row in rows:
            m = re.search(r'-ch\d+-(\d+)$', str(row.get("id") or ""))
            if m:
                try:
                    max_num = max(max_num, int(m.group(1)))
                except ValueError:
                    pass
        if max_num > self._counter:
            self._counter = max_num

    def reseed_counter_global(self):
        """Reseed the global trade-id counter from ALL rows (any status, any strategy).
        load_trades() excludes open/pending rows, so the per-runner reseed can miss
        live trades' ids and later reissue them — overwriting live rows."""
        if self._conn is None:
            return
        try:
            cur = self._conn.execute("SELECT id FROM trades")
            max_num = self._counter
            for (tid,) in cur.fetchall():
                m = re.search(r'-ch\d+-(\d+)$', str(tid or ""))
                if m:
                    try:
                        max_num = max(max_num, int(m.group(1)))
                    except ValueError:
                        pass
            self._counter = max_num
        except Exception as e:
            logger.error(f"Failed to reseed trade-id counter: {e}")

    def save_trade(self, trade):
        with self._lock:
            try:
                if hasattr(trade, '__dataclass_fields__'):
                    data = asdict(trade)
                else:
                    data = trade
                self._conn.execute(
                    "INSERT OR REPLACE INTO trades "
                    "(id, strategy, direction, entry, sl, tp, lots, open_time, close_time, pnl, status, position_id, _trail_px, _sl_dist, exit_price, exit_reason) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (data["id"], data["strategy"], data["direction"], data["entry"],
                     data["sl"], data["tp"], data["lots"], data["open_time"],
                     data.get("close_time"), data.get("pnl"), data.get("status", "open"),
                     data.get("position_id"), data.get("_trail_px", 0.0) or 0.0,
                     data.get("_sl_dist", 0.0) or 0.0,
                     data.get("exit_price"), data.get("exit_reason"))
                )
                self._conn.commit()
            except Exception as e:
                logger.error(f"Failed to save trade {data.get('id')}: {e}")

    def load_trades(self, strategy_name: str, limit: int = 50) -> list:
        if self._conn is None:
            return []
        try:
            cur = self._conn.execute(
                "SELECT * FROM trades WHERE strategy=? AND status NOT IN ('open', 'pending') "
                "ORDER BY COALESCE(close_time, open_time) DESC LIMIT ?",
                (strategy_name, limit)
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to load trades for {strategy_name}: {e}")
            return []

    def load_open_trade_by_id(self, trade_id: str) -> Optional[dict]:
        if self._conn is None:
            return None
        try:
            cur = self._conn.execute(
                "SELECT id, open_time, position_id, sl, tp, _trail_px, _sl_dist FROM trades WHERE id=?",
                (trade_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            return {"id": row[0], "open_time": row[1], "position_id": row[2],
                    "sl": row[3], "tp": row[4],
                    "_trail_px": row[5] or 0.0, "_sl_dist": row[6] or 0.0}
        except Exception as e:
            logger.error(f"Failed to load open trade {trade_id}: {e}")
            return None

    def update_trade_pnl(self, trade_id: str, pnl: float, status: str = "closed",
                         reason: str = None, exit_price: float = None,
                         exit_reason: str = None):
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE trades SET pnl=?, status=? WHERE id=?",
                    (pnl, status, trade_id)
                )
                if exit_price is not None:
                    self._conn.execute(
                        "UPDATE trades SET exit_price=?, exit_reason=? WHERE id=?",
                        (exit_price, exit_reason, trade_id)
                    )
                if reason:
                    self._conn.execute(
                        "UPDATE trades SET fail_reason=? WHERE id=?",
                        (reason, trade_id)
                    )
                self._conn.commit()
            except Exception as e:
                logger.error(f"Failed to update trade {trade_id}: {e}")

    def mark_stale_rows_failed(self, ids: list, reason: str = None) -> int:
        """Mark trade rows 'failed' by id (locked — the watchdog thread shares
        this connection with the poll thread)."""
        if not ids:
            return 0
        with self._lock:
            try:
                total = 0
                for tid in ids:
                    cur = self._conn.execute(
                        "UPDATE trades SET pnl=NULL, status='failed' WHERE id=?", (tid,))
                    total += cur.rowcount
                if reason:
                    for tid in ids:
                        self._conn.execute(
                            "UPDATE trades SET fail_reason=? WHERE id=?",
                            (reason, tid))
                self._conn.commit()
                return total
            except Exception as e:
                logger.error(f"Failed to mark rows failed: {e}")
                return 0

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None