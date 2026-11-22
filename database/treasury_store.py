"""Treasury persistence module for the trading engine."""
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Optional

from engine.models import Trade, ChallengeState

logger = logging.getLogger(__name__)


class TreasuryStore:
    """Manages treasury persistence using SQLite."""

    def __init__(self, db_path: str = "bot/data/treasuries.db"):
        self.db_path = db_path
        self._conn = None
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS treasuries (
                strategy TEXT PRIMARY KEY,
                treasury REAL NOT NULL DEFAULT 0.0
            )
        """)
        for col in ["total_spent", "total_earned", "trade_pnl"]:
            try:
                self._conn.execute(f"ALTER TABLE treasuries ADD COLUMN {col} REAL NOT NULL DEFAULT 0.0")
            except sqlite3.OperationalError:
                pass
        try:
            self._conn.execute("ALTER TABLE treasuries ADD COLUMN monthly_stats TEXT NOT NULL DEFAULT '{}'")
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute("ALTER TABLE treasuries ADD COLUMN daily_pnl TEXT DEFAULT '{}'")
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute("ALTER TABLE treasuries ADD COLUMN phase INTEGER NOT NULL DEFAULT 1")
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute("ALTER TABLE treasuries ADD COLUMN challenges TEXT NOT NULL DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute("ALTER TABLE treasuries ADD COLUMN account_id TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute("ALTER TABLE treasuries ADD COLUMN account_balance REAL")
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, runners: dict):
        if self._conn is None:
            return
        try:
            cursor = self._conn.cursor()
            for name, runner in runners.items():
                ch_list = []
                for c in runner.state.challenges:
                    cd = asdict(c)
                    cd.pop("_trading_day_set", None)
                    cd["_traded_day_set"] = sorted(cd.get("_traded_day_set") or [])
                    if cd.get("open_trade") and isinstance(cd["open_trade"], Trade):
                        cd["open_trade"] = asdict(cd["open_trade"])
                    ch_list.append(cd)
                cursor.execute(
                    "INSERT INTO treasuries "
                    "(strategy, treasury, total_spent, total_earned, trade_pnl, "
                    " monthly_stats, daily_pnl, phase, challenges, account_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(strategy) DO UPDATE SET "
                    "treasury=excluded.treasury, total_spent=excluded.total_spent, "
                    "total_earned=excluded.total_earned, trade_pnl=excluded.trade_pnl, "
                    "monthly_stats=excluded.monthly_stats, daily_pnl=excluded.daily_pnl, "
                    "phase=excluded.phase, challenges=excluded.challenges, "
                    "account_id=COALESCE(excluded.account_id, account_id)",
                    (name, runner.state.treasury, runner.state.total_spent,
                     runner.state.total_earned, runner.state.trade_pnl,
                     json.dumps(runner.state.monthly_stats),
                     json.dumps(runner.state.daily_pnl),
                     runner.state.current_challenge.phase if runner.state.current_challenge else 1,
                     json.dumps(ch_list),
                     getattr(runner.state, "account_id", None)),
                )
            self._conn.commit()
        except Exception as e:
            logger.error(f"[Treasury] Failed to save: {e}")

    def load(self, runners: dict):
        if self._conn is None:
            return
        try:
            cols = [d[1] for d in self._conn.execute("PRAGMA table_info(treasuries)").fetchall()]
            has_challenges = "challenges" in cols
            has_account = "account_id" in cols
            if has_challenges:
                sel = ("SELECT strategy, treasury, total_spent, total_earned, trade_pnl, "
                       "monthly_stats, daily_pnl, phase, challenges"
                       + (", account_id" if has_account else "") + " FROM treasuries")
                rows = self._conn.execute(sel).fetchall()
            else:
                sel = ("SELECT strategy, treasury, total_spent, total_earned, trade_pnl, "
                       "monthly_stats, daily_pnl, phase"
                       + (", account_id" if has_account else "") + " FROM treasuries")
                rows = self._conn.execute(sel).fetchall()
            loaded = 0
            for row in rows:
                name = row[0]
                if name in runners:
                    runner = runners[name]
                    runner.state.treasury = row[1]
                    runner.state.total_spent = row[2] if row[2] is not None else 0.0
                    runner.state.total_earned = row[3] if row[3] is not None else 0.0
                    runner.state.trade_pnl = row[4] if row[4] is not None else 0.0

                    try:
                        ms = json.loads(row[5]) if row[5] else {}
                        if isinstance(ms, dict):
                            for mk, mv in ms.items():
                                if isinstance(mv, dict):
                                    runner.state.monthly_stats[mk] = {
                                        "spent": mv.get("spent", 0),
                                        "earned": mv.get("earned", 0),
                                        "trade_pnl": mv.get("trade_pnl", 0),
                                        "withdrawn": mv.get("withdrawn", 0),
                                    }
                    except Exception as e:
                        logger.warning(f"[Treasury] Monthly stats merge failed for {name}: {e}")

                    try:
                        dp = json.loads(row[6]) if row[6] else {}
                        if isinstance(dp, dict):
                            from datetime import timedelta
                            cutoff = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=365)).strftime("%Y-%m-%d")
                            runner.state.daily_pnl = {k: v for k, v in dp.items() if k >= cutoff}
                    except Exception:
                        pass

                    if has_challenges and len(row) > 8 and row[8]:
                        try:
                            ch_data = json.loads(row[8])
                            restored = []
                            for cd in ch_data:
                                if cd.get("open_trade") and isinstance(cd["open_trade"], dict):
                                    cd["open_trade"] = Trade(**cd["open_trade"])
                                ots = cd.get("open_trades")
                                if isinstance(ots, list):
                                    cd["open_trades"] = [Trade(**ot) if isinstance(ot, dict) else ot for ot in ots]
                                # JSON turns int day-epoch keys into strings — restore
                                # them so day-keyed lookups (_day_cap, _day_pnl) work
                                for day_key in ("_day_cap", "_day_pnl"):
                                    d = cd.get(day_key)
                                    if isinstance(d, dict):
                                        cd[day_key] = {int(k): v for k, v in d.items()
                                                       if str(k).lstrip('-').isdigit()}
                                ch = ChallengeState(**cd)
                                ch._trading_day_set = set()
                                ch._traded_day_set = set(cd.get("_traded_day_set") or [])
                                restored.append(ch)
                            if restored:
                                runner.state.challenges = restored
                        except Exception as e:
                            logger.warning(f"[Treasury] Challenge restore failed for {name}: {e}")

                    ch = runner.state.current_challenge
                    if ch and not has_challenges:
                        ch.phase = row[7] if len(row) > 7 and row[7] is not None else 1

                    if has_account:
                        idx = 9 if has_challenges else 8
                        runner.state.account_id = (row[idx] if len(row) > idx and row[idx] else None)
                    loaded += 1
            if loaded:
                logger.info(f"[Treasury] Loaded {loaded} runner treasuries from DB")
        except Exception as e:
            logger.error(f"[Treasury] Failed to load: {e}")

    def get_account_balance(self, name: str) -> Optional[float]:
        """Stored account balance for a runner, or None."""
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT account_balance FROM treasuries WHERE strategy = ?", (name,)
            ).fetchone()
            return float(row[0]) if row and row[0] is not None else None
        except Exception:
            return None

    def set_account_balance(self, name: str, balance: float):
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT INTO treasuries (strategy, account_balance) VALUES (?, ?) "
                "ON CONFLICT(strategy) DO UPDATE SET account_balance=excluded.account_balance",
                (name, float(balance)),
            )
            self._conn.commit()
        except Exception as e:
            logger.error(f"[Treasury] set_account_balance failed: {e}")

    def get_account_id(self, name: str) -> Optional[str]:
        """Stored account_id for a runner, or None."""
        try:
            row = self._conn.execute(
                "SELECT account_id FROM treasuries WHERE strategy=?", (name,)).fetchone()
            return row[0] if row and row[0] else None
        except Exception:
            return None

    def set_account_id(self, name: str, account_id):
        try:
            self._conn.execute(
                "UPDATE treasuries SET account_id=? WHERE strategy=?",
                (str(account_id) if account_id is not None else None, name))
            self._conn.commit()
        except Exception as e:
            logger.error(f"[Treasury] Failed to set account_id for {name}: {e}")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None