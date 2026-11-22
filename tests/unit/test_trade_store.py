import os
import tempfile
import unittest

from database.trade_store import TradeStore
from engine.models import Trade


class TestTradeStore(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "test_trades.db")
        self.store = TradeStore(db_path=self.db_path)

    def tearDown(self):
        self.store.close()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_trade(self, **overrides):
        defaults = dict(
            id="t1", strategy="CO", direction=1,
            entry=100.0, sl=99.0, tp=103.0,
            lots=0.1, open_time="2025-01-15T10:00:00",
            status="open",
        )
        defaults.update(overrides)
        return Trade(**defaults)

    def test_init_creates_tables(self):
        self.assertIsNotNone(self.store._conn)

    def test_save_and_load_trade(self):
        t = self._make_trade(status="closed", close_time="2025-01-15T11:00:00", pnl=50.0)
        self.store.save_trade(t)
        rows = self.store.load_trades("CO")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "t1")

    def test_save_multiple_trades(self):
        for i in range(5):
            t = self._make_trade(id=f"t{i}", status="closed", close_time=f"2025-01-1{i}T10:00:00", pnl=float(i))
            self.store.save_trade(t)
        rows = self.store.load_trades("CO")
        self.assertEqual(len(rows), 5)

    def test_save_trade_update(self):
        t = self._make_trade()
        self.store.save_trade(t)
        t.pnl = 50.0
        t.status = "closed"
        self.store.save_trade(t)
        rows = self.store.load_trades("CO")
        self.assertEqual(rows[0]["pnl"], 50.0)

    def test_load_trades_empty(self):
        rows = self.store.load_trades("NONEXISTENT")
        self.assertEqual(rows, [])

    def test_load_trades_with_strategy_filter(self):
        t1 = self._make_trade(id="t1", strategy="CO", status="closed", pnl=10.0)
        t2 = self._make_trade(id="t2", strategy="Other", status="closed", pnl=20.0)
        self.store.save_trade(t1)
        self.store.save_trade(t2)
        rows = self.store.load_trades("CO")
        self.assertEqual(len(rows), 1)

    def test_load_trades_filters_open(self):
        t1 = self._make_trade(id="t1", status="open")
        t2 = self._make_trade(id="t2", status="closed", pnl=10.0)
        self.store.save_trade(t1)
        self.store.save_trade(t2)
        rows = self.store.load_trades("CO")
        self.assertEqual(len(rows), 1)

    def test_load_open_trade_by_id(self):
        t = self._make_trade()
        self.store.save_trade(t)
        result = self.store.load_open_trade_by_id("t1")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "t1")

    def test_load_open_trade_by_id_not_found(self):
        result = self.store.load_open_trade_by_id("nonexistent")
        self.assertIsNone(result)

    def test_update_trade_pnl(self):
        t = self._make_trade(status="closed", close_time="2025-01-15T11:00:00")
        self.store.save_trade(t)
        self.store.update_trade_pnl("t1", 75.0, "closed")
        rows = self.store.load_trades("CO")
        self.assertEqual(rows[0]["pnl"], 75.0)
        self.assertEqual(rows[0]["status"], "closed")

    def test_reseed_counter(self):
        rows = [{"id": "CO-ch1-5"}, {"id": "CO-ch1-10"}, {"id": "CO-ch1-3"}]
        self.store.reseed_counter(rows)
        self.assertEqual(self.store._counter, 10)

    def test_reseed_counter_counts_failed(self):
        rows = [{"id": "CO-ch1-5", "status": "closed"}, {"id": "CO-ch1-99", "status": "failed"}]
        self.store.reseed_counter(rows)
        self.assertEqual(self.store._counter, 99)

    def test_reseed_counter_global_covers_all_rows(self):
        # Simulate the live-row blind spot: reseed_counter (per-runner rows) misses
        # open rows; reseed_counter_global must see every row of every strategy.
        self.store.save_trade(self._make_trade(id="FR-ch1-76", status="open"))
        self.store.save_trade(self._make_trade(status="closed", close_time="2025-01-15T11:00:00"))
        self.store.reseed_counter([{"id": "CO-ch1-5", "status": "closed"}])
        self.store.reseed_counter_global()
        self.assertEqual(self.store._counter, 76)


if __name__ == "__main__":
    unittest.main()
