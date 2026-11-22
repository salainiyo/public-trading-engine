import json
import os
import tempfile
import unittest

from database.treasury_store import TreasuryStore
from engine.models import ChallengeState, Trade
from engine.strategy_runner import StrategyRunner


def _make_farm_config():
    from engine.market import FARMS
    return dict(FARMS[0])


class TestTreasuryStore(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "test_treasuries.db")
        self.store = TreasuryStore(db_path=self.db_path)

    def tearDown(self):
        self.store.close()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_init_creates_table(self):
        self.assertIsNotNone(self.store._conn)

    def test_save_and_load(self):
        runner = StrategyRunner(_make_farm_config())
        runner.state.treasury = 100.0
        self.store.save({"CO": runner})
        runner2 = StrategyRunner(_make_farm_config())
        self.store.load({"CO": runner2})
        self.assertEqual(runner2.state.treasury, 100.0)

    def test_save_multiple_strategies(self):
        r1 = StrategyRunner(_make_farm_config())
        r1.state.treasury = 50.0
        cfg2 = _make_farm_config()
        cfg2["name"] = "Other"
        r2 = StrategyRunner(cfg2)
        r2.state.treasury = 75.0
        self.store.save({"CO": r1, "Other": r2})
        r1b = StrategyRunner(_make_farm_config())
        r2b = StrategyRunner(cfg2)
        self.store.load({"CO": r1b, "Other": r2b})
        self.assertEqual(r1b.state.treasury, 50.0)
        self.assertEqual(r2b.state.treasury, 75.0)

    def test_save_overwrites(self):
        runner = StrategyRunner(_make_farm_config())
        runner.state.treasury = 100.0
        self.store.save({"CO": runner})
        runner.state.treasury = 200.0
        self.store.save({"CO": runner})
        runner2 = StrategyRunner(_make_farm_config())
        self.store.load({"CO": runner2})
        self.assertEqual(runner2.state.treasury, 200.0)

    def test_load_empty(self):
        runner = StrategyRunner(_make_farm_config())
        self.store.load({"CO": runner})
        self.assertEqual(runner.state.treasury, 0.0)

    def test_json_roundtrip(self):
        runner = StrategyRunner(_make_farm_config())
        runner.state.treasury = 150.0
        runner.state.total_spent = 34.0
        runner.state.total_earned = 500.0
        runner.state.monthly_stats = {"2025-01": {"spent": 17, "earned": 100, "trade_pnl": 50, "withdrawn": 0}}
        self.store.save({"CO": runner})
        runner2 = StrategyRunner(_make_farm_config())
        self.store.load({"CO": runner2})
        self.assertEqual(runner2.state.treasury, 150.0)
        self.assertEqual(runner2.state.total_spent, 34.0)
        self.assertIn("2025-01", runner2.state.monthly_stats)

    def test_challenge_restore(self):
        runner = StrategyRunner(_make_farm_config())
        ch = runner.state.current_challenge
        ch.phase = 2
        ch.equity = 5500.0
        ch.ch_trade_pnl = 500.0
        runner.state.treasury = 200.0
        self.store.save({"CO": runner})
        runner2 = StrategyRunner(_make_farm_config())
        self.store.load({"CO": runner2})
        restored = runner2.state.current_challenge
        self.assertEqual(restored.phase, 2)


if __name__ == "__main__":
    unittest.main()
