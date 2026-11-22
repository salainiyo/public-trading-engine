"""Integration tests — cross-module wiring, persistence, and snapshot."""
import os
import pytest

from events.event_bus import EventBus
from engine.market import FARMS
from engine.strategy_runner import StrategyRunner
from engine.signals import SIGNAL_MAP
from database.treasury_store import TreasuryStore


class TestIntegration:
    def test_event_bus_communication(self):
        bus = EventBus()
        received = []
        bus.subscribe("test_event", lambda et, d: received.append(d))
        bus.publish("test_event", {"value": 42})
        assert len(received) == 1
        assert received[0]["value"] == 42

    def test_initialize_components(self):
        from main import Bot
        assert Bot is not None
        assert hasattr(Bot, "__init__")

    def test_runners_created(self):
        assert len(FARMS) == 3
        names = [f["name"] for f in FARMS]
        assert "Goat 5K" in names
        assert "Aqua Standard 5K" in names
        assert "FundingPips 2-Step" in names
        for farm in FARMS:
            assert farm["eval_strat"] in SIGNAL_MAP
            assert farm["funded_strat"] in SIGNAL_MAP
            assert farm["stop_mode"] == "fixed"
            assert farm["stop_value"] == 25.0

    def test_save_restore_treasuries(self, tmp_path):
        db_path = str(tmp_path / "treasuries.db")
        store = TreasuryStore(db_path)
        try:
            runner = StrategyRunner(FARMS[0])
            runner.state.treasury = 100.0
            runner.state.total_spent = 17.0
            runner.state.total_earned = 50.0
            runner.state.trade_pnl = 30.0
            store.save({"Trail Champion": runner})

            store2 = TreasuryStore(db_path)
            runner2 = StrategyRunner(FARMS[0])
            try:
                store2.load({"Trail Champion": runner2})
                assert runner2.state.treasury == 100.0
                assert runner2.state.total_spent == 17.0
                assert runner2.state.total_earned == 50.0
                assert runner2.state.trade_pnl == 30.0
            finally:
                store2.close()
        finally:
            store.close()

    def test_full_snapshot(self):
        runner = StrategyRunner(FARMS[0])
        snap = runner.snapshot()
        assert isinstance(snap, dict)
        for key in (
            "strategy", "farm_type", "challenges", "active_count",
            "total_challenges", "total_spent", "total_earned", "treasury",
            "net_pnl", "trade_pnl", "total_blown", "recent_trades",
            "monthly_stats", "daily_pnl", "winrate",
        ):
            assert key in snap, f"Missing snapshot key: {key}"
        assert snap["strategy"] == "Goat 5K"
        assert snap["active_count"] >= 1
        assert snap["total_challenges"] >= 1
