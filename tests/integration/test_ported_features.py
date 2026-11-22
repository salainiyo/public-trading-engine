"""Ported-feature integration tests — lifecycle, reconciler, treasury, indicators."""
import os
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from engine.strategy_runner import StrategyRunner
from engine.reconciler import Reconciler
from engine.models import Trade, ChallengeState, StrategyState
from engine.market import FARMS, GOAT_FIRM_5K_PROFILE, COMMISSION, CONTRACT_SIZE
from engine.indicators import build_indicator_dict
from engine.signals import SIGNAL_MAP
from database.treasury_store import TreasuryStore
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner():
    return StrategyRunner(FARMS[0])


def _synthetic_bars(n=100, base=4300.0):
    rng = np.random.RandomState(42)
    opens = base + rng.randn(n).cumsum() * 2
    highs = opens + np.abs(rng.randn(n))
    lows = opens - np.abs(rng.randn(n))
    closes = opens + rng.randn(n) * 0.5
    volumes = rng.randint(100, 1000, n).astype(float)
    times = np.arange(n) * 3600 + 1_700_000_000
    return opens, highs, lows, closes, volumes, times


# ===========================================================================
# TestFullLifecycle
# ===========================================================================


class TestFullLifecycle:
    def test_init_to_entry_to_close(self):
        runner = _make_runner()
        ch = runner.state.current_challenge
        assert ch is not None
        assert ch.phase == 1
        assert ch.equity == 5000.0

        bar = {"open": 4300.0, "high": 4310.0, "low": 4290.0, "close": 4305.0}
        trade = runner._compute_entry(signal=1, bar=bar, atr_val=10.0)
        assert trade is not None
        assert trade.direction == 1
        assert trade.strategy == "Goat 5K"
        assert len(ch.open_trades) == 1
        assert ch.open_trades[0].id == trade.id
        assert trade.sl < trade.entry < trade.tp
        assert trade.entry - trade.sl == pytest.approx(25.0)
        assert trade.tp - trade.entry == pytest.approx(75.0)

        before_equity = ch.equity
        trade.pnl = 150.0
        now_dt = datetime.now(timezone.utc)
        runner.on_trade_closed(trade, now_dt)

        assert len(ch.open_trades) == 0
        pnl_net = 150.0 - COMMISSION * trade.lots
        assert ch.equity == pytest.approx(before_equity + pnl_net, abs=0.01)
        assert len(ch.trade_history) == 1

    def test_phase_transition_and_treasury(self):
        runner = _make_runner()
        ch = runner.state.current_challenge
        ib = GOAT_FIRM_5K_PROFILE["ib"]
        ch.ch_trade_pnl = ib * 0.10
        ch.trading_days = 3
        now_dt = datetime.now(timezone.utc)
        now_day = int(now_dt.timestamp() / 86400)
        advanced = runner._advance_phase(ch, now_day, now_dt)
        assert advanced is True
        assert ch.phase == 2
        assert ch.equity == ib
        assert ch.ch_trade_pnl == 0.0
        assert ch.trading_days == 0

    def test_trailing_stop_lifecycle(self):
        runner = _make_runner()
        runner.config["use_trail"] = True
        runner._last_bar_atr = 10.0
        ch = runner.state.current_challenge
        t = Trade(
            id="test-ch1-1", strategy="Continuation Open",
            direction=1, entry=4300.0, sl=4270.0, tp=4420.0,
            lots=0.01, open_time=datetime.now(timezone.utc).isoformat(),
            position_id=42,
        )
        t._sl_dist = 30.0
        t._trail_px = 4270.0
        ch.open_trades.append(t)

        bar = {"open": 4310.0, "high": 4330.0, "low": 4305.0, "close": 4325.0}
        closed = runner._update_trailing_stop_single(ch, t, bar, 10.0)
        assert closed is False
        assert t._trail_px > 4270.0


# ===========================================================================
# TestReconcilerIntegration
# ===========================================================================


class TestReconcilerIntegration:
    def test_reconcile_and_settle(self):
        rec = Reconciler()
        runner = _make_runner()
        runners = {"Continuation Open": runner}
        reconciled, settled, closed_stale, position_map = rec.reconcile([], runners)
        assert reconciled == 0
        assert settled == 0
        assert closed_stale == 0

    def test_manual_close_detection_full_cycle(self):
        rec = Reconciler()
        runner = _make_runner()
        ch = runner.state.current_challenge
        t = Trade(
            id="test-ch1-1", strategy="Continuation Open",
            direction=1, entry=4300.0, sl=4270.0, tp=4420.0,
            lots=0.01, open_time=datetime.now(timezone.utc).isoformat(),
            position_id=12345,
        )
        ch.open_trades.append(t)

        broker_positions = [{
            "id": 12345,
            "comment": "Continuation Open|test-ch1-1|ch_1",
            "side": "buy",
            "qty": 0.01,
            "openPrice": 4300.0,
            "closePrice": 4350.0,
            "status": "closed",
            "pnl": 50.0,
        }]

        runners = {"Continuation Open": runner}
        reconciled, settled, closed_stale, position_map = rec.reconcile(
            broker_positions, runners,
        )
        assert settled == 1
        assert len(ch.open_trades) == 0
        assert len(ch.trade_history) >= 1


# ===========================================================================
# TestTreasuryIntegration
# ===========================================================================


class TestTreasuryIntegration:
    def test_save_and_restore_state(self, tmp_path):
        db_path = str(tmp_path / "treasury_int.db")
        store = TreasuryStore(db_path)
        try:
            runner = _make_runner()
            runner.state.treasury = 100.0
            runner.state.total_spent = 17.0
            runner.state.total_earned = 50.0
            store.save({"Continuation Open": runner})

            store2 = TreasuryStore(db_path)
            runner2 = _make_runner()
            try:
                store2.load({"Continuation Open": runner2})
                assert runner2.state.treasury == 100.0
                assert runner2.state.total_spent == 17.0
                assert runner2.state.total_earned == 50.0
            finally:
                store2.close()
        finally:
            store.close()


# ===========================================================================
# TestIndicatorsIntegration
# ===========================================================================


class TestIndicatorsIntegration:
    def test_build_dict_matches_signal_requirements(self):
        opens, highs, lows, closes, volumes, times = _synthetic_bars(100)
        r = build_indicator_dict(opens, highs, lows, closes, volumes, times)
        assert set(r) == {
            "O", "H", "L", "C", "V", "time",
            "RSI14", "ATR14", "ATR10", "RB_H20", "RB_L20",
        }
        for key in r:
            assert len(r[key]) == 100
        assert not np.all(np.isnan(r["RSI14"][14:]))
        assert not np.all(np.isnan(r["ATR14"][14:]))
        assert not np.all(np.isnan(r["ATR10"][10:]))
        assert not np.all(np.isnan(r["RB_H20"][20:]))
        assert not np.all(np.isnan(r["RB_L20"][20:]))

    def test_signal_map_completeness(self):
        assert set(SIGNAL_MAP) == {"Supertrend RSI"}
        assert callable(SIGNAL_MAP["Supertrend RSI"])


class TestTrailStatePersistence:
    def test_trade_store_persists_trail_fields(self, tmp_path):
        from database.trade_store import TradeStore
        store = TradeStore(str(tmp_path / "trades.db"))
        try:
            t = Trade(id="TCH-ch1-1", strategy="Trail Champion", direction=1,
                      entry=4400.0, sl=4420.5, tp=1e9, lots=0.01,
                      open_time=datetime.now(timezone.utc).isoformat(),
                      position_id=99, _trail_px=4420.5, _sl_dist=20.5)
            store.save_trade(t)
            row = store.load_open_trade_by_id("TCH-ch1-1")
            assert row is not None
            assert row["_trail_px"] == 4420.5
            assert row["_sl_dist"] == 20.5
        finally:
            store.close()

    def test_trading_stop_updates_sl_and_persists(self):
        runner = _make_runner()
        runner.config["use_trail"] = True
        runner.config["trail_atr_mult"] = 1.0
        runner.config["trail_act"] = 0.5
        runner._last_bar_atr = 10.0
        ch = runner.state.current_challenge
        bar = {"open": 4300.0, "high": 4310.0, "low": 4295.0, "close": 4305.0}
        trade = runner._compute_entry(signal=1, bar=bar, atr_val=10.0)
        assert trade is not None
        trade.position_id = 123  # filled — trailing only applies to live positions
        initial_sl = trade.sl
        # First trailing call lazy-inits _sl_dist and _trail_px (no move yet)
        bar1 = {"open": 4300.0, "high": 4301.0, "low": 4299.0, "close": 4300.5}
        closed = runner._update_trailing_stop_single(ch, trade, bar1, 10.0)
        assert not closed
        assert trade._sl_dist > 0
        assert trade._trail_px == initial_sl
        # Move price up past activation (0.5R) -> trail should move
        bar2 = {"open": 4306.0, "high": 4315.0, "low": 4306.0, "close": 4314.0}
        closed = runner._update_trailing_stop_single(ch, trade, bar2, 10.0)
        assert not closed
        assert trade._trail_px > initial_sl
        assert trade.sl == round(trade._trail_px, 2)

    def test_trail_activation_not_reached_no_move(self):
        runner = _make_runner()
        runner.config["use_trail"] = True
        runner.config["trail_atr_mult"] = 1.0
        runner.config["trail_act"] = 0.5
        runner._last_bar_atr = 10.0
        ch = runner.state.current_challenge
        bar = {"open": 4300.0, "high": 4310.0, "low": 4295.0, "close": 4305.0}
        trade = runner._compute_entry(signal=1, bar=bar, atr_val=10.0)
        assert trade is not None
        trade.position_id = 123  # filled — trailing only applies to live positions
        initial_sl = trade.sl
        # Price moves only 0.2R from the LIMIT entry (< 0.5R activation) -> no trail
        bar2 = {"open": 4295.0, "high": 4297.0, "low": 4294.0, "close": 4296.0}
        closed = runner._update_trailing_stop_single(ch, trade, bar2, 10.0)
        assert not closed
        assert trade._trail_px == initial_sl
        assert trade.sl == initial_sl

    def test_treasury_roundtrip_restores_trade_objects_with_trail_state(self, tmp_path):
        store = TreasuryStore(str(tmp_path / "treasuries.db"))
        try:
            runner = StrategyRunner(FARMS[0])
            ch = runner.state.current_challenge
            t = Trade(id="TCH-ch1-9", strategy="Trail Champion", direction=-1,
                      entry=4309.68, sl=4349.58, tp=1e9, lots=0.01,
                      open_time=datetime.now(timezone.utc).isoformat(),
                      position_id=123, _trail_px=4345.0, _sl_dist=39.9)
            ch.open_trades.append(t)
            store.save({"Trail Champion": runner})

            store2 = TreasuryStore(str(tmp_path / "treasuries.db"))
            runner2 = StrategyRunner(FARMS[0])
            try:
                store2.load({"Trail Champion": runner2})
                ch2 = runner2.state.current_challenge
                assert len(ch2.open_trades) == 1
                ot = ch2.open_trades[0]
                assert isinstance(ot, Trade), f"expected Trade, got {type(ot).__name__}"
                assert ot.position_id == 123
                assert ot._trail_px == 4345.0
                assert ot._sl_dist == 39.9
            finally:
                store2.close()
        finally:
            store.close()
