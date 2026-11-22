"""Script-parity execution model (archive/lastly):
market entry, script sizing, traded-day validity, script daily DD, reset
payouts, add-on gating, balance overrides and exit classification."""
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from database.trade_store import TradeStore
from database.treasury_store import TreasuryStore
from engine.market import (
    FARMS, AQUA_FIRM_5K_PROFILE, FP_FIRM_5K_PROFILE, GOAT_FIRM_5K_PROFILE,
)
from engine.models import ChallengeState, Trade
from engine.reconciler import classify_exit
from engine.strategy_runner import StrategyRunner
from main import Bot


def _farm(tag="GOAT5", **over):
    f = dict(next(x for x in FARMS if x["tag"] == tag))
    f.update(over)
    return f


class TestScriptSizing(unittest.TestCase):
    def test_5k_sizing_with_fixed_25_dollar_stop(self):
        runner = StrategyRunner(_farm())
        ch = runner.state.current_challenge
        bar = {"open": 4300.0, "high": 4310.0, "low": 4290.0, "close": 4305.0}
        t = runner._compute_entry(signal=1, bar=bar, atr_val=10.0)
        self.assertIsNotNone(t)
        self.assertEqual(t.lots, 0.02)  # floor(0.01*5000/25) = 2 oz
        self.assertEqual(t.entry, 4300.0)  # market entry at the bar open
        self.assertAlmostEqual(t.sl, 4300.0 - 25.0)
        self.assertAlmostEqual(t.tp, 4300.0 + 75.0)  # RR 3

    def test_100k_uses_script_units(self):
        runner = StrategyRunner(_farm())
        ch = runner.state.current_challenge
        ch.equity = ch.phase_start_bal = ch.day_equity = 100000.0
        ch.dd_floor = 90000.0
        bar = {"open": 4300.0, "high": 4310.0, "low": 4290.0, "close": 4305.0}
        t = runner._compute_entry(signal=1, bar=bar, atr_val=10.0)
        self.assertIsNotNone(t)
        self.assertEqual(t.lots, 0.40)

    def test_ref_price_includes_open_unrealized(self):
        runner = StrategyRunner(_farm())
        ch = runner.state.current_challenge
        ch.equity = ch.phase_start_bal = ch.day_equity = 100000.0
        ch.dd_floor = 90000.0
        ch.open_trades = [Trade(id="old", strategy="Goat 5K", direction=1,
                                entry=4000.0, sl=3900.0, tp=4100.0, lots=1.0,
                                open_time=datetime.now(timezone.utc).isoformat(),
                                status="open", position_id=1)]
        bar = {"open": 4300.0, "high": 4310.0, "low": 4290.0, "close": 4305.0}
        t = runner._compute_entry(signal=1, bar=bar, atr_val=10.0, ref_price=4300.0)
        self.assertIsNotNone(t)
        self.assertEqual(t.lots, 0.52)

    def test_sizing_below_one_ounce_is_skipped(self):
        runner = StrategyRunner(_farm())
        ch = runner.state.current_challenge
        ch.equity = ch.phase_start_bal = ch.day_equity = 2000.0
        ch.dd_floor = 1800.0
        bar = {"open": 4300.0, "high": 4310.0, "low": 4290.0, "close": 4305.0}
        self.assertIsNone(runner._compute_entry(signal=1, bar=bar, atr_val=10.0))
        self.assertEqual(ch.open_trades, [])

    def test_best_open_r_is_max_and_ignores_pending(self):
        runner = StrategyRunner(_farm())
        ch = runner.state.current_challenge
        ch.open_trades = [
            Trade(id="a", strategy="Goat 5K", direction=1, entry=100.0, sl=90.0,
                  tp=110.0, lots=0.1, open_time="", status="open", position_id=1),
            Trade(id="b", strategy="Goat 5K", direction=1, entry=100.0, sl=90.0,
                  tp=110.0, lots=0.1, open_time="", status="pending"),
        ]
        self.assertAlmostEqual(runner._best_open_r(ch, 106.0), 0.6)


class TestTradedDayMode(unittest.TestCase):
    def test_any_pnl_day_counts_once(self):
        runner = StrategyRunner(_farm())
        ch = runner.state.current_challenge
        runner._add_closed_trade_day(ch, 100, 0.01)  # tiny PnL still counts
        runner._add_closed_trade_day(ch, 100, 5.0)   # same day, no double count
        runner._add_closed_trade_day(ch, 101, -1.0)
        self.assertEqual(ch.trading_days, 2)
        self.assertEqual(ch._traded_day_set, {100, 101})

    def test_refresh_counts_distinct_trade_days(self):
        runner = StrategyRunner(_farm())
        ch = runner.state.current_challenge
        day = int(datetime.now(timezone.utc).timestamp() / 86400) - 2
        ts = datetime.fromtimestamp(day * 86400, tz=timezone.utc).isoformat()
        ch.trade_history = [
            {"pnl": 1.0, "close_time": ts},
            {"pnl": 2.0, "close_time": ts},
        ]
        runner._refresh_qualifying_days(ch)
        self.assertEqual(ch.trading_days, 1)


class TestScriptDailyDD(unittest.TestCase):
    def test_day_rollover_uses_prior_eod_reference(self):
        runner = StrategyRunner(_farm())
        ch = runner.state.current_challenge
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        ch.day_start = now_day - 1
        ch._eod_ref = 5200.0
        ch.equity = 5100.0
        ch.day_equity = 5100.0
        ch.dd_floor = 4500.0
        runner._process_challenge_bar(ch, {"time": now.timestamp(), "close": 5100.0},
                                      10.0, 60, now_day, now)
        self.assertEqual(ch.day_equity, 5200.0)  # max(prev_eod, equity)

    def test_script_daily_floor_breach(self):
        runner = StrategyRunner(_farm())
        ch = runner.state.current_challenge
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        ch.day_start = now_day
        ch.day_equity = 5200.0
        ch.equity = 4900.0  # below 5200 * (1 - 0.04) = 4992
        ch.dd_floor = 4500.0
        runner._process_challenge_bar(ch, {"time": now.timestamp(), "close": 4900.0},
                                      10.0, 60, now_day, now)
        self.assertTrue(ch.blown)


class TestPayoutRetention(unittest.TestCase):
    def test_retain_mode_keeps_unpaid_profit(self):
        bot = MagicMock()
        runner = StrategyRunner(_farm(), bot=bot)
        ch = runner.state.current_challenge
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        ch.phase = 3
        ch.day_start = now_day
        ch.last_payout_day = now_day - 14
        ch.equity = 5600.0
        ch.ch_trade_pnl = 600.0
        ch.day_equity = 5500.0
        ch.dd_floor = 4500.0
        ch._traded_day_set = {now_day - 3, now_day - 2, now_day - 1}
        ch.trading_days = 3
        ch._last_payout_valid_idx = 0
        runner._process_challenge_bar(ch, {"time": now.timestamp(), "close": 5600.0},
                                      10.0, 60, now_day, now)
        bot.process_payout.assert_called_once()
        self.assertAlmostEqual(bot.process_payout.call_args[0][1], 480.0)
        self.assertEqual(ch.equity, 5120.0)
        self.assertEqual(ch.ch_trade_pnl, 0.0)
        self.assertEqual(ch.day_equity, 5500.0)  # day reference continuous
        self.assertAlmostEqual(ch.dd_floor, 4500.0)
        self.assertEqual(ch.last_payout_day, now_day)

    def test_all_firms_use_retain_mode(self):
        for profile in (GOAT_FIRM_5K_PROFILE, AQUA_FIRM_5K_PROFILE, FP_FIRM_5K_PROFILE):
            self.assertEqual(profile["payout_mode"], "retain")
            self.assertEqual(profile["first_payout_cap"], float("inf"))


class TestBalanceOverride(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.tstore = TreasuryStore(db_path=os.path.join(self._tmpdir, "t.db"))
        self.bot = Bot.__new__(Bot)
        self.bot.treasury_store = self.tstore
        self.bot._uploaded_credentials = {}

    def tearDown(self):
        self.tstore.close()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _runner(self):
        runner = StrategyRunner(_farm())
        runner._broker = {"account_id": 1}
        return runner

    def test_blank_balance_is_noop(self):
        runner = self._runner()
        self.bot._apply_uploaded_balance(runner, "GOAT5")
        self.assertEqual(runner.profile["ib"], 5000.0)

    def test_balance_scales_challenge_cost(self):
        runner = self._runner()
        self.bot._uploaded_credentials = {"GOAT5": {"balance": 10000.0,
                                                    "challenge_cost": None}}
        self.bot._apply_uploaded_balance(runner, "GOAT5")
        self.assertEqual(runner.profile["ib"], 10000.0)
        self.assertEqual(runner.profile["cc"], 34.0)  # 17 * 10000/5000
        # FARMS defaults untouched (per-runner copy)
        self.assertEqual(GOAT_FIRM_5K_PROFILE["ib"], 5000.0)

    def test_explicit_challenge_cost_wins(self):
        runner = self._runner()
        self.bot._uploaded_credentials = {"GOAT5": {"balance": 10000.0,
                                                    "challenge_cost": 55.0}}
        self.bot._apply_uploaded_balance(runner, "GOAT5")
        self.assertEqual(runner.profile["cc"], 55.0)

    def test_balance_change_resets_runner(self):
        runner = self._runner()
        self.bot._uploaded_credentials = {"GOAT5": {"balance": 10000.0}}
        self.bot._apply_uploaded_balance(runner, "GOAT5")
        self.bot._check_account_change(runner)
        self.assertEqual(len(runner.state.challenges), 1)
        self.assertEqual(runner.state.challenges[0].phase_start_bal, 10000.0)
        self.assertEqual(self.tstore.get_account_balance("Goat 5K"), 10000.0)
        # unchanged balance on the next boot is dormant
        runner.state.trade_pnl = 123.0
        self.bot._check_account_change(runner)
        self.assertEqual(runner.state.trade_pnl, 123.0)


class TestExitClassification(unittest.TestCase):
    def test_levels(self):
        t = {"tp": 110.0, "sl": 90.0}
        self.assertEqual(classify_exit(t, 110.5), "tp")
        self.assertEqual(classify_exit(t, 90.5), "sl")
        self.assertEqual(classify_exit(t, 100.0), "market")
        self.assertIsNone(classify_exit(t, None))

    def test_trade_store_roundtrip(self):
        tmp = tempfile.mkdtemp()
        store = TradeStore(os.path.join(tmp, "trades.db"))
        try:
            t = Trade(id="GOAT5-ch1-1", strategy="Goat 5K", direction=1,
                      entry=100.0, sl=90.0, tp=110.0, lots=0.1,
                      open_time="2026-01-01T00:00:00+00:00",
                      close_time="2026-01-01T01:00:00+00:00",
                      pnl=10.0, status="closed",
                      exit_price=110.0, exit_reason="tp")
            store.save_trade(t)
            rows = store.load_trades("Goat 5K")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["exit_price"], 110.0)
            self.assertEqual(rows[0]["exit_reason"], "tp")

            t2 = Trade(id="GOAT5-ch1-2", strategy="Goat 5K", direction=1,
                       entry=100.0, sl=90.0, tp=110.0, lots=0.1,
                       open_time="2026-01-01T02:00:00+00:00", status="open")
            store.save_trade(t2)
            store.update_trade_pnl("GOAT5-ch1-2", 10.0, status="closed",
                                   exit_price=110.0, exit_reason="tp")
            row = next(r for r in store.load_trades("Goat 5K") if r["id"] == "GOAT5-ch1-2")
            self.assertEqual(row["exit_price"], 110.0)
            self.assertEqual(row["exit_reason"], "tp")
        finally:
            store.close()
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
