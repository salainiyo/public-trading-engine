import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from engine.strategy_runner import StrategyRunner
from engine.models import Trade
from engine.market import (
    GOAT_FIRM_5K_PROFILE, AQUA_FIRM_5K_PROFILE, FP_FIRM_5K_PROFILE,
)


def _make_farm_config(**overrides):
    cfg = {
        "name": "Goat 5K",
        "tag": "GOAT5",
        "profile": GOAT_FIRM_5K_PROFILE,
        "broker": "goat100k",
        "broker_key": "herofx_goat",
        "mode": "rec",
        "eval_strat": "Supertrend RSI",
        "eval_tf": 60,
        "eval_rr": 3.0,
        "funded_strat": "Supertrend RSI",
        "funded_tf": 60,
        "funded_rr": 3.0,
        "eval_rpt": 0.01,
        "funded_rpt": 0.01,
        "stop_mode": "fixed",
        "stop_value": 25.0,
        "entry_slip": 0.0,
        "use_delayed_signal": True,
        "entry_delay_bars": 1,
        "max_hold_bars": 0,
        "max_concurrent": 2,
        "sizing_mode": "script",
        "addon_min_r": None,
        "early_close": True,
    }
    cfg.update(overrides)
    return cfg


def _bot():
    bot = MagicMock()
    bot.trade_store = MagicMock()
    bot.trade_store._next_trade_id.return_value = 1
    bot.treasury_store = MagicMock()
    return bot


class TestStrategyRunner(unittest.TestCase):
    def test_initialization(self):
        runner = StrategyRunner(_make_farm_config())
        self.assertEqual(runner.name, "Goat 5K")
        self.assertEqual(runner.state.challenges[0].phase, 1)
        self.assertEqual(runner.state.total_spent, GOAT_FIRM_5K_PROFILE["cc"])
        self.assertEqual(runner.state.challenges[0].equity, 5000.0)

    def test_add_new_challenge(self):
        runner = StrategyRunner(_make_farm_config())
        n_before = len(runner.state.challenges)
        runner._add_new_challenge()
        self.assertEqual(len(runner.state.challenges), n_before + 1)

    def test_advance_phase_1_to_2(self):
        runner = StrategyRunner(_make_farm_config())
        ch = runner.state.current_challenge
        ch.ch_trade_pnl = 450.0  # GOAT target: 8% of 5000 = 400
        ch.trading_days = 5
        now = datetime.now(timezone.utc)
        result = runner._advance_phase(ch, int(now.timestamp() / 86400), now)
        self.assertTrue(result)
        self.assertEqual(ch.phase, 2)
        self.assertEqual(ch.trading_days, 0)

    def test_advance_phase_1_insufficient_pnl_or_days(self):
        runner = StrategyRunner(_make_farm_config())
        ch = runner.state.current_challenge
        now = datetime.now(timezone.utc)
        ch.ch_trade_pnl = 350.0
        ch.trading_days = 5
        self.assertFalse(runner._advance_phase(ch, int(now.timestamp() / 86400), now))
        ch.ch_trade_pnl = 450.0
        ch.trading_days = 2
        self.assertFalse(runner._advance_phase(ch, int(now.timestamp() / 86400), now))
        self.assertEqual(ch.phase, 1)

    def test_advance_phase_2_to_3(self):
        runner = StrategyRunner(_make_farm_config())
        ch = runner.state.current_challenge
        ch.phase = 2
        ch.ch_trade_pnl = 350.0  # GOAT target: 6% of 5000 = 300
        ch.trading_days = 5
        now = datetime.now(timezone.utc)
        result = runner._advance_phase(ch, int(now.timestamp() / 86400), now)
        self.assertTrue(result)
        self.assertEqual(ch.phase, 3)
        self.assertIsNotNone(ch.funded_date)
        self.assertEqual(ch.last_payout_day, int(now.timestamp() / 86400))

    def test_compute_entry_buy(self):
        bot = _bot()
        runner = StrategyRunner(_make_farm_config(), bot=bot)
        ch = runner.state.current_challenge
        bar = {"open": 4300.0, "high": 4310.0, "low": 4290.0, "close": 4305.0}
        trade = runner._compute_entry(1, bar, atr_val=5.0, ch=ch)
        self.assertIsNotNone(trade)
        self.assertEqual(trade.direction, 1)
        self.assertEqual(trade.entry, 4300.0)
        self.assertEqual(trade.sl, 4275.0)
        self.assertEqual(trade.tp, 4375.0)
        self.assertEqual(len(ch.open_trades), 1)
        bot.trade_store.save_trade.assert_called_once_with(trade)

    def test_compute_entry_no_signal(self):
        runner = StrategyRunner(_make_farm_config(), bot=_bot())
        ch = runner.state.current_challenge
        bar = {"open": 4300.0, "high": 4310.0, "low": 4290.0, "close": 4305.0}
        self.assertIsNone(runner._compute_entry(0, bar, atr_val=5.0, ch=ch))
        self.assertEqual(ch.open_trades, [])

    def test_compute_entry_rejects_when_trade_open(self):
        runner = StrategyRunner(_make_farm_config(max_concurrent=1), bot=_bot())
        ch = runner.state.current_challenge
        ch.open_trades = [MagicMock()]
        bar = {"open": 4300.0, "high": 4310.0, "low": 4290.0, "close": 4305.0}
        self.assertIsNone(runner._compute_entry(1, bar, atr_val=5.0, ch=ch))

    def test_on_trade_closed(self):
        bot = _bot()
        runner = StrategyRunner(_make_farm_config(), bot=bot)
        ch = runner.state.current_challenge
        t = Trade(id="t1", strategy="Goat 5K", direction=1, entry=100.0, sl=99.0,
                  tp=103.0, lots=0.1, open_time=datetime.now(timezone.utc).isoformat(),
                  position_id=1)
        ch.open_trades = [t]
        t.pnl = 150.0
        runner.on_trade_closed(t, datetime.now(timezone.utc), ch)
        self.assertEqual(len(ch.open_trades), 0)
        self.assertEqual(len(ch.trade_history), 1)
        self.assertAlmostEqual(ch.equity, 5000.0 + 150.0 - 0.3)

    def test_on_trade_closed_blows_on_dd(self):
        bot = _bot()
        runner = StrategyRunner(_make_farm_config(), bot=bot)
        ch = runner.state.current_challenge
        ch.dd_floor = 4500.0
        ch.day_equity = 5000.0
        ch.equity = 4501.0
        t = Trade(id="t1", strategy="Goat 5K", direction=1, entry=100.0, sl=99.0,
                  tp=103.0, lots=1.0, open_time=datetime.now(timezone.utc).isoformat(),
                  position_id=1)
        ch.open_trades = [t]
        t.pnl = -600.0
        runner.on_trade_closed(t, datetime.now(timezone.utc), ch)
        self.assertTrue(ch.blown)
        bot._recycle.assert_called_once()

    def test_add_closed_trade_day(self):
        runner = StrategyRunner(_make_farm_config())
        ch = runner.state.current_challenge
        now_day = int(datetime.now(timezone.utc).timestamp() / 86400)
        runner._add_closed_trade_day(ch, now_day, 100.0)
        runner._add_closed_trade_day(ch, now_day, 5.0)
        self.assertEqual(ch._day_pnl[now_day], 105.0)
        self.assertEqual(ch.trading_days, 1)  # entry mode: a day counts once
        self.assertIn(now_day, ch._traded_day_set)

    def test_refresh_qualifying_days(self):
        runner = StrategyRunner(_make_farm_config())
        ch = runner.state.current_challenge
        now = datetime.now(timezone.utc)
        ch.trade_history = [{"close_time": now.isoformat(), "pnl": 100.0}]
        runner._refresh_qualifying_days(ch)
        self.assertEqual(ch.trading_days, 1)

    def test_apply_daily_profit_cap(self):
        profile = {**GOAT_FIRM_5K_PROFILE, "daily_profit_cap": 1000.0}
        runner = StrategyRunner(_make_farm_config(profile=profile))
        ch = runner.state.current_challenge
        ch.phase = 3
        now_day = int(datetime.now(timezone.utc).timestamp() / 86400)
        ch._day_pnl[now_day] = 4000.0
        old_equity = ch.equity
        runner._apply_daily_profit_cap(ch, now_day)
        self.assertAlmostEqual(ch.equity, old_equity - 3000.0)

    def test_phase_progress(self):
        runner = StrategyRunner(_make_farm_config())
        ch = runner.state.current_challenge
        ch.ch_trade_pnl = 200.0
        prog = StrategyRunner._phase_progress(ch, runner.profile, ch.ch_trade_pnl)
        self.assertEqual(prog["phase"], 1)
        self.assertAlmostEqual(prog["pct"], 50.0, places=1)

    def test_snapshot(self):
        runner = StrategyRunner(_make_farm_config())
        snap = runner.snapshot()
        self.assertEqual(snap["strategy"], "Goat 5K")
        self.assertIn("challenges", snap)
        self.assertIn("active_count", snap)
        self.assertEqual(snap["active_count"], 1)

    def test_win_rate(self):
        runner = StrategyRunner(_make_farm_config())
        ch = runner.state.current_challenge
        ch.trade_history = [{"pnl": 100.0}, {"pnl": -50.0}, {"pnl": 200.0}]
        self.assertAlmostEqual(runner._win_rate(), 66.7, places=1)

    def test_win_rate_empty(self):
        self.assertEqual(StrategyRunner(_make_farm_config())._win_rate(), 0.0)

    def test_load_trade_history_phase1(self):
        ts = MagicMock()
        ts.load_trades.return_value = []
        runner = StrategyRunner(_make_farm_config())
        runner._load_trade_history(ts)
        self.assertEqual(runner.state.trade_pnl, 0.0)

    def test_load_trade_history_phase2_no_pnl_reconstruct(self):
        ts = MagicMock()
        ts.load_trades.return_value = []
        runner = StrategyRunner(_make_farm_config())
        runner.state.current_challenge.phase = 2
        runner._load_trade_history(ts)
        self.assertEqual(runner.state.trade_pnl, 0.0)


class TestFixedDollarStopAndSizing(unittest.TestCase):
    def test_fixed_stop_produces_entry_pm25_sl_and_pm75_tp(self):
        runner = StrategyRunner(_make_farm_config(), bot=_bot())
        ch = runner.state.current_challenge
        bar = {"open": 4300.0, "high": 4320.0, "low": 4280.0, "close": 4305.0}
        long = runner._compute_entry(1, bar, atr_val=999.0, ch=ch)
        self.assertIsNotNone(long)
        self.assertAlmostEqual(long.entry, 4300.0)
        self.assertAlmostEqual(long.sl, 4275.0)
        self.assertAlmostEqual(long.tp, 4375.0)
        self.assertAlmostEqual(long.entry - long.sl, 25.0)
        self.assertAlmostEqual(long.tp - long.entry, 75.0)  # RR 3

        short = runner._compute_entry(-1, bar, atr_val=999.0, ch=ch)
        self.assertIsNotNone(short)
        self.assertAlmostEqual(short.sl, 4325.0)
        self.assertAlmostEqual(short.tp, 4225.0)
        self.assertAlmostEqual(short.sl - short.entry, 25.0)
        self.assertAlmostEqual(short.entry - short.tp, 75.0)

    def test_script_sizing_uses_one_percent_and_25_dollar_distance(self):
        runner = StrategyRunner(_make_farm_config(), bot=_bot())
        ch = runner.state.current_challenge
        bar = {"open": 4300.0, "high": 4320.0, "low": 4280.0, "close": 4305.0}
        trade = runner._compute_entry(1, bar, atr_val=1.0, ch=ch)
        self.assertAlmostEqual(trade.lots, 0.02)

        ch.equity = ch.day_equity = 100_000.0
        ch.dd_floor = 90_000.0
        ch.open_trades = []
        trade = runner._compute_entry(1, bar, atr_val=1.0, ch=ch)
        self.assertAlmostEqual(trade.lots, 0.40)

    def test_script_sizing_skips_below_one_ounce(self):
        runner = StrategyRunner(_make_farm_config(), bot=_bot())
        ch = runner.state.current_challenge
        ch.equity = ch.day_equity = 2000.0  # 1% = $20 -> 20/25 < 1 oz
        ch.dd_floor = 1800.0
        bar = {"open": 4300.0, "high": 4320.0, "low": 4280.0, "close": 4305.0}
        self.assertIsNone(runner._compute_entry(1, bar, atr_val=10.0, ch=ch))
        self.assertEqual(ch.open_trades, [])

        ch.equity = ch.day_equity = 2500.0  # 1% = $25 -> exactly 1 oz
        trade = runner._compute_entry(1, bar, atr_val=10.0, ch=ch)
        self.assertIsNotNone(trade)
        self.assertAlmostEqual(trade.lots, 0.01)

    def test_valid_day_mode_entry_marks_day_at_fill(self):
        runner = StrategyRunner(_make_farm_config(), bot=_bot())
        ch = runner.state.current_challenge
        bar = {"open": 4300.0, "high": 4320.0, "low": 4280.0, "close": 4305.0}
        fake_now = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
        with patch("engine.strategy_runner.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            trade = runner._compute_entry(1, bar, atr_val=10.0, ch=ch)
        self.assertIsNotNone(trade)
        expected_day = int(fake_now.timestamp() / 86400)
        self.assertIn(expected_day, ch._traded_day_set)
        self.assertEqual(ch.trading_days, 1)

    def test_other_valid_day_mode_does_not_mark_at_fill(self):
        profile = {**GOAT_FIRM_5K_PROFILE, "valid_day_mode": "pct"}
        runner = StrategyRunner(_make_farm_config(profile=profile), bot=_bot())
        ch = runner.state.current_challenge
        bar = {"open": 4300.0, "high": 4320.0, "low": 4280.0, "close": 4305.0}
        runner._compute_entry(1, bar, atr_val=10.0, ch=ch)
        self.assertEqual(ch._traded_day_set, set())
        self.assertEqual(ch.trading_days, 0)


class TestStrategyRunnerTrailingStop(unittest.TestCase):
    """Trailing remains a configurable (legacy) engine feature."""

    def _runner(self, **cfg_over):
        bot = _bot()
        bot.price_digits = 2
        cfg = _make_farm_config(use_trail=True, trail_atr_mult=3.0, trail_act=0.5,
                                **cfg_over)
        return StrategyRunner(cfg, bot=bot), bot

    @staticmethod
    def _trade(entry=2000.0, sl=1985.0, tp=2030.0, sl_dist=15.0, trail_px=1985.0,
               position_id=7, status="open"):
        t = Trade(id="t1", strategy="Goat 5K", direction=1, entry=entry, sl=sl,
                  tp=tp, lots=0.1, open_time=datetime.now(timezone.utc).isoformat(),
                  status=status, position_id=position_id)
        t._sl_dist = sl_dist
        t._trail_px = trail_px
        return t

    def test_trailing_stop_activation(self):
        runner, _ = self._runner()
        t = self._trade()
        runner.state.current_challenge.open_trades = [t]
        runner._last_bar_atr = 5.0
        bar = {"high": 2020.0, "low": 2000.0, "close": 2015.0}
        self.assertFalse(runner._update_trailing_stop_single(
            runner.state.current_challenge, t, bar, 5.0))
        self.assertGreater(t._trail_px, 1985.0)

    def test_trailing_stop_exit(self):
        runner, _ = self._runner()
        t = self._trade(sl=1990.0, sl_dist=10.0, trail_px=2005.0)
        ch = runner.state.current_challenge
        ch.open_trades = [t]
        runner._last_bar_atr = 5.0
        bar = {"high": 2004.0, "low": 2003.0, "close": 2003.5}
        self.assertTrue(runner._update_trailing_stop_single(ch, t, bar, 5.0))

    def test_trailing_stop_no_exit_when_above(self):
        runner, _ = self._runner()
        t = self._trade(sl=1990.0, sl_dist=10.0, trail_px=2005.0)
        ch = runner.state.current_challenge
        ch.open_trades = [t]
        runner._last_bar_atr = 5.0
        bar = {"high": 2020.0, "low": 2010.0, "close": 2015.0}
        self.assertFalse(runner._update_trailing_stop_single(ch, t, bar, 5.0))

    def test_trailing_stop_pushes_sl_to_broker(self):
        runner, bot = self._runner()
        t = self._trade(position_id=99)
        ch = runner.state.current_challenge
        ch.open_trades = [t]
        runner._last_bar_atr = 5.0
        bar = {"high": 2020.0, "low": 2000.0, "close": 2015.0}
        runner._update_trailing_stop_single(ch, t, bar, 5.0)
        bot.push_trail_sl_to_broker.assert_called_once()
        args = bot.push_trail_sl_to_broker.call_args[0]
        self.assertIs(args[0], runner)
        self.assertIs(args[1], t)

    def test_trailing_stop_skips_pending_limit(self):
        runner, bot = self._runner()
        t = self._trade(status="pending", position_id=None)
        ch = runner.state.current_challenge
        ch.open_trades = [t]
        runner._last_bar_atr = 5.0
        bar = {"high": 2020.0, "low": 2000.0, "close": 2015.0}
        self.assertFalse(runner._update_trailing_stop_single(ch, t, bar, 5.0))
        self.assertEqual(t._trail_px, 1985.0)
        self.assertEqual(t.sl, 1985.0)
        bot.push_trail_sl_to_broker.assert_not_called()

    def test_trailing_stop_no_push_without_position(self):
        runner, bot = self._runner()
        t = self._trade(position_id=None)
        ch = runner.state.current_challenge
        ch.open_trades = [t]
        runner._last_bar_atr = 5.0
        bar = {"high": 2020.0, "low": 2000.0, "close": 2015.0}
        runner._update_trailing_stop_single(ch, t, bar, 5.0)
        bot.push_trail_sl_to_broker.assert_not_called()


class TestFridayFlat(unittest.TestCase):
    """Venue rule exercised as a generic config flag (not tied to a profile)."""

    def _runner(self, phase):
        bot = _bot()
        runner = StrategyRunner(_make_farm_config(friday_flat=True), bot=bot)
        ch = runner.state.current_challenge
        ch.phase = phase
        t = Trade(id="t1", strategy="Goat 5K", direction=1, entry=2000.0,
                  sl=1985.0, tp=2030.0, lots=0.1,
                  open_time=datetime.now(timezone.utc).isoformat())
        ch.open_trades = [t]
        return bot, runner, ch, t

    def _bar_close(self, runner, ch, when):
        runner._process_challenge_bar(ch, {"time": int(when.timestamp())}, 15.0, 60,
                                      int(when.timestamp() / 86400), when)

    def test_funded_friday_19utc_force_closes(self):
        bot, runner, ch, t = self._runner(phase=3)
        self._bar_close(runner, ch, datetime(2026, 10, 9, 19, 0, tzinfo=timezone.utc))
        bot._close_trade_at_market.assert_called_once_with(t, runner, ch)
        bot._cancel_pending_for_challenge.assert_called_once_with(runner, ch)

    def test_eval_phase_also_flat(self):
        for phase in (1, 2):
            bot, runner, ch, t = self._runner(phase=phase)
            self._bar_close(runner, ch, datetime(2026, 10, 9, 19, 0, tzinfo=timezone.utc))
            bot._close_trade_at_market.assert_called_once_with(t, runner, ch)

    def test_weekend_window_boundaries(self):
        _, runner, _, _ = self._runner(phase=3)
        self.assertFalse(runner.is_weekend_flat_window(
            datetime(2026, 10, 9, 18, 59, tzinfo=timezone.utc)))
        self.assertTrue(runner.is_weekend_flat_window(
            datetime(2026, 10, 9, 19, 0, tzinfo=timezone.utc)))
        self.assertTrue(runner.is_weekend_flat_window(
            datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)))
        self.assertTrue(runner.is_weekend_flat_window(
            datetime(2026, 10, 11, 23, 0, tzinfo=timezone.utc)))
        self.assertFalse(runner.is_weekend_flat_window(
            datetime(2026, 10, 12, 0, 0, tzinfo=timezone.utc)))
        self.assertFalse(runner.is_weekend_flat_window(
            datetime(2026, 10, 9, 19, 30, tzinfo=timezone.utc), live=True))
        self.assertTrue(runner.is_weekend_flat_window(
            datetime(2026, 10, 9, 20, 0, tzinfo=timezone.utc), live=True))

    def test_funded_non_friday_no_close(self):
        bot, runner, ch, _ = self._runner(phase=3)
        self._bar_close(runner, ch, datetime(2026, 10, 12, 19, 0, tzinfo=timezone.utc))
        bot._close_trade_at_market.assert_not_called()

    def test_funded_friday_wrong_hour_no_close(self):
        bot, runner, ch, _ = self._runner(phase=3)
        self._bar_close(runner, ch, datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc))
        bot._close_trade_at_market.assert_not_called()

class TestFirmPhaseProgression(unittest.TestCase):
    def _runner(self, profile):
        return StrategyRunner(_make_farm_config(profile=profile, name="Firm 5K"))

    def test_all_profiles_initialize_consistently(self):
        for profile, dd_floor, daily_floor in (
            (GOAT_FIRM_5K_PROFILE, 4500.0, 4800.0),
            (AQUA_FIRM_5K_PROFILE, 4600.0, 4750.0),
            (FP_FIRM_5K_PROFILE, 4500.0, 4750.0),
        ):
            ch = self._runner(profile).state.current_challenge
            self.assertEqual(ch.equity, 5000.0)
            self.assertEqual(ch.phase, 1)
            self.assertAlmostEqual(ch.dd_floor, dd_floor)
            self.assertAlmostEqual(ch.day_equity * (1 - profile["dl"]), daily_floor)

    def test_aqua_phase1_to_2_and_insufficient_cases(self):
        runner = self._runner(AQUA_FIRM_5K_PROFILE)
        ch = runner.state.current_challenge
        now = datetime.now(timezone.utc)
        ch.ch_trade_pnl = 250.0  # below 8% target
        ch.trading_days = 5
        self.assertFalse(runner._advance_phase(ch, int(now.timestamp() / 86400), now))
        ch.ch_trade_pnl = 400.0
        ch.trading_days = 2
        self.assertFalse(runner._advance_phase(ch, int(now.timestamp() / 86400), now))
        ch.trading_days = 5
        self.assertTrue(runner._advance_phase(ch, int(now.timestamp() / 86400), now))
        self.assertEqual(ch.phase, 2)

    def test_aqua_phase2_to_3(self):
        runner = self._runner(AQUA_FIRM_5K_PROFILE)
        ch = runner.state.current_challenge
        ch.phase = 2
        ch.ch_trade_pnl = 250.0  # 5% target
        ch.trading_days = 5
        now = datetime.now(timezone.utc)
        self.assertTrue(runner._advance_phase(ch, int(now.timestamp() / 86400), now))
        self.assertEqual(ch.phase, 3)
        self.assertIsNotNone(ch.funded_date)

    def test_fp_phase_transitions(self):
        runner = self._runner(FP_FIRM_5K_PROFILE)
        ch = runner.state.current_challenge
        now = datetime.now(timezone.utc)
        ch.ch_trade_pnl = 450.0
        ch.trading_days = 5
        self.assertTrue(runner._advance_phase(ch, int(now.timestamp() / 86400), now))
        ch.phase = 2
        ch.ch_trade_pnl = 350.0
        ch.trading_days = 5
        self.assertTrue(runner._advance_phase(ch, int(now.timestamp() / 86400), now))
        self.assertEqual(ch.phase, 3)

    def test_phase_progress_and_snapshot(self):
        runner = self._runner(FP_FIRM_5K_PROFILE)
        ch = runner.state.current_challenge
        ch.ch_trade_pnl = 200.0
        prog = StrategyRunner._phase_progress(ch, runner.profile, ch.ch_trade_pnl)
        self.assertAlmostEqual(prog["pct"], 50.0, places=1)
        snap = runner.snapshot()
        self.assertAlmostEqual(snap["challenges"][0]["equity"], 5000.0)
        self.assertAlmostEqual(snap["challenges"][0]["balance"], 5000.0)


class TestDayWorstUpLReset(unittest.TestCase):
    """The intraday projection must never re-count a closed trade's loss."""

    def _setup(self):
        bot = _bot()
        runner = StrategyRunner(_make_farm_config(), bot=bot)
        ch = runner.state.current_challenge
        return bot, runner, ch

    @staticmethod
    def _trade(tid="t1", entry=2000.0):
        return Trade(id=tid, strategy="Goat 5K", direction=1, entry=entry,
                     sl=entry - 15.0, tp=entry + 45.0, lots=0.1,
                     open_time=datetime.now(timezone.utc).isoformat())

    def test_worst_upl_reset_when_flat(self):
        _, runner, ch = self._setup()
        t1 = self._trade()
        ch.open_trades = [t1]
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        ch.day_start = now_day
        ch.day_equity = ch.equity
        runner._process_challenge_bar(ch, {"time": int(now.timestamp()), "low": 1990.0,
                                           "high": 2005.0, "close": 2000.0},
                                      15.0, 60, now_day, now)
        self.assertAlmostEqual(ch._day_worst_upl, -100.0)
        self.assertFalse(ch.blown)
        t1.pnl = -80.0
        runner.on_trade_closed(t1, now, ch)
        self.assertEqual(len(ch.open_trades), 0)
        self.assertAlmostEqual(ch._day_worst_upl, 0.0)

    def test_new_trade_does_not_inherit_stale_worst(self):
        _, runner, ch = self._setup()
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        ch.day_start = now_day
        ch.day_equity = ch.equity
        t1 = self._trade()
        ch.open_trades = [t1]
        runner._process_challenge_bar(ch, {"time": int(now.timestamp()), "low": 1990.0,
                                           "high": 2005.0, "close": 2000.0},
                                      15.0, 60, now_day, now)
        t1.pnl = -80.0
        runner.on_trade_closed(t1, now, ch)
        t2 = self._trade(tid="t2", entry=2010.0)
        ch.open_trades = [t2]
        runner._process_challenge_bar(ch, {"time": int(now.timestamp()), "low": 2006.0,
                                           "high": 2015.0, "close": 2010.0},
                                      15.0, 60, now_day, now)
        self.assertAlmostEqual(ch._day_worst_upl, -40.0)
        self.assertAlmostEqual(ch.equity + ch._day_worst_upl, ch.equity - 40.0)


class TestPayoutRetention(unittest.TestCase):
    """Payouts retain the unpaid profit (equity -= payout)."""

    def _setup(self):
        bot = _bot()
        runner = StrategyRunner(_make_farm_config(), bot=bot)
        return bot, runner, runner.state.current_challenge

    def test_payout_retains_unpaid_profit(self):
        bot, runner, ch = self._setup()
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        ch.phase = 3
        ch.day_start = now_day
        ch.last_payout_day = now_day - 14
        ch.equity = 5600.0
        ch.day_equity = 5500.0
        ch.ch_trade_pnl = 600.0
        ch._day_cap = {now_day - 5: 40.0}
        ch._day_pnl = {now_day - 5: 40.0, now_day - 4: 30.0}
        ch._traded_day_set = set(range(now_day - 5, now_day - 2))
        ch._last_payout_valid_idx = 0
        runner.state.trade_pnl = 123.0
        runner._process_challenge_bar(ch, {"time": now.timestamp()}, 15.0, 60,
                                      now_day, now)
        bot.process_payout.assert_called_once()
        self.assertAlmostEqual(ch.equity, 5120.0)
        self.assertAlmostEqual(runner.state.total_earned, 480.0)
        self.assertAlmostEqual(ch.day_equity, 5500.0)
        self.assertEqual(ch._day_cap, {now_day - 5: 40.0})
        self.assertAlmostEqual(runner.state.trade_pnl, 123.0)
        self.assertAlmostEqual(ch.ch_trade_pnl, 0.0)
        self.assertEqual(ch.last_payout_day, now_day)


class TestEarlyCloseParity(unittest.TestCase):
    """Intra-bar touch of the 8% target, with today counted when partial books."""

    def _setup(self):
        bot = _bot()
        runner = StrategyRunner(_make_farm_config(), bot=bot)
        return bot, runner, runner.state.current_challenge

    @staticmethod
    def _ot(entry=5400.0, lots=0.01, tid="t-ec", position_id=77):
        t = Trade(id=tid, strategy="Goat 5K", direction=1, entry=entry,
                  sl=entry - 30.0, tp=entry + 90.0, lots=lots,
                  open_time=datetime.now(timezone.utc).isoformat())
        t.position_id = position_id
        return t

    def _prime(self, ch, trading_days):
        ch.phase = 1
        ch.equity = 5375.0
        ch.ch_trade_pnl = 375.0
        ch.trading_days = trading_days

    def test_extra_day_makes_two_days_sufficient(self):
        bot, runner, ch = self._setup()
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        self._prime(ch, trading_days=2)
        ch.open_trades = [self._ot()]
        self.assertTrue(runner._check_early_close(ch, {"close": 5480.0}, now_day, now))
        bot._close_trade_at_market.assert_called_once()

    def test_extra_day_cannot_lift_one_valid_day(self):
        bot, runner, ch = self._setup()
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        self._prime(ch, trading_days=1)
        ch.open_trades = [self._ot()]
        self.assertFalse(runner._check_early_close(ch, {"close": 5480.0}, now_day, now))
        bot._close_trade_at_market.assert_not_called()

    def test_live_price_path_uses_bid_for_longs(self):
        _, runner, ch = self._setup()
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        self._prime(ch, trading_days=2)
        ch.open_trades = [self._ot()]
        self.assertFalse(runner._check_early_close(ch, None, now_day, now,
                                                   bid=5410.0, ask=5420.0))
        self.assertTrue(runner._check_early_close(ch, {"close": 5480.0}, now_day, now))

    def test_aggregate_target_closes_all_filled_legs(self):
        bot, runner, ch = self._setup()
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        ch.phase = 1
        ch.equity = 5350.0
        ch.ch_trade_pnl = 350.0
        ch.trading_days = 2
        ch.open_trades = [self._ot(tid="t-a"), self._ot(tid="t-b")]
        self.assertTrue(runner._check_early_close(ch, {"close": 5430.0}, now_day, now))
        self.assertEqual(bot._close_trade_at_market.call_count, 2)
        closed = {c.args[0].id for c in bot._close_trade_at_market.call_args_list}
        self.assertEqual(closed, {"t-a", "t-b"})

    def test_target_already_met_flattens_remaining_legs(self):
        bot, runner, ch = self._setup()
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        ch.phase = 1
        ch.equity = 5400.0
        ch.ch_trade_pnl = 400.0
        ch.trading_days = 3
        ch.open_trades = [self._ot(tid="t-a"), self._ot(tid="t-b")]
        self.assertTrue(runner._check_early_close(ch, None, now_day, now))
        self.assertEqual(bot._close_trade_at_market.call_count, 2)

    def test_early_close_waits_for_inflight_close(self):
        bot, runner, ch = self._setup()
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        self._prime(ch, trading_days=2)
        ch.open_trades = [self._ot()]
        bot._pending_orders = {"t-ec_close": {"kind": "close"}}
        self.assertTrue(runner._check_early_close(ch, {"close": 5480.0}, now_day, now))
        bot._close_trade_at_market.assert_not_called()

    def test_early_close_cancels_pending_entries(self):
        bot, runner, ch = self._setup()
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        self._prime(ch, trading_days=2)
        ch.open_trades = [self._ot(tid="t-filled"),
                          self._ot(tid="t-pending", position_id=None)]
        self.assertTrue(runner._check_early_close(ch, {"close": 5480.0}, now_day, now))
        bot._cancel_pending_for_challenge.assert_called_once_with(runner, ch)
        bot._close_trade_at_market.assert_called_once()
        self.assertEqual(bot._close_trade_at_market.call_args.args[0].id, "t-filled")

    def test_early_close_config_false_disables_path(self):
        runner = StrategyRunner(_make_farm_config(early_close=False), bot=_bot())
        ch = runner.state.current_challenge
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        self._prime(ch, trading_days=3)
        ch.open_trades = [self._ot()]
        self.assertFalse(runner._check_early_close(ch, {"close": 5480.0}, now_day, now))


class TestPhaseAdvanceWithOpenTrades(unittest.TestCase):
    def _mk(self):
        return StrategyRunner(_make_farm_config(early_close=False), bot=_bot())

    @staticmethod
    def _ot(tid):
        t = Trade(id=tid, strategy="Goat 5K", direction=1, entry=5400.0,
                  sl=5370.0, tp=5490.0, lots=0.01,
                  open_time=datetime.now(timezone.utc).isoformat())
        t.position_id = 77
        return t

    def test_phase_does_not_advance_with_open_trade(self):
        runner = self._mk()
        ch = runner.state.current_challenge
        now = datetime.now(timezone.utc)
        ch.ch_trade_pnl = 500.0
        ch.trading_days = 3
        ch.open_trades = [self._ot("t-open")]
        self.assertFalse(runner._advance_phase(ch, int(now.timestamp() / 86400), now))
        self.assertEqual(ch.phase, 1)

    def test_close_waits_for_other_leg_then_advances(self):
        runner = self._mk()
        ch = runner.state.current_challenge
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        ch.equity = 5350.0
        ch.ch_trade_pnl = 350.0
        ch.trading_days = 2
        ch._trading_day_set = {now_day - 2, now_day - 1}
        ch._traded_day_set = {now_day - 2, now_day - 1}
        t1 = self._ot("t-1")
        t2 = self._ot("t-2")
        ch.open_trades = [t1, t2]
        t1.pnl = 20.0
        runner.on_trade_closed(t1, now, ch)
        self.assertEqual(ch.phase, 1)
        self.assertEqual([t.id for t in ch.open_trades], ["t-2"])
        t2.pnl = 60.0
        runner.on_trade_closed(t2, now, ch)
        self.assertEqual(ch.phase, 2)
        self.assertEqual(ch.open_trades, [])


class TestPerTradeLossCap(unittest.TestCase):
    """max_loss_per_trade is a blow rule when configured on a profile."""

    def _setup(self):
        profile = {**GOAT_FIRM_5K_PROFILE, "max_loss_per_trade": 0.01}
        bot = _bot()
        runner = StrategyRunner(_make_farm_config(profile=profile), bot=bot)
        return bot, runner, runner.state.current_challenge

    @staticmethod
    def _ot(entry=4300.0, lots=0.1):
        return Trade(id="t-mlpt", strategy="Goat 5K", direction=1, entry=entry,
                     sl=entry - 30.0, tp=entry + 60.0, lots=lots,
                     open_time=datetime.now(timezone.utc).isoformat())

    def test_intraday_bar_extreme_blows(self):
        _, runner, ch = self._setup()
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        ch.day_start = now_day
        ch.day_equity = ch.equity
        ch.open_trades = [self._ot()]
        runner._process_challenge_bar(ch, {"time": now.timestamp(), "low": 4294.0,
                                           "high": 4310.0, "close": 4300.0},
                                      10.0, 60, now_day, now)
        self.assertTrue(ch.blown)

    def test_close_path_blows(self):
        _, runner, ch = self._setup()
        t = self._ot()
        ch.open_trades = [t]
        t.pnl = -60.0
        runner.on_trade_closed(t, datetime.now(timezone.utc), ch)
        self.assertTrue(ch.blown)

    def test_small_loss_does_not_blow(self):
        _, runner, ch = self._setup()
        now = datetime.now(timezone.utc)
        now_day = int(now.timestamp() / 86400)
        ch.day_start = now_day
        ch.day_equity = ch.equity
        ch.open_trades = [self._ot()]
        runner._process_challenge_bar(ch, {"time": now.timestamp(), "low": 4296.0,
                                           "high": 4310.0, "close": 4300.0},
                                      10.0, 60, now_day, now)
        self.assertFalse(ch.blown)


class TestSwapParity(unittest.TestCase):
    """Kernel numba:1733: swap USD/oz per calendar night, x3 on Wednesdays."""

    @staticmethod
    def _swap(direction, open_day, close_day, rate):
        return StrategyRunner._swap_usd(
            direction, 0.01, f"2026-10-{open_day:02d}T12:00:00+00:00",
            datetime(2026, 10, close_day, 12, tzinfo=timezone.utc), rate)

    def test_wednesday_triple_night(self):
        self.assertAlmostEqual(self._swap(1, 5, 8, -0.12), -0.60)

    def test_short_flips_sign(self):
        self.assertAlmostEqual(self._swap(-1, 5, 8, -0.12), 0.60)

    def test_same_day_exit_is_free(self):
        self.assertEqual(self._swap(1, 5, 5, -0.12), 0.0)

    def test_swap_folded_into_close_pnl(self):
        profile = {**GOAT_FIRM_5K_PROFILE, "swap_rate": -0.12}
        runner = StrategyRunner(_make_farm_config(profile=profile), bot=_bot())
        ch = runner.state.current_challenge
        t = Trade(id="t-swap", strategy="Goat 5K", direction=1, entry=5400.0,
                  sl=5370.0, tp=5490.0, lots=0.01,
                  open_time="2026-10-05T12:00:00+00:00")
        ch.open_trades = [t]
        t.pnl = -10.0
        runner.on_trade_closed(t, datetime(2026, 10, 8, 12, tzinfo=timezone.utc), ch)
        self.assertAlmostEqual(t.pnl, -10.63)  # -10 - 0.03 commission - 0.60 swap
        self.assertAlmostEqual(ch.equity, 5000.0 - 10.63)

    def test_default_firm_swap_rate_is_minus_five_cents(self):
        self.assertEqual(GOAT_FIRM_5K_PROFILE["swap_rate"], -0.05)
        self.assertEqual(AQUA_FIRM_5K_PROFILE["swap_rate"], -0.05)
        self.assertEqual(FP_FIRM_5K_PROFILE["swap_rate"], -0.05)
        self.assertAlmostEqual(self._swap(1, 5, 8, -0.05), -0.25)


class TestPhaseTransitionAccounting(unittest.TestCase):
    """No eval surplus is booked; the 2->3 transition books the fee refund."""

    @staticmethod
    def _mk(profile=None):
        return StrategyRunner(_make_farm_config(profile=profile) if profile else _make_farm_config())

    def test_phase1_to_2_books_no_surplus(self):
        runner = self._mk()
        ch = runner.state.current_challenge
        now = datetime.now(timezone.utc)
        ch.ch_trade_pnl = 500.0
        ch.trading_days = 3
        self.assertTrue(runner._advance_phase(ch, int(now.timestamp() / 86400), now))
        self.assertEqual(ch.phase, 2)
        self.assertAlmostEqual(ch.equity, 5000.0)
        self.assertAlmostEqual(runner.state.total_earned, 0.0)

    def test_phase2_to_3_books_fee_refund(self):
        runner = self._mk()
        ch = runner.state.current_challenge
        now = datetime.now(timezone.utc)
        ch.phase = 2
        ch.ch_trade_pnl = 350.0
        ch.trading_days = 3
        self.assertTrue(runner._advance_phase(ch, int(now.timestamp() / 86400), now))
        self.assertEqual(ch.phase, 3)
        self.assertAlmostEqual(ch.equity, 5000.0)
        self.assertAlmostEqual(runner.state.total_earned, 17.0)

    def test_each_firm_refunds_its_own_fee(self):
        for profile in (GOAT_FIRM_5K_PROFILE, AQUA_FIRM_5K_PROFILE, FP_FIRM_5K_PROFILE):
            runner = self._mk(profile)
            ch = runner.state.current_challenge
            ch.phase = 2
            ch.ch_trade_pnl = 400.0
            ch.trading_days = 3
            now = datetime.now(timezone.utc)
            self.assertTrue(runner._advance_phase(ch, int(now.timestamp() / 86400), now))
            self.assertAlmostEqual(runner.state.total_earned, profile["cc"])


if __name__ == "__main__":
    unittest.main()
