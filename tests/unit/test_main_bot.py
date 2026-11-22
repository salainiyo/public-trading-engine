import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone

from main import Bot, _row_get, _to_float, _parse_ts, _parse_epoch, _parse_comment, _select_account
from api.trade_locker_client import TradeLockerError
from database.trade_store import TradeStore
from engine.models import Trade


class TestHelpers(unittest.TestCase):
    def test_row_get_basic(self):
        row = {"id": 123, "name": "test"}
        self.assertEqual(_row_get(row, "id"), 123)

    def test_row_get_case_insensitive(self):
        row = {"PositionId": 42}
        self.assertEqual(_row_get(row, "positionid"), 42)

    def test_row_get_fallback(self):
        row = {"b": 2}
        self.assertEqual(_row_get(row, "a", "b", "c"), 2)

    def test_row_get_missing(self):
        row = {"a": 1}
        self.assertIsNone(_row_get(row, "x", "y"))

    def test_row_get_none(self):
        self.assertIsNone(_row_get(None, "a"))

    def test_to_float_valid(self):
        self.assertAlmostEqual(_to_float("3.14"), 3.14)

    def test_to_float_invalid(self):
        self.assertIsNone(_to_float("abc"))

    def test_parse_ts_none(self):
        self.assertIsNone(_parse_ts(None))

    def test_parse_ts_iso_string(self):
        dt = _parse_ts("2025-01-15T10:30:00+00:00")
        self.assertIsInstance(dt, datetime)

    def test_parse_ts_naive_iso_gets_utc(self):
        dt = _parse_ts("2025-01-15T10:30:00")
        self.assertIsNotNone(dt)
        self.assertIsNotNone(dt.tzinfo)
        self.assertEqual(dt.utcoffset().total_seconds(), 0)

    def test_parse_ts_int_epoch_s(self):
        dt = _parse_ts(1700000000)
        self.assertIsInstance(dt, datetime)
        self.assertEqual(dt.year, 2023)

    def test_parse_ts_int_epoch_ms(self):
        dt = _parse_ts(1700000000000)
        self.assertIsInstance(dt, datetime)

    def test_parse_epoch_none(self):
        self.assertIsNone(_parse_epoch(None))

    def test_parse_epoch_int(self):
        self.assertEqual(_parse_epoch(1700000000), 1700000000)

    def test_parse_epoch_ms(self):
        self.assertEqual(_parse_epoch(1700000000000), 1700000000)

    def test_parse_comment_valid(self):
        s, tid, ch = _parse_comment("CO|trade-1|ch_2")
        self.assertEqual(s, "CO")
        self.assertEqual(tid, "trade-1")
        self.assertEqual(ch, 2)

    def test_parse_comment_no_pipe(self):
        s, tid, ch = _parse_comment("nocomment")
        self.assertIsNone(s)

    def test_parse_comment_no_ch(self):
        s, tid, ch = _parse_comment("CO|trade-1|bad")
        self.assertEqual(s, "CO")
        self.assertIsNone(ch)

    def test_parse_comment_empty(self):
        s, tid, ch = _parse_comment("")
        self.assertIsNone(s)


class TestResolveRunnerName(unittest.TestCase):
    def _make_bot(self):
        bot = Bot.__new__(Bot)
        bot.runners = {}
        return bot

    def test_exact_match(self):
        bot = self._make_bot()
        bot.runners = {"Continuation Open": MagicMock()}
        self.assertEqual(bot._resolve_runner_name("Continuation Open"), "Continuation Open")

    def test_tag_match(self):
        bot = self._make_bot()
        runner = MagicMock()
        runner.config = {"tag": "CO", "eval_strat": "X", "funded_strat": "Y"}
        bot.runners = {"Continuation Open": runner}
        self.assertEqual(bot._resolve_runner_name("CO"), "Continuation Open")

    def test_no_match(self):
        bot = self._make_bot()
        bot.runners = {"A": MagicMock()}
        self.assertEqual(bot._resolve_runner_name("NOPE"), "NOPE")

    def test_alias_match(self):
        bot = self._make_bot()
        runner = MagicMock()
        runner.config = {"tag": "PP", "eval_strat": "Positional Pivot", "funded_strat": "Positional Pivot", "aliases": ["Continuation Open"]}
        bot.runners = {"Positional Pivot": runner}
        self.assertEqual(bot._resolve_runner_name("Continuation Open"), "Positional Pivot")


class TestFindChallenge(unittest.TestCase):
    def _make_bot(self):
        bot = Bot.__new__(Bot)
        return bot

    def test_find_by_num(self):
        bot = self._make_bot()
        ch1 = MagicMock(challenge_num=1)
        ch2 = MagicMock(challenge_num=2)
        runner = MagicMock()
        runner.state.challenges = [ch1, ch2]
        self.assertIs(bot._find_challenge(runner, 2), ch2)

    def test_find_fallback(self):
        bot = self._make_bot()
        ch = MagicMock(challenge_num=99)
        runner = MagicMock()
        runner.state.challenges = [ch]
        runner.state.current_challenge = ch
        self.assertIs(bot._find_challenge(runner, 1), ch)


class TestComputeHoldBars(unittest.TestCase):
    def _make_bot(self):
        bot = Bot.__new__(Bot)
        return bot

    def test_basic(self):
        bot = self._make_bot()
        runner = MagicMock()
        runner.config = {"eval_tf": 60, "funded_tf": 60}
        ch = MagicMock(phase=1)
        now = datetime.now(timezone.utc).isoformat()
        result = bot._compute_hold_bars(runner, ch, now)
        self.assertEqual(result, 0)

    def test_empty(self):
        bot = self._make_bot()
        runner = MagicMock()
        ch = MagicMock(phase=1)
        result = bot._compute_hold_bars(runner, ch, "")
        self.assertEqual(result, 0)


class TestBarPipeline(unittest.TestCase):
    def _make_bot(self):
        bot = Bot.__new__(Bot)
        bot.brokers = {
            "test_broker": {
                "buffers": {60: {"opens": [], "highs": [], "lows": [], "closes": [], "volumes": [], "times": []}},
                "current_bars": {60: None},
                "last_bar_times": {60: None},
                "current_price": 0.0,
                "last_bid": None,
                "last_ask": None,
                "last_poll_time": None,
            }
        }
        bot.runners = {}
        bot._pending_signals = {}
        return bot

    def test_on_quote_builds_bar(self):
        bot = self._make_bot()
        with patch("main.xau_market_open", return_value=True), \
             patch("main.time.time", return_value=1000000):
            bot._on_quote(100.0, "test_broker")
        bar = bot.brokers["test_broker"]["current_bars"][60]
        self.assertIsNotNone(bar)
        self.assertEqual(bar["open"], 100.0)

    def test_on_quote_updates_existing_bar(self):
        bot = self._make_bot()
        with patch("main.xau_market_open", return_value=True), \
             patch("main.time.time", return_value=1000):
            bot._on_quote(100.0, "test_broker")
            bot._on_quote(102.0, "test_broker")
        bar = bot.brokers["test_broker"]["current_bars"][60]
        self.assertEqual(bar["high"], 102.0)
        self.assertEqual(bar["low"], 100.0)

    def test_close_bar_appends_to_buffer(self):
        bot = self._make_bot()
        bar = {"open": 100, "high": 105, "low": 98, "close": 103, "volume": 10, "time": 1000}
        import numpy as np
        with patch("main.build_indicator_dict") as mock_build:
            mock_build.return_value = {"ATR14": MagicMock(__getitem__=lambda s, k: np.array([1.0]))}
            bot._close_bar(60, bar, "test_broker")
        buf = bot.brokers["test_broker"]["buffers"][60]
        self.assertEqual(len(buf["opens"]), 1)

    def test_close_bar_skips_partial(self):
        bot = self._make_bot()
        bar = {"open": 100, "high": 105, "low": 98, "close": 103,
               "volume": 10, "time": 1000, "_partial": True}
        with patch("main.build_indicator_dict") as mock_build:
            bot._close_bar(60, bar, "test_broker")
        buf = bot.brokers["test_broker"]["buffers"][60]
        self.assertEqual(len(buf["opens"]), 0)
        mock_build.assert_not_called()

    def test_request_history_seeds_in_progress_bar(self):
        bot = Bot.__new__(Bot)
        now = 1700000000
        hour = now // 3600 * 3600
        client = MagicMock()
        client.history.return_value = [
            {"t": hour - 7200, "o": 90, "h": 92, "l": 89, "c": 91, "v": 5},
            {"t": hour - 3600, "o": 91, "h": 93, "l": 90, "c": 92, "v": 6},
            {"t": hour, "o": 92, "h": 94, "l": 91, "c": 93, "v": 2},
        ]
        bot.brokers = {"test_broker": {
            "client": client, "account_id": 1, "route_id": 1, "instrument_id": 1,
            "buffers": {60: {"opens": [], "highs": [], "lows": [], "closes": [],
                             "volumes": [], "times": []}},
            "current_bars": {60: None},
            "last_bar_times": {60: None},
        }}
        bot.runners = {}
        bot._pending_signals = {}
        with patch("main.time.time", return_value=now):
            bot._request_history("test_broker")
        broker = bot.brokers["test_broker"]
        buf = broker["buffers"][60]
        # The in-progress hour bar is the live current bar, not a buffer row.
        self.assertEqual(buf["times"], [hour - 7200, hour - 3600])
        self.assertEqual(broker["current_bars"][60]["time"], hour)
        self.assertEqual(broker["current_bars"][60]["close"], 93)
        self.assertEqual(broker["last_bar_times"][60], hour)

        with patch("main.xau_market_open", return_value=True), \
             patch("main.time.time", return_value=hour + 60):
            bot._on_quote(95.0, "test_broker")
        self.assertEqual(broker["current_bars"][60]["high"], 95.0)
        self.assertEqual(len(buf["times"]), 2)  # same hour: no duplicate bar

        with patch("main.xau_market_open", return_value=True), \
             patch("main.time.time", return_value=hour + 3600):
            bot._on_quote(96.0, "test_broker")
        # The seeded bar closes exactly once, with no duplicate timestamp.
        self.assertEqual(buf["times"], [hour - 7200, hour - 3600, hour])

    def test_request_history_keeps_older_unclosed_current_bar(self):
        bot = Bot.__new__(Bot)
        now = 1700000000
        hour = now // 3600 * 3600
        client = MagicMock()
        client.history.return_value = [
            {"t": hour - 7200, "o": 90, "h": 92, "l": 89, "c": 91, "v": 5},
            {"t": hour, "o": 92, "h": 94, "l": 91, "c": 93, "v": 2},
        ]
        stale = {"open": 91, "high": 93, "low": 90, "close": 92,
                 "volume": 6, "time": hour - 3600}
        bot.brokers = {"test_broker": {
            "client": client, "account_id": 1, "route_id": 1, "instrument_id": 1,
            "buffers": {60: {"opens": [], "highs": [], "lows": [], "closes": [],
                             "volumes": [], "times": []}},
            "current_bars": {60: dict(stale)},
            "last_bar_times": {60: hour - 3600},
        }}
        bot.runners = {}
        bot._pending_signals = {}
        with patch("main.time.time", return_value=now):
            bot._request_history("test_broker")
        broker = bot.brokers["test_broker"]
        # The hour rolled while the quote feed was down: keep the older bar
        # for _on_quote to close instead of silently dropping it.
        self.assertEqual(broker["current_bars"][60]["time"], hour - 3600)
        self.assertEqual(broker["last_bar_times"][60], hour - 3600)

    def test_first_bar_without_history_seed_is_partial(self):
        bot = self._make_bot()
        now = 1700000000
        hour = now // 3600 * 3600
        with patch("main.xau_market_open", return_value=True), \
             patch("main.time.time", return_value=hour + 120):
            bot._on_quote(100.0, "test_broker")
        bar = bot.brokers["test_broker"]["current_bars"][60]
        self.assertTrue(bar.get("_partial"))
        with patch("main.xau_market_open", return_value=True), \
             patch("main.time.time", return_value=hour + 3600):
            bot._on_quote(101.0, "test_broker")
        buf = bot.brokers["test_broker"]["buffers"][60]
        self.assertEqual(buf["times"], [])  # partial bar dropped

    def test_poll_quotes_updates_price(self):
        bot = self._make_bot()
        mock_client = MagicMock()
        mock_client.quotes.return_value = {"bid": 100.0, "ask": 101.0}
        bot.brokers["test_broker"]["client"] = mock_client
        bot.brokers["test_broker"]["instrument_id"] = 1
        bot.brokers["test_broker"]["route_id"] = 1
        with patch("main.xau_market_open", return_value=True), \
             patch("main.time.time", return_value=1000):
            bot._poll_quotes()
        self.assertAlmostEqual(bot.brokers["test_broker"]["current_price"], 100.5)


class TestOrderExecution(unittest.TestCase):
    def _make_bot(self):
        bot = Bot.__new__(Bot)
        mock_client = MagicMock()
        bot.brokers = {
            "test_broker": {
                "client": mock_client,
                "account_id": 1,
                "instrument_id": 1,
                "route_id": 1,
                "trade_route_id": 1,
                "price_digits": 2,
                "live": False,
            }
        }
        bot._pending_orders = {}
        bot._close_attempt_ts = {}
        bot.trade_store = MagicMock()
        return bot

    def _make_runner(self, broker_key="test_broker"):
        runner = MagicMock()
        runner.config = {"tag": "CO"}
        runner._broker = self.bot.brokers[broker_key]
        runner._broker_key = broker_key
        return runner

    def setUp(self):
        self.bot = self._make_bot()

    def test_place_order_sync_success(self):
        bot = self.bot
        client = bot.brokers["test_broker"]["client"]
        client.place_order.return_value = "ord-123"
        trade = MagicMock()
        trade.direction = 1
        trade.lots = 0.1
        trade.entry = 100.0
        trade.sl = 99.0
        trade.tp = 103.0
        trade.id = "test-trade-1"
        runner = self._make_runner()
        ch = MagicMock()
        ch.challenge_num = 1
        bot._place_order_sync(trade, runner, ch)
        client.place_order.assert_called_once()

    def test_place_order_sync_strategy_id_too_long(self):
        bot = self.bot
        client = bot.brokers["test_broker"]["client"]
        trade = MagicMock()
        trade.id = "x" * 50
        trade.direction = 1
        trade.lots = 0.1
        trade.entry = 100.0
        trade.sl = 99.0
        trade.tp = 103.0
        trade.open_time = "2025-01-15T10:00:00"
        runner = self._make_runner()
        ch = MagicMock()
        ch.challenge_num = 1
        ch.open_trades = [trade]
        bot._place_order_sync(trade, runner, ch)
        client.place_order.assert_not_called()

    def test_close_trade_at_market(self):
        bot = self.bot
        client = bot.brokers["test_broker"]["client"]
        trade = MagicMock()
        trade.id = "t1"
        trade.position_id = 99
        trade.lots = 0.1
        runner = self._make_runner()
        runner.name = "test"
        ch = MagicMock()
        ch.open_trades = [trade]
        bot._close_trade_at_market(trade, runner, ch)
        client.close_position.assert_called_once_with(99, qty=0.1)
        self.assertIsNotNone(bot._orders_history_due)

    def test_pending_close_broker_keys_only_close_orders(self):
        bot = self.bot
        runner = self._make_runner()
        runner.name = "TestRunner"
        bot.runners = {"TestRunner": runner}
        bot._pending_orders = {
            "t1_close": {"kind": "close", "runner_name": "TestRunner"},
            "t2": {"kind": "limit_open", "runner_name": "TestRunner"},
        }
        self.assertEqual(bot._pending_close_broker_keys(), {"test_broker"})

    def test_prune_settled_close_pending(self):
        bot = self.bot
        bot.runners = {}
        bot._pending_orders = {"t1_close": {"kind": "close"}}
        bot._close_attempt_ts = {"t1": 123.0}
        bot._prune_settled_close_pending()
        self.assertNotIn("t1_close", bot._pending_orders)
        self.assertNotIn("t1", bot._close_attempt_ts)

    def test_close_trade_at_market_duplicate_guard(self):
        bot = self.bot
        client = bot.brokers["test_broker"]["client"]
        bot._pending_orders = {"t1_close": {"kind": "close"}}
        trade = MagicMock()
        trade.id = "t1"
        trade.position_id = 99
        runner = self._make_runner()
        ch = MagicMock()
        ch.open_trades = [trade]
        bot._close_trade_at_market(trade, runner, ch)
        client.close_position.assert_not_called()

    def test_session_closed_requeues_signal(self):
        bot = self.bot
        client = bot.brokers["test_broker"]["client"]
        client.place_order.side_effect = TradeLockerError("trading out of session is forbidden")
        bot._pending_signals = {}
        trade = MagicMock()
        trade.direction = -1
        trade.lots = 0.1
        trade.entry = 100.0
        trade.sl = 101.0
        trade.tp = 90.0
        trade.id = "test-trade-1"
        trade.strategy = "CO"
        runner = self._make_runner()
        runner.name = "TestRunner"
        ch = MagicMock()
        ch.challenge_num = 1
        ch.open_trades = [trade]
        bot._place_order_sync(trade, runner, ch, 15.0)
        # Session-rejected re-queues are marked ready immediately (pattern_time=0)
        self.assertEqual(bot._pending_signals.get("TestRunner", {}).get(1), (-1, 15.0, 0, 100.0))
        client.place_order.assert_called_once()

    def test_push_trail_sl_to_broker_success(self):
        bot = self.bot
        client = bot.brokers["test_broker"]["client"]
        trade = MagicMock()
        trade.id = "t1"
        trade.position_id = 99
        trade.sl = 4152.98
        trade.strategy = "Trail Champion"
        runner = self._make_runner()
        bot.push_trail_sl_to_broker(runner, trade)
        client.modify_position.assert_called_once_with(99, stop_loss=4152.98)

    def test_push_trail_sl_to_broker_session_closed_sets_cooldown(self):
        bot = self.bot
        client = bot.brokers["test_broker"]["client"]
        client.modify_position.side_effect = TradeLockerError(
            "SL orders are forbidden by trade session status")
        trade = MagicMock()
        trade.id = "t1"
        trade.position_id = 99
        trade.sl = 4152.98
        trade.strategy = "Trail Champion"
        runner = self._make_runner()
        bot.push_trail_sl_to_broker(runner, trade)
        self.assertGreater(bot.brokers["test_broker"]["session_cooldown_until"], time.time())

    def test_push_trail_sl_to_broker_error_no_crash(self):
        bot = self.bot
        client = bot.brokers["test_broker"]["client"]
        client.modify_position.side_effect = TradeLockerError("some other failure")
        trade = MagicMock()
        trade.id = "t1"
        trade.position_id = 99
        trade.sl = 4152.98
        trade.strategy = "Trail Champion"
        runner = self._make_runner()
        bot.push_trail_sl_to_broker(runner, trade)
        client.modify_position.assert_called_once()

    def test_push_trail_sl_to_broker_skips_without_position(self):
        bot = self.bot
        client = bot.brokers["test_broker"]["client"]
        trade = MagicMock()
        trade.id = "t1"
        trade.position_id = None
        trade.sl = 4152.98
        runner = self._make_runner()
        bot.push_trail_sl_to_broker(runner, trade)
        client.modify_position.assert_not_called()


class TestPendingOrderWatchdog(unittest.TestCase):
    def _make_bot(self):
        bot = Bot.__new__(Bot)
        bot._pending_orders = {}
        bot._close_attempt_ts = {}
        bot.reconcile = MagicMock()
        return bot

    def test_fresh_order_not_flagged(self):
        import time
        bot = self._make_bot()
        bot._pending_orders = {"trade1": {"runner_name": "r", "kind": "open", "sent_at": time.time()}}
        bot._check_pending_orders()
        bot.reconcile.assert_not_called()

    def test_stale_open_order_triggers_reconcile(self):
        import time
        bot = self._make_bot()
        bot._pending_orders = {"trade1": {"runner_name": "r", "kind": "open", "sent_at": time.time() - 100}}
        bot._check_pending_orders()
        bot.reconcile.assert_called_once()

    def test_stale_close_order_triggers_reconcile(self):
        import time
        bot = self._make_bot()
        bot._pending_orders = {"trade1": {"runner_name": "r", "kind": "close", "sent_at": time.time() - 100}}
        bot._check_pending_orders()
        bot.reconcile.assert_called_once()


class TestSLSlippageCheck(unittest.TestCase):
    def setUp(self):
        self.bot = self._make_bot()

    def _make_bot(self):
        bot = Bot.__new__(Bot)
        mock_client = MagicMock()
        bot.brokers = {
            "test_broker": {
                "client": mock_client,
                "account_id": 1,
                "instrument_id": 1,
                "route_id": 1,
                "trade_route_id": 1,
                "price_digits": 2,
                "live": False,
                "current_price": 100.0,
                "last_bid": 100.0,
                "last_ask": 100.5,
            }
        }
        bot._pending_orders = {}
        bot._close_attempt_ts = {}
        bot.trade_store = MagicMock()
        return bot

    def _make_runner(self, trades, blown=False):
        runner = MagicMock()
        runner.config = {"tag": "CO"}
        runner._broker = self.bot.brokers["test_broker"]
        runner._broker_key = "test_broker"
        ch = MagicMock()
        ch.blown = blown
        ch.open_trades = trades
        runner.state.challenges = [ch]
        return runner

    def _make_trade(self, direction=1, sl=99.0, position_id=99):
        trade = MagicMock()
        trade.id = "t1"
        trade.direction = direction
        trade.sl = sl
        trade.position_id = position_id
        trade.lots = 0.1
        return trade

    def test_long_breach_forces_exit(self):
        bot = self.bot
        trade = self._make_trade(direction=1, sl=100.5)
        runner = self._make_runner([trade])
        bot.runners = {"r": runner}
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        bot.brokers["test_broker"]["client"].close_position.assert_called_once_with(99, qty=0.1)

    def test_long_no_breach_no_exit(self):
        bot = self.bot
        trade = self._make_trade(direction=1, sl=95.0)
        runner = self._make_runner([trade])
        bot.runners = {"r": runner}
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        bot.brokers["test_broker"]["client"].close_position.assert_not_called()

    def test_short_breach_forces_exit(self):
        bot = self.bot
        trade = self._make_trade(direction=-1, sl=100.0)
        runner = self._make_runner([trade])
        bot.runners = {"r": runner}
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        bot.brokers["test_broker"]["client"].close_position.assert_called_once_with(99, qty=0.1)

    def test_short_no_breach_no_exit(self):
        bot = self.bot
        trade = self._make_trade(direction=-1, sl=105.0)
        runner = self._make_runner([trade])
        bot.runners = {"r": runner}
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        bot.brokers["test_broker"]["client"].close_position.assert_not_called()

    def test_early_close_skipped_when_config_disabled(self):
        bot = self.bot
        trade = self._make_trade(direction=1, sl=95.0)  # no SL breach
        runner = self._make_runner([trade])
        runner.config = {"tag": "CO", "early_close": False}
        runner.state.challenges[0].phase = 1
        runner._check_early_close = MagicMock()
        bot.runners = {"r": runner}
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        runner._check_early_close.assert_not_called()

    def test_long_uses_bid_short_uses_ask(self):
        bot = self.bot
        bot.brokers["test_broker"]["last_bid"] = 98.9
        bot.brokers["test_broker"]["last_ask"] = 100.4
        # Long with sl=99.0: bid 98.9 <= 99.0 → breach
        trade_long = self._make_trade(direction=1, sl=99.0)
        runner = self._make_runner([trade_long])
        bot.runners = {"r": runner}
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        bot.brokers["test_broker"]["client"].close_position.assert_called_once_with(99, qty=0.1)
        # Short with sl=100.0: ask 100.4 >= 100.0 → breach
        bot.brokers["test_broker"]["client"].close_position.reset_mock()
        trade_short = self._make_trade(direction=-1, sl=100.0)
        trade_short.id = "t2"
        runner2 = self._make_runner([trade_short])
        bot.runners = {"r": runner2}
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        bot.brokers["test_broker"]["client"].close_position.assert_called_once_with(99, qty=0.1)

    def test_position_not_open_on_broker_skipped(self):
        bot = self.bot
        trade = self._make_trade(direction=1, sl=105.0, position_id=None)
        runner = self._make_runner([trade])
        bot.runners = {"r": runner}
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        bot.brokers["test_broker"]["client"].close_position.assert_not_called()

    def test_blown_challenge_skipped(self):
        bot = self.bot
        trade = self._make_trade(direction=1, sl=105.0)
        runner = self._make_runner([trade], blown=True)
        bot.runners = {"r": runner}
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        bot.brokers["test_broker"]["client"].close_position.assert_not_called()

    def test_market_closed_skipped(self):
        bot = self.bot
        trade = self._make_trade(direction=1, sl=105.0)
        runner = self._make_runner([trade])
        bot.runners = {"r": runner}
        with patch("main.xau_market_open", return_value=False):
            bot._sl_slippage_check()
        bot.brokers["test_broker"]["client"].close_position.assert_not_called()

    def test_pending_close_suppresses_spam(self):
        bot = self.bot
        trade = self._make_trade(direction=1, sl=100.5)
        runner = self._make_runner([trade])
        bot.runners = {"r": runner}
        bot._pending_orders = {"t1_close": {"kind": "close", "sent_at": time.time()}}
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        bot.brokers["test_broker"]["client"].close_position.assert_not_called()

    def test_no_price_skipped(self):
        bot = self.bot
        bot.brokers["test_broker"]["current_price"] = 0.0
        bot.brokers["test_broker"]["last_bid"] = None
        bot.brokers["test_broker"]["last_ask"] = None
        trade = self._make_trade(direction=1, sl=105.0)
        runner = self._make_runner([trade])
        bot.runners = {"r": runner}
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        bot.brokers["test_broker"]["client"].close_position.assert_not_called()

    def test_runner_other_broker_skipped(self):
        bot = self.bot
        trade = self._make_trade(direction=1, sl=105.0)
        runner = self._make_runner([trade])
        runner._broker_key = "other_broker"
        bot.runners = {"r": runner}
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        bot.brokers["test_broker"]["client"].close_position.assert_not_called()


class TestCleanupOrphanRows(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = TradeStore(db_path=os.path.join(self._tmpdir, "t.db"))
        self.bot = Bot.__new__(Bot)
        self.bot.trade_store = self.store
        self.bot._pending_orders = {}
        self.bot._close_attempt_ts = {}

    def tearDown(self):
        self.store.close()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _runner_with_open(self, tid):
        runner = MagicMock()
        ch = MagicMock()
        t = MagicMock()
        t.id = tid
        ch.open_trades = [t]
        runner.state.challenges = [ch]
        return runner

    def _save_stale_row(self, tid):
        t = Trade(id=tid, strategy="CO", direction=1, entry=100.0, sl=99.0, tp=103.0,
                  lots=0.1, open_time=(datetime.now(timezone.utc) - timedelta(hours=3)).isoformat())
        self.store.save_trade(t)

    def test_marks_stale_orphan_failed(self):
        self._save_stale_row("orphan-1")
        self.bot.runners = {}
        self.bot._cleanup_orphan_rows()
        rows = self.store.load_trades("CO")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")

    def test_skips_memory_known_ids(self):
        self._save_stale_row("live-1")
        self.bot.runners = {"r": self._runner_with_open("live-1")}
        self.bot._cleanup_orphan_rows()
        rows = self.store.load_trades("CO")
        self.assertEqual(len(rows), 0)

    def test_skips_recent_rows(self):
        t = Trade(id="recent-1", strategy="CO", direction=1, entry=100.0, sl=99.0, tp=103.0,
                  lots=0.1, open_time=datetime.now(timezone.utc).isoformat())
        self.store.save_trade(t)
        self.bot.runners = {}
        self.bot._cleanup_orphan_rows()
        rows = self.store.load_trades("CO")
        self.assertEqual(len(rows), 0)

    def test_skips_pending_order_ids(self):
        self._save_stale_row("pending-1")
        self.bot._pending_orders = {"pending-1": {"kind": "open", "sent_at": time.time()}}
        self.bot.runners = {}
        self.bot._cleanup_orphan_rows()
        rows = self.store.load_trades("CO")
        self.assertEqual(len(rows), 0)


class TestDelayedSignalSessionCooldown(unittest.TestCase):
    def _make_bot(self, cooldown_until):
        bot = Bot.__new__(Bot)
        bot.brokers = {
            "test_broker": {
                "client": MagicMock(),
                "account_id": 1,
                "instrument_id": 1,
                "route_id": 1,
                "trade_route_id": 1,
                "price_digits": 2,
                "live": False,
                "current_price": 100.0,
                "session_cooldown_until": cooldown_until,
                "buffers": {60: {}},
                "current_bars": {60: None},
                "last_bar_times": {60: (int(time.time()) // 3600) * 3600 - 3600},
            }
        }
        bot._pending_signals = {}
        bot._pending_orders = {}
        bot._close_attempt_ts = {}
        bot.trade_store = MagicMock()
        return bot

    def _make_runner(self):
        runner = MagicMock()
        runner.config = {"tag": "CO", "eval_tf": 60, "funded_tf": 60}
        runner._broker_key = "test_broker"
        runner._last_bar_atr = 99.0  # decoy: A[p+1] must be IGNORED for entries
        ch = MagicMock()
        ch.challenge_num = 1
        ch.phase = 1
        ch.blown = False
        ch.open_trades = []
        runner.state.challenges = [ch]
        return runner

    def test_cooldown_keeps_signal_pending(self):
        bot = self._make_bot(time.time() + 1800)
        runner = self._make_runner()
        bot.runners = {"TestRunner": runner}
        # pattern is 2 bars old → ready, but the session cooldown re-stores it
        pattern_time = (int(time.time()) // 3600) * 3600 - 7200
        bot._pending_signals = {"TestRunner": {1: (1, 15.0, pattern_time, 100.0)}}
        with patch("main.xau_market_open", return_value=True):
            bot._on_quote(100.0, "test_broker")
        self.assertEqual(bot._pending_signals.get("TestRunner", {}).get(1),
                         (1, 15.0, pattern_time, 100.0))

    def test_unready_signal_waits_two_bars(self):
        # Pattern only 1 bar old → kernel fires at pattern+1, so at pattern+2
        # open the signal must still be pending (not consumed).
        bot = self._make_bot(0)
        runner = self._make_runner()
        bot.runners = {"TestRunner": runner}
        pattern_time = (int(time.time()) // 3600) * 3600 - 3600
        bot._pending_signals = {"TestRunner": {1: (1, 15.0, pattern_time, 100.0)}}
        with patch("main.xau_market_open", return_value=True):
            bot._on_quote(100.0, "test_broker")
        runner._compute_entry.assert_not_called()
        self.assertEqual(bot._pending_signals.get("TestRunner", {}).get(1),
                         (1, 15.0, pattern_time, 100.0))

    def test_entry_delay_bars_config_consumes_after_one_bar(self):
        # The archive/lastly farms fire at pattern+1: bar_start == pattern+1
        # is ready with entry_delay_bars=1.
        bot = self._make_bot(0)
        runner = self._make_runner()
        runner.config["entry_delay_bars"] = 1
        runner._max_concurrent.return_value = 1
        runner._compute_entry.return_value = None
        bot.runners = {"TestRunner": runner}
        pattern_time = (int(time.time()) // 3600) * 3600 - 3600
        bot._pending_signals = {"TestRunner": {1: (1, 15.0, pattern_time, 100.0)}}
        with patch("main.xau_market_open", return_value=True):
            bot._on_quote(100.0, "test_broker")
        runner._compute_entry.assert_called_once()
        self.assertEqual(bot._pending_signals.get("TestRunner", {}), {})

    def test_addon_gate_drops_signal_below_half_r(self):
        bot = self._make_bot(0)
        runner = self._make_runner()
        runner.config.update({"entry_delay_bars": 1, "addon_min_r": 0.5})
        runner._max_concurrent.return_value = 2
        runner._best_open_r.return_value = 0.2
        ch = runner.state.challenges[0]
        ch.open_trades = [MagicMock(status="open")]
        bot.runners = {"TestRunner": runner}
        pattern_time = (int(time.time()) // 3600) * 3600 - 3600
        bot._pending_signals = {"TestRunner": {1: (1, 15.0, pattern_time, 100.0)}}
        with patch("main.xau_market_open", return_value=True):
            bot._on_quote(100.0, "test_broker")
        runner._compute_entry.assert_not_called()
        self.assertEqual(bot._pending_signals.get("TestRunner", {}), {})

    def test_no_cooldown_consumes_signal(self):
        bot = self._make_bot(0)
        runner = self._make_runner()
        runner._broker = bot.brokers["test_broker"]
        runner._max_concurrent.return_value = 1
        trade = MagicMock()
        trade.id = "t1"
        trade.direction = 1
        trade.lots = 0.1
        trade.entry = 100.0
        trade.sl = 99.0
        trade.tp = 103.0
        trade.strategy = "CO"
        runner._compute_entry.return_value = trade
        bot.runners = {"TestRunner": runner}
        pattern_time = (int(time.time()) // 3600) * 3600 - 7200
        bot._pending_signals = {"TestRunner": {1: (1, 15.0, pattern_time, 100.0)}}
        with patch("main.xau_market_open", return_value=True):
            bot._on_quote(100.0, "test_broker")
        runner._compute_entry.assert_called_once()
        # Entry ATR must be the stored PATTERN ATR (kernel A[p] = ATRV[1:][i]),
        # not runner._last_bar_atr (A[p+1], the decoy 99.0 above)
        self.assertEqual(runner._compute_entry.call_args[0][2], 15.0)
        self.assertNotIn("TestRunner", bot._pending_signals)


class TestPendingSignalGates(unittest.TestCase):
    def _bot(self):
        bot = Bot.__new__(Bot)
        bot._pending_signals = {}
        return bot

    def _runner(self):
        runner = MagicMock()
        runner.config = {"tag": "CO", "eval_tf": 60, "funded_tf": 60}
        runner.state.active_challenges = []
        return runner

    @staticmethod
    def _bar(bar_time):
        return {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
                "volume": 10, "time": bar_time}

    def test_atr_seed_on_first_bar(self):
        # After a restart _last_bar_atr is None — seed it from A[j-1] so the
        # first trailing update uses the previous bar's ATR (backtest parity).
        bot = self._bot()
        runner = self._runner()
        runner._last_bar_atr = None
        r = {"ATR14": [14.0, 15.0, 16.0], "C": [1.0, 2.0, 3.0]}
        bar_time = (int(time.time()) // 3600) * 3600 - 3600
        bot._process_runner_bar("R", runner, self._bar(bar_time), 16.0, 60, r)
        self.assertEqual(runner._last_bar_atr, 15.0)


class TestSelectAccount(unittest.TestCase):
    def _client(self, accounts):
        client = MagicMock()
        client.all_accounts.return_value = accounts
        return client

    def _accounts(self):
        return [
            {"id": 111111111, "accNum": 1, "status": "ACTIVE", "accountBalance": 5000},
            {"id": 222222222, "accNum": 2, "status": "ACTIVE", "accountBalance": 100000},
        ]

    def test_first_word(self):
        aid, anum = _select_account(self._client(self._accounts()), {"account": "first"})
        self.assertEqual(str(aid), "111111111")
        self.assertEqual(str(anum), "1")

    def test_second_word(self):
        aid, anum = _select_account(self._client(self._accounts()), {"account": "second"})
        self.assertEqual(str(aid), "222222222")
        self.assertEqual(str(anum), "2")

    def test_numeric_index(self):
        aid, _ = _select_account(self._client(self._accounts()), {"account": 2})
        self.assertEqual(str(aid), "222222222")

    def test_account_id_exact_in_account_field(self):
        aid, _ = _select_account(self._client(self._accounts()), {"account": "222222222"})
        self.assertEqual(str(aid), "222222222")

    def test_account_id_field_takes_precedence(self):
        aid, _ = _select_account(self._client(self._accounts()),
                                 {"account_id": 111111111, "account": "second"})
        self.assertEqual(str(aid), "111111111")

    def test_acc_num_takes_precedence_over_account(self):
        aid, _ = _select_account(self._client(self._accounts()),
                                 {"acc_num": 2, "account": "first"})
        self.assertEqual(str(aid), "222222222")

    def test_fallback_first(self):
        aid, _ = _select_account(self._client(self._accounts()), {})
        self.assertEqual(str(aid), "111111111")

    def test_index_out_of_range_falls_back_to_first(self):
        aid, _ = _select_account(self._client(self._accounts()), {"account": "9"})
        self.assertEqual(str(aid), "111111111")

    def test_inactive_status_selected_with_warning(self):
        accounts = [{"id": 1, "accNum": 1, "status": "INACTIVE"}]
        aid, _ = _select_account(self._client(accounts), {"account": "first"})
        self.assertEqual(str(aid), "1")

    def test_no_accounts_raises(self):
        with self.assertRaises(RuntimeError):
            _select_account(self._client([]), {})


class TestUploadedCredentials(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._old_data_dir = os.environ.get("GOATBOT_DATA_DIR")
        os.environ["GOATBOT_DATA_DIR"] = self._tmpdir
        os.environ.pop("CREDENTIALS_KEY", None)
        from config.credentials import CredentialsStore
        self.store = CredentialsStore(db_path=os.path.join(self._tmpdir, "credentials.db"))
        self.bot = Bot.__new__(Bot)
        self.bot._credentials_store = self.store
        self.bot._uploaded_credentials = {}
        self.bot.data_dir = self._tmpdir

    def tearDown(self):
        self.store.close()
        if self._old_data_dir is None:
            os.environ.pop("GOATBOT_DATA_DIR", None)
        else:
            os.environ["GOATBOT_DATA_DIR"] = self._old_data_dir
        os.environ.pop("CREDENTIALS_KEY", None)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_apply_builds_broker_configs(self):
        self.store.save_credentials("GOAT5", "a@b.com", "pw", "srv.example.com", "second")
        configs = {}
        self.bot._apply_uploaded_credentials(configs)
        self.assertIn("cred_GOAT5", configs)
        cfg = configs["cred_GOAT5"]
        self.assertEqual(cfg["email"], "a@b.com")
        self.assertEqual(cfg["account"], 2)
        self.assertEqual(cfg["env_prefix"], "GOAT5")
        self.assertIn("GOAT5", self.bot._uploaded_credentials)

    def test_empty_store_dormant(self):
        configs = {"existing": {"email": "x"}}
        self.bot._apply_uploaded_credentials(configs)
        self.assertNotIn("cred_GOAT5", configs)
        self.assertEqual(self.bot._uploaded_credentials, {})

    def test_no_duplicate_config(self):
        self.store.save_credentials("GOAT5", "a@b.com", "pw", "srv", 1)
        configs = {}
        self.bot._apply_uploaded_credentials(configs)
        self.bot._apply_uploaded_credentials(configs)
        self.assertEqual(sum(1 for k in configs if k == "cred_GOAT5"), 1)


class TestAssignRunnersToBrokers(unittest.TestCase):
    def test_uploaded_credentials_override_farms(self):
        bot = Bot.__new__(Bot)
        bot._uploaded_credentials = {"TCH": {"email": "x"}}
        bot.brokers = {
            "herofx_goat": {"account_id": 1},
            "cred_TCH": {"account_id": 2},
        }
        runner = MagicMock()
        runner.config = {"tag": "TCH", "broker_key": "herofx_goat"}
        runner2 = MagicMock()
        runner2.config = {"tag": "FR", "broker_key": "herofx_goat"}
        bot.runners = {"Trail Champion": runner, "Fractal": runner2}
        bot._assign_runners_to_brokers()
        self.assertEqual(runner.config["broker_key"], "cred_TCH")
        self.assertEqual(runner._broker, bot.brokers["cred_TCH"])
        self.assertEqual(runner2._broker, bot.brokers["herofx_goat"])


class TestAccountChangeReset(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        from database.treasury_store import TreasuryStore
        self.tstore = TreasuryStore(db_path=os.path.join(self._tmpdir, "treasuries.db"))
        self.bot = Bot.__new__(Bot)
        self.bot.treasury_store = self.tstore
        self.bot._uploaded_credentials = {}

    def tearDown(self):
        self.tstore.close()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_runner(self, account_id):
        from engine.models import StrategyState
        runner = MagicMock()
        runner.name = "Trail Champion"
        runner._broker = {"account_id": account_id}
        runner.state = StrategyState(name="Trail Champion")
        return runner

    def test_first_boot_persists_account(self):
        runner = self._make_runner(111111111)
        self.tstore.save({"Trail Champion": runner})
        self.bot._check_account_change(runner)
        self.assertEqual(self.tstore.get_account_id("Trail Champion"), "111111111")
        # no reset — state untouched
        self.assertEqual(len(runner.state.challenges), 0)

    def test_same_account_no_reset(self):
        runner = self._make_runner(111111111)
        self.tstore.save({"Trail Champion": runner})
        self.bot._check_account_change(runner)
        runner.state.trade_pnl = 500.0
        self.bot._check_account_change(runner)
        self.assertEqual(runner.state.trade_pnl, 500.0)

    def test_changed_account_resets_fresh(self):
        runner = self._make_runner(111111111)
        self.tstore.save({"Trail Champion": runner})
        self.bot._check_account_change(runner)
        runner2 = self._make_runner(222222222)
        runner2.state.trade_pnl = 500.0
        runner2.state.total_spent = 34.0
        runner2._add_new_challenge.side_effect = lambda: runner2.state.challenges.append("new-ch")
        self.bot._check_account_change(runner2)
        # fresh state
        self.assertEqual(runner2.state.trade_pnl, 0.0)
        self.assertEqual(runner2.state.total_spent, 0.0)
        self.assertEqual(runner2.state.challenges, ["new-ch"])  # new challenge created
        self.assertEqual(self.tstore.get_account_id("Trail Champion"), "222222222")


class TestWeekendFlatEntryBlock(unittest.TestCase):
    def test_friday_window_drops_pending_signal(self):
        bot = Bot.__new__(Bot)
        bot.runners = {}
        bot._pending_signals = {"FPST5": {1: (-1, 10.0, 0, 3500.0)}}
        runner = MagicMock()
        runner.config = {"friday_flat": True, "tag": "FPST5"}
        runner.is_weekend_flat_window.return_value = True
        friday_bar = {"time": int(datetime(2026, 10, 9, 20, 0,
                                           tzinfo=timezone.utc).timestamp())}
        bot._process_runner_bar("FPST5", runner, friday_bar, 15.0, 60, {})
        runner.on_bar_close.assert_called_once()
        runner._compute_entry.assert_not_called()
        self.assertNotIn("FPST5", bot._pending_signals)

    def test_place_order_guard_blocks_entry(self):
        bot = Bot.__new__(Bot)
        bot._discard_failed_entry = MagicMock()
        runner = MagicMock()
        runner.config = {"friday_flat": True, "tag": "FPST5"}
        runner.is_weekend_flat_window.return_value = True
        trade = MagicMock()
        trade.id = "FPST5-ch1-87"
        trade.strategy = "FundingPips 2-Step"
        bot._place_order_sync(trade, runner)
        bot._discard_failed_entry.assert_called_once_with(
            runner, trade, "weekend_flat_window")


class TestWeekendFlatSafety(unittest.TestCase):
    def test_backstop_flattens_open_trade(self):
        bot = Bot.__new__(Bot)
        bot._pending_signals = {"FPST5": {1: (-1, 10.0, 0, 3500.0)}}
        bot._pending_orders = {}
        bot._cancel_pending_for_challenge = MagicMock()
        bot._close_trade_at_market = MagicMock()
        trade = MagicMock()
        trade.id = "FPST5-ch1-86"
        trade.status = "open"
        ch = MagicMock()
        ch.blown = False
        ch.open_trades = [trade]
        runner = MagicMock()
        runner.name = "FPST5"
        runner.config = {"friday_flat": True, "tag": "FPST5"}
        runner.is_weekend_flat_window.return_value = True
        runner.state.active_challenges = [ch]
        bot.runners = {"FPST5": runner}
        with patch("main.xau_market_open", return_value=True), \
                patch("main.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 10, 9, 20, 0,
                                                tzinfo=timezone.utc)
            bot._enforce_weekend_flat()
        bot._cancel_pending_for_challenge.assert_called_once_with(runner, ch)
        bot._close_trade_at_market.assert_called_once_with(trade, runner, ch)
        self.assertNotIn("FPST5", bot._pending_signals)


class TestLimitLifecycleFixes(unittest.TestCase):
    """Regression tests for the 2026-10-15/16 production incidents:
    float lot qty rejected by the broker, cancel-404 freeing a live GTC slot,
    and a stale broker SL on limit adoption."""

    def _make_bot(self):
        bot = Bot.__new__(Bot)
        mock_client = MagicMock()
        bot.brokers = {
            "test_broker": {
                "client": mock_client,
                "account_id": 1,
                "instrument_id": 1,
                "route_id": 1,
                "trade_route_id": 1,
                "price_digits": 2,
                "live": False,
            }
        }
        bot._pending_orders = {}
        bot._close_attempt_ts = {}
        bot._broker_sl = {}
        bot._last_trail_ts = {}
        bot.trade_store = MagicMock()
        return bot

    def _make_runner(self, bot, trades):
        runner = MagicMock()
        runner.name = "CO"
        runner.config = {"tag": "GOAT5"}
        runner._broker = bot.brokers["test_broker"]
        runner._broker_key = "test_broker"
        ch = MagicMock()
        ch.challenge_num = 1
        ch.blown = False
        ch.open_trades = trades
        runner.state.challenges = [ch]
        return runner, ch

    def _make_trade(self, lots=0.1, sl=99.0, position_id=None, status="pending"):
        trade = MagicMock()
        trade.id = "GOAT5-ch1-9"
        trade.strategy = "Goat 5K"
        trade.direction = 1
        trade.lots = lots
        trade.entry = 100.0
        trade.sl = sl
        trade.tp = 106.0
        trade.open_time = "2026-10-15T10:00:00+00:00"
        trade.position_id = position_id
        trade.status = status
        return trade

    def test_place_order_rounds_float_qty_artifact(self):
        bot = self._make_bot()
        client = bot.brokers["test_broker"]["client"]
        client.place_order.return_value = "ord-1"
        trade = self._make_trade(lots=0.7000000000000001)
        runner, ch = self._make_runner(bot, [trade])
        bot._place_order_sync(trade, runner, ch)
        body = client.place_order.call_args[0][1]
        self.assertEqual(body["qty"], 0.7)
        self.assertEqual(trade.lots, 0.7)

    def test_place_order_rejects_sub_step_qty(self):
        bot = self._make_bot()
        client = bot.brokers["test_broker"]["client"]
        trade = self._make_trade(lots=0.004)
        runner, ch = self._make_runner(bot, [trade])
        bot._place_order_sync(trade, runner, ch)
        client.place_order.assert_not_called()
        bot.trade_store.update_trade_pnl.assert_called()
        self.assertEqual(bot.trade_store.update_trade_pnl.call_args[0][2], "failed")

    def test_cancel_404_but_order_still_live_keeps_slot(self):
        bot = self._make_bot()
        client = bot.brokers["test_broker"]["client"]
        client.cancel_order.side_effect = TradeLockerError("HTTP 404: order not found")
        client.columns = {"orders": "orders"}
        client.orders.return_value = [{"id": "ord-1", "strategyId": "GOAT5|GOAT5-ch1-9|ch_1"}]
        trade = self._make_trade()
        runner, _ = self._make_runner(bot, [trade])
        bot.runners = {"CO": runner}
        info = {"runner_name": "CO", "kind": "limit_open", "order_id": "ord-1"}
        ret = bot._cancel_limit_order("GOAT5-ch1-9", info)
        self.assertFalse(ret)  # keep tracked, retry later
        self.assertIn("cancel_retry_at", info)
        self.assertIn(trade, runner.state.challenges[0].open_trades)
        self.assertIsNone(trade.position_id)

    def test_cancel_404_and_order_gone_discards(self):
        bot = self._make_bot()
        client = bot.brokers["test_broker"]["client"]
        client.cancel_order.side_effect = TradeLockerError("HTTP 404: order not found")
        client.columns = {"orders": "orders"}
        client.orders.return_value = []
        client.positions.return_value = []
        trade = self._make_trade()
        runner, _ = self._make_runner(bot, [trade])
        bot.runners = {"CO": runner}
        info = {"runner_name": "CO", "kind": "limit_open", "order_id": "ord-1"}
        ret = bot._cancel_limit_order("GOAT5-ch1-9", info)
        self.assertTrue(ret)

    def test_on_limit_fill_pushes_stale_broker_sl(self):
        bot = self._make_bot()
        trade = self._make_trade(sl=4184.41, position_id=42, status="open")
        runner, _ = self._make_runner(bot, [trade])
        bot.runners = {"CO": runner}
        bot._broker_sl["GOAT5-ch1-9"] = 4151.30  # order placed with the old SL
        bot.push_trail_sl_to_broker = MagicMock()
        bot._on_limit_fill("GOAT5-ch1-9", 4151.30)
        bot.push_trail_sl_to_broker.assert_called_once()
        self.assertNotIn("GOAT5-ch1-9", bot._pending_orders)

    def test_on_limit_fill_no_push_when_armed_matches(self):
        bot = self._make_bot()
        trade = self._make_trade(sl=4184.41, position_id=42, status="open")
        runner, _ = self._make_runner(bot, [trade])
        bot.runners = {"CO": runner}
        bot.push_trail_sl_to_broker = MagicMock()
        bot._on_limit_fill("GOAT5-ch1-9", 4184.41)
        bot.push_trail_sl_to_broker.assert_not_called()

    def test_adopt_filled_limit_every_poll(self):
        bot = self._make_bot()
        client = bot.brokers["test_broker"]["client"]
        client.columns = {"positions": "positions"}
        client.positions.return_value = [
            {"id": 777, "comment": "GOAT5|GOAT5-ch1-9|ch_1", "stopLoss": 99.0}
        ]
        trade = self._make_trade(sl=101.0)  # memory trail != order SL
        runner, _ = self._make_runner(bot, [trade])
        bot.runners = {"CO": runner}
        bot._pending_orders = {"GOAT5-ch1-9": {"runner_name": "CO", "kind": "limit_open",
                                               "order_id": "ord-1", "sent_at": 0}}
        bot.push_trail_sl_to_broker = MagicMock()
        bot._adopt_filled_limits()
        self.assertEqual(trade.position_id, 777)
        self.assertEqual(trade.status, "open")
        bot.trade_store.save_trade.assert_called()
        self.assertNotIn("GOAT5-ch1-9", bot._pending_orders)
        bot.push_trail_sl_to_broker.assert_called_once()

    def test_watchdog_skips_when_broker_stop_armed(self):
        bot = self._make_bot()
        broker = bot.brokers["test_broker"]
        broker.update({"current_price": 100.0, "last_bid": 98.0, "last_ask": 98.5,
                       "last_poll_time": time.time()})
        client = broker["client"]
        trade = self._make_trade(sl=99.0, position_id=42, status="open")
        trade.direction = 1
        runner, _ = self._make_runner(bot, [trade])
        bot.runners = {"CO": runner}
        bot._broker_sl["GOAT5-ch1-9"] = 99.0  # armed at the trail level
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        client.close_position.assert_not_called()

    def test_watchdog_forces_exit_when_stop_not_armed(self):
        bot = self._make_bot()
        broker = bot.brokers["test_broker"]
        broker.update({"current_price": 100.0, "last_bid": 98.0, "last_ask": 98.5,
                       "last_poll_time": time.time()})
        client = broker["client"]
        trade = self._make_trade(sl=99.0, position_id=42, status="open")
        runner, ch = self._make_runner(bot, [trade])
        runner._check_early_close = MagicMock()
        ch.phase = 3
        bot.runners = {"CO": runner}
        bot._broker_sl["GOAT5-ch1-9"] = 95.0  # broker stop far below the trail
        with patch("main.xau_market_open", return_value=True):
            bot._sl_slippage_check()
        client.close_position.assert_called_once_with(42, qty=trade.lots)

    def test_intrabar_trail_ratchets_and_throttles(self):
        bot = self._make_bot()
        trade = self._make_trade(sl=99.0, position_id=42, status="open")
        trade._trail_px = 98.0
        runner = MagicMock()
        runner.config = {"tag": "GOAT5", "use_trail": True}
        runner._next_trail_px.return_value = 99.5
        bar = {"high": 102.0, "low": 101.0}
        bot._ratchet_live_trail(runner, trade, bar, "test_broker")
        self.assertEqual(trade._trail_px, 99.5)
        runner._apply_trail_move.assert_called_once()
        runner._next_trail_px.reset_mock()
        bot._ratchet_live_trail(runner, trade, bar, "test_broker")
        runner._next_trail_px.assert_not_called()  # throttled (10s)

if __name__ == "__main__":
    unittest.main()
