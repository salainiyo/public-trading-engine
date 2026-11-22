import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from engine.reconciler import Reconciler
from engine.models import Trade


class TestReconcilerReconcile(unittest.TestCase):
    def test_parse_comment(self):
        s, tid, ch = Reconciler._parse_comment("CO|trade-1|ch_2")
        self.assertEqual(s, "CO")
        self.assertEqual(tid, "trade-1")
        self.assertEqual(ch, 2)

    def test_parse_comment_no_pipe(self):
        s, tid, ch = Reconciler._parse_comment("nocomment")
        self.assertIsNone(s)
        self.assertIsNone(tid)
        self.assertIsNone(ch)

    def test_parse_ts_garbage_year(self):
        self.assertIsNone(Reconciler._parse_ts("1789-03-08T07:40:00"))
        self.assertIsNone(Reconciler._parse_ts(-5_000_000_000))

    def test_parse_ts_naive_string_gets_utc(self):
        dt = Reconciler._parse_ts("2026-09-28T10:00:00")
        self.assertIsNotNone(dt)
        self.assertIsNotNone(dt.tzinfo)

    def test_reconcile_match_persists_position_id(self):
        ts = MagicMock()
        ts.load_open_trade_by_id.return_value = None
        reconciler = Reconciler(trade_store=ts)
        runner = MagicMock()
        runner.config = {"eval_strat": "CO", "funded_strat": "CO", "tag": "CO"}
        ch = MagicMock()
        ch.challenge_num = 1
        t = Trade(id="t1", strategy="CO", direction=1, entry=100.0, sl=99.0, tp=103.0,
                  lots=0.1, open_time=datetime.now(timezone.utc).isoformat())
        ch.open_trades = [t]
        runner.state.challenges = [ch]
        runner.state.current_challenge = ch
        pos = {
            "id": 100,
            "comment": "CO|t1|ch_1",
            "qty": 0.1,
            "side": "buy",
            "openPrice": 100.0,
            "stopLoss": 99.0,
            "takeProfit": 103.0,
            "pnl": 0.0,
            "openTime": datetime.now(timezone.utc).isoformat(),
        }
        reconciler.reconcile([pos], {"CO": runner})
        # position_id must be persisted to the DB so _cleanup_orphan_rows
        # cannot misidentify this live trade as an orphan
        self.assertEqual(t.position_id, 100)
        ts.save_trade.assert_called()

    def test_closed_uses_broker_close_time(self):
        # close_time must come from the broker row, not the detection time —
        # a settlement pass used to stamp every row with "now" (identical
        # timestamps for trades closed days apart).
        reconciler = Reconciler()
        runner = MagicMock()
        runner.config = {"eval_strat": "CO", "funded_strat": "CO", "tag": "CO"}
        ch = MagicMock()
        ch.challenge_num = 1
        t = Trade(id="t1", strategy="CO", direction=1, entry=100.0, sl=99.0,
                  tp=103.0, lots=0.1,
                  open_time="2026-10-02T06:00:00+00:00", position_id=100)
        ch.open_trades = [t]
        runner.state.challenges = [ch]
        runner.state.current_challenge = ch
        pos = {
            "id": 100,
            "comment": "CO|t1|ch_1",
            "qty": 0.1,
            "side": "buy",
            "openPrice": 100.0,
            "closePrice": 104.0,
            "status": "closed",
            "openTime": "2026-10-02T06:00:00+00:00",
            "closeTime": "2026-10-02T07:15:00+00:00",
            "pnl": 40.0,
        }
        reconciler.reconcile([pos], {"CO": runner})
        runner.on_trade_closed.assert_called_once()
        close_dt = runner.on_trade_closed.call_args[0][1]
        self.assertEqual(close_dt.hour, 7)
        self.assertEqual(close_dt.minute, 15)

    def test_adoption_passes_broker_sl_to_on_fill(self):
        reconciler = Reconciler()
        runner = MagicMock()
        runner.config = {"eval_strat": "CO", "funded_strat": "CO", "tag": "CO"}
        ch = MagicMock()
        ch.challenge_num = 1
        t = Trade(id="t1", strategy="CO", direction=1, entry=100.0, sl=99.0,
                  tp=103.0, lots=0.1,
                  open_time="2026-10-15T10:00:00+00:00", status="pending")
        ch.open_trades = [t]
        runner.state.challenges = [ch]
        runner.state.current_challenge = ch
        pos = {
            "id": 100,
            "comment": "CO|t1|ch_1",
            "qty": 0.1,
            "side": "buy",
            "openPrice": 100.0,
            "stopLoss": 99.0,
            "takeProfit": 103.0,
            "openTime": "2026-10-15T10:00:00+00:00",
        }
        on_fill = MagicMock()
        reconciler.reconcile([pos], {"CO": runner}, on_fill=on_fill)
        on_fill.assert_called_once_with("t1", 99.0)
        self.assertEqual(t.position_id, 100)
        self.assertEqual(t.status, "open")

    def test_reconcile_never_loosens_trailed_sl(self):
        # A stale broker SL (failed trail push) must not roll memory back past
        # the trailed level — the next trail push re-syncs the broker.
        ts = MagicMock()
        ts.load_open_trade_by_id.return_value = None
        reconciler = Reconciler(trade_store=ts)
        runner = MagicMock()
        runner.config = {"eval_strat": "CO", "funded_strat": "CO", "tag": "CO"}
        ch = MagicMock()
        ch.challenge_num = 1
        t = Trade(id="t1", strategy="CO", direction=1, entry=100.0, sl=101.0, tp=200.0,
                  lots=0.1, open_time=datetime.now(timezone.utc).isoformat())
        t.position_id = 100
        t._trail_px = 101.0
        ch.open_trades = [t]
        runner.state.challenges = [ch]
        runner.state.current_challenge = ch
        pos = {
            "id": 100,
            "comment": "CO|t1|ch_1",
            "qty": 0.1,
            "side": "buy",
            "openPrice": 100.0,
            "stopLoss": 99.0,  # stale initial SL, looser than the trail
            "takeProfit": 0.0,
            "pnl": 0.0,
            "openTime": datetime.now(timezone.utc).isoformat(),
        }
        reconciler.reconcile([pos], {"CO": runner})
        self.assertEqual(t.sl, 101.0)

    def test_reconcile_tightens_sl_from_broker(self):
        # A TIGHTER broker SL (higher for a long) is adopted normally.
        ts = MagicMock()
        ts.load_open_trade_by_id.return_value = None
        reconciler = Reconciler(trade_store=ts)
        runner = MagicMock()
        runner.config = {"eval_strat": "CO", "funded_strat": "CO", "tag": "CO"}
        ch = MagicMock()
        ch.challenge_num = 1
        t = Trade(id="t1", strategy="CO", direction=1, entry=100.0, sl=99.0, tp=200.0,
                  lots=0.1, open_time=datetime.now(timezone.utc).isoformat())
        t.position_id = 100
        t._trail_px = 101.0
        ch.open_trades = [t]
        runner.state.challenges = [ch]
        runner.state.current_challenge = ch
        pos = {
            "id": 100,
            "comment": "CO|t1|ch_1",
            "qty": 0.1,
            "side": "buy",
            "openPrice": 100.0,
            "stopLoss": 101.5,  # tighter than the trail → adopt
            "takeProfit": 0.0,
            "pnl": 0.0,
            "openTime": datetime.now(timezone.utc).isoformat(),
        }
        reconciler.reconcile([pos], {"CO": runner})
        self.assertEqual(t.sl, 101.5)

    def test_adoption_clamps_sl_to_persisted_trail(self):
        # New-to-memory position adopted after a restart: broker SL may be
        # stale/looser than the DB-persisted trail — clamp, don't roll back.
        ts = MagicMock()
        ts.load_open_trade_by_id.return_value = {
            "_sl_dist": 2.0, "_trail_px": 101.0, "sl": 99.0, "tp": 0.0,
        }
        reconciler = Reconciler(trade_store=ts)
        runner = MagicMock()
        runner.config = {"eval_strat": "CO", "funded_strat": "CO", "tag": "CO"}
        ch = MagicMock()
        ch.challenge_num = 1
        ch.open_trades = []  # not in memory → adoption path
        runner.state.challenges = [ch]
        runner.state.current_challenge = ch
        pos = {
            "id": 100,
            "comment": "CO|t9|ch_1",
            "qty": 0.1,
            "side": "buy",
            "openPrice": 100.0,
            "stopLoss": 99.0,  # stale initial SL
            "takeProfit": 0.0,
            "pnl": 0.0,
            "openTime": datetime.now(timezone.utc).isoformat(),
        }
        reconciler.reconcile([pos], {"CO": runner})
        self.assertEqual(len(ch.open_trades), 1)
        self.assertEqual(ch.open_trades[0]._trail_px, 101.0)
        self.assertEqual(ch.open_trades[0].sl, 101.0)

    def test_reconcile_open_position(self):
        ts = MagicMock()
        ts.load_open_trade_by_id.return_value = None
        reconciler = Reconciler(trade_store=ts)
        runner = MagicMock()
        runner.config = {"eval_strat": "CO", "funded_strat": "CO", "tag": "CO"}
        ch = MagicMock()
        ch.challenge_num = 1
        ch.open_trades = []
        runner.state.challenges = [ch]
        runner.state.current_challenge = ch
        pos = {
            "id": 100,
            "comment": "CO|t1|ch_1",
            "qty": 0.1,
            "side": "buy",
            "openPrice": 100.0,
            "stopLoss": 99.0,
            "takeProfit": 103.0,
            "pnl": 0.0,
            "openTime": datetime.now(timezone.utc).isoformat(),
        }
        reconciled, settled, closed, pmap = reconciler.reconcile([pos], {"CO": runner})
        self.assertGreaterEqual(reconciled + settled, 0)

    def test_reconcile_creates_new_from_broker(self):
        ts = MagicMock()
        ts.load_open_trade_by_id.return_value = None
        ts.save_trade.return_value = None
        reconciler = Reconciler(trade_store=ts)
        runner = MagicMock()
        runner.config = {"eval_strat": "CO", "funded_strat": "CO", "tag": "CO"}
        ch = MagicMock()
        ch.challenge_num = 1
        ch.open_trades = []
        runner.state.challenges = [ch]
        runner.state.current_challenge = ch
        pos = {
            "id": 200,
            "comment": "CO|t1|ch_1",
            "qty": 0.2,
            "side": "sell",
            "openPrice": 200.0,
            "openTime": datetime.now(timezone.utc).isoformat(),
        }
        reconciled, settled, closed, pmap = reconciler.reconcile([pos], {"CO": runner})
        self.assertEqual(len(ch.open_trades), 1)
        self.assertEqual(ch.open_trades[0].direction, -1)

    def test_reconcile_detects_manual_close(self):
        ts = MagicMock()
        reconciler = Reconciler(trade_store=ts)
        runner = MagicMock()
        runner.config = {"eval_strat": "CO", "funded_strat": "CO", "tag": "CO"}
        ot = MagicMock()
        ot.id = "t1"
        ot.position_id = 999
        ot.status = "open"
        ch = MagicMock()
        ch.challenge_num = 1
        ch.blown = False
        ch.open_trades = [ot]
        runner.state.challenges = [ch]
        runner.state.current_challenge = ch
        reconciled, settled, closed, pmap = reconciler.reconcile([], {"CO": runner})
        self.assertEqual(closed, 1)
        self.assertEqual(ot.status, "reconciled_unknown")


class TestReconcilerSettle(unittest.TestCase):
    def test_settle_idempotent(self):
        ts = MagicMock()
        reconciler = Reconciler(trade_store=ts)
        runner = MagicMock()
        runner.config = {"eval_strat": "CO"}
        ch = MagicMock()
        ch.blown = False
        ch.open_trades = []
        ch.trade_history = [{"id": "t1", "position_id": "100", "entry": 100.0, "direction": 1, "lots": 0.1, "pnl": 10.0, "status": "closed"}]
        runner.state.challenges = [ch]
        fills = [{"positionId": 100, "status": "filled", "filledQty": 0.1, "avgPrice": 100.1, "side": "sell", "strategyId": "CO|t1|ch_1"}]
        updated = reconciler.settle_manual_closes(fills, {"CO": runner}, {"100": ("CO", 1)})
        self.assertEqual(updated, 1)

    def test_settle_manual_close_with_fills(self):
        ts = MagicMock()
        reconciler = Reconciler(trade_store=ts)
        runner = MagicMock()
        runner.config = {"eval_strat": "CO"}
        ot = MagicMock()
        ot.id = "t2"
        ot.position_id = 200
        ot.entry = 100.0
        ot.direction = 1
        ot.lots = 0.1
        ot.pnl = None
        ot.status = "open"
        ch = MagicMock()
        ch.blown = False
        ch.open_trades = [ot]
        ch.trade_history = []
        runner.state.challenges = [ch]
        fills = [{"positionId": 200, "status": "filled", "filledQty": 0.1, "avgPrice": 110.0, "side": "sell", "strategyId": "CO|t2|ch_1"}]
        updated = reconciler.settle_manual_closes(fills, {"CO": runner}, {"200": ("CO", 1)})
        self.assertEqual(updated, 1)


if __name__ == "__main__":
    unittest.main()
