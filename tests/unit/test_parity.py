"""Parity engine + live-vs-backtest comparison tests.

The archive scripts live outside the Docker image and require numba, so these
tests exercise the pure-Python port in `parity/` on deterministic synthetic
bars (no archive import).
"""
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from engine.indicators import build_indicator_dict
from parity.checker import build_report, compare_trades, normalize_live
from parity.engine import COST_PER_OZ, run_trades
from main import Bot


SIGNAL = "Supertrend RSI"


def _bars(n=400, start=1760000000, base=4000.0):
    """Rising zig-zag bars: real down-ticks keep Cutler RSI finite, and the
    steady new highs fire the Supertrend RSI breakout pattern."""
    op, hi, lo, cl, vol, times = [], [], [], [], [], []
    for i in range(n):
        c = base + 5.0 * (i // 2) - (2.0 if i % 2 else 0.0)
        op.append(c - 1.0)
        hi.append(c + 4.0)
        lo.append(c - 4.0)
        cl.append(c)
        vol.append(1000)
        times.append(start + i * 3600)
    return {"open": op, "high": hi, "low": lo, "close": cl,
            "volume": vol, "time": times}


def _sim_kwargs(**over):
    kw = dict(rr=3.0, stop_mode="fixed", stop_value=25.0,
              risk_pct=0.01, initial=5000.0)
    kw.update(over)
    return kw


def _iso(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


class TestRunTrades(unittest.TestCase):
    def test_supertrend_rsi_produces_trades(self):
        r = run_trades(_bars(), SIGNAL, **_sim_kwargs())
        self.assertGreater(len(r["trades"]), 0)
        self.assertEqual(r["bars"], 400)
        self.assertTrue(all(t["dir"] == 1 for t in r["trades"]))

    def test_defaults_are_rr3_one_percent_atr_stop(self):
        defaults = run_trades(_bars(), SIGNAL)
        explicit = run_trades(_bars(), SIGNAL, rr=3.0, risk_pct=0.01,
                              stop_mode="atr", atr_mult=3.0)
        self.assertEqual(
            [(t["entry_time"], t["exit_time"], t["pnl"]) for t in defaults["trades"]],
            [(t["entry_time"], t["exit_time"], t["pnl"]) for t in explicit["trades"]],
        )
        self.assertTrue(defaults["trades"])

    def test_entry_stop_risk_match_script_model(self):
        bars = _bars()
        r = run_trades(bars, SIGNAL, **_sim_kwargs())
        idx_by_time = {t: i for i, t in enumerate(bars["time"])}
        self.assertTrue(r["trades"])
        for t in r["trades"]:
            i = idx_by_time[t["entry_time"]]
            self.assertGreater(i, 0)
            self.assertAlmostEqual(t["entry"], bars["open"][i], places=6)
            self.assertAlmostEqual(t["risk"],
                                   t["units"] * 25.0 + COST_PER_OZ * t["units"],
                                   places=4)
            self.assertAlmostEqual(t["r"], t["pnl"] / t["risk"], places=3)
            self.assertAlmostEqual(t["exit"], t["entry"] + 75.0, places=6)

    def test_atr_stop_mode_still_available(self):
        bars = _bars()
        r = run_trades(bars, SIGNAL, rr=3.0, atr_mult=3.0,
                       risk_pct=0.01, stop_mode="atr", initial=5000.0)
        ind = build_indicator_dict(
            [float(x) for x in bars["open"]], [float(x) for x in bars["high"]],
            [float(x) for x in bars["low"]], [float(x) for x in bars["close"]],
            [float(x) for x in bars["volume"]],
            [int(x) for x in bars["time"]])
        atr = ind["ATR14"]
        idx_by_time = {t: i for i, t in enumerate(bars["time"])}
        for t in r["trades"]:
            i = idx_by_time[t["entry_time"]]
            dist = 3.0 * atr[i - 1]
            self.assertAlmostEqual(t["risk"],
                                   t["units"] * dist + COST_PER_OZ * t["units"],
                                   places=3)

    def test_min_units_floor_and_skip_below_min(self):
        r = run_trades(_bars(), SIGNAL, **_sim_kwargs())
        self.assertTrue(all(t["units"] >= 1 for t in r["trades"]))
        self.assertEqual(r["trades"][0]["lots"], 0.02)
        skipped = run_trades(_bars(), SIGNAL, **_sim_kwargs(initial=100.0,
                                                            skip_below_min=True))
        self.assertEqual(skipped["trades"], [])
        forced = run_trades(_bars(), SIGNAL, **_sim_kwargs(initial=100.0))
        self.assertTrue(all(t["units"] >= 1 for t in forced["trades"]))

    def test_max_two_concurrent_positions(self):
        bars = _bars()
        r = run_trades(bars, SIGNAL, **_sim_kwargs(initial=100000.0, max_pos=2))
        spans = [(t["entry_time"], t["exit_time"]) for t in r["trades"]]
        self.assertTrue(spans)
        for s, _ in spans:
            overlapping = sum(1 for s2, e2 in spans if s2 <= s < e2)
            self.assertLessEqual(overlapping, 2)

    def test_addon_min_r_gates_second_position(self):
        bars = _bars()
        gated = run_trades(bars, SIGNAL, **_sim_kwargs(initial=100000.0,
                                                        addon_min_r=99.0))
        self.assertTrue(gated["trades"])
        spans = [(t["entry_time"], t["exit_time"]) for t in gated["trades"]]
        for s, _ in spans:
            self.assertLessEqual(sum(1 for s2, e2 in spans if s2 <= s < e2), 1)

    def test_partial_r_emits_partial_exits(self):
        r = run_trades(_bars(), SIGNAL, **_sim_kwargs(partial_r=1.5))
        self.assertTrue(any(t["exit_reason"].startswith("partial")
                            for t in r["trades"]))

    def test_unknown_signal_is_noop(self):
        r = run_trades(_bars(), "No Such Signal", initial=5000.0)
        self.assertEqual(r["trades"], [])
        self.assertEqual(r["final_balance"], 5000.0)

    def test_start_index_discards_earlier_state(self):
        bars = _bars()
        full = run_trades(bars, SIGNAL, **_sim_kwargs(initial=100000.0))
        self.assertTrue(full["trades"])
        first_idx = bars["time"].index(full["trades"][0]["entry_time"])
        start = first_idx + 1
        aligned = run_trades(bars, SIGNAL, **_sim_kwargs(initial=100000.0,
                                                         start_index=start))
        self.assertTrue(all(t["entry_time"] >= bars["time"][start]
                            for t in aligned["trades"]))


class TestCompareTrades(unittest.TestCase):
    def _live(self, **over):
        t = {"id": "GOAT5-ch1-1", "status": "closed", "direction": 1,
             "entry": 4000.0, "sl": 3975.0, "lots": 0.1, "pnl": 250.0,  # R = 1
             "open_time": _iso(1760000000), "close_time": _iso(1760003600),
             "exit_price": 4010.0, "exit_reason": "tp"}
        t.update(over)
        return t

    def _bt(self, **over):
        t = {"dir": 1, "entry": 4000.0, "exit": 4010.0,
             "entry_time": 1760000000, "exit_time": 1760003600,
             "r": 1.0, "exit_reason": "tp"}
        t.update(over)
        return t

    def test_normalize_skips_open_and_unsettled(self):
        live = [self._live(), self._live(status="open"),
                self._live(pnl=None, id="x")]
        self.assertEqual(len(normalize_live(live)), 1)

    def test_matched_trade_ok(self):
        c = compare_trades([self._live()], [self._bt()])
        self.assertEqual(c["matched"], 1)
        self.assertEqual(c["live_only"], 0)
        self.assertEqual(c["bt_only"], 0)
        self.assertEqual(c["mean_r_delta"], 0.0)
        self.assertEqual(build_report("Goat 5K", "GOAT", c)["status"], "OK")

    def test_direction_mismatch_flags_drift(self):
        c = compare_trades([self._live(direction=-1)], [self._bt()])
        self.assertEqual(c["signal_mismatches"], 1)
        self.assertEqual(c["matched"], 0)
        self.assertEqual(build_report("Goat 5K", "GOAT", c)["status"], "DRIFT")

    def test_large_r_delta_flags_drift(self):
        c = compare_trades([self._live(pnl=300.0)], [self._bt(r=1.0)])
        self.assertGreater(abs(c["mean_r_delta"]), 0.15)
        self.assertEqual(build_report("Goat 5K", "GOAT", c)["status"], "DRIFT")

    def test_partial_exit_is_expected_divergence(self):
        bt = self._bt(exit_reason="partial_tp", r=3.0)
        c = compare_trades([self._live(pnl=100.0)], [bt])
        self.assertEqual(c["expected_divergence"], 1)
        self.assertIsNone(c["mean_r_delta"])
        report = build_report("Goat 5K", "GOAT", c)
        self.assertEqual(report["status"], "OK")
        self.assertTrue(report["trades"][0]["expected"])

    def test_no_live_trades_is_no_data(self):
        c = compare_trades([], [self._bt()])
        self.assertEqual(c["live_trades"], 0)
        self.assertEqual(build_report("Goat 5K", "GOAT", c)["status"], "NO DATA")

    def test_single_live_trade_is_no_data(self):
        c = compare_trades([self._live()], [self._bt(entry_time=1760007200)])
        self.assertEqual(c["live_trades"], 1)
        self.assertEqual(build_report("Goat 5K", "GOAT", c)["status"], "NO DATA")

    def test_one_bar_execution_grace_matches(self):
        c = compare_trades([self._live(open_time=_iso(1760003605))],
                           [self._bt(entry_time=1760000000)])
        self.assertEqual(c["matched"], 1)
        self.assertEqual(c["live_only"], 0)

    def test_four_bar_gap_is_live_only(self):
        c = compare_trades([self._live(open_time=_iso(1760000000 + 4 * 3600))],
                           [self._bt(entry_time=1760000000)])
        self.assertEqual(c["matched"], 0)
        self.assertEqual(c["live_only"], 1)

    def test_small_sample_one_match_allows_one_unmatched(self):
        c = {"live_trades": 2, "matched": 1, "live_only": 1, "bt_only": 1,
             "signal_mismatches": 0, "mean_r_delta": None}
        self.assertEqual(build_report("Goat 5K", "GOAT", c)["status"], "OK")

    def test_small_sample_zero_matches_is_drift(self):
        c = {"live_trades": 2, "matched": 0, "live_only": 2, "bt_only": 2,
             "signal_mismatches": 0, "mean_r_delta": None}
        self.assertEqual(build_report("Goat 5K", "GOAT", c)["status"], "DRIFT")

    def test_error_report_drift(self):
        report = build_report("Goat 5K", "GOAT", {}, error="boom")
        self.assertEqual(report["status"], "DRIFT")
        self.assertEqual(report["error"], "boom")


class TestRunParityCheck(unittest.TestCase):
    @staticmethod
    def _runner():
        runner = MagicMock()
        runner.name = "Goat 5K"
        runner.config = {
            "tag": "GOAT5", "eval_strat": SIGNAL, "eval_tf": 60,
            "eval_rr": 3.0, "stop_mode": "fixed", "stop_value": 25.0,
            "eval_rpt": 0.01, "addon_min_r": None,
        }
        runner.profile = {"ib": 5000.0, "rpt": 0.01}
        runner._max_concurrent.return_value = 2
        return runner

    def test_live_iso_open_time_is_normalized(self):
        bot = Bot.__new__(Bot)
        bot.parity_reports = {}
        bars = _bars()
        bot._fetch_parity_bars = lambda runner, cache: bars
        runner = self._runner()
        bot._collect_live_trades = lambda r, since: [{
            "id": "x", "status": "closed", "direction": 1, "entry": 4000.0,
            "sl": 3975.0, "lots": 0.1, "pnl": 10.0,
            "open_time": _iso(bars["time"][10]),
            "close_time": _iso(bars["time"][11]),
            "exit_price": 4010.0, "exit_reason": "tp"}]
        bot._run_parity_check(runner, {})
        report = bot.parity_reports["Goat 5K"]
        self.assertIsNone(report["error"])
        self.assertEqual(report["live_trades"], 1)

    def test_parity_check_starts_flat_at_first_live_bar(self):
        bot = Bot.__new__(Bot)
        bot.parity_reports = {}
        bars = _bars()
        bot._fetch_parity_bars = lambda runner, cache: bars
        runner = self._runner()
        first_bar = bars["time"][20]
        bot._collect_live_trades = lambda r, since: [{
            "id": "x", "status": "closed", "direction": 1, "entry": 4000.0,
            "sl": 3975.0, "lots": 0.1, "pnl": 10.0,
            "open_time": _iso(first_bar + 5),
            "close_time": _iso(bars["time"][21]),
            "exit_price": 4010.0, "exit_reason": "tp"}]
        with patch("main.run_trades", return_value={"trades": [], "bars": len(bars["time"])}) as rt:
            bot._run_parity_check(runner, {})
        self.assertEqual(rt.call_args.kwargs["start_index"], 20)
        self.assertEqual(rt.call_args.kwargs["stop_mode"], "fixed")
        self.assertEqual(rt.call_args.kwargs["stop_value"], 25.0)
        self.assertNotIn("partial_r", rt.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
