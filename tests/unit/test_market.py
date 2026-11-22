import unittest
from datetime import datetime, timezone

import engine.market as market
from engine.market import (
    xau_market_open, FARMS, GOAT_FIRM_5K_PROFILE, AQUA_FIRM_5K_PROFILE,
    FP_FIRM_5K_PROFILE, COMMISSION, CONTRACT_SIZE, MIN_LOT, MAX_LOT, LOT_STEP,
    FUNDED_RISK_PER_TRADE, PROFIT_SPLIT, DAILY_PROFIT_CAP, MIN_TRADING_DAYS,
    QUALIFYING_DAY_PCT, PAYOUT_FREQ_DAYS, PAYOUT_VALID_DAYS, FIRST_PAYOUT_CAP_N,
    FIRST_PAYOUT_CAP, LEVERAGE_EVAL, LEVERAGE_FUNDED, SYMBOL,
)


class TestXauMarketOpen(unittest.TestCase):
    def test_monday_open(self):
        self.assertTrue(xau_market_open(datetime(2025, 1, 13, 10, 0, tzinfo=timezone.utc)))

    def test_friday_before_close(self):
        self.assertTrue(xau_market_open(datetime(2025, 1, 10, 20, 0, tzinfo=timezone.utc)))

    def test_friday_after_close(self):
        self.assertFalse(xau_market_open(datetime(2025, 1, 10, 21, 5, tzinfo=timezone.utc)))

    def test_saturday_closed(self):
        self.assertFalse(xau_market_open(datetime(2025, 1, 11, 10, 0, tzinfo=timezone.utc)))

    def test_sunday_before_open(self):
        self.assertFalse(xau_market_open(datetime(2025, 1, 12, 21, 0, tzinfo=timezone.utc)))

    def test_sunday_after_open(self):
        self.assertTrue(xau_market_open(datetime(2025, 1, 12, 22, 10, tzinfo=timezone.utc)))

    def test_market_constants_match_boundaries(self):
        self.assertEqual(market.MARKET_CLOSED_FRI_UTC, (21, 5))
        self.assertEqual(market.MARKET_OPEN_SUN_UTC, (22, 10))


class TestConstants(unittest.TestCase):
    def test_trading_constants(self):
        self.assertEqual(MIN_TRADING_DAYS, 3)
        self.assertEqual(QUALIFYING_DAY_PCT, 0.005)
        self.assertEqual(PROFIT_SPLIT, 0.80)
        self.assertEqual(PAYOUT_FREQ_DAYS, 14)
        self.assertEqual(PAYOUT_VALID_DAYS, 4)
        self.assertEqual(FIRST_PAYOUT_CAP_N, 2)
        self.assertEqual(FIRST_PAYOUT_CAP, 200.0)
        self.assertEqual(DAILY_PROFIT_CAP, 3000.0)

    def test_execution_constants(self):
        self.assertEqual(COMMISSION, 3.0)
        self.assertEqual(CONTRACT_SIZE, 100)
        self.assertEqual(MIN_LOT, 0.01)
        self.assertEqual(MAX_LOT, 50.0)
        self.assertEqual(LOT_STEP, 0.01)
        self.assertEqual(market.MAX_HOLD_BARS, 480)
        self.assertEqual(FUNDED_RISK_PER_TRADE, 0.01)
        self.assertEqual(LEVERAGE_EVAL, 100)
        self.assertEqual(LEVERAGE_FUNDED, 50)
        self.assertEqual(SYMBOL, "XAUUSD")


class TestFirmProfiles(unittest.TestCase):
    def test_only_three_profiles_exported(self):
        exported = [n for n in dir(market) if n.endswith("_PROFILE")]
        self.assertEqual(
            set(exported),
            {"GOAT_FIRM_5K_PROFILE", "AQUA_FIRM_5K_PROFILE", "FP_FIRM_5K_PROFILE"},
        )

    def test_retired_profiles_gone(self):
        retired = (
            "GOAT_5K_PROFILE", "AQUAFUNDED_5K_PROFILE", "GOAT_100K_PROFILE",
            "AQUAFUNDED_STANDARD_5K_PROFILE", "AQUAFUNDED_ELITE_5K_PROFILE",
            "AQUAFUNDED_STANDARD_100K_PROFILE", "FUNDINGPIPS_2STEP_5K_PROFILE",
            "FUNDINGPIPS_ZERO_5K_PROFILE", "GOAT_INSTANT_100K_PROFILE",
        )
        for name in retired:
            self.assertFalse(hasattr(market, name), name)

    def test_common_profile_rules(self):
        for name, p in (
            ("GOAT", GOAT_FIRM_5K_PROFILE),
            ("AQUA", AQUA_FIRM_5K_PROFILE),
            ("FP", FP_FIRM_5K_PROFILE),
        ):
            self.assertEqual(p["ib"], 5000.0, name)
            self.assertEqual(p["sm"], "atr", name)
            self.assertEqual(p["rpt"], 0.01, name)
            self.assertEqual(p["min_valid_days"], 3, name)
            self.assertEqual(p["valid_day_mode"], "entry", name)
            self.assertEqual(p["first_payout_cap"], float("inf"), name)
            self.assertEqual(p["daily_profit_cap"], float("inf"), name)
            self.assertEqual(p["daily_dd_mode"], "script", name)
            self.assertEqual(p["payout_freq_days"], 14, name)
            self.assertEqual(p["payout_mode"], "retain", name)
            self.assertEqual(p["payout_min_valid_days"], 0, name)
            self.assertEqual(p["swap_rate"], -0.05, name)

    def test_firm_fees_targets_and_drawdowns(self):
        expected = (
            (GOAT_FIRM_5K_PROFILE, 17.0, 0.08, 0.06, 0.10, 0.04, 0.80),
            (AQUA_FIRM_5K_PROFILE, 39.0, 0.08, 0.05, 0.08, 0.05, 0.90),
            (FP_FIRM_5K_PROFILE, 36.0, 0.08, 0.05, 0.10, 0.05, 0.80),
        )
        for p, cc, p1, p2, md, dl, split in expected:
            self.assertEqual(p["cc"], cc, p)
            self.assertEqual(p["p1"], p1, p)
            self.assertEqual(p["p2"], p2, p)
            self.assertEqual(p["md"], md, p)
            self.assertEqual(p["dl"], dl, p)
            self.assertEqual(p["profit_split"], split, p)

    def test_derived_levels(self):
        ib = GOAT_FIRM_5K_PROFILE["ib"]
        self.assertAlmostEqual(ib * (1 - GOAT_FIRM_5K_PROFILE["md"]), 4500.0)
        self.assertAlmostEqual(ib * (1 + GOAT_FIRM_5K_PROFILE["p1"]), 5400.0)
        self.assertAlmostEqual(ib * (1 + GOAT_FIRM_5K_PROFILE["p2"]), 5300.0)
        self.assertAlmostEqual(ib * GOAT_FIRM_5K_PROFILE["dl"], 200.0)
        self.assertAlmostEqual(ib * (1 + AQUA_FIRM_5K_PROFILE["p2"]), 5250.0)


class TestFarms(unittest.TestCase):
    def test_three_firms_with_correct_tags(self):
        self.assertEqual(len(FARMS), 3)
        self.assertEqual([f["tag"] for f in FARMS], ["GOAT5", "AQUA5", "FPST5"])
        self.assertEqual(
            [f["name"] for f in FARMS],
            ["Goat 5K", "Aqua Standard 5K", "FundingPips 2-Step"],
        )

    def test_profile_and_broker_wiring(self):
        expected = (
            ("GOAT5", GOAT_FIRM_5K_PROFILE, "Goat 5K"),
            ("AQUA5", AQUA_FIRM_5K_PROFILE, "Aqua Standard 5K"),
            ("FPST5", FP_FIRM_5K_PROFILE, "FundingPips 2-Step"),
        )
        for tag, profile, name in expected:
            f = next(f for f in FARMS if f["tag"] == tag)
            self.assertIs(f["profile"], profile, tag)
            self.assertEqual(f["name"], name, tag)
            self.assertEqual(f["broker_key"], "herofx_goat", tag)
            self.assertEqual(f["broker"], "goat100k", tag)
            self.assertEqual(f["mode"], "rec", tag)
            self.assertEqual(f["max_acc"], 80, tag)

    def test_all_farms_use_one_shared_strategy_config(self):
        for f in FARMS:
            tag = f["tag"]
            self.assertEqual(f["eval_strat"], "Supertrend RSI", tag)
            self.assertEqual(f["funded_strat"], "Supertrend RSI", tag)
            self.assertEqual(f["eval_tf"], 60, tag)
            self.assertEqual(f["funded_tf"], 60, tag)
            self.assertEqual(f["eval_rr"], 3.0, tag)
            self.assertEqual(f["funded_rr"], 3.0, tag)
            self.assertEqual(f["stop_mode"], "fixed", tag)
            self.assertEqual(f["stop_value"], 25.0, tag)
            self.assertEqual(f["entry_slip"], 0.0, tag)
            self.assertEqual(f["eval_rpt"], 0.01, tag)
            self.assertEqual(f["funded_rpt"], 0.01, tag)
            self.assertIsNone(f["addon_min_r"], tag)

    def test_execution_flags(self):
        for f in FARMS:
            tag = f["tag"]
            self.assertTrue(f["use_delayed_signal"], tag)
            self.assertEqual(f["entry_delay_bars"], 1, tag)
            self.assertFalse(f["use_limit_entry"], tag)
            self.assertFalse(f["use_trail"], tag)
            self.assertEqual(f["max_hold_bars"], 0, tag)
            self.assertEqual(f["max_concurrent"], 2, tag)
            self.assertEqual(f["sizing_mode"], "script", tag)
            self.assertTrue(f["early_close"], tag)

    def test_retired_farm_tags_gone(self):
        for tag in ("AQEL5", "AQUA10", "FPZRO", "GOATI1", "FR", "CO", "TCH"):
            self.assertFalse([f for f in FARMS if f["tag"] == tag], tag)


if __name__ == "__main__":
    unittest.main()
