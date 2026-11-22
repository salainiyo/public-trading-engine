import unittest

import numpy as np

from engine.indicators import (
    rsi, atr, atr_simple, rolling_max, rolling_min, build_indicator_dict,
)


class TestRsi(unittest.TestCase):
    def test_warmup_and_length(self):
        arr = np.array([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
                        46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03,
                        46.41, 46.22, 45.64, 44.09, 44.34, 44.09, 43.61, 44.33,
                        44.83, 45.10, 45.42, 45.84, 46.08], dtype=float)
        result = rsi(arr, 14)
        self.assertEqual(len(result), len(arr))
        self.assertTrue(np.isnan(result[:14]).all())
        self.assertFalse(np.isnan(result[14]))

    def test_cutler_rsi_matches_manual_calculation(self):
        arr = np.array([100.0 + 3 * (i // 2) - (1.0 if i % 2 else 0.0)
                        for i in range(20)])
        d = np.diff(arr[:15])
        expected = 100 - 100 / (1 + d.clip(min=0).mean() / (-d).clip(min=0).mean())
        result = rsi(arr, 14)
        self.assertAlmostEqual(result[14], expected)
        self.assertGreater(result[14], 60.0)

    def test_flat_series_is_nan(self):
        result = rsi(np.full(20, 50.0), 14)
        self.assertTrue(np.all(np.isnan(result[13:])))


class TestAtr(unittest.TestCase):
    @staticmethod
    def _series():
        h = np.array([110.0, 112.0, 115.0, 113.0, 116.0, 118.0, 117.0, 119.0,
                      121.0, 120.0, 122.0, 124.0, 123.0, 125.0, 127.0, 126.0])
        l = np.array([108.0, 109.0, 111.0, 110.0, 112.0, 114.0, 113.0, 115.0,
                      117.0, 116.0, 118.0, 120.0, 119.0, 121.0, 123.0, 122.0])
        c = np.array([109.0, 111.0, 114.0, 112.0, 115.0, 117.0, 115.0, 118.0,
                      120.0, 118.0, 121.0, 123.0, 121.0, 124.0, 126.0, 124.0])
        return h, l, c

    def test_warmup_and_length(self):
        h, l, c = self._series()
        result = atr(h, l, c, 14)
        self.assertEqual(len(result), len(h))
        self.assertTrue(np.isnan(result[:13]).all())
        self.assertFalse(np.isnan(result[13]))

    def test_wilder_recurrence(self):
        h, l, c = self._series()
        n = 14
        tr = np.fmax.reduce([
            h - l,
            np.abs(h - np.concatenate([[np.nan], c[:-1]])),
            np.abs(l - np.concatenate([[np.nan], c[:-1]])),
        ])
        first = tr[:n].mean()
        result = atr(h, l, c, n)
        self.assertAlmostEqual(result[n - 1], first)
        self.assertAlmostEqual(result[n], (first * (n - 1) + tr[n]) / n)
        self.assertAlmostEqual(result[-1], (result[-2] * (n - 1) + tr[-1]) / n)


class TestAtrSimple(unittest.TestCase):
    def test_matches_rolling_tr_mean(self):
        h = np.array([110.0, 112.0, 115.0, 113.0, 116.0, 118.0, 117.0, 119.0,
                      121.0, 120.0, 122.0, 124.0, 123.0, 125.0, 127.0, 126.0])
        l = np.array([108.0, 109.0, 111.0, 110.0, 112.0, 114.0, 113.0, 115.0,
                      117.0, 116.0, 118.0, 120.0, 119.0, 121.0, 123.0, 122.0])
        c = np.array([109.0, 111.0, 114.0, 112.0, 115.0, 117.0, 115.0, 118.0,
                      120.0, 118.0, 121.0, 123.0, 121.0, 124.0, 126.0, 124.0])
        result = atr_simple(h, l, c, 10)
        self.assertEqual(len(result), len(h))
        self.assertTrue(np.isnan(result[:9]).all())
        prev_c = np.concatenate([[np.nan], c[:-1]])
        tr = np.fmax.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
        self.assertAlmostEqual(result[9], tr[:10].mean())
        self.assertAlmostEqual(result[-1], tr[-10:].mean())


class TestRollingExtrema(unittest.TestCase):
    def test_rolling_max(self):
        arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = rolling_max(arr, 3)
        self.assertTrue(np.isnan(result[:2]).all())
        self.assertAlmostEqual(result[2], 3.0)
        self.assertAlmostEqual(result[3], 5.0)
        self.assertAlmostEqual(result[4], 5.0)

    def test_rolling_min(self):
        arr = np.array([5.0, 3.0, 4.0, 1.0, 2.0])
        result = rolling_min(arr, 3)
        self.assertTrue(np.isnan(result[:2]).all())
        self.assertAlmostEqual(result[2], 3.0)
        self.assertAlmostEqual(result[3], 1.0)
        self.assertAlmostEqual(result[4], 1.0)


class TestBuildIndicatorDict(unittest.TestCase):
    @staticmethod
    def _bars(n=60, seed=7):
        rng = np.random.RandomState(seed)
        opens = 100.0 + rng.randn(n).cumsum()
        highs = opens + np.abs(rng.randn(n))
        lows = opens - np.abs(rng.randn(n))
        closes = (opens + highs + lows) / 3.0
        volumes = rng.uniform(100, 1000, n)
        times = np.arange(n, dtype=np.int64) * 3600 + 1_700_000_000
        return opens, highs, lows, closes, volumes, times

    def test_exact_key_set_and_lengths(self):
        bars = self._bars()
        r = build_indicator_dict(*bars)
        self.assertEqual(
            set(r),
            {"O", "H", "L", "C", "V", "time",
             "RSI14", "ATR14", "ATR10", "RB_H20", "RB_L20"},
        )
        for key, val in r.items():
            self.assertEqual(len(val), 60, key)
        self.assertEqual(r["time"].dtype, np.dtype(np.int64))
        self.assertEqual(r["C"].dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(r["C"], bars[3])

    def test_indicators_match_direct_calls(self):
        opens, highs, lows, closes, volumes, times = self._bars()
        r = build_indicator_dict(opens, highs, lows, closes, volumes, times)
        np.testing.assert_allclose(r["RSI14"], rsi(closes, 14))
        np.testing.assert_allclose(r["ATR14"], atr(highs, lows, closes, 14))
        np.testing.assert_allclose(r["ATR10"], atr_simple(highs, lows, closes, 10))
        np.testing.assert_allclose(r["RB_H20"], rolling_max(highs, 20))
        np.testing.assert_allclose(r["RB_L20"], rolling_min(lows, 20))
        self.assertTrue(np.isnan(r["RB_H20"][:19]).all())
        self.assertFalse(np.isnan(r["RB_H20"][19:]).any())


if __name__ == "__main__":
    unittest.main()
