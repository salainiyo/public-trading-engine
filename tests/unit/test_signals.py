import unittest

import numpy as np
import pandas as pd

from engine.indicators import build_indicator_dict
from engine.signals import sig_supertrend_rsi, SIGNAL_MAP


def _indicator_dict(n, *, close=None, rsi=None):
    """Deterministic indicator dict with fully controlled breakout/extrema."""
    close = np.full(n, 100.0) if close is None else np.asarray(close, dtype=float)
    high = close + 1.0
    low = close - 1.0
    return {
        "O": close.copy(), "H": high, "L": low, "C": close,
        "V": np.ones(n), "time": np.arange(n, dtype=np.int64),
        "ATR10": np.full(n, 1.0),
        "RSI14": np.full(n, 50.0) if rsi is None else np.asarray(rsi, dtype=float),
        "RB_H20": np.full(n, 100.0),
        "RB_L20": np.full(n, 100.0),
    }


class TestSigSupertrendRsi(unittest.TestCase):
    def test_warmup_guard(self):
        r = _indicator_dict(30, close=np.arange(30.0) + 100.0, rsi=np.full(30, 80.0))
        r["RB_H20"] = np.full(30, -np.inf)
        for i in range(21):
            self.assertEqual(sig_supertrend_rsi(r, i), 0)

    def test_long_signal(self):
        r = _indicator_dict(30, rsi=np.full(30, 75.0))
        r["C"][25] = 110.0  # above prior 20-bar high and above the lower band
        self.assertEqual(sig_supertrend_rsi(r, 25), 1)

    def test_short_signal(self):
        r = _indicator_dict(30, rsi=np.full(30, 25.0))
        r["C"][25] = 90.0  # below prior 20-bar low and below the lower band
        self.assertEqual(sig_supertrend_rsi(r, 25), -1)

    def test_no_signal_rsi_neutral(self):
        r = _indicator_dict(30, rsi=np.full(30, 50.0))
        r["C"][25] = 110.0  # breakout satisfied, but RSI neither > 60 nor < 40
        self.assertEqual(sig_supertrend_rsi(r, 25), 0)

    def test_no_signal_without_breakout(self):
        r = _indicator_dict(30, rsi=np.full(30, 75.0))
        self.assertEqual(sig_supertrend_rsi(r, 25), 0)  # close == prior high/low

    def test_nan_guard(self):
        r = _indicator_dict(30, rsi=np.full(30, 75.0))
        r["C"][25] = 110.0
        r["RSI14"][25] = np.nan
        self.assertEqual(sig_supertrend_rsi(r, 25), 0)
        r["RSI14"][25] = 75.0
        r["ATR10"][25] = np.nan
        self.assertEqual(sig_supertrend_rsi(r, 25), 0)

    def test_falls_back_to_computed_arrays(self):
        r = _indicator_dict(30, rsi=np.full(30, 75.0))
        r["C"][25] = 110.0
        for key in ("ATR10", "RB_H20", "RB_L20"):
            del r[key]
        self.assertEqual(sig_supertrend_rsi(r, 25), 1)

    def test_signal_map_membership(self):
        self.assertEqual(set(SIGNAL_MAP), {"Supertrend RSI"})
        self.assertIs(SIGNAL_MAP["Supertrend RSI"], sig_supertrend_rsi)


def _mixed_bars(n=120):
    """Deterministic bars containing both long and short pattern bars.

    A rising leg produces RSI>60 breakouts. A quiet decline pushes RSI<40,
    and one wide-range bar closing near its low satisfies the below-band
    condition required for a short.
    """
    up_end = 4000.0 + 2.0 * 29  # last close of the rising leg
    close = np.zeros(n, dtype=float)
    for i in range(n):
        if i < 30:
            close[i] = 4000.0 + 2.0 * i - (3.0 if i % 5 == 0 else 0.0)
        elif i < 55:
            close[i] = up_end - 0.8 * (i - 30)
        elif i == 55:
            close[i] = up_end - 0.8 * 25 - 5.0
        else:
            close[i] = up_end - 0.8 * 25 - 5.0 - 0.5 * (i - 55)
    half_range = np.where(np.arange(n) < 30, 2.0, 0.1)
    high = close + half_range
    low = close - half_range
    high[55] = close[55] + 19.95
    low[55] = close[55] - 0.05
    return close, high, low


def _reference_shifted(close, high, low):
    """Inline pandas port of archive strategy_23_supertrend_rsi.

    ATR10 is a simple rolling mean of TR, RSI14 is Cutler's, the 20-bar
    rolling high/low are from the pattern bar, and the pattern is shifted by
    one bar (live entry happens at the next bar's open).
    """
    s = pd.Series(close)
    h = pd.Series(high)
    l = pd.Series(low)
    d = s.diff()
    g = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + g / loss.replace(0, np.nan))
    tr = pd.concat([h - l, (h - s.shift(1)).abs(), (l - s.shift(1)).abs()],
                   axis=1).max(axis=1)
    atr10 = tr.rolling(10).mean()
    band = (h + l) / 2 - 3 * atr10
    st = s > band
    bo = (s > h.rolling(20).max().shift(1)) & st & (rsi > 60)
    bd = (s < l.rolling(20).min().shift(1)) & ~st & (rsi < 40)
    pattern = pd.Series(0, index=s.index)
    pattern[bo] = 1
    pattern[bd] = -1
    return pattern.shift(1).fillna(0).astype(int).values


class TestSupertrendRsiMatchesArchiveReference(unittest.TestCase):
    def test_bar_by_bar_against_pandas_reference(self):
        close, high, low = _mixed_bars()
        r = build_indicator_dict(
            close - 1.0, high, low, close, np.ones(len(close)),
            np.arange(len(close)) * 3600,
        )
        ref = _reference_shifted(close, high, low)
        actual = np.array([sig_supertrend_rsi(r, i) for i in range(len(close))])
        np.testing.assert_array_equal(actual[:-1], ref[1:])
        self.assertIn(1, ref)
        self.assertIn(-1, ref)


if __name__ == "__main__":
    unittest.main()
