"""Technical indicators used by the Supertrend RSI strategy."""
import numpy as np
import pandas as pd


def rsi(arr, n=14):
    d = pd.Series(arr).diff()
    g = d.clip(lower=0).rolling(n, min_periods=n).mean()
    l = (-d.clip(upper=0)).rolling(n, min_periods=n).mean()
    return (100 - 100 / (1 + g / l.replace(0, np.nan))).values


def atr(h, l, c, n=14):
    tr = pd.concat([
        pd.Series(h - l),
        (pd.Series(h) - pd.Series(c).shift(1)).abs(),
        (pd.Series(l) - pd.Series(c).shift(1)).abs()
    ], axis=1).max(axis=1).values
    out = np.full(len(tr), np.nan)
    if len(tr) >= n:
        out[n - 1] = tr[:n].mean()
        for j in range(n, len(tr)):
            out[j] = (out[j - 1] * (n - 1) + tr[j]) / n
    return out


def atr_simple(h, l, c, n=10):
    tr = pd.concat([
        pd.Series(h - l),
        (pd.Series(h) - pd.Series(c).shift(1)).abs(),
        (pd.Series(l) - pd.Series(c).shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean().values


def rolling_max(arr, n):
    return pd.Series(arr).rolling(n, min_periods=n).max().values


def rolling_min(arr, n):
    return pd.Series(arr).rolling(n, min_periods=n).min().values


def build_indicator_dict(opens, highs, lows, closes, volumes, times):
    """Build the indicator arrays needed by the signal and the risk engine."""
    r = {
        'O': np.array(opens, dtype=np.float64),
        'H': np.array(highs, dtype=np.float64),
        'L': np.array(lows, dtype=np.float64),
        'C': np.array(closes, dtype=np.float64),
        'V': np.array(volumes, dtype=np.float64),
        'time': np.array(times, dtype=np.int64),
    }
    r['RSI14'] = rsi(r['C'], 14)
    r['ATR14'] = atr(r['H'], r['L'], r['C'], 14)
    r['ATR10'] = atr_simple(r['H'], r['L'], r['C'], 10)
    r['RB_H20'] = rolling_max(r['H'], 20)
    r['RB_L20'] = rolling_min(r['L'], 20)
    return r
