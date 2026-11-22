"""Signal function for the trading engine."""
import numpy as np

from engine.indicators import rolling_max, rolling_min, atr_simple


def sig_supertrend_rsi(r, i):
    """ATR(10) lower-band regime filter + RSI(14) 20-bar breakout.

    Long when the close is above the prior 20-bar high, above the ATR lower
    band and RSI > 60. Short when it is below the prior 20-bar low, below the
    band and RSI < 40. The signal fires on the pattern bar and the live bot
    enters at the next bar's open (use_delayed_signal / entry_delay_bars=1),
    matching the backtest's ``sig.shift(1)``.
    """
    if i < 21:
        return 0
    H = r['H']
    L = r['L']
    C = r['C']
    a10 = r.get('ATR10')
    if a10 is None:
        a10 = atr_simple(H, L, C, 10)
    rh20 = r.get('RB_H20')
    rl20 = r.get('RB_L20')
    if rh20 is None:
        rh20 = rolling_max(H, 20)
    if rl20 is None:
        rl20 = rolling_min(L, 20)
    if (np.isnan(a10[i]) or np.isnan(rh20[i - 1]) or np.isnan(rl20[i - 1])
            or np.isnan(r['RSI14'][i])):
        return 0
    mid = (H[i] + L[i]) / 2
    lo = mid - 3 * a10[i]
    uptrend = C[i] > lo
    if C[i] > rh20[i - 1] and uptrend and r['RSI14'][i] > 60:
        return 1
    if C[i] < rl20[i - 1] and not uptrend and r['RSI14'][i] < 40:
        return -1
    return 0


SIGNAL_MAP = {
    'Supertrend RSI': sig_supertrend_rsi,
}
