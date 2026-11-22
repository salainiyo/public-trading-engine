"""Market constants and configuration for the trading engine."""
from datetime import datetime


# ── Market constants ──────────────────────────────────────────────────────────
MIN_TRADING_DAYS = 3
QUALIFYING_DAY_PCT = 0.005
PROFIT_SPLIT = 0.80
PAYOUT_FREQ_DAYS = 14
PAYOUT_VALID_DAYS = 4
FIRST_PAYOUT_CAP_N = 2
FIRST_PAYOUT_CAP = 200.0
DAILY_PROFIT_CAP = 3000.0
COMMISSION = 3.0
CONTRACT_SIZE = 100

MIN_LOT = 0.01
MAX_LOT = 50.0
LOT_STEP = 0.01
MAX_HOLD_BARS = 480

MARKET_CLOSED_FRI_UTC = (21, 5)
MARKET_OPEN_SUN_UTC = (22, 10)
FRIDAY_FLAT_BAR_UTC = (19, 0)

STALE_SPOT_SECONDS = 90
STALE_RECONNECT_COOLDOWN = 300

LEVERAGE_EVAL = 100
LEVERAGE_FUNDED = 50
MAX_CAP = 80

SYMBOL = "XAUUSD"

FUNDED_RISK_PER_TRADE = 0.01

POLL_INTERVAL = 5
RECONCILE_POLL_EVERY = 60
ORDER_HISTORY_POLL_EVERY = 60
ORDER_HISTORY_FAST_RETRY = 10
OPEN_ORDER_TIMEOUT = 45
CLOSE_ORDER_TIMEOUT = 45
SL_CHECK_INTERVAL = 1

LOOKBACK_DAYS = {5: 7, 15: 14, 30: 30, 60: 60, 240: 120}


# ── Firm rulebooks (from scripts/supertrend_rsi_propfirm.py) ─────────────────
# Common execution model for all three firms: Supertrend ATR-band + RSI
# breakout on H1, signal at bar t-1, MARKET fill at bar t open, fixed $25
# stop, RR 3, 1% risk, max 2 independent positions, no add-on/trail/news
# filter. A trading day is a calendar day with an executed entry. Payouts
# retain the unpaid portion of equity and are paid every 14 days.
GOAT_FIRM_5K_PROFILE = {
    'ib': 5000.0,
    'md': 0.10,
    'dl': 0.04,
    'p1': 0.08,
    'p2': 0.06,
    'cc': 17.0,
    'sm': 'atr',
    'rpt': 0.01,
    'min_valid_days': 3,
    'valid_day_mode': 'entry',
    'profit_split': 0.80,
    'first_payout_cap': float('inf'),
    'daily_profit_cap': float('inf'),
    'daily_dd_mode': 'script',
    'payout_freq_days': 14,
    'payout_mode': 'retain',
    'payout_min_valid_days': 0,
    'swap_rate': -0.05,
}

AQUA_FIRM_5K_PROFILE = {
    'ib': 5000.0,
    'md': 0.08,
    'dl': 0.05,
    'p1': 0.08,
    'p2': 0.05,
    'cc': 39.0,
    'sm': 'atr',
    'rpt': 0.01,
    'min_valid_days': 3,
    'valid_day_mode': 'entry',
    'profit_split': 0.90,
    'first_payout_cap': float('inf'),
    'daily_profit_cap': float('inf'),
    'daily_dd_mode': 'script',
    'payout_freq_days': 14,
    'payout_mode': 'retain',
    'payout_min_valid_days': 0,
    'swap_rate': -0.05,
}

FP_FIRM_5K_PROFILE = {
    'ib': 5000.0,
    'md': 0.10,
    'dl': 0.05,
    'p1': 0.08,
    'p2': 0.05,
    'cc': 36.0,
    'sm': 'atr',
    'rpt': 0.01,
    'min_valid_days': 3,
    'valid_day_mode': 'entry',
    'profit_split': 0.80,
    'first_payout_cap': float('inf'),
    'daily_profit_cap': float('inf'),
    'daily_dd_mode': 'script',
    'payout_freq_days': 14,
    'payout_mode': 'retain',
    'payout_min_valid_days': 0,
    'swap_rate': -0.05,
}

FARMS = [
    {
        'name': 'Goat 5K',
        'tag': 'GOAT5',
        'desc': 'Supertrend ATR-band + RSI breakout (1H), MARKET entry, '
                'fixed $25 stop, RR 3, max 2, 1% risk',
        'profile': GOAT_FIRM_5K_PROFILE,
        'broker_key': 'herofx_goat',
        'broker': 'goat100k',
        'mode': 'rec',
        'max_acc': 80,
        'eval_strat': 'Supertrend RSI',
        'eval_tf': 60,
        'eval_rr': 3.0,
        'stop_mode': 'fixed',
        'stop_value': 25.0,
        'entry_slip': 0.0,
        'funded_strat': 'Supertrend RSI',
        'funded_tf': 60,
        'funded_rr': 3.0,
        'eval_rpt': 0.01,
        'funded_rpt': 0.01,
        'use_delayed_signal': True,
        'entry_delay_bars': 1,
        'use_trail': False,
        'use_limit_entry': False,
        'max_hold_bars': 0,
        'max_concurrent': 2,
        'addon_min_r': None,
        'sizing_mode': 'script',
        'early_close': True,
    },
    {
        'name': 'Aqua Standard 5K',
        'tag': 'AQUA5',
        'desc': 'Supertrend ATR-band + RSI breakout (1H), MARKET entry, '
                'fixed $25 stop, RR 3, max 2, 1% risk',
        'profile': AQUA_FIRM_5K_PROFILE,
        'broker_key': 'herofx_goat',
        'broker': 'goat100k',
        'mode': 'rec',
        'max_acc': 80,
        'eval_strat': 'Supertrend RSI',
        'eval_tf': 60,
        'eval_rr': 3.0,
        'stop_mode': 'fixed',
        'stop_value': 25.0,
        'entry_slip': 0.0,
        'funded_strat': 'Supertrend RSI',
        'funded_tf': 60,
        'funded_rr': 3.0,
        'eval_rpt': 0.01,
        'funded_rpt': 0.01,
        'use_delayed_signal': True,
        'entry_delay_bars': 1,
        'use_trail': False,
        'use_limit_entry': False,
        'max_hold_bars': 0,
        'max_concurrent': 2,
        'addon_min_r': None,
        'sizing_mode': 'script',
        'early_close': True,
    },
    {
        'name': 'FundingPips 2-Step',
        'tag': 'FPST5',
        'desc': 'Supertrend ATR-band + RSI breakout (1H), MARKET entry, '
                'fixed $25 stop, RR 3, max 2, 1% risk',
        'profile': FP_FIRM_5K_PROFILE,
        'broker_key': 'herofx_goat',
        'broker': 'goat100k',
        'mode': 'rec',
        'max_acc': 80,
        'eval_strat': 'Supertrend RSI',
        'eval_tf': 60,
        'eval_rr': 3.0,
        'stop_mode': 'fixed',
        'stop_value': 25.0,
        'entry_slip': 0.0,
        'funded_strat': 'Supertrend RSI',
        'funded_tf': 60,
        'funded_rr': 3.0,
        'eval_rpt': 0.01,
        'funded_rpt': 0.01,
        'use_delayed_signal': True,
        'entry_delay_bars': 1,
        'use_trail': False,
        'use_limit_entry': False,
        'max_hold_bars': 0,
        'max_concurrent': 2,
        'addon_min_r': None,
        'sizing_mode': 'script',
        'early_close': True,
    },
]


# ── Parity checker (live vs standalone backtest) ────────────────────────────
PARITY_INTERVAL_SECONDS = 6 * 3600
PARITY_LOOKBACK_DAYS = 45
PARITY_MATCH_BAR_SECONDS = 3600
PARITY_R_TOL = 0.15


def xau_market_open(dt_utc: datetime) -> bool:
    wd = dt_utc.weekday()
    if wd == 5:
        return False
    if wd == 4 and (dt_utc.hour, dt_utc.minute) >= MARKET_CLOSED_FRI_UTC:
        return False
    if wd == 6 and (dt_utc.hour, dt_utc.minute) < MARKET_OPEN_SUN_UTC:
        return False
    return True
