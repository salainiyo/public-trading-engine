"""Core data models for the trading engine."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Trade:
    id: str
    strategy: str
    direction: int
    entry: float
    sl: float
    tp: float
    lots: float
    open_time: str
    close_time: Optional[str] = None
    pnl: Optional[float] = None
    status: str = "open"
    position_id: Optional[int] = None
    _hold_bars: int = 0
    _trail_px: float = 0.0
    _sl_dist: float = 0.0
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None


@dataclass
class ChallengeState:
    strategy: str
    challenge_num: int
    equity: float
    balance: float
    phase_start_bal: float
    dd_floor: float
    day_equity: float
    phase: int = 1
    day_start: int = 0
    trading_days: int = 0
    _trading_day_set: set = field(default_factory=set)
    _traded_day_set: set = field(default_factory=set)
    _eod_ref: float = 0.0
    _day_pnl: dict = field(default_factory=dict)
    total_payouts: float = 0.0
    blown: bool = False
    funded_date: Optional[str] = None
    equity_peak: float = 0.0
    recycled: bool = False
    open_trades: list = field(default_factory=list)
    trade_history: list = field(default_factory=list)
    last_payout_day: int = 0
    ch_trade_pnl: float = 0.0
    _day_worst_upl: float = 0.0
    trail_locked: bool = False
    _payouts_run: int = 0
    _last_payout_valid_idx: int = 0
    _day_cap: dict = field(default_factory=dict)

    @property
    def open_trade(self):
        """Backward-compatible: returns the first open trade, or None."""
        return self.open_trades[0] if self.open_trades else None

    @open_trade.setter
    def open_trade(self, value):
        """Backward-compatible setter: sets/replaces the single open trade."""
        if value is None:
            self.open_trades = []
        elif not self.open_trades:
            self.open_trades = [value]
        else:
            self.open_trades[0] = value


@dataclass
class StrategyState:
    name: str
    challenges: list = field(default_factory=list)
    total_spent: float = 0.0
    total_earned: float = 0.0
    trade_pnl: float = 0.0
    treasury: float = 0.0
    monthly_stats: dict = field(default_factory=dict)
    daily_pnl: dict = field(default_factory=dict)
    last_eval_time: Optional[int] = None
    last_exec_time: Optional[int] = None
    account_id: Optional[str] = None
    blown_pnl: float = 0.0

    @property
    def current_challenge(self):
        active = [c for c in self.challenges if not c.blown]
        return active[0] if active else (self.challenges[-1] if self.challenges else None)

    @property
    def active_challenges(self):
        return [c for c in self.challenges if not c.blown]
