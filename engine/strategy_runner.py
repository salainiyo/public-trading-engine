"""Strategy runner module for the trading engine."""
import logging
import math
import re
import copy
import numpy as np
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Optional

from engine.models import Trade, ChallengeState, StrategyState
from engine.market import (
    MIN_TRADING_DAYS, QUALIFYING_DAY_PCT, COMMISSION, CONTRACT_SIZE,
    MIN_LOT, MAX_LOT, LOT_STEP, FUNDED_RISK_PER_TRADE,
    LEVERAGE_EVAL, LEVERAGE_FUNDED, DAILY_PROFIT_CAP,
    FIRST_PAYOUT_CAP_N, FIRST_PAYOUT_CAP, PAYOUT_FREQ_DAYS, PAYOUT_VALID_DAYS,
    PROFIT_SPLIT, MAX_HOLD_BARS, FRIDAY_FLAT_BAR_UTC
)

logger = logging.getLogger(__name__)


class StrategyRunner:
    def __init__(self, farm_config: dict, broadcast_fn=None, bot=None):
        self.config = farm_config
        self.name = farm_config['name']
        self.farm_type = farm_config['mode']
        self.profile = copy.deepcopy(farm_config['profile'])
        self.config['profile'] = self.profile
        self.broadcast = broadcast_fn or (lambda: None)
        self.bot = bot
        self._broker = None  # Set by Bot after broker connect
        self._broker_key = None  # Set by Bot after broker connect
        self.state = StrategyState(name=self.name)
        self._add_new_challenge()
        self._last_month = datetime.now(timezone.utc).month
        self._last_bar_atr = None

    def _max_concurrent(self):
        return self.config.get("max_concurrent", 1)

    def is_weekend_flat_window(self, dt, live: bool = False) -> bool:
        """True when an FP flat-rule runner must be flat / block entries.

        Friday's 19:00 bar (20:00 UTC close) is the last pre-weekend bar; the
        bar-time cutoff is 19:00 so the bar-close path flattens before the
        21:05 market close. Wall-clock checks use 20:00 UTC. Saturday and
        Sunday are blocked entirely; Monday 00:00 UTC resumes.
        """
        if not self.config.get("friday_flat"):
            return False
        wd = dt.weekday()
        if wd == 4:
            cutoff = (20, 0) if live else FRIDAY_FLAT_BAR_UTC
            return (dt.hour, dt.minute) >= cutoff
        return wd in (5, 6)

    def _best_open_r(self, ch, price):
        """Best open-trade R at `price` (script `best_profit_R`, a MAX despite
        the archive comment saying "worst"). R uses the initial SL distance."""
        best = -float("inf")
        for ot in ch.open_trades:
            if getattr(ot, "status", None) == "pending":
                continue
            sl_dist = getattr(ot, "_sl_dist", 0.0) or abs(ot.entry - ot.sl)
            if sl_dist <= 0:
                continue
            r = (price - ot.entry) * ot.direction / sl_dist
            best = max(best, r)
        return best

    def _load_trade_history(self, trade_store):
        rows = trade_store.load_trades(self.name, limit=500)
        trade_store.reseed_counter_global()
        for ch in self.state.challenges:
            ch.trade_history = []
            ch._trading_day_set = set()
            ch.open_trades = []
        if rows:
            for ch in self.state.challenges:
                if ch.phase <= 1:
                    ch.ch_trade_pnl = 0.0
            if any(c.phase <= 1 for c in self.state.challenges):
                self.state.trade_pnl = 0.0
                for ym in self.state.monthly_stats:
                    self.state.monthly_stats[ym]["trade_pnl"] = 0.0
                self.state.daily_pnl = {}
        for t in reversed(rows):
            m = re.search(r'-ch(\d+)-', t.get("id", ""))
            if not m:
                ch = self.state.current_challenge
                if ch is None:
                    continue
                ch_num = ch.challenge_num
            else:
                ch_num = int(m.group(1))
            ch = next((c for c in self.state.challenges if c.challenge_num == ch_num), None)
            if ch is None:
                continue
            ch.trade_history.append(t)
            pnl = t.get("pnl", 0.0) or 0.0
            if ch.phase <= 1:
                # Eval-phase PnL re-derived from history (the challenge compounds)
                ch.ch_trade_pnl += pnl
                ch.equity = ch.phase_start_bal + ch.ch_trade_pnl
            # ALL rows (any phase, incl. funded/blown) contribute to strategy-level
            # stats so a restart doesn't silently drop funded-phase figures
            self.state.trade_pnl += pnl
            ct = t.get("close_time")
            if ct:
                try:
                    dt = datetime.fromisoformat(ct)
                    ym = dt.strftime("%Y-%m")
                    day_str = dt.strftime("%Y-%m-%d")
                    if ym not in self.state.monthly_stats:
                        self.state.monthly_stats[ym] = {"spent": 0, "earned": 0, "trade_pnl": 0, "withdrawn": 0}
                    self.state.monthly_stats[ym]["trade_pnl"] += pnl
                    self.state.daily_pnl[day_str] = self.state.daily_pnl.get(day_str, 0.0) + pnl
                except Exception:
                    pass
        # blown_pnl (display): cumulative realized PnL of blown challenges
        blown_total = 0.0
        for ch in self.state.challenges:
            if ch.blown:
                blown_total += sum((t.get("pnl", 0.0) or 0.0) for t in ch.trade_history)
        self.state.blown_pnl = blown_total
        for ch in self.state.challenges:
            self._refresh_qualifying_days(ch)
        self._phase_integrity_check()

    def _phase_integrity_check(self):
        instant = self.profile.get("mode") == "instant"
        for ch in self.state.challenges:
            if ch.blown:
                continue
            if instant:
                # Instant venues are funded from the start (no eval history to
                # verify): force phase 3 and repair any pre-instant state
                if ch.phase != 3:
                    logger.warning(f"[{self.name}] Ch#{ch.challenge_num} instant venue "
                                   f"restored to funded (phase {ch.phase} -> 3); "
                                   f"resetting challenge accounting to fresh")
                    ch.phase = 3
                    ch.phase_start_bal = self.profile["ib"]
                    ch.dd_floor = self.profile["ib"] * (1 - self.profile.get("trail_dd", self.profile["md"]))
                    ch.equity_peak = max(ch.equity, self.profile["ib"])
                    ch.equity = self.profile["ib"]
                    ch.day_equity = self.profile["ib"]
                    ch.ch_trade_pnl = 0.0
                if ch.last_payout_day is None:
                    ch.last_payout_day = 0  # never paid yet — payout clock starts at funding
                continue
            if ch.phase != 3:
                continue
            if not ch.funded_date:
                do_downgrade = True
                phantom_bonus = sum(t.get("pnl", 0.0) or 0.0 for t in ch.trade_history)
            else:
                try:
                    funded_dt = datetime.fromisoformat(ch.funded_date)
                except Exception:
                    funded_dt = None
                pre_funding_pnl = 0.0
                for t in ch.trade_history:
                    ct = t.get("close_time")
                    if ct and funded_dt:
                        try:
                            if datetime.fromisoformat(ct) < funded_dt:
                                pre_funding_pnl += t.get("pnl", 0.0) or 0.0
                        except Exception:
                            pre_funding_pnl += t.get("pnl", 0.0) or 0.0
                    else:
                        pre_funding_pnl += t.get("pnl", 0.0) or 0.0
                needed = ch.phase_start_bal * (self.profile["p1"] + self.profile["p2"])
                do_downgrade = pre_funding_pnl < needed
                phantom_bonus = pre_funding_pnl if do_downgrade else 0.0
            if do_downgrade:
                logger.warning(f"[{self.name}] Ch#{ch.challenge_num} Phase 3 downgraded to Phase 2")
                ib = self.profile["ib"]
                ch.phase = 2
                ch.funded_date = None
                ch.last_payout_day = None
                ch.phase_start_bal = ib
                ch.equity = ib
                ch.day_equity = ib
                ch.dd_floor = ib * (1 - self.profile["md"])
                ch.ch_trade_pnl = 0.0
                ch.trading_days = 0
                ch._trading_day_set.clear()
                ch._day_pnl.clear()
                self.state.trade_pnl = 0.0
                self.state.daily_pnl = {}
                for ym in self.state.monthly_stats:
                    self.state.monthly_stats[ym]["trade_pnl"] = 0.0
                if phantom_bonus and phantom_bonus > 0:
                    self.state.total_earned -= phantom_bonus
                    ym = datetime.now(timezone.utc).strftime("%Y-%m")
                    self.state.monthly_stats.setdefault(ym, {"spent": 0, "earned": 0, "trade_pnl": 0, "withdrawn": 0})
                    self.state.monthly_stats[ym]["earned"] -= phantom_bonus

    def _add_new_challenge(self):
        n = len(self.state.challenges) + 1
        ib = self.profile["ib"]
        instant = self.profile.get("mode") == "instant"
        floor_frac = self.profile.get("trail_dd", self.profile["md"]) if instant else self.profile["md"]
        ch = ChallengeState(
            strategy=self.name, challenge_num=n,
            equity=ib, balance=ib, phase_start_bal=ib,
            dd_floor=ib * (1 - floor_frac), day_equity=ib,
            phase=3 if instant else 1,
            equity_peak=ib,
        )
        self.state.challenges.append(ch)
        cost = self.profile["cc"]
        self.state.total_spent += cost
        ym = datetime.now(timezone.utc).strftime("%Y-%m")
        if ym not in self.state.monthly_stats:
            self.state.monthly_stats[ym] = {"spent": 0, "earned": 0, "trade_pnl": 0, "withdrawn": 0}
        self.state.monthly_stats[ym]["spent"] = self.state.monthly_stats[ym].get("spent", 0) + cost

    def _add_closed_trade_day(self, ch, now_day, pnl):
        ch._day_pnl[now_day] = ch._day_pnl.get(now_day, 0.0) + pnl
        if self.profile.get("valid_day_mode") in ("traded", "entry"):
            # A valid day is any day with trade activity. The entry-mode set is
            # also populated at fill time in _compute_entry.
            ch._traded_day_set.add(now_day)
            ch.trading_days = len(ch._traded_day_set)
            return
        threshold = self.profile["ib"] * self.profile.get("valid_day_pct", QUALIFYING_DAY_PCT)
        if ch._day_pnl[now_day] >= threshold:
            ch._trading_day_set.add(now_day)
        ch.trading_days = len(ch._trading_day_set)

    def _apply_daily_profit_cap(self, ch, now_day):
        if ch.phase != 3 or ch.blown:
            return
        daily_cap = self.profile.get("daily_profit_cap", DAILY_PROFIT_CAP)
        if daily_cap == float('inf') or daily_cap <= 0:
            return
        realized = ch._day_pnl.get(now_day, 0.0)
        target = max(realized - daily_cap, 0.0)
        prev = ch._day_cap.get(now_day, 0.0)
        adj = target - prev
        if abs(adj) < 0.005:
            return
        ch._day_cap[now_day] = target
        ch.equity -= adj
        ch.ch_trade_pnl -= adj
        self.state.trade_pnl -= adj
        day_dt = datetime.fromtimestamp(now_day * 86400, tz=timezone.utc)
        day_str = day_dt.strftime("%Y-%m-%d")
        self.state.daily_pnl[day_str] = self.state.daily_pnl.get(day_str, 0.0) - adj
        ym = day_dt.strftime("%Y-%m")
        ms = self.state.monthly_stats.setdefault(ym, {"spent": 0, "earned": 0, "trade_pnl": 0, "withdrawn": 0})
        ms["trade_pnl"] = ms.get("trade_pnl", 0) - adj

    def _refresh_qualifying_days(self, ch):
        day_pnl = {}
        for t in ch.trade_history:
            ct = t.get("close_time")
            pnl = t.get("pnl", 0.0) or 0.0
            if not ct:
                continue
            try:
                day = int(datetime.fromisoformat(ct).timestamp() / 86400)
            except Exception:
                continue
            day_pnl[day] = day_pnl.get(day, 0.0) + pnl
        ch._day_pnl = day_pnl
        if self.profile.get("valid_day_mode") in ("traded", "entry"):
            # Every day with a trade counts; keep the persisted set for entry
            # days whose fills predate the persisted trade history.
            ch._traded_day_set |= set(day_pnl)
            ch.trading_days = len(ch._traded_day_set)
            return
        threshold = self.profile["ib"] * self.profile.get("valid_day_pct", QUALIFYING_DAY_PCT)
        ch._trading_day_set = set(d for d, p in day_pnl.items() if p >= threshold)
        ch.trading_days = len(ch._trading_day_set)

    def on_bar_close(self, bar, atr, period_minutes):
        now_ts = bar["time"]
        now_day = int(now_ts / 86400)
        now_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
        for ch in list(self.state.active_challenges):
            self._process_challenge_bar(ch, bar, atr, period_minutes, now_day, now_dt)
        # Auto-rebuy: if no active challenges left, create a new one
        if not self.state.active_challenges:
            self._add_new_challenge()
        self._last_bar_atr = atr

    def _process_challenge_bar(self, ch, bar, atr, period_minutes, now_day, now_dt):
        if ch.blown:
            return
        tag = self.config.get('tag', self.name)
        if now_day != ch.day_start:
            ch.day_start = now_day
            if self.profile.get("daily_dd_mode") == "script" and ch._eod_ref > 0:
                # Script parity: day_ref = max(balance, equity) at the prior
                # day's last bar (prev_ref in archive/lastly). _eod_ref is
                # written at the end of every bar close.
                ch.day_equity = max(ch._eod_ref, ch.equity)
            else:
                ch.day_equity = ch.equity
            ch._day_worst_upl = 0.0

        # FundingPips weekend rule (venue rule, not backtest parity): all
        # phases must be flat before the weekend — force-close any held
        # position and block new entries from Friday's 19:00 bar (20:00 UTC)
        # through Sunday. Fire at the close of Friday's 19:00 bar while the
        # session is still open, and again at the 20:00 bar close as a
        # backstop. fp_2step_5k.py deliberately holds through the weekend, so
        # this is a documented divergence.
        if self.is_weekend_flat_window(now_dt) and ch.open_trades:
            logger.info(f"[{tag}] Friday flat (venue rule) — force-closing "
                        f"{len(ch.open_trades)} open trade(s) before weekend")
            if self.bot:
                self.bot._cancel_pending_for_challenge(self, ch)
                for ot in list(ch.open_trades):
                    self.bot._close_trade_at_market(ot, self, ch)
                self.broadcast()
            return

        # Intraday DD check across ALL open trades
        # Filled trades only — the kernel's path starts at the fill bar
        # (unfilled pending limits have no path contribution).
        mlpt = self.profile.get("max_loss_per_trade")
        if ch.open_trades:
            bar_worst = None
            pertrade_breach = False
            for ot in ch.open_trades:
                if ot.status == "pending":
                    continue
                extreme = bar.get("low" if ot.direction == 1 else "high", bar.get("close", ot.entry))
                upl = (extreme - ot.entry) * ot.direction * ot.lots * CONTRACT_SIZE
                ch._day_worst_upl = min(ch._day_worst_upl, upl)
                # Kernel eq_worst = balance + wR×risk uses THIS bar's worst
                # (path_worst[pp]), not the day-cumulative min — with the
                # instant floor ratcheting between bars they can diverge.
                bar_worst = upl if bar_worst is None else min(bar_worst, upl)
                # Per-trade loss cap as a BLOW rule (kernel numba:1917):
                #   wR × risk ≤ −mlpt × trade_start_balance → account blown.
                # ch.equity == entry balance while the trade is open (bookings
                # happen only at close; max_concurrent=1) — comment the
                # invariant when touching this.
                if mlpt and ot.status == "open" and upl <= -mlpt * ch.equity:
                    pertrade_breach = True
            if bar_worst is not None:
                if self.profile.get('daily_dd_mode') == 'fixed':
                    daily_floor = ch.day_equity - self.profile["dl"] * self.profile["ib"]
                else:
                    daily_floor = ch.day_equity * (1 - self.profile["dl"])
                proj = ch.equity + bar_worst
                breach = None
                if proj <= ch.dd_floor or proj <= daily_floor:
                    breach = (f"intraday DD: proj={proj:.2f} dd_floor={ch.dd_floor:.2f} "
                              f"daily={daily_floor:.2f}")
                elif pertrade_breach:
                    breach = (f"pertrade mlpt: worst_upl={bar_worst:.2f} <= "
                              f"-{mlpt:.4f} × eq={ch.equity:.2f}")
                if breach:
                    ch.blown = True
                    logger.warning(f"[{tag}] Ch#{ch.challenge_num} BLOWN ({breach}) "
                                   f"eq={ch.equity:.2f} worst_upl={ch._day_worst_upl:.2f} "
                                   f"open_trades={len(ch.open_trades)}")
                    if self.bot:
                        self.bot._cancel_pending_for_challenge(self, ch)
                        for ot in list(ch.open_trades):
                            self.bot._close_trade_at_market(ot, self, ch)
                        self.bot.treasury_store.save({self.name: self})
                        self.bot._recycle()
                    self.broadcast()
                    return

        # Instant mode: trailing max-loss ratchet (kernel numba:1906-1933).
        # ORDER MATTERS: the breach checks above ran against the PREVIOUS
        # floor; only now does the floor ratchet up on this bar's best
        # excursion (eq_best = balance + bR×risk, raise-only, no ib cap —
        # the breakeven lock below subsumes it).
        if self.profile.get("mode") == "instant" and ch.open_trades:
            ib = self.profile["ib"]
            trail_dd = self.profile.get("trail_dd", 0.05)
            best_upl = 0.0
            got = False
            for ot in ch.open_trades:
                if ot.status == "pending":
                    continue
                extreme = bar.get("high" if ot.direction == 1 else "low", bar.get("close", ot.entry))
                upl = (extreme - ot.entry) * ot.direction * ot.lots * CONTRACT_SIZE
                best_upl = upl if not got else max(best_upl, upl)
                got = True
            if got:
                eq_best = ch.equity + best_upl
                if not getattr(ch, "trail_locked", False):
                    new_floor = eq_best - trail_dd * ib
                    if new_floor > ch.dd_floor:
                        logger.info(f"[{tag}] Ch#{ch.challenge_num} trailing floor raised: "
                                    f"{ch.dd_floor:.2f} -> {new_floor:.2f} "
                                    f"(eq_best={eq_best:.2f} trail_dd={trail_dd})")
                        ch.dd_floor = new_floor
                    # Breakeven lock (kernel numba:1928-1932): once eq_best
                    # reaches ib + trail_dd×ib, freeze the floor at >= ib —
                    # the ratchet stops for the rest of the period (reset
                    # only by a payout or a blown challenge).
                    if eq_best >= ib + trail_dd * ib:
                        if ib > ch.dd_floor:
                            ch.dd_floor = ib
                        ch.trail_locked = True

        # Static DD check (no open trades)
        if not ch.open_trades:
            eff = ch.equity
            if self.profile.get('daily_dd_mode') == 'fixed':
                daily_floor = ch.day_equity - self.profile["dl"] * self.profile["ib"]
            else:
                daily_floor = ch.day_equity * (1 - self.profile["dl"])
            if eff <= ch.dd_floor or eff <= daily_floor:
                ch.blown = True
                tag = self.config.get('tag', self.name)
                logger.warning(f"[{tag}] Ch#{ch.challenge_num} BLOWN (static DD): "
                               f"eq={eff:.2f} dd_floor={ch.dd_floor:.2f} daily={daily_floor:.2f}")
                if self.bot:
                    self.bot._recycle()
                self.broadcast()
                return

        # Phase 3 payout (only when no open trades)
        if period_minutes == self.config["funded_tf"] and ch.phase == 3 and not ch.open_trades:
            if ch.last_payout_day is None:
                ch.last_payout_day = 0  # guard: never-paid instant challenge
            days_since = now_day - ch.last_payout_day
            ib = self.profile["ib"]
            effective_equity = ch.equity  # kernel `balance` (retained post-payout)
            _valid_days = (len(ch._traded_day_set)
                           if self.profile.get("valid_day_mode") in ("traded", "entry")
                           else len(ch._trading_day_set))
            valid_since = _valid_days - ch._last_payout_valid_idx
            min_valid = self.profile.get("payout_min_valid_days",
                                         self.profile.get("min_valid_days", PAYOUT_VALID_DAYS))
            payout_freq = self.profile.get("payout_freq_days", PAYOUT_FREQ_DAYS)
            if days_since >= payout_freq and effective_equity > ib and valid_since >= min_valid:
                # Consistency gate (instant venues with a rule): the best day's
                # profit since the last payout must be <= consistency × the total
                consistency_ok = True
                consistency = self.profile.get("consistency")
                if consistency and consistency < 1.0:
                    since = [(d, p) for d, p in ch._day_pnl.items() if d > ch.last_payout_day]
                    total_since = sum(p for _, p in since)
                    best_since = max((p for _, p in since), default=0.0)
                    consistency_ok = not (total_since > 0 and best_since > consistency * total_since)
                if consistency_ok:
                    profit_split = self.profile.get("profit_split", PROFIT_SPLIT)
                    payout = (effective_equity - ib) * profit_split
                    if ch._payouts_run < FIRST_PAYOUT_CAP_N:
                        first_payout_cap = self.profile.get("first_payout_cap", FIRST_PAYOUT_CAP)
                        if payout > first_payout_cap:
                            logger.info(f"[{tag}] Ch#{ch.challenge_num} payout capped: "
                                        f"${payout:.2f} -> ${first_payout_cap:.2f} (first-payout cap, "
                                        f"payout #{ch._payouts_run + 1})")
                        payout = min(payout, first_payout_cap)
                    ch._payouts_run += 1
                    ch.total_payouts += payout
                    if self.bot:
                        self.bot.process_payout(self.name, payout)
                    self.state.total_earned += payout
                    logger.info(f"[{tag}] Ch#{ch.challenge_num} PAYOUT ${payout:.2f} "
                                f"(days_since={days_since} freq={payout_freq} valid={valid_since}/"
                                f"{min_valid} eff={effective_equity:.2f} ib={ib:.2f} "
                                f"consistency_gate={'passed' if consistency_ok else 'n/a'})")
                    if self.profile.get("payout_mode") == "reset":
                        # Script parity (archive/lastly): after a payout the
                        # account restarts at the initial balance (the paid
                        # profit leaves; nothing is retained).
                        ch.equity = ib
                        ch.ch_trade_pnl = 0.0
                        ch.day_equity = ib
                        ch._eod_ref = ib
                        ch.equity_peak = ib
                        ch.dd_floor = ib * (1 - self.profile["md"])
                    else:
                        # Backtest parity (kernel numba:1769/1997): balance -= payout
                        # — the account KEEPS ib + profit×(1−split); only the paid
                        # part leaves. day_equity / _day_cap / cumulative trade_pnl
                        # are untouched (kernel's day_ref, day_realized and total
                        # accounting are continuous across a payout).
                        ch.equity -= payout
                        ch.ch_trade_pnl = 0.0
                        # Floor reset after payout: instant mode resets the trailing peak too
                        if self.profile.get("mode") == "instant":
                            ch.equity_peak = ib
                            ch.dd_floor = ib * (1 - self.profile.get("trail_dd", 0.05))
                            ch.trail_locked = False
                        else:
                            ch.dd_floor = ib * (1 - self.profile["md"])
                    ch.last_payout_day = now_day
                    ch._last_payout_valid_idx = _valid_days
                    ym = now_dt.strftime("%Y-%m")
                    self.state.monthly_stats.setdefault(ym, {"spent": 0, "earned": 0, "trade_pnl": 0, "withdrawn": 0})
                    self.state.monthly_stats[ym]["earned"] += payout
                    self.state.monthly_stats[ym]["withdrawn"] += payout
                    self.broadcast()
                else:
                    consistency = self.profile.get("consistency")
                    if consistency and consistency < 1.0:
                        since = [(d, p) for d, p in ch._day_pnl.items() if d > ch.last_payout_day]
                        best = max((p for _, p in since), default=0.0)
                        total = sum(p for _, p in since)
                        logger.info(f"[{tag}] Ch#{ch.challenge_num} payout held by consistency "
                                    f"gate: best day ${best:.2f} > {consistency} × total ${total:.2f} "
                                    f"— waiting for more days")

        # Early close check (Phase 1/2 only, check ALL open trades)
        closed_any = False
        if (ch.phase in (1, 2) and ch.open_trades
                and self.config.get("early_close", True)):
            closed_any = self._check_early_close(ch, bar, now_day, now_dt)

        # Trailing stop (check ALL open trades)
        if not closed_any and ch.open_trades and self.config.get("use_trail"):
            trade_tf = self.config["eval_tf"] if ch.phase < 3 else self.config["funded_tf"]
            if period_minutes == trade_tf:
                for ot in list(ch.open_trades):
                    if self._update_trailing_stop_single(ch, ot, bar, atr):
                        closed_any = True

        # Hold bars / max-hold (check ALL open trades)
        if ch.open_trades:
            trade_tf = self.config["eval_tf"] if ch.phase < 3 else self.config["funded_tf"]
            if period_minutes == trade_tf:
                max_hold = self.config.get("max_hold_bars", MAX_HOLD_BARS)
                if max_hold and max_hold > 0:
                    for ot in list(ch.open_trades):
                        if getattr(ot, "position_id", None) is None:
                            continue  # pending limit: the kernel's hold counts fill bars only
                        ot._hold_bars += 1
                        if ot._hold_bars >= max_hold:
                            if self.bot:
                                self.bot._close_trade_at_market(ot, self, ch)

        # Phase advance (only when no open trades)
        if not ch.blown and not ch.open_trades:
            if self._advance_phase(ch, now_day, now_dt):
                if self.bot:
                    self.bot.treasury_store.save({self.name: self})
                self.broadcast()

        # End-of-bar reference for the script daily-DD mode: the prior day's
        # last bar's max(balance, equity incl. unrealized) (prev_ref).
        if not ch.blown:
            eq_incl = ch.equity
            for ot in ch.open_trades:
                if getattr(ot, "status", None) == "pending":
                    continue
                px = bar.get("close", ot.entry)
                eq_incl += (px - ot.entry) * ot.direction * ot.lots * CONTRACT_SIZE
            ch._eod_ref = max(ch.equity, eq_incl)
            if (self.profile.get("valid_day_mode") in ("traded", "entry")
                    and any(getattr(ot, "status", None) != "pending"
                            for ot in ch.open_trades)):
                ch._traded_day_set.add(now_day)
                ch.trading_days = len(ch._traded_day_set)

    def _flatten_phase_exposure(self, ch, filled, pending):
        """Cancel pending entries and market-close every filled leg.

        The phase target is an account-level condition. With max_concurrent > 1,
        closing only the leg that happens to cross it leaves the other leg
        exposed and can give the target back — or let the next phase start
        with a prior-phase position still open.
        """
        if not filled and not pending:
            return False
        if self.bot:
            if pending:
                self.bot._cancel_pending_for_challenge(self, ch)
            for ot in filled:
                self.bot._close_trade_at_market(ot, self, ch)
        return True

    def _check_early_close(self, ch, bar, now_day, now_dt, bid=None, ask=None):
        """Backtest parity — kernel early close (numba:1686-1700):
        fire when the phase target is reached, counting TODAY as a valid day
        when the partial books it (`extra`), gate on
        valid_days + extra >= min_valid_days.

        `bid`/`ask` (1s watchdog path): project at the LIVE price for
        intra-bar touch semantics (kernel uses the bar's best — `eq_best`);
        otherwise the bar close is used (bar-close backstop).

        With more than one filled leg the target is evaluated on the SUM of
        their projected PnL and every leg is flattened. The kernel is
        single-position, so this is the live multi-position extension of its
        account-level target check.
        """
        if not self.config.get("early_close", True):
            return False
        if ch.blown or ch.phase not in (1, 2) or not ch.open_trades:
            return False
        ib = self.profile["ib"]
        eff = ch.equity  # kernel `balance`
        target = ch.phase_start_bal * (1 + (self.profile["p1"] if ch.phase == 1 else self.profile["p2"]))
        filled = [ot for ot in ch.open_trades if ot.position_id is not None]
        pending = [ot for ot in ch.open_trades if ot.position_id is None]
        if not filled and not pending:
            return False
        # The 1s watchdog calls this every second. Once every filled leg has a
        # close order in flight, wait for settlement instead of logging the
        # same target hit (and re-entering the close guard) on every tick.
        already_closing = bool(
            filled and self.bot and all(
                f"{ot.id}_close" in getattr(self.bot, "_pending_orders", {})
                for ot in filled))
        if already_closing and not pending:
            return True
        if eff >= target:
            # The target is already booked (e.g. a TP closed one leg) but the
            # phase still has exposure. Lock it: cancel entries and flatten.
            tag = self.config.get('tag', self.name)
            logger.info(f"[{tag}] Ch#{ch.challenge_num} early close: target already "
                        f"met (eq={eff:.2f} target={target:.2f}) — flattening "
                        f"{len(filled)} filled / {len(pending)} pending")
            return self._flatten_phase_exposure(ch, filled, pending)
        if bar is None and (bid is None or ask is None):
            return False
        total_net = 0.0
        for ot in filled:
            if bid is not None and ask is not None:
                close_price = bid if ot.direction == 1 else ask
            elif bar is not None:
                close_price = bar.get("close")
            else:
                close_price = None
            if close_price is None:
                return False
            gross = (close_price - ot.entry) * ot.direction * ot.lots * CONTRACT_SIZE
            total_net += gross - COMMISSION * ot.lots
        projected = eff + total_net
        if projected < target:
            return False
        # Days gate: existing valid days + TODAY (kernel `extra`). The kernel
        # computes `extra` unconditionally (it may double-count a today that
        # is already in valid_days) — replicated exactly for parity. The
        # partial is capped at the remaining gap, matching the kernel's
        # partial_R = min(needed_R, bR).
        threshold = ib * self.profile.get("valid_day_pct", QUALIFYING_DAY_PCT)
        partial = min(max(target - eff, 0.0), max(total_net, 0.0))
        today_pnl = ch._day_pnl.get(now_day, 0.0) + partial
        extra = 1 if today_pnl >= threshold else 0
        min_valid = self.profile.get("min_valid_days", MIN_TRADING_DAYS)
        if ch.trading_days + extra < min_valid:
            return False
        tag = self.config.get('tag', self.name)
        logger.info(f"[{tag}] Ch#{ch.challenge_num} early close: projected="
                    f"{projected:.2f} target={target:.2f} legs={len(filled)} "
                    f"(+{len(pending)} pending)")
        return self._flatten_phase_exposure(ch, filled, pending)

    def _update_trailing_stop(self, ch, bar, atr_val):
        """Legacy: update trailing stop for ch.open_trade (single-trade compat)."""
        ot = ch.open_trade
        if ot is None:
            return False
        return self._update_trailing_stop_single(ch, ot, bar, atr_val)

    def _next_trail_px(self, ot, hi, lo, atr_val=None):
        """Candidate trail from a bar's best excursion (kernel numba:1906-1933
        ratchets from the bar's high/low). Returns None when the trail does not
        improve (raise-only). Shared by the bar-close and live (intrabar) paths."""
        if ot._sl_dist <= 0:
            ot._sl_dist = abs(ot.entry - ot.sl)
        sl_dist = ot._sl_dist
        if sl_dist <= 0:
            return None
        if ot._trail_px <= 0:
            ot._trail_px = ot.sl
        trail_mult = self.config.get("trail_atr_mult", 3.0)
        trail_act = self.config.get("trail_act", 0.5)
        prev_atr = self._last_bar_atr if self._last_bar_atr and not np.isnan(self._last_bar_atr) else atr_val
        if not prev_atr or np.isnan(prev_atr):
            prev_atr = sl_dist / max(self.config.get("sl_atr_mult", 1.5), 1e-9)
        if ot.direction == 1:
            if (hi - ot.entry) < trail_act * sl_dist:
                return None
            new_trail = hi - trail_mult * prev_atr
            return new_trail if new_trail > ot._trail_px else None
        if (ot.entry - lo) < trail_act * sl_dist:
            return None
        new_trail = lo + trail_mult * prev_atr
        return new_trail if (new_trail < ot._trail_px or ot._trail_px <= 0) else None

    def _trail_digits(self):
        return getattr(self.bot, "price_digits", 2) if self.bot else 2

    def _apply_trail_move(self, ch, ot, new_trail):
        digits = self._trail_digits()
        ot._trail_px = new_trail
        ot.sl = round(ot._trail_px, digits)
        tag = self.config.get('tag', self.name)
        logger.info(f"[{tag}] Trail {ot.id}: SL {ot.sl:.2f} "
                    f"(dist={ot._sl_dist:.2f} trail={ot._trail_px:.2f})")
        if self.bot and self.bot.trade_store:
            self.bot.trade_store.save_trade(ot)
        if self.bot and ot.position_id is not None:
            self.bot.push_trail_sl_to_broker(self, ot)

    def _update_trailing_stop_single(self, ch, ot, bar, atr_val):
        # Only a FILLED position trails: a pending limit has no broker position
        # to protect, and ratcheting from pre-fill bars corrupts the trail that
        # applies on adoption (the resting order still carries the initial SL).
        if getattr(ot, "position_id", None) is None:
            return False
        hi = bar.get("high", bar.get("close", ot.entry))
        lo = bar.get("low", bar.get("close", ot.entry))
        if ot._sl_dist <= 0:
            ot._sl_dist = abs(ot.entry - ot.sl)
        if ot._sl_dist <= 0:
            return False
        if ot._trail_px <= 0:
            ot._trail_px = ot.sl
        if ot.direction == 1:
            breached = lo <= ot._trail_px
        else:
            breached = hi >= ot._trail_px
        if breached:
            # Stop-fill parity: if the broker stop is armed at the trail, let it
            # fill there (kernel stop-fill model) instead of dumping at market.
            if self.bot and getattr(self.bot, "_stop_is_armed", None) and self.bot._stop_is_armed(ot):
                return True
            if self.bot:
                self.bot._close_trade_at_market(ot, self, ch)
            return True
        new_trail = self._next_trail_px(ot, hi, lo, atr_val)
        if new_trail is not None:
            self._apply_trail_move(ch, ot, new_trail)
        return False

    def _advance_phase(self, ch, now_day, now_dt):
        if ch.open_trades:
            # A phase reset while a prior-phase position is still live would
            # book that position's later PnL into the new phase.
            return False
        # Backtest parity (kernel numba:1787-1812): phase transitions zero the
        # balance WITHOUT booking the surplus as earned — only the 2→3 challenge
        # FEE REFUND is booked (kernel `total_refunds += fee`).
        eff = self.profile["ib"] + ch.ch_trade_pnl
        min_valid = self.profile.get("min_valid_days", MIN_TRADING_DAYS)
        if ch.phase == 1 and eff >= ch.phase_start_bal * (1 + self.profile["p1"]):
            if ch.trading_days >= min_valid:
                ib = self.profile["ib"]
                profit = eff - ib
                ch.phase = 2
                tag = self.config.get('tag', self.name)
                logger.info(f"[{tag}] Ch#{ch.challenge_num} PHASE 1→2 (profit={profit:+.2f})")
                ch.phase_start_bal = ib
                ch.equity = ib
                ch.day_equity = ib
                ch.dd_floor = ib * (1 - self.profile["md"])
                self.state.trade_pnl -= ch.ch_trade_pnl
                ch.ch_trade_pnl = 0.0
                ch.trading_days = 0
                ch._trading_day_set.clear()
                ch._traded_day_set.clear()
                ch._eod_ref = ib
                ch._day_pnl.clear()
                return True
        elif ch.phase == 2 and eff >= ch.phase_start_bal * (1 + self.profile["p2"]):
            if ch.trading_days >= min_valid:
                ib = self.profile["ib"]
                profit = eff - ib
                # Kernel `total_refunds += fee` — the challenge fee is
                # refunded on funding; it is the only surplus booked.
                cc = self.profile.get("cc", 0.0)
                if cc > 0:
                    self.state.total_earned += cc
                    ym = now_dt.strftime("%Y-%m")
                    self.state.monthly_stats.setdefault(ym, {"spent": 0, "earned": 0, "trade_pnl": 0, "withdrawn": 0})
                    self.state.monthly_stats[ym]["earned"] += cc
                ch.phase = 3
                tag = self.config.get('tag', self.name)
                logger.info(f"[{tag}] Ch#{ch.challenge_num} PHASE 2→3 FUNDED! (profit={profit:+.2f} fee_refund={cc:+.2f})")
                ch.funded_date = now_dt.isoformat()
                ch.phase_start_bal = ib
                ch.equity = ib
                ch.day_equity = ib
                ch.dd_floor = ib * (1 - self.profile["md"])
                self.state.trade_pnl -= ch.ch_trade_pnl
                ch.ch_trade_pnl = 0.0
                ch.trading_days = 0
                ch._trading_day_set.clear()
                ch._traded_day_set.clear()
                ch._eod_ref = ib
                ch._day_pnl.clear()
                ch.last_payout_day = now_day
                ch._payouts_run = 0
                ch._last_payout_valid_idx = 0
                ch._day_cap = {}
                return True
        return False

    def _mark_entry_day(self, ch):
        if self.profile.get("valid_day_mode") != "entry":
            return
        day = int(datetime.now(timezone.utc).timestamp() / 86400)
        ch._traded_day_set.add(day)
        ch.trading_days = len(ch._traded_day_set)

    def _compute_entry(self, signal, bar, atr_val=0.0, ch=None, ref_price=None):
        if ch is None:
            ch = self.state.current_challenge
        max_conc = self._max_concurrent()
        if ch is None or ch.blown or len(ch.open_trades) >= max_conc or signal == 0:
            return None
        rr = self.config["eval_rr"] if ch.phase < 3 else self.config["funded_rr"]
        rpt = self.config.get("eval_rpt", self.profile["rpt"]) if ch.phase < 3 else self.config.get("funded_rpt", self.profile.get("rpt", FUNDED_RISK_PER_TRADE))
        o = bar["open"]
        use_limit = bool(self.config.get("use_limit_entry"))
        if use_limit:
            # LIMIT entry (backtest limit engine): pullback limit at the next
            # bar's open ∓ lf × ATR, maker fill, waits up to limit_wait bars
            lf = self.config.get("limit_frac", 0.75)
            ep = o - lf * atr_val if signal == 1 else o + lf * atr_val
        else:
            slip = self.config.get("entry_slip", 0.0)
            ep = o + (slip if signal == 1 else -slip)
        if self.config.get("stop_mode") == "fixed":
            sl_dist = float(self.config.get("stop_value", 25.0))
            if sl_dist <= 0:
                return None
            sl = ep - sl_dist if signal == 1 else ep + sl_dist
            tp = ep + sl_dist * rr if signal == 1 else ep - sl_dist * rr
        elif self.profile["sm"] == "fixed_0.5":
            sl_dist = abs(ep) * self.config.get("stop_pct", 0.005)
            if sl_dist <= 0:
                return None
            sl = ep - sl_dist if signal == 1 else ep + sl_dist
            tp = ep + sl_dist * rr if signal == 1 else ep - sl_dist * rr
        else:
            if np.isnan(atr_val) or atr_val <= 0:
                return None
            atr_mult = self.config.get("sl_atr_mult", 1.5)
            sl = ep - atr_val * atr_mult if signal == 1 else ep + atr_val * atr_mult
            tp = ep + atr_val * atr_mult * rr if signal == 1 else ep - atr_val * atr_mult * rr
            sl_dist = abs(ep - sl)
        eff = ch.equity  # kernel `balance` (realized): DD headroom + leverage base
        if self.config.get("sizing_mode") == "script":
            # Script parity: units = floor(risk% × equity_incl / stop distance)
            # in integer ounces. The backtest skips the entry when the result is
            # below its 0.01-lot minimum instead of forcing a 1-oz position.
            ref = ref_price if ref_price is not None else ep
            eq_incl = eff
            for ot in ch.open_trades:
                if getattr(ot, "status", None) == "pending":
                    continue
                eq_incl += (ref - ot.entry) * ot.direction * ot.lots * CONTRACT_SIZE
            units = int(math.floor(rpt * eq_incl / max(sl_dist, 1e-9)))
            if units < 1:
                return None
            lots = round(units / CONTRACT_SIZE, 2)
            lots = min(lots, MAX_LOT)
            leverage = LEVERAGE_EVAL if ch.phase < 3 else LEVERAGE_FUNDED
            max_by_lev = (eff * leverage) / max(ep * CONTRACT_SIZE, 1e-9)
            lots = round(min(lots, max_by_lev, MAX_LOT), 2)
            if lots < MIN_LOT:
                return None
            tid = self.bot.trade_store._next_trade_id() if self.bot and self.bot.trade_store else 1
            tag = self.config.get('tag', self.name)
            t = Trade(id=f"{tag}-ch{ch.challenge_num}-{tid}", strategy=self.name,
                      direction=signal, entry=round(ep, 5), sl=round(sl, 5), tp=round(tp, 5),
                      lots=lots, open_time=datetime.now(timezone.utc).isoformat(),
                      status="pending" if use_limit else "open")
            ch.open_trades.append(t)
            self._mark_entry_day(ch)
            d = "LONG" if signal == 1 else "SHORT"
            logger.info(f"[{tag}] {'LIMIT ' if use_limit else ''}Entry {d} {t.id}: "
                        f"ep={ep:.2f} sl={sl:.2f} tp={tp:.2f} lots={lots} "
                        f"units={units}oz atr={atr_val:.2f} rr={rr}")
            if self.bot and self.bot.trade_store:
                self.bot.trade_store.save_trade(t)
            self.broadcast()
            return t
        dh = max(eff - ch.dd_floor, 0)
        # Daily-loss headroom: use the SAME floor formula the enforcement uses
        # (fixed-mode profiles floor at day_equity - dl×ib, not the fraction)
        if self.profile.get('daily_dd_mode') == 'fixed':
            dlh = max(eff - (ch.day_equity - self.profile["dl"] * self.profile["ib"]), 0)
        else:
            dlh = max(eff - ch.day_equity * (1 - self.profile["dl"]), 0)
        # Backtest parity (kernel numba:1644):
        #   risk = eval_risk × initial_balance   (phases 1-2: FIXED on ib)
        #   risk = funded_risk × balance         (phase 3: current balance)
        # — eval risk must NOT compound with eval-phase P&L.
        risk_base = self.profile["ib"] if ch.phase < 3 else ch.equity
        risk_amt = risk_base * rpt
        # Per-trade loss cap (instant venues): risk never exceeds mlpt × initial
        mlpt = self.profile.get("max_loss_per_trade")
        if mlpt:
            risk_amt = min(risk_amt, mlpt * self.profile["ib"])
        er = max(min(risk_amt, min(dh, dlh) * 0.5), 0)
        if er <= 0:
            # No risk budget (near a DD floor) — do not open a position
            logger.info(f"[{self.config.get('tag', self.name)}] Entry skipped: no risk "
                        f"budget (er=0, eff={eff:.2f} dd_floor={ch.dd_floor:.2f} "
                        f"dh={dh:.2f} dlh={dlh:.2f})")
            return None
        value_per_lot = CONTRACT_SIZE * sl_dist
        raw = er / (value_per_lot + 1e-9)
        lots = max(MIN_LOT, round(raw / LOT_STEP) * LOT_STEP)
        lots = min(lots, MAX_LOT)
        leverage = LEVERAGE_EVAL if ch.phase < 3 else LEVERAGE_FUNDED
        max_by_lev = (eff * leverage) / max(ep * CONTRACT_SIZE, 1e-9)
        lots = min(lots, max_by_lev, MAX_LOT)
        # Broker lot step is 0.01: float artifacts (0.7000000000000001) get
        # rejected with "order amount is not multiple of lot step". Round to
        # the step before the row is saved and the order is sent.
        lots = round(lots, 2)
        if lots < MIN_LOT:
            return None
        tid = self.bot.trade_store._next_trade_id() if self.bot and self.bot.trade_store else 1
        tag = self.config.get('tag', self.name)
        t = Trade(id=f"{tag}-ch{ch.challenge_num}-{tid}", strategy=self.name,
                  direction=signal, entry=round(ep, 5), sl=round(sl, 5), tp=round(tp, 5),
                  lots=lots, open_time=datetime.now(timezone.utc).isoformat(),
                  status="pending" if use_limit else "open")
        ch.open_trades.append(t)
        self._mark_entry_day(ch)
        d = "LONG" if signal == 1 else "SHORT"
        logger.info(f"[{tag}] {'LIMIT ' if use_limit else ''}Entry {d} {t.id}: "
                    f"ep={ep:.2f} sl={sl:.2f} tp={tp:.2f} lots={lots} atr={atr_val:.2f} rr={rr}")
        if self.bot and self.bot.trade_store:
            self.bot.trade_store.save_trade(t)
        self.broadcast()
        return t

    def manual_entry(self, direction, lots, price, ch=None):
        if ch is None:
            ch = self.state.current_challenge
        max_conc = self._max_concurrent()
        if ch is None or ch.blown or len(ch.open_trades) >= max_conc or direction not in (1, -1) or lots <= 0:
            return None
        rr = self.config["eval_rr"] if ch.phase < 3 else self.config["funded_rr"]
        ep = price + (0.05 if direction == 1 else -0.05)
        atr_val = ep * 0.01
        atr_mult = self.config.get("sl_atr_mult", 1.5)
        sl = ep - atr_val * atr_mult if direction == 1 else ep + atr_val * atr_mult
        tp = ep + atr_val * atr_mult * rr if direction == 1 else ep - atr_val * atr_mult * rr
        lots = min(max(round(lots, 2), 0), MAX_LOT)
        if lots < MIN_LOT:
            return None
        tid = self.bot.trade_store._next_trade_id() if self.bot and self.bot.trade_store else 1
        tag = self.config.get('tag', self.name)
        t = Trade(id=f"{tag}-ch{ch.challenge_num}-manual-{tid}", strategy=self.name,
                  direction=direction, entry=round(ep, 5), sl=round(sl, 5), tp=round(tp, 5),
                  lots=lots, open_time=datetime.now(timezone.utc).isoformat())
        ch.open_trades.append(t)
        if self.bot and self.bot.trade_store:
            self.bot.trade_store.save_trade(t)
        self.broadcast()
        return t

    @staticmethod
    def _swap_usd(direction, lots, open_time, close_dt, swap_rate):
        """Kernel swap (numba:1733): −rate USD/oz per calendar night held,
        ×3 on Wednesday nights, summed n_nights = exit_day − entry_day.
        Returns a signed PnL delta (positive rate + long = credit)."""
        if not swap_rate or not open_time:
            return 0.0
        try:
            ot_dt = datetime.fromisoformat(open_time)
            if ot_dt.tzinfo is None:
                ot_dt = ot_dt.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return 0.0
        ed = int(ot_dt.timestamp() // 86400)
        xd = int(close_dt.timestamp() // 86400)
        n_nights = xd - ed
        if n_nights <= 0:
            return 0.0
        mult = 0
        for kk in range(1, n_nights + 1):
            mult += 3 if (ed + kk + 3) % 7 == 2 else 1
        return swap_rate * (lots * CONTRACT_SIZE) * mult * direction

    def on_trade_closed(self, trade, now_dt, ch=None):
        if ch is None:
            ch = self.state.current_challenge
        if ch is None:
            return
        tag = self.config.get('tag', self.name)
        # Remove from open_trades ALWAYS (blown challenges too): the position is
        # gone from the broker — keeping it creates phantom open trades and
        # phantom unrealized PnL until restart.
        ch.open_trades = [t for t in ch.open_trades if t.id != trade.id]
        if not ch.open_trades:
            # Backtest parity: the kernel's intraday projection is
            # balance + wR×risk where balance ALREADY includes this trade's
            # realized loss. Keeping the closed trade's worst excursion here
            # would count the same loss twice against the DD floor when the
            # next trade opens later today.
            ch._day_worst_upl = 0.0
        now_day = int(now_dt.timestamp() / 86400)
        if trade.pnl is None:
            # Placeholder close (position vanished; fill-price settlement follows
            # via orders history): book nothing yet, keep the row discoverable
            trade.close_time = now_dt.isoformat()
            trade.status = "reconciled_unknown"
            ch.trade_history.append(asdict(trade))
            logger.info(f"[{tag}] {trade.id} placeholder close (position vanished) — "
                        f"PnL pending fill-price settlement")
            return
        # Swap (2-step profiles; kernel numba:1733) — booked at exit, joins the
        # balance before the floor checks. Instant venues pass no swap_rate.
        swap = self._swap_usd(trade.direction, trade.lots, trade.open_time,
                              now_dt, self.profile.get("swap_rate", 0.0))
        pnl_net = (trade.pnl if trade.pnl is not None else 0) - COMMISSION * trade.lots + swap
        start_bal = ch.equity  # entry balance (bookings happen only at close)
        trade.pnl = pnl_net
        ch.equity += pnl_net
        logger.info(f"[{tag}] Closed {trade.id}: pnl={pnl_net:+.2f} eq={ch.equity:.2f} "
                    f"ch_pnl={ch.ch_trade_pnl + pnl_net:+.2f} blown={ch.blown}")
        self._add_closed_trade_day(ch, now_day, pnl_net)
        ch.ch_trade_pnl += pnl_net
        self.state.trade_pnl += pnl_net
        ym = now_dt.strftime("%Y-%m")
        self.state.monthly_stats.setdefault(ym, {"spent": 0, "earned": 0, "trade_pnl": 0, "withdrawn": 0})
        self.state.monthly_stats[ym]["trade_pnl"] = self.state.monthly_stats[ym].get("trade_pnl", 0) + pnl_net
        day_str = now_dt.strftime("%Y-%m-%d")
        self.state.daily_pnl[day_str] = self.state.daily_pnl.get(day_str, 0.0) + pnl_net
        self._apply_daily_profit_cap(ch, now_day)
        if ch.blown:
            # Already blown (e.g. intraday DD): pnl is booked to stats above but
            # the challenge stays dead — no recycle/phase logic re-runs
            ch.trade_history.append(asdict(trade))
            if self.bot:
                self.bot.treasury_store.save({self.name: self})
            self.broadcast()
            return
        if self.profile.get('daily_dd_mode') == 'fixed':
            daily_floor = ch.day_equity - self.profile["dl"] * self.profile["ib"]
        else:
            daily_floor = ch.day_equity * (1 - self.profile["dl"])
        # Per-trade loss cap as a BLOW rule (kernel numba:1917):
        # realized ≤ −mlpt × trade_start_balance → account blown.
        mlpt = self.profile.get("max_loss_per_trade")
        pertrade_breach = bool(mlpt and pnl_net <= -mlpt * start_bal)
        if ch.equity <= ch.dd_floor or ch.equity <= daily_floor or pertrade_breach:
            ch.blown = True
            if pertrade_breach:
                reason = (f"pertrade mlpt: pnl={pnl_net:+.2f} <= "
                          f"-{mlpt:.4f} × start={start_bal:.2f}")
            else:
                reason = (f"close-triggered: eq={ch.equity:.2f} "
                          f"dd_floor={ch.dd_floor:.2f} daily_floor={daily_floor:.2f}")
            logger.warning(f"[{tag}] Ch#{ch.challenge_num} BLOWN ({reason})")
            ch.trade_history.append(asdict(trade))
            if self.bot:
                self.bot.treasury_store.save({self.name: self})
                self.bot._recycle()
            self.broadcast()
            return
        ch.trade_history.append(asdict(trade))
        if (ch.phase in (1, 2) and ch.open_trades
                and self.config.get("early_close", True)):
            # A normal TP/SL may have put the phase over target while another
            # leg is still live: flatten that leg before phase advance.
            self._check_early_close(ch, None, now_day, now_dt)
        self._advance_phase(ch, now_day, now_dt)
        if self.bot:
            self.bot.treasury_store.save({self.name: self})
        self.broadcast()

    def snapshot(self):
        challenges = self.state.challenges
        active = [c for c in challenges if not c.blown]
        total_blown = sum(1 for c in challenges if c.blown)
        all_trades = []
        open_trade_dicts = []
        for c in challenges:
            all_trades.extend(list(c.trade_history))
            for ot in c.open_trades:
                otd = self._open_trade_with_pnl_single(ot)
                if otd:
                    if ot.status == "pending":
                        # A placed limit is a live order: show it immediately
                        # (badge on the dashboard) with no unrealized PnL.
                        otd["unrealized_pnl"] = 0.0
                    all_trades.append(otd)
                    open_trade_dicts.append(otd)
        unrealized = sum(otd.get("unrealized_pnl", 0) for otd in open_trade_dicts)
        live_trade_pnl = self.state.trade_pnl + unrealized
        net = live_trade_pnl - self.state.total_spent
        # Failed entries (never traded) are excluded from every display list;
        # pending limits stay visible (they are live broker orders).
        all_trades = [t for t in all_trades
                      if (t.get("status") or "").lower() != "failed"]
        open_ids = set(otd.get("id") for otd in open_trade_dicts)
        closed_trades = [t for t in all_trades if t.get("id") not in open_ids]
        _by_id = {}
        for t in closed_trades:
            tid = t.get("id")
            key = t.get("close_time") or t.get("open_time") or ""
            if tid not in _by_id or key > _by_id[tid][0]:
                _by_id[tid] = (key, t)
        closed_sorted = sorted([t for _, t in _by_id.values()],
                               key=lambda t: t.get("close_time") or t.get("open_time") or "", reverse=True)
        recent_with_open = list(open_trade_dicts) + closed_sorted[:20]
        return {
            "strategy": self.name, "farm_type": self.farm_type,
            "broker": self.config.get("broker", ""),
            "challenges": [self._challenge_snapshot(c) for c in challenges],
            "active_count": len(active), "total_challenges": len(challenges),
            "total_spent": round(self.state.total_spent, 2),
            "total_earned": round(self.state.total_earned, 2),
            "treasury": round(self.state.treasury, 2),
            "net_pnl": round(net, 2), "trade_pnl": round(self.state.trade_pnl, 2),
            "blown_pnl": round(self.state.blown_pnl, 2),
            "unrealized_pnl": round(unrealized, 2),
            "trading_net": round(live_trade_pnl, 2), "total_blown": total_blown,
            "recent_trades": recent_with_open,
            "monthly_stats": {k: dict(v) for k, v in list(self.state.monthly_stats.items())},
            "daily_pnl": {k: round(v, 2) for k, v in list(self.state.daily_pnl.items())},
            "winrate": self._win_rate(),
            "phase1_target": self.profile.get("p1"),
            "phase2_target": self.profile.get("p2"),
            "current_timeframe": self._current_timeframe(),
        }

    def _challenge_snapshot(self, ch):
        ib = self.profile["ib"]
        eff = ch.equity  # display equity (kernel `balance` — survives payout retention)
        # open positions + placed limits (pending: no unrealized PnL yet)
        open_dicts = [self._open_trade_with_pnl_single(ot) for ot in ch.open_trades]
        open_dicts = [d for d in open_dicts if d]
        for d in open_dicts:
            if d.get("status") == "pending":
                d["unrealized_pnl"] = 0.0
        total_unrealized = sum(d.get("unrealized_pnl", 0) for d in open_dicts)
        return {
            "challenge_num": ch.challenge_num, "phase": ch.phase,
            "equity": round(eff, 2), "balance": round(ib, 2),
            "dd_floor": round(ch.dd_floor, 2), "blown": ch.blown,
            "funded_date": ch.funded_date,
            "trade_pnl": round(ch.ch_trade_pnl, 2),
            "trading_net": round(ch.ch_trade_pnl + total_unrealized, 2),
            "trading_days": ch.trading_days, "open_trades": open_dicts,
            "open_trade": open_dicts[0] if open_dicts else None,
            "total_payouts": round(ch.total_payouts, 2),
            "phase_progress": self._phase_progress(ch, self.profile, ch.ch_trade_pnl),
        }

    @staticmethod
    def _phase_progress(ch, profile, trade_pnl=0):
        effective_equity = ch.phase_start_bal + trade_pnl
        if ch.phase in (1, 2):
            key = "p1" if ch.phase == 1 else "p2"
            p = profile.get(key)
            if p is None:
                pct = 100.0  # no eval target on this profile — nothing to progress toward
            else:
                target = ch.phase_start_bal * (1 + p)
                denom = target - ch.phase_start_bal
                pct = 100.0 if denom == 0 else (effective_equity - ch.phase_start_bal) / denom * 100
        else:
            pct = 100.0
        return {"pct": round(max(0, min(100, pct)), 1), "phase": ch.phase}

    def _open_trade_with_pnl(self, ch):
        """Legacy compat: returns first open trade dict."""
        if not ch or not ch.open_trades:
            return None
        return self._open_trade_with_pnl_single(ch.open_trades[0])

    def _open_trade_with_pnl_single(self, ot):
        if not ot:
            return None
        t = asdict(ot)
        price = 0.0
        if self.bot and self._broker_key and hasattr(self.bot, 'brokers'):
            broker = self.bot.brokers.get(self._broker_key)
            if broker:
                price = broker.get("current_price", 0.0)
        if price and price > 0:
            t["unrealized_pnl"] = round((price - ot.entry) * ot.direction * ot.lots * CONTRACT_SIZE, 2)
        return t

    def _win_rate(self):
        wins = total = 0
        for ch in self.state.challenges:
            for t in ch.trade_history:
                pnl = t.get("pnl")
                if pnl is not None:
                    total += 1
                    if pnl > 0:
                        wins += 1
        return round(wins / total * 100, 1) if total > 0 else 0.0

    def _current_timeframe(self):
        ch = self.state.current_challenge
        if ch is None:
            return f"{self.config['eval_tf']}m"
        return f"{self.config['funded_tf']}m" if ch.phase == 3 else f"{self.config['eval_tf']}m"

    def _challenge_timeframe(self, ch):
        if ch is None or ch.blown:
            return f"{self.config['eval_tf']}m"
        return f"{self.config['funded_tf']}m" if ch.phase == 3 else f"{self.config['eval_tf']}m"
