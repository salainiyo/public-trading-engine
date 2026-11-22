#!/usr/bin/env python3
"""Standalone, self-contained backtest of strategy_23_supertrend_rsi for the three
prop-firm 2-step 5K challenges (GoatFunded, FundingPips, AquaFunded).

Everything is inlined below: the contract kernel, the strategy, the rulebooks,
the ATR/calendar preparation and the block-bootstrap challenge evaluator.  No
module from the pipeline's src/ is imported -- only numpy/pandas/numba.

Usage (from the repository root):
    python3 scripts/supertrend_rsi_propfirm.py --data path/to/XAU_H1.csv
    python3 scripts/supertrend_rsi_propfirm.py --slice test --paths 1000

Requires numpy, pandas and numba. The default configuration is a fixed $25
stop, RR 3, 1% risk and max_pos 2.  --slice test runs the one-shot
2016-2026 evaluation; treat that as a report only, never as a selection input.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("NUMBA_CACHE_DIR", str(
    Path(__file__).resolve().parents[1] / "data" / "cache" / "numba"))
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- rulebooks
FIRMS = {
    "GoatFunded": dict(fee=17.0, target1=0.08, target2=0.06, daily_dd=0.04,
                       max_dd=0.10, min_days=3, funded_min_days=0,
                       profit_split=0.80, payout_days=14, initial_balance=5000.0,
                       leverage_eval=100, leverage_funded=50, margin_buffer=0.10,
                       heat_fraction=0.50, commission_per_lot=0.0,
                       lot_size=100.0, min_lot=0.01, lot_step=0.01),
    "FundingPips": dict(fee=36.0, target1=0.08, target2=0.05, daily_dd=0.05,
                        max_dd=0.10, min_days=3, funded_min_days=0,
                        profit_split=0.80, payout_days=14, initial_balance=5000.0,
                        leverage_eval=100, leverage_funded=50, margin_buffer=0.10,
                        heat_fraction=0.50, commission_per_lot=0.0,
                        lot_size=100.0, min_lot=0.01, lot_step=0.01),
    "AquaFunded": dict(fee=39.0, target1=0.08, target2=0.05, daily_dd=0.05,
                       max_dd=0.08, min_days=3, funded_min_days=0,
                       profit_split=0.90, payout_days=14, initial_balance=5000.0,
                       leverage_eval=100, leverage_funded=50, margin_buffer=0.10,
                       heat_fraction=0.50, commission_per_lot=0.0,
                       lot_size=100.0, min_lot=0.01, lot_step=0.01),
}

# stop-name -> (stop_mode, stop_value, exit_type, trail_mult, partial_r)
STOP_MODES = {
    "fixed_25": (1, 25.0, 0, 1.5, 1.0),
    "fixed_50": (1, 50.0, 0, 1.5, 1.0),
    "fixed_100": (1, 100.0, 0, 1.5, 1.0),
    "atr_1.0": (0, 1.0, 0, 1.5, 1.0),
    "atr_1.5": (0, 1.5, 0, 1.5, 1.0),
    "atr_2.0": (0, 2.0, 0, 1.5, 1.0),
    "trail_2.5": (0, 2.0, 1, 2.5, 1.0),
    "be_1.0": (0, 2.0, 2, 1.5, 1.0),
    "be_trail_1.0": (0, 2.0, 3, 1.5, 1.0),
}

SPREAD = 0.35
SLIPPAGE = 0.10
SWAP = -0.05
HORIZON_DAYS = 180
BLOCK_DAYS = 10


# ===== INLINED CONTRACT ENGINE (verbatim) =====
"""Contract-compliant no-lookahead simulation and challenge evaluation.

All per-bar work runs through ``@njit(cache=True)`` nopython kernels.  The
engine implements the AGENTS.md rules:

* signal/ATR at bar ``t`` come from ``t-1``; the fill is ``open[t]``;
* entries/adds are processed before exits, so an intrabar exit cannot free a
  slot for a same-bar entry/add;
* stop-first when stop and target are both inside a bar;
* gap-through-stop fills at ``open[t]``;
* trailing / breakeven updates happen on bar close and affect later bars;
* spread, slippage, swap and commission are applied as declared;
* concurrency, pyramid adds, heat cap and margin guard are enforced;
* the challenge evaluator chains P1 -> P2 -> funded with fresh accounts,
  static max drawdown, ``max(balance, equity)`` daily references, calendar-day
  trading-day counting and calendar-day payouts.

The legacy ``btcore`` kernels are not used by this module.
"""
import numpy as np
from numba import njit

EPS = 1e-12

# Output tuple layout for simulate_contract_multi.
SIM_N = 11
SIM_SKIPPED_HEAT = 11
SIM_SKIPPED_MARGIN = 12
SIM_NADDS = 13
SIM_MAX_CONCURRENT = 14
SIM_PEAK_RISK = 15
SIM_WORST_DAY = 16


@njit(cache=True)
def simulate_contract_multi(op, hi, lo, cl, atr, dow, sig, day, start, end,
                            initial, risk, rr, stop_mode, stop_value,
                            exit_type, trail_mult, partial_r, spread,
                            slippage, swap, commission, max_pos, addon_mode,
                            addon_trigger_r, addon_size, max_adds, heat_cap,
                            leverage, margin_buffer, lot_size, min_lot,
                            lot_step, gap_reserve):
    """Single- or multi-position contract simulator.

    stop_mode: 0 = ``stop_value * ATR[t-1]``, 1 = fixed USD/oz ``stop_value``.
    exit_type: 0 plain SL/TP, 1 trailing, 2 partial+BE, 3 partial+BE then trail.
    addon_mode: 0 none (independent positions up to max_pos), 1 pyramid.
    """
    n = end - start
    eq_close = np.empty(n, np.float64)
    eq_worst = np.empty(n, np.float64)
    bal_close = np.empty(n, np.float64)
    cap = n * max_pos + 4
    t_entry = np.empty(cap, np.int64)
    t_exit = np.empty(cap, np.int64)
    t_dir = np.empty(cap, np.int8)
    t_pnl = np.empty(cap, np.float64)
    t_r = np.empty(cap, np.float64)
    t_spread = np.empty(cap, np.float64)
    t_slip = np.empty(cap, np.float64)
    t_swap = np.empty(cap, np.float64)
    t_comm = np.empty(cap, np.float64)
    nt = 0

    p_active = np.zeros(max_pos, np.int8)
    p_dir = np.zeros(max_pos, np.int8)
    p_entry = np.zeros(max_pos, np.float64)
    p_stop = np.zeros(max_pos, np.float64)
    p_target = np.zeros(max_pos, np.float64)
    p_units = np.zeros(max_pos, np.float64)
    p_init_units = np.zeros(max_pos, np.float64)
    p_dist = np.zeros(max_pos, np.float64)
    p_eidx = np.zeros(max_pos, np.int64)
    p_partial = np.zeros(max_pos, np.int8)
    p_gross = np.zeros(max_pos, np.float64)
    p_cost = np.zeros(max_pos, np.float64)
    p_cspread = np.zeros(max_pos, np.float64)
    p_cslip = np.zeros(max_pos, np.float64)
    p_cswap = np.zeros(max_pos, np.float64)
    p_ccomm = np.zeros(max_pos, np.float64)
    p_risk = np.zeros(max_pos, np.float64)

    balance = initial
    chain_adds = 0
    n_adds = 0
    n_skipped_heat = 0
    n_skipped_margin = 0
    max_concurrent = 0
    peak_open_risk = 0.0
    worst_day_loss = 0.0
    prev_day = day[start]
    day_start_equity = initial
    day_min_equity = initial
    comm_per_oz = commission / lot_size if lot_size > 0 else 0.0
    min_units = max(1.0, lot_size * min_lot)
    lot_step_oz = max(EPS, lot_size * lot_step)

    for k in range(n):
        t = start + k

        # ---------------------------------------------------------- overnight swap
        if day[t] != prev_day:
            if worst_day_loss < (day_start_equity - day_min_equity):
                worst_day_loss = day_start_equity - day_min_equity
            day_start_equity = balance
            day_min_equity = balance
            mult = 3.0 if dow[t - 1] == 2 else 1.0
            for j in range(max_pos):
                if p_active[j] == 0:
                    continue
                sw = abs(swap) * p_units[j] * mult
                balance -= sw
                p_cost[j] += sw
                p_cswap[j] += sw
            prev_day = day[t]

        # ------------------------------------------------ start-of-bar position state
        open_count = 0
        long_count = 0
        short_count = 0
        unreal_total = 0.0
        open_risk = 0.0
        used_margin = 0.0
        be_same_dir = False
        trigger_same_dir = False
        for j in range(max_pos):
            if p_active[j] == 0:
                continue
            open_count += 1
            if p_dir[j] == 1:
                long_count += 1
            else:
                short_count += 1
            ur = p_units[j] * (cl[t - 1] - p_entry[j]) * p_dir[j]
            unreal_total += ur
            dist_stop = abs(cl[t - 1] - p_stop[j])
            open_risk += p_units[j] * dist_stop
            used_margin += p_units[j] * cl[t - 1] / leverage
            at_be = ((p_dir[j] == 1 and p_stop[j] >= p_entry[j]) or
                     (p_dir[j] == -1 and p_stop[j] <= p_entry[j]))
            # The add trigger is measured on the original unit's profit, not
            # the post-partial remainder, otherwise a 50% partial makes the
            # trigger unreachable before the target.
            unit_ur = p_init_units[j] * (cl[t - 1] - p_entry[j]) * p_dir[j]
            prof_r = unit_ur / p_risk[j] if p_risk[j] > EPS else 0.0
            if at_be:
                be_same_dir = True
            if at_be and prof_r >= addon_trigger_r:
                trigger_same_dir = True
        if open_risk > peak_open_risk:
            peak_open_risk = open_risk
        if open_count > max_concurrent:
            max_concurrent = open_count

        # --------------------------------------------------------------- entries/adds
        s = sig[t - 1] if t > start else 0
        if s != 0 and atr[t - 1] == atr[t - 1]:
            if stop_mode == 1:
                d = stop_value
            else:
                d = stop_value * atr[t - 1]
            if d > EPS:
                direction = 1 if s > 0 else -1
                same_dir = ((direction == 1 and short_count == 0) or
                            (direction == -1 and long_count == 0))
                allow = False
                is_add = False
                if open_count == 0:
                    allow = True
                elif open_count < max_pos and same_dir:
                    if addon_mode == 0:
                        allow = True
                    elif addon_mode == 1:
                        if be_same_dir and trigger_same_dir and chain_adds < max_adds:
                            allow = True
                            is_add = True
                if allow:
                    if is_add:
                        cash = addon_size * risk * balance
                    else:
                        cash = risk * balance
                    if cash < 0.0:
                        cash = 0.0
                    raw_units = np.floor(cash / d)
                    u = np.floor(raw_units / lot_step_oz) * lot_step_oz
                    if u < min_units:
                        # Allow the minimum lot only when its risk is within
                        # 10% of the configured budget.  This keeps a viable
                        # strategy alive after a small loss without silently
                        # doubling risk when the stop is too wide for the
                        # account; infeasible risk/stop pairs are skipped.
                        if cash > 0.0 and min_units * d <= cash * 1.10:
                            u = min_units
                        else:
                            u = 0.0
                    if u >= min_units:
                        new_risk = u * d
                        if heat_cap > 0.0 and (open_risk + new_risk + gap_reserve) > heat_cap:
                            n_skipped_heat += 1
                        else:
                            equity_now = balance + unreal_total
                            required_margin = u * op[t] / leverage
                            free_margin = equity_now - used_margin
                            if required_margin > (1.0 - margin_buffer) * free_margin:
                                n_skipped_margin += 1
                            else:
                                slot = -1
                                for j in range(max_pos):
                                    if p_active[j] == 0:
                                        slot = j
                                        break
                                if slot >= 0:
                                    entry_spread = u * spread * 0.5
                                    entry_slip = u * slippage
                                    entry_comm = u * comm_per_oz * 0.5
                                    entry_cost = entry_spread + entry_slip + entry_comm
                                    p_active[slot] = 1
                                    p_dir[slot] = direction
                                    p_entry[slot] = op[t] + direction * (spread * 0.5 + slippage)
                                    p_stop[slot] = p_entry[slot] - direction * d
                                    p_target[slot] = p_entry[slot] + direction * rr * d
                                    p_units[slot] = u
                                    p_init_units[slot] = u
                                    p_dist[slot] = d
                                    p_eidx[slot] = t
                                    p_partial[slot] = 0
                                    p_gross[slot] = 0.0
                                    p_cost[slot] = entry_cost
                                    p_cspread[slot] = entry_spread
                                    p_cslip[slot] = entry_slip
                                    p_cswap[slot] = 0.0
                                    p_ccomm[slot] = entry_comm
                                    p_risk[slot] = u * d + entry_cost
                                    balance -= entry_cost
                                    open_risk += new_risk
                                    used_margin += u * op[t] / leverage
                                    if is_add:
                                        n_adds += 1
                                        chain_adds += 1
                                    else:
                                        chain_adds = 0

        # ------------------------------------------------------------------- exits
        for j in range(max_pos):
            if p_active[j] == 0:
                continue
            direction = p_dir[j]
            entry = p_entry[j]
            stop = p_stop[j]
            target = p_target[j]
            units = p_units[j]
            init_dist = p_dist[j]
            exited = False
            exit_price = 0.0

            if direction == 1:
                if op[t] <= stop:
                    exit_price = op[t]
                    exited = True
                elif op[t] >= target:
                    exit_price = op[t]
                    exited = True
                else:
                    hit_stop = lo[t] <= stop
                    hit_target = hi[t] >= target
                    plevel = entry + partial_r * init_dist
                    hit_partial = ((exit_type >= 2) and p_partial[j] == 0
                                   and hi[t] >= plevel)
                    if hit_stop:
                        exit_price = stop
                        exited = True
                    elif hit_partial and rr > partial_r:
                        take = units * 0.5
                        gross = take * (plevel - entry)
                        cost = (take * spread * 0.5 + take * slippage
                                + take * comm_per_oz * 0.5)
                        balance += gross - cost
                        p_gross[j] += gross
                        p_cost[j] += cost
                        p_cspread[j] += take * spread * 0.5
                        p_cslip[j] += take * slippage
                        p_ccomm[j] += take * comm_per_oz * 0.5
                        units -= take
                        p_partial[j] = 1
                        stop = entry
                        if lo[t] <= entry:
                            exit_price = entry
                            exited = True
                        elif hi[t] >= target:
                            exit_price = target
                            exited = True
                    elif hit_target:
                        exit_price = target
                        exited = True
            else:
                if op[t] >= stop:
                    exit_price = op[t]
                    exited = True
                elif op[t] <= target:
                    exit_price = op[t]
                    exited = True
                else:
                    hit_stop = hi[t] >= stop
                    hit_target = lo[t] <= target
                    plevel = entry - partial_r * init_dist
                    hit_partial = ((exit_type >= 2) and p_partial[j] == 0
                                   and lo[t] <= plevel)
                    if hit_stop:
                        exit_price = stop
                        exited = True
                    elif hit_partial and rr > partial_r:
                        take = units * 0.5
                        gross = take * (entry - plevel)
                        cost = (take * spread * 0.5 + take * slippage
                                + take * comm_per_oz * 0.5)
                        balance += gross - cost
                        p_gross[j] += gross
                        p_cost[j] += cost
                        p_cspread[j] += take * spread * 0.5
                        p_cslip[j] += take * slippage
                        p_ccomm[j] += take * comm_per_oz * 0.5
                        units -= take
                        p_partial[j] = 1
                        stop = entry
                        if hi[t] >= entry:
                            exit_price = entry
                            exited = True
                        elif lo[t] <= target:
                            exit_price = target
                            exited = True
                    elif hit_target:
                        exit_price = target
                        exited = True

            if exited:
                gross = units * (exit_price - entry) * direction
                exit_spread = units * spread * 0.5
                exit_slip = units * slippage
                exit_comm = units * comm_per_oz * 0.5
                exit_cost = exit_spread + exit_slip + exit_comm
                balance += gross - exit_cost
                p_gross[j] += gross
                p_cost[j] += exit_cost
                p_cspread[j] += exit_spread
                p_cslip[j] += exit_slip
                p_ccomm[j] += exit_comm
                total_pnl = p_gross[j] - p_cost[j]
                risk_cash = p_init_units[j] * p_dist[j] + p_cost[j]
                t_entry[nt] = p_eidx[j]
                t_exit[nt] = t
                t_dir[nt] = direction
                t_pnl[nt] = total_pnl
                t_r[nt] = total_pnl / risk_cash if risk_cash > EPS else 0.0
                t_spread[nt] = p_cspread[j]
                t_slip[nt] = p_cslip[j]
                t_swap[nt] = p_cswap[j]
                t_comm[nt] = p_ccomm[j]
                nt += 1
                p_active[j] = 0
                if open_count <= 1:
                    chain_adds = 0
                open_count -= 1
            else:
                p_units[j] = units
                p_stop[j] = stop
                if exit_type == 1 or (exit_type == 3 and p_partial[j] == 1):
                    if atr[t] == atr[t]:
                        if direction == 1:
                            ns = cl[t] - trail_mult * atr[t]
                            if ns > p_stop[j]:
                                p_stop[j] = ns
                        else:
                            ns = cl[t] + trail_mult * atr[t]
                            if ns < p_stop[j]:
                                p_stop[j] = ns

        # ------------------------------------------------------------------ equity
        bal_close[k] = balance
        unreal = 0.0
        worst = 0.0
        for j in range(max_pos):
            if p_active[j] == 1:
                if p_dir[j] == 1:
                    unreal += p_units[j] * (cl[t] - p_entry[j])
                    adverse = lo[t] - p_entry[j]
                    if adverse > 0.0:
                        adverse = 0.0
                    worst += p_units[j] * adverse
                else:
                    unreal += p_units[j] * (p_entry[j] - cl[t])
                    adverse = p_entry[j] - hi[t]
                    if adverse > 0.0:
                        adverse = 0.0
                    worst += p_units[j] * adverse
        eq_close[k] = balance + unreal
        eq_worst[k] = balance + worst
        if eq_worst[k] < day_min_equity:
            day_min_equity = eq_worst[k]

    if worst_day_loss < (day_start_equity - day_min_equity):
        worst_day_loss = day_start_equity - day_min_equity

    return (eq_close, eq_worst, bal_close,
            t_entry[:nt], t_exit[:nt], t_dir[:nt], t_pnl[:nt], t_r[:nt],
            t_spread[:nt], t_slip[:nt], t_swap[:nt], t_comm[:nt],
            n_skipped_heat, n_skipped_margin, n_adds, max_concurrent,
            peak_open_risk, worst_day_loss)


def simulate_contract(op, hi, lo, cl, atr, dow, sig, day, start, end, initial,
                      risk, rr, stop_mode, stop_value, exit_type, trail_mult,
                      partial_r, spread, slippage, swap, commission=0.0):
    """Backward-compatible single-position entry point.

    Returns the first 11 fields (equity, trades, costs) so existing callers and
    tests keep working.  Use :func:`simulate_contract_multi` for concurrency
    metrics and the additional counters.
    """
    out = simulate_contract_multi(
        op, hi, lo, cl, atr, dow, sig, day, start, end, initial, risk, rr,
        stop_mode, stop_value, exit_type, trail_mult, partial_r, spread,
        slippage, swap, commission, 1, 0, 0.0, 1.0, 0, 0.0, 100.0, 0.10,
        100.0, 0.01, 0.01, 0.0)
    return out[:SIM_N]


@njit(cache=True)
def evaluate_path(eq_close, eq_worst, bal_close, day, entry_flags, initial,
                  target1, target2, daily_dd, max_dd, min_days, fee, split,
                  payout_every, horizon_bars, funded_min_days,
                  daily_denom_mode):
    """Chain P1 -> P2 -> funded over one equity path.

    ``entry_flags[k]`` is True on a bar with an executed entry, so a trading
    day is a calendar day with at least one entry.  ``daily_denom_mode`` 0 uses
    the stricter ``min(initial, daily_ref)``; 1 uses ``daily_ref``.

    Returns ``(passed, n_payouts, net_profit, pass_bar, pass_day, fund_bar,
    calendar_days, daily_breach, max_breach, n_traded_days)``.
    """
    n = min(len(eq_close), horizon_bars)
    if n < 2:
        return (False, 0, -fee, -1, -1, -1, 0, False, False, 0)
    is_eod = np.empty(n, np.bool_)
    is_eod[:-1] = day[1:n] != day[:n - 1]
    is_eod[n - 1] = True

    phase = 0
    bal = initial - fee
    ref = initial
    eod_ref = initial
    traded = 0
    last_traded_day = -1
    last_day = -1
    peak = initial
    worst_dd = 0.0
    pass_bar = -1
    pass_day = -1
    fund_bar = -1
    daily_breach = False
    max_breach = False
    net = -fee
    refunded = False
    payouts = 0
    p_start = 0
    baseline = initial
    last_payout_bar = -1
    target = target1
    just_reset = False
    n_traded_total = 0

    for k in range(n):
        d = day[k]
        if d != last_day:
            traded += 0
            last_day = d
            ref = eod_ref
        eq = eq_close[k]
        w = eq_worst[k]
        bal = bal_close[k]
        if entry_flags[k] and d != last_traded_day:
            traded += 1
            n_traded_total += 1
            last_traded_day = d
        if w > peak:
            peak = w
        dd = (peak - w) / peak if peak > 0 else 0.0
        if dd > worst_dd:
            worst_dd = dd
        floor = initial * (1.0 - max_dd)
        if w <= floor:
            max_breach = True
            break
        denom = ref
        if daily_denom_mode == 0 and initial < denom:
            denom = initial
        if denom > 0 and (ref - w) >= daily_dd * denom:
            daily_breach = True
            break
        if phase < 2 and eq >= initial * (1.0 + target) and traded >= min_days:
            if phase == 0:
                pass_bar = k
                pass_day = d
                phase = 1
                target = target2
                bal = initial
                peak = initial
                traded = 0
                last_traded_day = -1
                last_day = -1
                ref = initial
                eod_ref = initial
                p_start = k + 1
                just_reset = True
            else:
                fund_bar = k
                phase = 2
                bal = initial
                peak = initial
                baseline = initial
                eod_ref = initial
                ref = initial
                traded = 0
                last_traded_day = -1
                last_day = -1
                p_start = k + 1
                just_reset = True
        elif (phase == 2 and (k - p_start) > 0
              and ((k - p_start) % payout_every) == 0
              and traded >= funded_min_days):
            profit = eq - baseline
            if profit > 0:
                net += profit * split
                payouts += 1
                if not refunded:
                    net += fee
                    refunded = True
                baseline = eq
                last_payout_bar = k
        if is_eod[k]:
            if just_reset:
                eod_ref = initial
                just_reset = False
            else:
                eod_ref = bal if bal > eq else eq

    if phase == 2 and (n - 1) > last_payout_bar:
        profit = eq_close[n - 1] - baseline
        if profit > 0:
            net += profit * split
            payouts += 1
            if not refunded:
                net += fee
                refunded = True
    passed = phase == 2
    calendar_days = (pass_day - day[0]) if pass_day >= 0 else horizon_bars
    return (passed, payouts, net, pass_bar, pass_day, fund_bar,
            calendar_days, daily_breach, max_breach, n_traded_total)


def entry_flags_from_trades(t_entry, start, horizon):
    """Build the per-bar entry mask used by :func:`evaluate_path`."""
    flags = np.zeros(horizon, dtype=np.bool_)
    for t in t_entry:
        k = int(t) - int(start)
        if 0 <= k < horizon:
            flags[k] = True
    return flags


def daily_pnl_series(eq_close, eq_worst, day, start, end, entry_flags=None):
    """Return per-calendar-day close/worst PnL relative to the prior close.

    The series is used by the §11 circular block bootstrap.  ``day`` must be
    the integer calendar-day array aligned with the equity arrays.
    ``entry_flags`` (per bar, optional) marks executed entries so the bootstrap
    can preserve the min-trading-days property of each sampled day.
    """
    n = end - start
    if n <= 0:
        return (np.empty(0, np.float64), np.empty(0, np.float64),
                np.empty(0, np.int32), np.empty(0, np.bool_))
    days = []
    close_pnl = []
    worst_pnl = []
    entry_days = []
    prev_close = eq_close[start]
    i = start
    while i < end:
        d = day[i]
        j = i
        while j + 1 < end and day[j + 1] == d:
            j += 1
        last_close = eq_close[j]
        day_worst = eq_worst[i]
        had_entry = False
        for k in range(i, j + 1):
            if eq_worst[k] < day_worst:
                day_worst = eq_worst[k]
            if entry_flags is not None and entry_flags[k]:
                had_entry = True
        days.append(int(d))
        close_pnl.append(float(last_close - prev_close))
        worst_pnl.append(float(day_worst - prev_close))
        entry_days.append(had_entry)
        prev_close = last_close
        i = j + 1
    return (np.asarray(close_pnl, np.float64),
            np.asarray(worst_pnl, np.float64),
            np.asarray(days, np.int32),
            np.asarray(entry_days, np.bool_))

# ===== INLINED STRATEGY + HELPERS (verbatim) =====
def _rsi(s, n=14):
    d = s.diff(); g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def _atr(df, n=14):
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(),
                    (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def strategy_23_supertrend_rsi(df):
    atr = _atr(df, 10); lo = ((df['High']+df['Low'])/2)-3*atr
    hi = ((df['High']+df['Low'])/2)+3*atr
    st = df['Close'] > lo; r = _rsi(df['Close'], 14)
    r20 = df['High'].rolling(20).max(); bo = (df['Close'] > r20.shift(1)) & st & (r > 60)
    l20 = df['Low'].rolling(20).min(); bd = (df['Close'] < l20.shift(1)) & ~st & (r < 40)
    sig = pd.Series(0, index=df.index); sig[bo] = 1; sig[bd] = -1
    return sig.shift(1).fillna(0).astype(int)

# ---------------------------------------------------------------- data/arrays
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return df


def prepare_arrays(df, atr_period=14):
    h = df["High"].to_numpy(np.float64)
    l = df["Low"].to_numpy(np.float64)
    c = df["Close"].to_numpy(np.float64)
    o = df["Open"].to_numpy(np.float64)
    pc = np.empty_like(c)
    pc[0] = c[0]
    pc[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.empty_like(c)
    atr[:] = np.nan
    if len(tr) >= atr_period:
        atr[atr_period - 1] = tr[:atr_period].mean()
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period
    day = (df["Date"].dt.tz_convert("UTC").dt.normalize()
           - pd.Timestamp("1970-01-01", tz="UTC")).dt.days.to_numpy(np.int32)
    return o, h, l, c, atr, day


def slice_bounds(dates, a, b):
    return (int(np.searchsorted(dates.values, np.datetime64(a))),
            int(np.searchsorted(dates.values, np.datetime64(b))))


# ---------------------------------------------------------------- one firm run
def run_firm(name, firm, cfg, arrays, sig, start, end, n_paths, seed):
    o, h, l, c, atr, dow, day = arrays
    smode, sval, etype, tr, pr = STOP_MODES[cfg["stop"]]
    heat = firm["heat_fraction"] * firm["daily_dd"] * firm["initial_balance"]
    lev = min(firm["leverage_eval"], firm["leverage_funded"])
    out = simulate_contract_multi(
        o, h, l, c, atr, dow, sig, day, start, end,
        firm["initial_balance"], cfg["risk"], cfg["rr"],
        smode, sval, etype, tr, pr,
        SPREAD, SLIPPAGE, SWAP, firm["commission_per_lot"],
        cfg["max_pos"], 0 if cfg["addon"] == "none" else 1,
        cfg["addon_trigger_r"], cfg["addon_size"], cfg["max_adds"],
        heat, lev, firm["margin_buffer"], firm["lot_size"],
        firm["min_lot"], firm["lot_step"], 0.0)
    eq, eqw, bal, t_en = out[0], out[1], out[2], out[3]
    flags = entry_flags_from_trades(t_en, start, end - start)
    pc, pw, days, edays = daily_pnl_series(eq, eqw, day[start:end], 0,
                                           end - start, flags)

    horizon = min(HORIZON_DAYS, len(pc))
    replay = evaluate_path(
        eq, eqw, bal, day[start:end], flags, firm["initial_balance"],
        firm["target1"], firm["target2"], firm["daily_dd"], firm["max_dd"],
        firm["min_days"], firm["fee"], firm["profit_split"],
        firm["payout_days"], end - start, firm["funded_min_days"], 0)

    rng = np.random.default_rng(seed)
    payouts, nets, pdays, passes, dbreach, mbreach = [], [], [], [], [], []
    for _ in range(n_paths):
        idx = []
        while len(idx) < horizon:
            s0 = int(rng.integers(0, len(pc)))
            idx.extend([(s0 + b) % len(pc) for b in range(BLOCK_DAYS)])
        idx = np.asarray(idx[:horizon], np.int64)
        b_pc, b_pw, b_ed = pc[idx], pw[idx], edays[idx]
        eqc = np.empty(horizon)
        eqworst = np.empty(horizon)
        run = firm["initial_balance"]
        for k in range(horizon):
            eqworst[k] = run + b_pw[k]
            run += b_pc[k]
            eqc[k] = run
        res = evaluate_path(
            eqc, eqworst, eqc, np.arange(horizon, dtype=np.int32), b_ed,
            firm["initial_balance"], firm["target1"], firm["target2"],
            firm["daily_dd"], firm["max_dd"], firm["min_days"], firm["fee"],
            firm["profit_split"], firm["payout_days"], horizon,
            firm["funded_min_days"], 0)
        passes.append(bool(res[0]))
        payouts.append(float(res[1]))
        nets.append(float(res[2]))
        pdays.append(float(res[6]) if res[0] else float(horizon))
        dbreach.append(bool(res[7]))
        mbreach.append(bool(res[8]))

    return {
        "firm": name,
        "challenge_fee": firm["fee"],
        "historical_pass": bool(replay[0]),
        "historical_payouts": int(replay[1]),
        "historical_net": float(replay[2]),
        "historical_pass_days": int(replay[6]),
        "mc_paths": int(n_paths),
        "mc_n_payouts": float(np.mean(payouts)),
        "mc_net_profit": float(np.mean(nets)),
        "mc_avg_days_to_pass": float(np.mean(pdays)),
        "mc_pass_rate": float(np.mean(passes)),
        "mc_daily_breach_rate": float(np.mean(dbreach)),
        "mc_max_breach_rate": float(np.mean(mbreach)),
        "trades": int(len(out[5])),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeframe", choices=["h1", "m15"], default="h1")
    p.add_argument("--data", default=None)
    p.add_argument("--slice", choices=["learn", "test", "full"], default="learn")
    p.add_argument("--stop", choices=sorted(STOP_MODES), default="fixed_25")
    p.add_argument("--rr", type=float, default=3.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--max-pos", type=int, default=2)
    p.add_argument("--addon", choices=["none", "pyramid"], default="none")
    p.add_argument("--paths", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    data = Path(args.data) if args.data else ROOT / "data" / (
        f"XAU_{args.timeframe.upper()}.csv")
    cfg = dict(stop=args.stop, rr=args.rr, risk=args.risk, max_pos=args.max_pos,
               addon=args.addon, addon_trigger_r=1.0, addon_size=0.5, max_adds=1)

    df = load_data(data)
    o, h, l, c, atr, day = prepare_arrays(df)
    dow = df.Date.dt.dayofweek.to_numpy(np.int32)
    arrays = (o, h, l, c, atr, dow, day)
    sig = pd.Series(strategy_23_supertrend_rsi(df), index=df.index).fillna(0)
    sig = sig.to_numpy(np.int8)

    if args.slice == "learn":
        s, e = slice_bounds(df.Date, "2026-01-01", "2026-10-31")
    elif args.slice == "test":
        s, e = slice_bounds(df.Date, "2016-01-01", "2026-01-01")
    else:
        s, e = 0, len(df)
    print(f"data={data.name} slice={args.slice} bars={e-s} "
          f"({df.Date.iloc[s]} -> {df.Date.iloc[e-1]})")
    print(f"config={cfg} paths={args.paths} seed={args.seed}\n")

    results = [run_firm(fid, firm, cfg, arrays, sig, s, e, args.paths, args.seed)
               for fid, firm in FIRMS.items()]
    for r in results:
        print(f"{r['firm']:13s} payouts={r['mc_n_payouts']:5.2f}  "
              f"net=${r['mc_net_profit']:8.0f}  "
              f"days={r['mc_avg_days_to_pass']:5.0f}  "
              f"pass={r['mc_pass_rate']*100:4.0f}%  "
              f"breach(D/M)={r['mc_daily_breach_rate']*100:.0f}/"
              f"{r['mc_max_breach_rate']*100:.0f}%  "
              f"hist_payouts={r['historical_payouts']}  trades={r['trades']}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"config": cfg, "slice": args.slice, "results": results}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
