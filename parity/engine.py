"""Pure-Python port of the standalone Supertrend RSI backtest trade engine.

The standalone backtest runs a numba kernel. Docker images do not include
`numba`, so this module re-implements its trade engine without that dependency.

Faithful to the backtest kernel:
  - signal at bar t uses `sig[t-1]` (the backtest `shift(1)`s the pattern)
  - entry at `open[t]`; stop is either `atr_mult * ATR[t-1]` or a fixed
    dollar distance, target = `rr` × stop
  - gap exits at the open, then stop before target intrabar
  - max `max_pos` concurrent; when `addon_min_r` is set, a second position
    requires the best open R >= that threshold
  - cost `COST` USD/oz deducted at entry, risk = units×dist + cost

With `skip_below_min=True` an entry whose size floors below `min_units` is
skipped, matching the backtest's 0.01-lot minimum.
"""
import math

import numpy as np

from engine.indicators import build_indicator_dict
from engine.signals import SIGNAL_MAP

COST_PER_OZ = 0.35


def _f(x, default=0.0):
    try:
        v = float(x)
        return v if not math.isnan(v) else default
    except (TypeError, ValueError):
        return default


def run_trades(bars, signal_name, *, rr=3.0, atr_mult=3.0, risk_pct=0.01,
               stop_mode="atr", stop_value=25.0,
               max_pos=2, addon_min_r=None, initial=5000.0,
               cost=COST_PER_OZ, min_units=1, contract_size=100,
               partial_r=None, skip_below_min=False, start_index=1):
    """Simulate the archive kernel over `bars`.

    `bars` is a dict of equal-length sequences: open, high, low, close,
    volume, time (epoch seconds). Returns dict with `trades`, `final_balance`.

    `start_index` starts the trade loop at that bar with a FLAT state (no
    positions) while indicators still use the full warmup history. The parity
    checker uses it to align the backtest with a freshly reset live account:
    the backtest's open positions/balance at the live start are discarded.
    """
    op = [_f(x) for x in bars["open"]]
    hi = [_f(x) for x in bars["high"]]
    lo = [_f(x) for x in bars["low"]]
    cl = [_f(x) for x in bars["close"]]
    vol = [_f(x, 1.0) or 1.0 for x in bars.get("volume", [1] * len(op))]
    times = [int(x) for x in bars.get("time", [0] * len(op))]
    n = len(op)
    if n < 3 or signal_name not in SIGNAL_MAP:
        return {"trades": [], "final_balance": initial, "bars": n}

    ind = build_indicator_dict(np.asarray(op), np.asarray(hi), np.asarray(lo),
                               np.asarray(cl), np.asarray(vol),
                               np.asarray(times, dtype=np.int64))
    atr = ind.get("ATR14")
    sig_fn = SIGNAL_MAP[signal_name]
    sig = []
    for i in range(n):
        try:
            sig.append(int(sig_fn(ind, i)))
        except Exception:
            sig.append(0)

    balance = float(initial)
    positions = []
    trades = []

    def _close(pos, t, px, reason):
        nonlocal balance
        realized = pos["units"] * (px - pos["entry"]) * pos["dir"]
        balance += pos["partial_real"] + realized
        total = pos["partial_real"] + realized - pos["cost"]
        risk = pos["units0"] * pos["dist"] + pos["cost"]
        trades.append({
            "entry_time": pos["entry_time"],
            "exit_time": times[t],
            "dir": pos["dir"],
            "entry": pos["entry"],
            "exit": px,
            "units": pos["units0"],
            "lots": round(pos["units0"] / contract_size, 4),
            "pnl": round(total, 4),
            "r": round(total / risk, 4) if risk > 0 else 0.0,
            "risk": round(risk, 4),
            "exit_reason": reason,
        })

    start = max(1, int(start_index))
    for t in range(start, n):
        # ── manage open positions on bar t (script order: gap, stop, target) ──
        for pos in list(positions):
            d = pos["dir"]
            stop, target = pos["stop"], pos["target"]
            exited = None
            px = 0.0
            if d == 1:
                if op[t] <= stop:
                    px, exited = op[t], "gap_sl"
                elif op[t] >= target:
                    px, exited = op[t], "gap_tp"
                elif lo[t] <= stop:
                    px, exited = stop, "sl"
                elif partial_r is not None and pos["partial_real"] == 0.0 and hi[t] >= pos["entry"] + partial_r * pos["dist"]:
                    plevel = pos["entry"] + partial_r * pos["dist"]
                    take = pos["units"] * 0.5
                    pos["partial_real"] += take * (plevel - pos["entry"])
                    pos["units"] -= take
                    pos["stop"] = pos["entry"]
                    if lo[t] <= pos["entry"]:
                        px, exited = pos["entry"], "partial_be"
                    else:
                        px, exited = target, "partial_tp"
                elif hi[t] >= target:
                    px, exited = target, "tp"
            else:
                if op[t] >= stop:
                    px, exited = op[t], "gap_sl"
                elif op[t] <= target:
                    px, exited = op[t], "gap_tp"
                elif hi[t] >= stop:
                    px, exited = stop, "sl"
                elif partial_r is not None and pos["partial_real"] == 0.0 and lo[t] <= pos["entry"] - partial_r * pos["dist"]:
                    plevel = pos["entry"] - partial_r * pos["dist"]
                    take = pos["units"] * 0.5
                    pos["partial_real"] += take * (pos["entry"] - plevel)
                    pos["units"] -= take
                    pos["stop"] = pos["entry"]
                    if hi[t] >= pos["entry"]:
                        px, exited = pos["entry"], "partial_be"
                    else:
                        px, exited = target, "partial_tp"
                elif lo[t] <= target:
                    px, exited = target, "tp"
            if exited:
                _close(pos, t, px, exited)
                positions.remove(pos)

        # ── best open R at this bar's close (script `best_profit_R`) ──
        best_r = -1e18
        unreal_total = 0.0
        long_count = short_count = 0
        for pos in positions:
            unreal_total += pos["units"] * (cl[t] - pos["entry"]) * pos["dir"]
            risk = pos["units0"] * pos["dist"] + pos["cost"]
            ur = ((pos["units0"] * (cl[t] - pos["entry"]) * pos["dir"] + pos["partial_real"])
                  / risk) if risk > 0 else 0.0
            best_r = max(best_r, ur)
            if pos["dir"] == 1:
                long_count += 1
            else:
                short_count += 1
        open_count = len(positions)

        # ── entry: signal from the previous bar, fill at this bar's open ──
        s = sig[t - 1] if t > 1 else 0
        if s == 0:
            continue
        if stop_mode == "fixed":
            dist = float(stop_value)
        else:
            atr_prev = _f(atr[t - 1]) if atr is not None else 0.0
            if atr_prev <= 0:
                continue
            dist = atr_mult * atr_prev
        if dist <= 0:
            continue
        if open_count == 0:
            allow = True
        elif open_count < max_pos:
            allow = True if addon_min_r is None else best_r >= addon_min_r
        else:
            allow = False
        if not allow:
            continue
        cash = risk_pct * (balance + unreal_total)
        units = math.floor(cash / dist) if cash > 0 else 0
        if units < min_units:
            if skip_below_min:
                continue
            units = min_units
        units = float(units)
        cost_total = cost * units
        balance -= cost_total
        positions.append({
            "dir": 1 if s > 0 else -1,
            "entry": op[t],
            "stop": op[t] - (1 if s > 0 else -1) * dist,
            "target": op[t] + (1 if s > 0 else -1) * rr * dist,
            "units": units,
            "units0": units,
            "dist": dist,
            "cost": cost_total,
            "partial_real": 0.0,
            "entry_time": times[t],
        })

    return {"trades": trades, "final_balance": round(balance, 4), "bars": n}
