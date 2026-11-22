"""Live-vs-backtest trade comparison for the parity panel."""
from datetime import datetime, timezone

from engine.market import CONTRACT_SIZE

MATCH_BAR_SECONDS = 3600
MATCH_GRACE_SECONDS = 120  # live orders fill a few seconds after the bar open
ENTRY_TOL = 1.0
EXIT_TOL = 2.0
R_TOL = 0.15


def _epoch(ts):
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return int(ts)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (TypeError, ValueError):
        return None


def normalize_live(trades):
    """Runner trade dicts/Trade objects -> comparable dicts."""
    out = []
    for t in trades:
        get = t.get if isinstance(t, dict) else lambda k, d=None: getattr(t, k, d)
        status = str(get("status") or "").lower()
        if status != "closed" or get("pnl") is None:
            continue
        entry = get("entry")
        sl = get("sl")
        lots = get("lots") or 0.0
        try:
            risk = abs(float(entry) - float(sl)) * float(lots) * CONTRACT_SIZE
        except (TypeError, ValueError):
            risk = 0.0
        pnl = float(get("pnl") or 0.0)
        out.append({
            "id": get("id"),
            "dir": int(get("direction") or 0),
            "entry": float(entry or 0.0),
            "exit": get("exit_price"),
            "open_time": _epoch(get("open_time")),
            "close_time": _epoch(get("close_time")),
            "pnl": pnl,
            "risk": risk,
            "r": (pnl / risk) if risk > 0 else 0.0,
            "exit_reason": get("exit_reason"),
        })
    return [t for t in out if t["open_time"] is not None and t["dir"] in (1, -1)]


def compare_trades(live, bt, *, bar_seconds=MATCH_BAR_SECONDS,
                   grace_seconds=MATCH_GRACE_SECONDS,
                   entry_tol=ENTRY_TOL, exit_tol=EXIT_TOL, r_tol=R_TOL):
    """Greedy nearest-time match by direction, then price/R deltas.

    Live market orders are sent on the first poll after the bar open, so a
    legitimate one-bar difference is `bar_seconds + a few seconds`; the grace
    keeps those matches instead of reporting them as drift.
    """
    live = normalize_live(live)
    max_dt = bar_seconds + grace_seconds
    used = set()
    matched = []
    live_only = []
    for lt in live:
        best_i = None
        best_dt = None
        for i, bt_t in enumerate(bt):
            if i in used or bt_t["dir"] != lt["dir"]:
                continue
            dt = abs(bt_t["entry_time"] - lt["open_time"])
            if dt <= max_dt and (best_dt is None or dt < best_dt):
                best_i, best_dt = i, dt
        if best_i is None:
            live_only.append(lt)
            continue
        used.add(best_i)
        bt_t = bt[best_i]
        matched.append({
            "live": lt,
            "bt": bt_t,
            "entry_delta": round(lt["entry"] - bt_t["entry"], 4),
            "exit_delta": (round(float(lt["exit"]) - float(bt_t["exit"]), 4)
                           if lt["exit"] is not None else None),
            "r_delta": round(lt["r"] - bt_t["r"], 4),
        })
    bt_only = [t for i, t in enumerate(bt) if i not in used]

    # Expected divergence: the archive kernel's 1.5R partial is a same-bar
    # artifact (the 1R TP is closer, so a live broker fills the whole position
    # at 1R). The bot models the realizable exit; do not flag it as drift.
    expected = 0
    for m in matched:
        if str(m["bt"].get("exit_reason", "")).startswith("partial"):
            m["expected"] = True
            expected += 1

    # Direction mismatches: nearest bt trade by time (any direction)
    mismatches = 0
    for lt in live:
        near = min(bt, key=lambda b: abs(b["entry_time"] - lt["open_time"]), default=None)
        if near is not None and abs(near["entry_time"] - lt["open_time"]) <= max_dt:
            if near["dir"] != lt["dir"]:
                mismatches += 1

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "live_trades": len(live),
        "bt_trades": len(bt),
        "matched": len(matched),
        "live_only": len(live_only),
        "bt_only": len(bt_only),
        "signal_mismatches": mismatches,
        "mean_entry_delta": _mean([m["entry_delta"] for m in matched]),
        "mean_exit_delta": _mean([m["exit_delta"] for m in matched]),
        "mean_r_delta": _mean([m["r_delta"] for m in matched if not m.get("expected")]),
        "expected_divergence": expected,
        "matched_detail": matched,
        "live_only_detail": live_only,
        "bt_only_detail": bt_only,
        "grace_seconds": grace_seconds,
        "entry_tol": entry_tol,
        "exit_tol": exit_tol,
        "r_tol": r_tol,
    }


def build_report(strategy, firm, comparison, *, window=None, bars=0,
                 error=None):
    """Dashboard-ready report with an OK/DRIFT verdict."""
    c = comparison or {}
    matched = c.get("matched", 0)
    live_only = c.get("live_only", 0)
    bt_only = c.get("bt_only", 0)
    mean_r = c.get("mean_r_delta")
    total = max(matched + live_only, 1)
    unmatched = live_only + bt_only
    # A single unmatched trade is not a sample: never call it DRIFT on
    # unmatched counts alone (wait for a second trade). A single MATCHED trade
    # is still a valid comparison; a direction mismatch or R drift still flags.
    no_live = (error is None and c.get("live_trades", 0) < 2
               and matched == 0
               and c.get("signal_mismatches", 0) == 0
               and (mean_r is None or abs(mean_r) <= c.get("r_tol", R_TOL)))
    allowed = max(1, int(0.2 * total))
    # Small samples: with at least one confirmed match, tolerate one extra
    # unmatched pair instead of flagging a 2-trade comparison as drift.
    if matched >= 1 and total <= 3:
        allowed = max(allowed, 2)
    ok = (error is None and c.get("signal_mismatches", 0) == 0
          and unmatched <= allowed
          and (mean_r is None or abs(mean_r) <= c.get("r_tol", R_TOL)))
    status = "NO DATA" if no_live else ("OK" if ok else "DRIFT")
    detail = []
    for m in c.get("matched_detail", []):
        detail.append({
            "status": "matched",
            "id": m["live"].get("id"),
            "dir": m["live"]["dir"],
            "open_time": m["live"]["open_time"],
            "entry_delta": m["entry_delta"],
            "exit_delta": m["exit_delta"],
            "r_delta": m["r_delta"],
            "expected": bool(m.get("expected")),
        })
    for t in c.get("live_only_detail", []):
        detail.append({
            "status": "live_only",
            "id": t.get("id"),
            "dir": t["dir"],
            "open_time": t["open_time"],
            "exit_reason": t.get("exit_reason"),
        })
    for t in c.get("bt_only_detail", []):
        detail.append({
            "status": "bt_only",
            "dir": t["dir"],
            "open_time": t["entry_time"],
            "exit_reason": t.get("exit_reason"),
        })
    return {
        "strategy": strategy,
        "firm": firm,
        "status": status,
        "error": error,
        "last_run": datetime.now(timezone.utc).isoformat(),
        "window": window,
        "bars": bars,
        "live_trades": c.get("live_trades", 0),
        "bt_trades": c.get("bt_trades", 0),
        "matched": matched,
        "live_only": live_only,
        "bt_only": bt_only,
        "signal_mismatches": c.get("signal_mismatches", 0),
        "mean_entry_delta": c.get("mean_entry_delta"),
        "mean_exit_delta": c.get("mean_exit_delta"),
        "mean_r_delta": mean_r,
        "expected_divergence": c.get("expected_divergence", 0),
        "trades": detail[:100],
    }
