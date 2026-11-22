"""Reconciler module — manual-close detection and fill-price PnL settlement."""
import logging
from datetime import datetime, timezone
from dataclasses import asdict

from engine.models import Trade
from engine.market import COMMISSION, CONTRACT_SIZE

logger = logging.getLogger(__name__)


def classify_exit(trade, price, tol=2.0):
    """Exit reason for parity: sl / tp / market, by closeness to the levels."""
    if price is None:
        return None
    get = trade.get if isinstance(trade, dict) else lambda k, d=None: getattr(trade, k, d)
    tp = get("tp"); sl = get("sl")
    try:
        if tp is not None and abs(float(price) - float(tp)) <= tol:
            return "tp"
        if sl is not None and abs(float(price) - float(sl)) <= tol:
            return "sl"
    except (TypeError, ValueError):
        pass
    return "market"


class Reconciler:
    """Handles position reconciliation and fill-price PnL settlement."""

    def __init__(self, trade_store=None, treasury_store=None):
        self.trade_store = trade_store
        self.treasury_store = treasury_store

    def reconcile(self, broker_positions, runners, pending_ids=None, on_fill=None):
        """Main reconcile entry point. Detects manually-closed positions.

        pending_ids: trade ids with an in-flight order (limit limits legitimately
        wait with no position_id — the post-pass must not close them).
        on_fill: callback(trade_id) invoked when a pending limit is adopted
        (lets the bot drop its pending-order entry)."""
        reconciled = 0
        settled = 0
        closed_stale = 0
        broker_open_pids = set()
        settled_pids = set()
        now_dt = datetime.now(timezone.utc)
        open_ids = set()
        position_map = {}
        pending = set(pending_ids) if pending_ids else set()

        for row in broker_positions:
            pos_id = row.get("id") or row.get("positionId") or row.get("ticket")
            if pos_id is None:
                continue
            comment = str(row.get("comment") or row.get("label") or row.get("commentId") or row.get("strategyId") or "").strip()
            strat_name, trade_id, ch_num = self._parse_comment(comment)
            if not strat_name:
                continue
            if strat_name not in runners:
                strat_name = self._resolve_runner_name(strat_name, runners)
            if strat_name not in runners:
                continue
            runner = runners[strat_name]
            ch = self._find_challenge(runner, ch_num)
            position_map[str(pos_id)] = (strat_name, ch_num)

            qty = self._to_float(row.get("qty") or row.get("quantity") or row.get("volume") or row.get("openQty"))
            lots = qty or 0.0
            side_raw = row.get("side")
            direction = 1 if str(side_raw).lower().startswith("buy") or str(side_raw) == "1" else -1
            open_price = self._to_float(row.get("openPrice") or row.get("price") or row.get("avgPrice"))
            close_price = self._to_float(row.get("closePrice"))
            status = str(row.get("status") or "").lower()
            sl = self._to_float(row.get("stopLoss") or row.get("sl"))
            tp = self._to_float(row.get("takeProfit") or row.get("tp"))
            pnl = self._to_float(row.get("pnl") or row.get("profit") or row.get("unrealizedPl"))
            open_ts_raw = row.get("openTime") or row.get("openDate") or row.get("createdDate")
            # Note: updatedAt is NOT treated as a close marker — open positions
            # may carry it (e.g. SL modifications touch updatedAt)
            is_closed = ("closed" in status or "close" in status
                         or close_price is not None)

            if is_closed:
                settled_pids.add(str(pos_id))
                if ch is None or not ch.open_trades:
                    continue
                # Find the matching open trade by position_id
                matched_trade = None
                for ot in ch.open_trades:
                    if ot.position_id is not None and str(ot.position_id) == str(pos_id):
                        matched_trade = ot
                        break
                if matched_trade is None:
                    continue
                matched_trade.pnl = round(pnl if pnl is not None else 0.0, 2)
                if close_price is not None:
                    matched_trade.exit_price = close_price
                    matched_trade.exit_reason = classify_exit(matched_trade, close_price)
                # Prefer the broker's actual close time over "when detected":
                # identical detection timestamps across trades made the
                # dashboard show simultaneous/delayed exits that never happened.
                close_dt = (self._parse_ts(row.get("closeTime") or row.get("closeDate")
                                           or row.get("closedDate")) or now_dt)
                matched_trade.close_time = close_dt.isoformat()
                matched_trade.status = "closed"
                closed_trade = matched_trade
                runner.on_trade_closed(matched_trade, close_dt, ch)
                if self.trade_store:
                    self.trade_store.save_trade(closed_trade)
                settled += 1
                continue

            open_ids.add(pos_id)
            broker_open_pids.add(str(pos_id))
            # Find matching open trade by position_id or trade_id
            if ch is not None and ch.open_trades:
                matched = False
                for ot in ch.open_trades:
                    if (ot.position_id is not None and str(ot.position_id) == str(pos_id)):
                        tf = runner.config.get("eval_tf", 60)
                        ot._hold_bars = self._compute_hold_bars(ch, open_ts_raw, tf)
                        self._sync_broker_sl(ot, sl)
                        if tp is not None and tp != 0:
                            ot.tp = tp
                        matched = True
                        break
                    if trade_id and ot.id == trade_id:
                        ot.position_id = pos_id
                        tf = runner.config.get("eval_tf", 60)
                        ot._hold_bars = self._compute_hold_bars(ch, open_ts_raw, tf)
                        self._sync_broker_sl(ot, sl)
                        if tp is not None and tp != 0:
                            ot.tp = tp
                        matched = True
                        break
                if matched:
                    # A pending limit order just filled: the position is open now
                    if getattr(ot, "status", None) == "pending":
                        ot.status = "open"
                    if on_fill is not None:
                        try:
                            on_fill(ot.id, sl)
                        except TypeError:
                            on_fill(ot.id)
                        except Exception:
                            pass
                    # Persist position_id (and broker sl/tp) so the DB row stays in
                    # sync with memory — otherwise _cleanup_orphan_rows later
                    # misidentifies this live trade as an orphan (pid NULL + open).
                    if self.trade_store:
                        self.trade_store.save_trade(ot)
                    continue
            if ch is None:
                ch = runner.state.current_challenge
                if ch is None:
                    continue
            db_row = self.trade_store.load_open_trade_by_id(trade_id) if trade_id and self.trade_store else None
            open_ts = self._parse_ts(open_ts_raw)
            if open_ts is not None:
                open_str = open_ts.isoformat()
            else:
                open_str = (db_row["open_time"] if db_row and db_row["open_time"]
                            else datetime.now(timezone.utc).isoformat())
            db_sl = db_row.get("sl") if db_row else None
            db_tp = db_row.get("tp") if db_row else None
            t = Trade(
                id=trade_id or f"reconciled-{pos_id}",
                strategy=strat_name, direction=direction,
                entry=round(open_price, 5) if open_price else 0.0,
                sl=round(sl, 5) if sl else (round(db_sl, 5) if db_sl else 0.0),
                tp=round(tp, 5) if tp else (round(db_tp, 5) if db_tp else 0.0),
                lots=lots, open_time=open_str, position_id=pos_id,
            )
            ch.open_trades.append(t)
            tf = runner.config.get("eval_tf", 60)
            t._hold_bars = self._compute_hold_bars(ch, open_str, tf)
            # Restore trail state from the persisted DB row (survives restarts)
            if db_row:
                if db_row.get("_sl_dist") and db_row["_sl_dist"] > 0:
                    t._sl_dist = db_row["_sl_dist"]
                if db_row.get("_trail_px") and db_row["_trail_px"] > 0:
                    t._trail_px = db_row["_trail_px"]
            if t._trail_px <= 0:
                t._trail_px = t.sl
            else:
                # Broker SL may be staler/looser than the persisted trail
                # (e.g. a failed trail push before a restart) — never roll back
                # past the trailed level; the next push re-syncs the broker.
                Reconciler._sync_broker_sl(t, t.sl)
            if self.trade_store:
                self.trade_store.save_trade(t)
            reconciled += 1

        # Post-pass: close trades that are no longer in broker's open set
        for name, runner in runners.items():
            for ch in runner.state.challenges:
                if ch.blown or not ch.open_trades:
                    continue
                for ot in list(ch.open_trades):
                    pid = ot.position_id
                    if ot.id in pending:
                        continue  # in-flight order (e.g. waiting limit) — not an orphan
                    if pid is not None and str(pid) in broker_open_pids:
                        continue
                    if pid is not None and str(pid) in settled_pids:
                        continue
                    if pid is not None:
                        ot.pnl = None
                        ot.close_time = now_dt.isoformat()
                        ot.status = "reconciled_unknown"
                        closed_trade = ot
                        runner.on_trade_closed(ot, now_dt, ch)
                        if self.trade_store:
                            self.trade_store.save_trade(closed_trade)
                        closed_stale += 1
                    elif pid is None:
                        try:
                            open_ts = datetime.fromisoformat(ot.open_time)
                            age = (datetime.now(timezone.utc) - open_ts).total_seconds()
                        except Exception:
                            age = 999
                        if age > 90:
                            ot.close_time = now_dt.isoformat()
                            ot.status = "reconciled_unknown"
                            runner.on_trade_closed(ot, now_dt, ch)
                            if self.trade_store:
                                self.trade_store.save_trade(ot)
                            closed_stale += 1

        return reconciled, settled, closed_stale, position_map

    def settle_manual_closes(self, order_rows, runners, position_map):
        """Settle PnL for trades closed outside the bot using fill prices."""
        updated = 0
        fills_by_pos = {}
        for row in order_rows:
            pos_id = row.get("positionId") or row.get("id")
            if pos_id is None:
                continue
            status = str(row.get("status") or "").lower()
            if status != "filled":
                continue
            qty = self._to_float(row.get("filledQty") or row.get("qty"))
            if not qty:
                continue
            price = self._to_float(row.get("avgPrice") or row.get("price"))
            if not price:
                continue
            side_raw = str(row.get("side") or "").lower()
            ts = self._parse_ts(row.get("createdDate") or row.get("lastModified"))
            fills_by_pos.setdefault(pos_id, []).append({
                "side": side_raw, "qty": qty, "price": price, "ts": ts,
                "strategyId": str(row.get("strategyId", "") or ""),
            })

        for pos_id, fills in fills_by_pos.items():
            fills.sort(key=lambda f: (f["ts"] or datetime.min.replace(tzinfo=timezone.utc)))

            def _realized(entry, direction):
                open_qty = 0.0
                open_cost = 0.0
                gross = 0.0
                closed_qty = 0
                last_close_ts = None
                close_notional = 0.0
                for f in fills:
                    fd = 1 if f["side"].startswith("buy") else -1
                    q = f["qty"]
                    p = f["price"]
                    if fd == direction:
                        open_qty += q
                        open_cost += p * q
                    else:
                        avg = (open_cost / open_qty) if open_qty > 0 else (entry or 0.0)
                        cq = min(q, open_qty) if open_qty > 0 else q
                        gross += (p - avg) * direction * cq
                        open_qty = max(open_qty - cq, 0.0)
                        open_cost = (avg * open_qty) if open_qty > 0 else 0.0
                        closed_qty += cq
                        close_notional += p * cq
                        last_close_ts = f["ts"]
                if closed_qty <= 0:
                    return None, None, None
                return (gross * CONTRACT_SIZE, last_close_ts,
                        close_notional / closed_qty)

            # Find matching open trade or trade_history entry
            matched = False
            for name, runner in runners.items():
                for ch in runner.state.challenges:
                    if ch.blown and not ch.open_trades:
                        continue
                    for ot in list(ch.open_trades):
                        if str(ot.position_id) == str(pos_id):
                            gross, last_ts, exit_px = _realized(ot.entry, ot.direction)
                            if gross is None:
                                break
                            close_dt = last_ts or datetime.now(timezone.utc)
                            ot.pnl = round(gross, 2)
                            ot.status = "closed"
                            ot.close_time = close_dt.isoformat()
                            if exit_px is not None:
                                ot.exit_price = exit_px
                                ot.exit_reason = classify_exit(ot, exit_px)
                            runner.on_trade_closed(ot, close_dt, ch)
                            if self.trade_store:
                                self.trade_store.save_trade(ot)
                            updated += 1
                            matched = True
                            break
                    if matched:
                        break
                    for t in ch.trade_history:
                        if str(t.get("position_id")) == str(pos_id):
                            gross, last_ts, exit_px = _realized(t.get("entry") or 0.0, t.get("direction", 1))
                            if gross is None:
                                break
                            close_dt = last_ts or datetime.now(timezone.utc)
                            lots = t.get("lots") or 0.0
                            net_pnl = round(gross, 2) - COMMISSION * lots
                            old_pnl = t.get("pnl") or 0
                            delta = net_pnl - (float(old_pnl) if old_pnl else 0.0)
                            already_settled = delta == 0 and str(t.get("status")) == "closed"
                            t["pnl"] = net_pnl
                            t["status"] = "closed"
                            if not already_settled:
                                ch.equity += delta
                                ch.ch_trade_pnl += delta
                                runner.state.trade_pnl += delta
                                if self.trade_store:
                                    self.trade_store.update_trade_pnl(
                                        t.get("id"), net_pnl, "closed",
                                        exit_price=exit_px,
                                        exit_reason=classify_exit(t, exit_px))
                                runner._refresh_qualifying_days(ch)
                                updated += 1
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break
        return updated

    @staticmethod
    def _sync_broker_sl(ot, sl):
        """Adopt the broker's SL into memory — but never LOOSEN a trailed stop.
        A stale broker SL (e.g. a failed trail push) must not roll memory back
        to the initial level; the next trail push re-syncs the broker."""
        if sl is None or sl == 0:
            return
        trail = float(getattr(ot, "_trail_px", 0) or 0)
        if trail > 0 and ((ot.direction == 1 and sl < trail)
                          or (ot.direction == -1 and sl > trail)):
            sl = trail
        ot.sl = sl

    @staticmethod
    def _parse_comment(comment: str) -> tuple:
        if not comment or "|" not in comment:
            return (None, None, None)
        parts = comment.split("|")
        strat = parts[0]
        tid = parts[1] if len(parts) > 1 else None
        ch_num = None
        if len(parts) > 2 and parts[2].startswith("ch_"):
            try:
                ch_num = int(parts[2][3:])
            except (ValueError, IndexError):
                pass
        return (strat, tid, ch_num)

    @staticmethod
    def _resolve_runner_name(name: str, runners: dict) -> str:
        if name in runners:
            return name
        for nm, rn in runners.items():
            cfg = rn.config
            if cfg.get("eval_strat") == name or cfg.get("funded_strat") == name or cfg.get("tag") == name:
                return nm
            if name in cfg.get("aliases", []):
                return nm
        return name

    @staticmethod
    def _find_challenge(runner, challenge_num):
        for c in runner.state.challenges:
            if c.challenge_num == challenge_num:
                return c
        return runner.state.current_challenge

    @staticmethod
    def _to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_ts(v):
        if v is None:
            return None
        try:
            if isinstance(v, (int, float)):
                ms = int(v)
                if ms > 10_000_000_000:
                    ms = ms // 1000
                dt = datetime.fromtimestamp(ms, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(v))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            # Sanity guard: reject nonsense epochs/years (e.g. '1789-...' misparses)
            if dt.year < 2000 or dt.year > 2100:
                return None
            return dt
        except Exception:
            return None

    @staticmethod
    def _compute_hold_bars(ch, open_ts_raw, tf_min=60):
        if not open_ts_raw:
            return 0
        try:
            entry = datetime.fromisoformat(str(open_ts_raw))
            elapsed = (datetime.now(timezone.utc) - entry).total_seconds()
            return max(0, int(elapsed // (tf_min * 60)))
        except Exception:
            return 0
