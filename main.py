#!/usr/bin/env python3
"""Main entry point for the trading engine — modular architecture.

Multi-broker support: each strategy routes to its own broker connection.
Bar pipeline, quote polling, order routing, and reconcile are per-broker.
"""
import asyncio
import collections
import logging
import sqlite3
import sys
import os
import time
import threading
import traceback
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from api.trade_locker_client import TradeLockerClient, TradeLockerError, RESOLUTIONS
from database.trade_store import TradeStore
from database.treasury_store import TreasuryStore
from engine.strategy_runner import StrategyRunner
from engine.reconciler import Reconciler, classify_exit
from engine.indicators import build_indicator_dict
from engine.market import (
    FARMS, SYMBOL, CONTRACT_SIZE, COMMISSION,
    POLL_INTERVAL, RECONCILE_POLL_EVERY, ORDER_HISTORY_POLL_EVERY,
    ORDER_HISTORY_FAST_RETRY,
    OPEN_ORDER_TIMEOUT, CLOSE_ORDER_TIMEOUT, SL_CHECK_INTERVAL,
    LOOKBACK_DAYS, STALE_SPOT_SECONDS, MAX_CAP, xau_market_open,
    PARITY_INTERVAL_SECONDS, PARITY_LOOKBACK_DAYS, PARITY_R_TOL,
)
from engine.signals import SIGNAL_MAP
from events.event_bus import EventBus
from events.websocket_broadcaster import WebSocketBroadcaster
from parity.engine import run_trades
from parity.checker import compare_trades, build_report, normalize_live

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Live (intrabar) trail ratchet: minimum seconds between broker SL pushes
INTRABAR_TRAIL_INTERVAL = 10


class _WSLogHandler(logging.Handler):
    """Pushes log records to the WS broadcaster so the dashboard can show them."""

    def __init__(self, broadcaster: WebSocketBroadcaster):
        super().__init__()
        self._broadcaster = broadcaster

    def emit(self, record):
        try:
            self._broadcaster.push_log(
                level=record.levelname,
                text=self.format(record),
            )
        except Exception:
            pass


def _env_value(key: str):
    """os.environ first, then the nearest .env (local `python main.py` runs)."""
    v = os.environ.get(key)
    if v not in (None, ""):
        return v
    try:
        from config.loader import _find_env_file, _parse_env_file
        path = _find_env_file()
        if path:
            return _parse_env_file(path).get(key)
    except Exception:
        pass
    return None


def _maybe_reset_state(data_dir: str) -> bool:
    """One-shot fresh start driven by RESET_STATE (e.g. RESET_STATE=1).

    Archives trades.db/treasuries.db under bot/data/archive_reset_<utc> and
    removes them so the next boot creates empty DBs (fresh challenges, zero
    treasury/PnL/history). Credentials, keys and tokens are KEPT.

    The flag is consumed via bot/data/.reset_state_done: the same value will not
    wipe again on a later restart. Delete that file (or use a new value) to
    reset a second time.
    """
    raw = _env_value("RESET_STATE")
    val = str(raw or "").strip()
    if val.lower() in ("", "0", "false", "no", "off"):
        return False
    os.makedirs(data_dir, exist_ok=True)
    marker = os.path.join(data_dir, ".reset_state_done")
    if os.path.exists(marker):
        try:
            with open(marker, encoding="utf-8") as fh:
                done = fh.read().strip()
        except OSError:
            done = ""
        if done == val:
            logger.warning(f"[reset] RESET_STATE={val} already applied — skipping "
                           f"(delete {marker} or use a new value to reset again)")
            return False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    archive_dir = os.path.join(data_dir, f"archive_reset_{stamp}")
    os.makedirs(archive_dir, exist_ok=True)
    archived, failed = [], False
    for name in ("trades.db", "treasuries.db"):
        src = os.path.join(data_dir, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(archive_dir, name)
        try:
            src_con = sqlite3.connect(src)
            try:
                dst_con = sqlite3.connect(dst)
                try:
                    src_con.backup(dst_con)
                finally:
                    dst_con.close()
            finally:
                src_con.close()
        except sqlite3.Error as e:
            logger.error(f"[reset] failed to archive {src}: {e}")
            failed = True
            continue
        for suffix in ("", "-wal", "-shm"):
            path = src + suffix
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as e:
                logger.error(f"[reset] failed to remove {path}: {e}")
                failed = True
        archived.append(name)
    if failed:
        logger.error("[reset] incomplete — marker not written; fix permissions and restart")
        return False
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(val)
    logger.warning(f"[reset] RESET_STATE={val}: fresh start — archived "
                   f"{archived or 'nothing'} to {archive_dir} (credentials/tokens kept)")
    return True


def _parse_strategy_brokers() -> dict:
    """Parse the optional STRATEGY_BROKERS override (tag:broker_key, comma-separated).

    Works from the .env file and os.environ. Absent -> {} (FARMS defaults used).
    """
    override = os.environ.get("STRATEGY_BROKERS")
    if not override:
        try:
            from config.loader import _find_env_file, _parse_env_file
            env_path = _find_env_file()
            if env_path:
                env_vars = _parse_env_file(env_path)
                override = env_vars.get("STRATEGY_BROKERS")
        except Exception:
            override = None
    out = {}
    if not override:
        return out
    from engine.market import FARMS
    known_tags = {f.get("tag") for f in FARMS}
    for pair in override.split(","):
        pair = pair.strip()
        if ":" in pair:
            tag, bkey = pair.split(":", 1)
            tag = tag.strip()
            out[tag] = bkey.strip()
            if tag not in known_tags:
                logging.getLogger(__name__).warning(
                    f"[config] STRATEGY_BROKERS tag '{tag}' matches no farm — ignored")
    return out


def _select_account(client, config):
    """Select ONE account from all_accounts() matching target id/acc_num/account.

    The `account` field accepts 'first'/'second'/... (word), '1'/'2'/... (1-based
    index), or an exact account id — one account per credentials, never all.
    Returns (account_id, acc_num).
    """
    try:
        accounts = client.all_accounts()
    except TradeLockerError as e:
        raise RuntimeError(f"all-accounts failed: {e}")
    if not accounts:
        raise RuntimeError("No accounts returned")
    target_id = config.get("account_id")
    target_acc = config.get("acc_num")
    target_account = config.get("account")
    chosen = None
    for a in accounts:
        if target_id is not None and str(a.get("id")) == str(target_id):
            chosen = a
            break
        if target_acc is not None and str(a.get("accNum")) == str(target_acc):
            chosen = a
            break
    if chosen is None and target_account is not None:
        s = str(target_account).strip().lower()
        for a in accounts:
            if s == str(a.get("id")).lower() or s == str(a.get("accNum")):
                chosen = a
                break
        if chosen is None:
            from config.credentials import normalize_account
            idx = normalize_account(target_account)
            if idx <= len(accounts):
                chosen = accounts[idx - 1]
                logger.info(f"Account field '{target_account}' -> index {idx} of {len(accounts)}")
            else:
                logger.warning(f"Account index {idx} out of range "
                               f"({len(accounts)} account(s)) — using first")
    if chosen is None:
        chosen = accounts[0]
        logger.info(f"No account match for id={target_id} accNum={target_acc} "
                    f"account={target_account} — using first")
    account_id = chosen.get("id")
    acc_num = chosen.get("accNum")
    if account_id is None:
        raise RuntimeError("Account missing 'id' in response")
    client.acc_num = acc_num
    status = chosen.get("status")
    if status and str(status).upper() != "ACTIVE":
        logger.warning(f"Account {account_id} status is {status!r} (not ACTIVE)")
    logger.info(f"Using account id={account_id} accNum={acc_num}")
    return account_id, acc_num


def _resolve_symbol(client, account_id):
    """Resolve XAUUSD instrument. Returns (instrument_id, route_id, trade_route_id, digits)."""
    try:
        instruments = client.instruments(account_id)
    except TradeLockerError as e:
        raise RuntimeError(f"instruments fetch failed: {e}")
    instrument_id = None
    route_id = None
    trade_route_id = None
    for inst in instruments or []:
        sym = str(_row_get(inst, "symbol", "tradableInstrumentName", "name") or "").upper()
        norm = sym[:-2] if sym.endswith(".X") else sym
        if norm == SYMBOL:
            instrument_id = inst.get("tradableInstrumentId") or inst.get("id")
            info_route = None
            trade_r = None
            for rt in inst.get("routes") or []:
                rid = rt.get("id") if isinstance(rt, dict) else rt
                rtype = str(rt.get("type", "")).upper() if isinstance(rt, dict) else ""
                if rtype == "INFO":
                    info_route = rid
                elif rtype == "TRADE":
                    trade_r = rid
            route_id = info_route or trade_r or inst.get("routeId")
            trade_route_id = trade_r or route_id
            break
    if instrument_id is None:
        raise RuntimeError(f"Symbol {SYMBOL} not found in instruments")
    digits = 2
    try:
        det = client.instrument_details(instrument_id, route_id)
        d = _to_float(_row_get(det, "digits", "priceDigits"))
        if d is not None:
            digits = int(d)
    except TradeLockerError as e:
        logger.warning(f"instrument_details failed ({e}) — defaulting digits=2")
    logger.info(f"Resolved {SYMBOL}: tradable={instrument_id} "
                f"info_route={route_id} trade_route={trade_route_id} "
                f"digits={digits}")
    return instrument_id, route_id, trade_route_id, digits


def _row_get(row, *candidates):
    if not row or not isinstance(row, dict):
        return None
    lower = {str(k).lower(): v for k, v in row.items()}
    for c in candidates:
        v = lower.get(c.lower())
        if v is not None and str(v) != "":
            return v
    return None


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_ts(v) -> Optional[datetime]:
    if v is None:
        return None
    try:
        if isinstance(v, (int, float)):
            ms = int(v)
            if ms > 10_000_000_000:
                ms = ms // 1000
            dt = datetime.fromtimestamp(ms, tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        # Broker timestamps can be naive ISO strings: normalize to UTC so fills
        # from different rows remain comparable when sorted.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt.year < 2000 or dt.year > 2100:
            return None
        return dt
    except Exception:
        return None


def _parse_epoch(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        ms = int(v)
        return ms // 1000 if ms > 10_000_000_000 else ms
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


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


class Bot:
    """Orchestrator that owns multiple broker clients, stores, runners, bar pipeline, and the poll loop."""

    def __init__(self, data_dir: str = "bot/data"):
        self.data_dir = data_dir
        _maybe_reset_state(data_dir)
        self.trade_store = TradeStore(db_path=os.path.join(data_dir, "trades.db"))
        self.treasury_store = TreasuryStore(db_path=os.path.join(data_dir, "treasuries.db"))
        self.reconciler = Reconciler(trade_store=self.trade_store)
        self.event_bus = EventBus()
        self.ws_broadcaster = WebSocketBroadcaster()
        self._log_queue: collections.deque = collections.deque(maxlen=500)

        # Multi-broker state: broker_key → {client, account_id, ...}
        self.brokers = {}

        # Uploaded credentials (1:1 model) — dormant until entries are uploaded
        self._credentials_store = None
        self._uploaded_credentials: dict = {}

        # Polling state
        self._poll_thread = None
        self._poll_running = False
        self._pending_orders = {}  # trade_id -> {runner_name, kind, sent_at}
        self._pending_signals = {}  # name -> {ch_num: (sig, atr, pattern_time, pattern_close)}
        self._close_attempt_ts = {}  # trade_id -> last close-attempt time (retry cooldown)
        self._orders_history_due = None  # fast settlement deadline after a close order
        self._close_lock = threading.Lock()  # serializes closes (poll thread vs 1s watchdog)
        self._broker_sl = {}  # trade_id -> SL armed at the broker (last successful push)
        self._last_trail_ts = {}  # trade_id -> ts of the last live (intrabar) trail push

        # Parity checker (live vs archive/lastly backtest)
        self.parity_reports = {}
        self._last_parity_run = 0.0
        self._parity_lock = threading.Lock()
        self.ws_broadcaster._parity_provider = lambda: self.parity_reports
        self.ws_broadcaster._parity_trigger = self._trigger_parity

        # Runners — FARMS copies (never mutate the global), with the optional
        # .env-driven strategy→broker override applied
        strat_brokers = _parse_strategy_brokers()
        self.runners = {}
        for farm in FARMS:
            f = dict(farm)
            tag = f.get("tag")
            if tag and tag in strat_brokers:
                f["broker_key"] = strat_brokers[tag]
            name = f["name"]
            runner = StrategyRunner(f, broadcast_fn=lambda nm=name: self._broadcast(nm), bot=self)
            self.runners[name] = runner

        # Load treasuries then trade history
        self.treasury_store.load(self.runners)
        for name, runner in self.runners.items():
            runner._load_trade_history(self.trade_store)
        self._save_treasuries()

        logger.info(f"Bot initialized with {len(self.runners)} strategies: {list(self.runners.keys())}")

    def _resolve_runner_name(self, name: str) -> str:
        if name in self.runners:
            return name
        for nm, rn in self.runners.items():
            cfg = rn.config
            if cfg.get("eval_strat") == name or cfg.get("funded_strat") == name or cfg.get("tag") == name:
                return nm
            if name in cfg.get("aliases", []):
                return nm
        return name

    def _find_challenge(self, runner, challenge_num: int):
        for c in runner.state.challenges:
            if c.challenge_num == challenge_num:
                return c
        return runner.state.current_challenge

    def _find_challenge_by_pos_id(self, pos_id: int, runner):
        for c in runner.state.challenges:
            for ot in c.open_trades:
                if str(ot.position_id) == str(pos_id):
                    return c
        return None

    def _compute_hold_bars(self, runner: StrategyRunner, ch, open_ts: str) -> int:
        if not open_ts:
            return 0
        try:
            tf_min = runner.config["eval_tf"] if ch.phase < 3 else runner.config["funded_tf"]
            entry = datetime.fromisoformat(open_ts)
            elapsed = (datetime.now(timezone.utc) - entry).total_seconds()
            return max(0, int(elapsed // (tf_min * 60)))
        except Exception:
            return 0

    def _on_tokens(self, env_prefix: str):
        """Return a callback that persists tokens for the given env_prefix."""
        def _callback(access_token: str, refresh_token: Optional[str]):
            from config.tokens import _persist_tokens
            _persist_tokens(env_prefix, access_token, refresh_token)
            logger.info(f"[TradeLocker] Persisted tokens for {env_prefix}")
        return _callback

    # ── Connect sequence ──────────────────────────────────────────────────
    def connect(self, all_configs: dict):
        """Connect to all brokers. all_configs: {broker_key: config_dict}."""
        # Start WS server
        if not self.ws_broadcaster._started:
            self.ws_broadcaster.start()
            _ws_handler = _WSLogHandler(self.ws_broadcaster)
            _ws_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logging.getLogger().addHandler(_ws_handler)
        self.ws_broadcaster.restart_fn = self._graceful_restart

        # Uploaded credentials (1:1 model): each strategy tag with stored
        # credentials gets its own broker config (dormant until uploaded)
        self._apply_uploaded_credentials(all_configs)

        # Collect all active timeframes from farms
        all_tfs = set()
        for farm in FARMS:
            all_tfs.add(farm["eval_tf"])
            all_tfs.add(farm["funded_tf"])

        for broker_key, cfg in all_configs.items():
            env_prefix = cfg.get("env_prefix", broker_key.upper())
            logger.info(f"[{broker_key}] Connecting to {cfg.get('server', '?')} "
                        f"as {cfg.get('email', '?')}...")

            client = TradeLockerClient(
                live=cfg.get("live", False),
                email=cfg.get("email"),
                password=cfg.get("password"),
                server=cfg.get("server"),
                access_token=cfg.get("access_token"),
                refresh_token=cfg.get("refresh_token"),
                on_tokens=self._on_tokens(env_prefix),
            )
            if not client.ensure_token():
                logger.error(f"[{broker_key}] Auth failed — skipping this broker")
                continue

            try:
                client.config()
            except TradeLockerError as e:
                logger.info(f"[{broker_key}] config fetch skipped (non-critical)")

            try:
                account_id, acc_num = _select_account(client, cfg)
            except RuntimeError as e:
                logger.error(f"[{broker_key}] Account selection failed: {e} — skipping")
                continue

            try:
                instrument_id, route_id, trade_route_id, digits = _resolve_symbol(client, account_id)
            except RuntimeError as e:
                logger.error(f"[{broker_key}] Symbol resolution failed: {e} — skipping")
                continue

            # Build bar buffers for this broker
            buffers = {
                tf: {"opens": [], "highs": [], "lows": [], "closes": [], "volumes": [], "times": []}
                for tf in all_tfs
            }
            current_bars = {tf: None for tf in all_tfs}
            last_bar_times = {tf: None for tf in all_tfs}

            self.brokers[broker_key] = {
                "client": client,
                "account_id": account_id,
                "acc_num": acc_num,
                "instrument_id": instrument_id,
                "route_id": route_id,
                "trade_route_id": trade_route_id,
                "price_digits": digits,
                "buffers": buffers,
                "current_bars": current_bars,
                "last_bar_times": last_bar_times,
                "live": cfg.get("live", False),
                "last_poll_time": None,
                "current_price": 0.0,
                "last_bid": None,
                "last_ask": None,
                "session_cooldown_until": 0,
            }

            # Fetch history for this broker
            self._request_history(broker_key)

            logger.info(f"[{broker_key}] Connected to "
                        f"{'LIVE' if cfg.get('live') else 'DEMO'} "
                        f"account {account_id}")

            # Brief pause between brokers to avoid TradeLocker rate limits
            time.sleep(3)

        if not self.brokers:
            raise RuntimeError("No brokers connected successfully")

        # Assign broker refs to runners (uploaded credentials override FARMS/env)
        self._assign_runners_to_brokers()

        # Account-change reset (1:1 model): a runner whose configured account
        # differs from its stored one starts fresh (new challenge = new account)
        for name, runner in self.runners.items():
            self._check_account_change(runner)

        # Initial quote poll + reconcile
        self._poll_quotes()
        self.reconcile()
        self._save_treasuries()
        self._broadcast()
        self._start_polling()
        logger.info(f"Connected to {len(self.brokers)} broker(s): {list(self.brokers.keys())}")

    def _request_history(self, broker_key: str):
        broker = self.brokers[broker_key]
        client = broker["client"]
        account_id = broker["account_id"]
        route_id = broker["route_id"]
        instrument_id = broker["instrument_id"]
        buffers = broker["buffers"]
        current_bars = broker.get("current_bars")
        last_bar_times = broker.get("last_bar_times")

        for period in sorted(buffers.keys()):
            resolution = RESOLUTIONS.get(period, "1H")
            now = time.time()
            to_ms = int(now * 1000)
            from_ms = to_ms - LOOKBACK_DAYS.get(period, 60) * 86400 * 1000
            bars = None
            for attempt in range(4):
                try:
                    bars = client.history(instrument_id, route_id,
                                          resolution, from_ms, to_ms)
                    break
                except TradeLockerError as e:
                    logger.warning(f"[{broker_key}] History attempt {attempt + 1}/4 failed "
                                   f"for {period}m: {str(e)[:160]}")
                    time.sleep(3.0 * (attempt + 1))
            if bars is None:
                logger.error(f"[{broker_key}] History fetch FAILED for {period}m after retries")
                continue
            now = time.time()  # refresh after retries (may cross a bar boundary)
            period_sec = period * 60
            current_start = int(now // period_sec) * period_sec
            closed = []
            in_progress = None
            seen = set()
            for b in bars or []:
                o = _to_float(_row_get(b, "open", "o"))
                h = _to_float(_row_get(b, "high", "h"))
                l = _to_float(_row_get(b, "low", "l"))
                c = _to_float(_row_get(b, "close", "c"))
                if o is None or h is None or l is None or c is None:
                    continue
                ts = _parse_epoch(_row_get(b, "time", "t") or b.get("timestamp"))
                if ts is None or ts in seen:
                    continue
                seen.add(ts)
                row = {
                    "open": o, "high": h, "low": l, "close": c,
                    "volume": _to_float(_row_get(b, "volume", "v")) or 1,
                    "time": ts,
                }
                # The history endpoint includes the current in-progress bar.
                # Keep it as the live current bar instead of putting it in the
                # buffer: otherwise the first quote poll starts a second bar at
                # the same timestamp and the first signal is evaluated on a
                # partial/duplicated bar (spurious entry at boot).
                if ts >= current_start:
                    in_progress = row
                else:
                    closed.append(row)
            closed.sort(key=lambda r: r["time"])
            buf = buffers[period]
            for k in ("opens", "highs", "lows", "closes", "volumes", "times"):
                buf[k] = []
            for row in closed[-300:]:
                buf["opens"].append(row["open"])
                buf["highs"].append(row["high"])
                buf["lows"].append(row["low"])
                buf["closes"].append(row["close"])
                buf["volumes"].append(row["volume"])
                buf["times"].append(row["time"])
            if current_bars is not None and last_bar_times is not None:
                cb = current_bars.get(period)
                if in_progress is not None:
                    # Do not replace an older unclosed bar (e.g. the hour
                    # rolled while the quote feed was down): let the next
                    # _on_quote close it first so no bar is lost.
                    if cb is None or cb["time"] >= current_start:
                        current_bars[period] = dict(in_progress)
                        last_bar_times[period] = in_progress["time"]
            n = len(buf["closes"])
            logger.info(f"[{broker_key}] Loaded {n} bars for {period}m timeframe.")
            if n < 220:
                logger.warning(f"[{broker_key}] Low bar count ({n}) for {period}m — "
                               f"indicators may produce NaN at startup")

    # ── Polling loop ──────────────────────────────────────────────────────
    def _start_polling(self):
        if self._poll_thread is not None:
            return
        self._poll_running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="poll")
        self._poll_thread.start()
        logger.info("Polling loop started.")
        if not getattr(self, "_sl_watch_thread", None):
            self._sl_watch_stop = threading.Event()
            self._sl_watch_thread = threading.Thread(
                target=self._sl_watch_loop, daemon=True, name="sl-watch")
            self._sl_watch_thread.start()
            logger.info(f"SL slippage watchdog started ({SL_CHECK_INTERVAL}s interval).")

    def _sl_watch_loop(self):
        """Stop-loss slippage watchdog: checks every second that open trades whose
        SL should have been hit are actually closed; force-exits those that aren't."""
        while not self._sl_watch_stop.wait(SL_CHECK_INTERVAL):
            try:
                self._sl_slippage_check()
            except Exception as e:
                logger.error(f"[SL-watch] check failed: {e}", exc_info=True)

    def _enforce_weekend_flat(self):
        """Backstop for the FP flat rule: any position still open during the
        Friday 20:00 UTC -> Monday 00:00 UTC window is flattened. Covers a
        restart between the 19:00 bar close and market close, and a failed
        close-order retry. Also drops any queued delayed-entry signal so it
        cannot fire at Sunday's market open."""
        now_dt = datetime.now(timezone.utc)
        if not xau_market_open(now_dt):
            return
        for runner in list(self.runners.values()):
            if not isinstance(runner.config, dict) or not runner.config.get("friday_flat"):
                continue
            if not runner.is_weekend_flat_window(now_dt, live=True):
                continue
            self._pending_signals.pop(runner.name, None)
            tag = runner.config.get("tag", runner.name)
            for ch in list(runner.state.active_challenges):
                if ch.blown or not ch.open_trades:
                    continue
                pending_entries = [
                    ot for ot in ch.open_trades
                    if getattr(ot, "status", None) == "pending"
                ]
                todo = [
                    ot for ot in ch.open_trades
                    if getattr(ot, "status", None) != "pending"
                    and f"{ot.id}_close" not in self._pending_orders
                ]
                if not pending_entries and not todo:
                    continue
                logger.warning(f"[{tag}] Weekend flat safety — flattening "
                               f"{len(todo)} filled + {len(pending_entries)} pending")
                self._cancel_pending_for_challenge(runner, ch)
                for ot in todo:
                    self._close_trade_at_market(ot, runner, ch)

    def _sl_slippage_check(self):
        if not xau_market_open(datetime.now(timezone.utc)):
            return
        now = time.time()
        for bkey, broker in self.brokers.items():
            price = broker.get("current_price", 0.0)
            if not price or price <= 0:
                continue
            # Stale feed (market just closed / poll stalled) — don't force-close
            # on prices that may be minutes old
            last_poll = broker.get("last_poll_time")
            if last_poll is not None and now - last_poll > STALE_SPOT_SECONDS:
                continue
            bid = broker.get("last_bid") or price
            ask = broker.get("last_ask") or price
            for runner in self.runners.values():
                if runner._broker_key != bkey:
                    continue
                for ch in runner.state.challenges:
                    if ch.blown or not ch.open_trades:
                        continue
                    # Backtest parity — the kernel ratchets the trail from the
                    # bar's BEST excursion and then checks the stop (numba:1906-
                    # 1933). Refreshing the trail intrabar from the live bar's
                    # running high/low reproduces that ordering instead of
                    # stopping out on a retrace against a stale (previous-bar)
                    # trail while the kernel would have followed the new low.
                    cb = None
                    if isinstance(runner.config, dict) and runner.config.get("use_trail"):
                        tf = runner.config["eval_tf"] if ch.phase < 3 else runner.config.get("funded_tf", 60)
                        cb = broker.get("current_bars", {}).get(tf)
                    for ot in list(ch.open_trades):
                        if ot.position_id is None:
                            continue  # not yet open on the broker
                        if cb is not None:
                            self._ratchet_live_trail(runner, ot, cb, bkey)
                        if ot.direction == 1:
                            should_hit = bid <= ot.sl
                            px = bid
                        else:
                            should_hit = ask >= ot.sl
                            px = ask
                        if should_hit:
                            if f"{ot.id}_close" in self._pending_orders:
                                continue  # close already requested — wait for it (no spam)
                            if self._stop_is_armed(ot):
                                # The broker stop sits exactly at this SL: it
                                # fills at the trail (kernel stop-fill model),
                                # no market dump needed.
                                continue
                            tag = runner.config.get('tag', runner.name) if isinstance(runner.config, dict) else runner.name
                            logger.warning(
                                f"[{tag}] SL breach not executed — forcing exit {ot.id} "
                                f"(sl={ot.sl:.2f} {'bid' if ot.direction == 1 else 'ask'}={px:.2f})")
                            self._close_trade_at_market(ot, runner, ch)
                    # Early close — intra-bar touch of the phase target at the
                    # LIVE price (kernel eq_best, numba:1693). SL first, then
                    # target: matches the kernel's per-bar order (blown → touch).
                    if (ch.phase in (1, 2)
                            and runner.config.get("early_close", True)):
                        now_dt = datetime.now(timezone.utc)
                        runner._check_early_close(ch, None, int(now_dt.timestamp() / 86400),
                                                  now_dt, bid=bid, ask=ask)

    def _ratchet_live_trail(self, runner, ot, bar, bkey):
        """Intrabar trail ratchet (kernel numba:1906-1933) — throttled: the stop
        is pushed to the broker at most every INTRABAR_TRAIL_INTERVAL seconds, so
        the armed stop follows the running best excursion of the live bar."""
        if ot.position_id is None:
            return
        last_ts = getattr(self, "_last_trail_ts", None)
        if last_ts is None:
            self._last_trail_ts = last_ts = {}
        if time.time() - last_ts.get(ot.id, 0) < INTRABAR_TRAIL_INTERVAL:
            return
        hi, lo = bar.get("high"), bar.get("low")
        if hi is None or lo is None:
            return
        try:
            new = runner._next_trail_px(ot, hi, lo)
        except Exception:
            return
        if new is None or abs(new - ot._trail_px) < 0.05:
            return
        digits = self.brokers.get(bkey, {}).get("price_digits", 2)
        ot._trail_px = new
        ot.sl = round(new, digits)
        runner._apply_trail_move(None, ot, new)
        last_ts[ot.id] = time.time()

    def _poll_loop(self):
        count = 0
        while self._poll_running:
            try:
                self._poll_once(count)
            except Exception as e:
                logger.error(f"[poll] error: {e}")
                traceback.print_exc()
            count += 1
            time.sleep(POLL_INTERVAL)

    def _poll_once(self, count):
        if not self.brokers:
            return
        self._enforce_weekend_flat()
        self._check_pending_orders()
        self._poll_quotes()
        due = getattr(self, "_orders_history_due", None)
        if due is not None and time.time() >= due:
            keys = self._pending_close_broker_keys()
            if keys:
                self._fetch_orders_history(keys)
                self._orders_history_due = time.time() + ORDER_HISTORY_FAST_RETRY
            else:
                self._orders_history_due = None
        self._broadcast()
        if count % RECONCILE_POLL_EVERY == 1:
            self.reconcile()
        if count % ORDER_HISTORY_POLL_EVERY == (ORDER_HISTORY_POLL_EVERY // 2) + 1:
            self._fetch_orders_history()
        if count % RECONCILE_POLL_EVERY == (RECONCILE_POLL_EVERY // 4) + 1:
            for bkey, broker in self.brokers.items():
                # Threshold BELOW the steady-state trim (300): the catchup only
                # fires when the buffer genuinely dropped (gap/refetch shortfall)
                short = [tf for tf, buf in broker["buffers"].items() if len(buf["closes"]) < 250]
                if short:
                    logger.info(f"[{bkey}] [history-catchup] refetching {sorted(short)}")
                    self._request_history(bkey)
        if count % 12 == 0:
            self._log_heartbeat()
        if time.time() - self._last_parity_run >= PARITY_INTERVAL_SECONDS:
            self._run_parity_all()

    def _check_pending_orders(self):
        now = time.time()
        now_dt = datetime.now(timezone.utc)
        # Fast adoption: a limit that fills between reconciles (up to 5 min)
        # must become a live trade immediately — the dashboard otherwise shows
        # the entry minutes late and the stale order SL is never re-synced.
        try:
            self._adopt_filled_limits()
        except Exception as e:
            logger.warning(f"[pending] fill adoption failed: {e}")
        stale = []
        for key, info in list(self._pending_orders.items()):
            if info.get("cancel_retry_at", 0) > now:
                continue  # cancel retry backoff (order confirmed still live)
            if info.get("kind") == "limit_open" and not xau_market_open(now_dt):
                # The fill window counts TRADE bars (kernel fills w ∈ [p+2..p+4];
                # after a weekend the next bars are Sunday's) — pause the
                # deadline across the closure so the limit survives the gap
                # exactly like the backtest's next bars.
                ref = info.setdefault("closed_ref", now)
                info["sent_at"] += max(now - ref, 0.0)
                info["closed_ref"] = now
                continue
            info.pop("closed_ref", None)
            age = now - info["sent_at"]
            timeout = info.get("timeout") or (
                CLOSE_ORDER_TIMEOUT if info.get("kind") == "close" else OPEN_ORDER_TIMEOUT)
            if age > timeout:
                logger.warning(f"[{info['runner_name']}] Pending {info.get('kind')} {key} "
                               f"unresolved for {age:.0f}s — forcing reconcile")
                stale.append((key, info))
        for key, info in stale:
            if info.get("kind") == "limit_open":
                # Cancel FIRST; drop the entry only once the broker call
                # resolved. A failed cancel must be retried (popping here
                # used to orphan a live GTC order with no retry).
                if self._cancel_limit_order(key, info):
                    self._pending_orders.pop(key, None)
            else:
                self._pending_orders.pop(key, None)
        if stale:
            self.reconcile()

    def _order_still_live(self, broker, trade_id, order_id):
        """After an ambiguous cancel 404, ask the broker's ACTIVE orders whether
        the order still exists. Returns True (live), False (gone), None (unknown).

        A 404 does NOT prove the order is gone: a still-resting GTC order can
        return "not found" from the cancel route and fill hours later (2026-10-15:
        FPZRO-ch1-195 was declared expired at 06:00 and filled at 19:52)."""
        client = broker.get("client")
        account_id = broker.get("account_id")
        if client is None:
            return None
        try:
            if not getattr(client, "columns", {}).get("orders"):
                try:
                    client.config()
                except TradeLockerError:
                    pass
            if not getattr(client, "columns", {}).get("orders"):
                return None
            rows = client.orders(account_id) or []
        except TradeLockerError as e:
            logger.warning(f"[pending] active-orders lookup failed: {e}")
            return None
        for o in rows:
            oid = o.get("id") or o.get("orderId") or o.get("order_id")
            sid = str(o.get("strategyId") or o.get("comment") or o.get("label") or "")
            if (order_id is not None and oid is not None and str(oid) == str(order_id)):
                return True
            if trade_id and trade_id in sid:
                return True
        return False

    def _cancel_limit_order(self, trade_id, info) -> bool:
        """An unfilled limit order timed out: cancel it at the broker, discard the
        entry, free the runner's slot. The broker cancel happens even if the trade
        row is gone (an orphaned GTC order must not survive); a trade that FILLED
        (position_id set) is never discarded — the reconciler adopted it live.
        Returns True when the pending entry may be dropped, False when the broker
        cancel failed and the caller should retry next cycle."""
        runner = self.runners.get(info.get("runner_name"))
        if runner is None:
            logger.warning(f"[pending] limit cancel for {trade_id}: unknown runner "
                           f"{info.get('runner_name')}")
            return True
        tag = runner.config.get('tag', runner.name) if isinstance(runner.config, dict) else runner.name
        trade = None
        for c in runner.state.challenges:
            for t in c.open_trades:
                if t.id == trade_id:
                    trade = t
                    break
            if trade is not None:
                break
        broker = runner._broker
        # Always attempt the broker cancel (the GTC order must not survive)
        cancel_gone = False
        if broker is not None and info.get("order_id"):
            try:
                broker["client"].cancel_order(broker["account_id"], info["order_id"])
                logger.info(f"[{tag}] Limit order {trade_id} unfilled after wait — "
                            f"cancelled at broker (order_id={info['order_id']})")
            except TradeLockerError as e:
                msg = str(e)
                if "404" in msg or "not found" in msg.lower():
                    # 404 is AMBIGUOUS: the order either expired/purged at the
                    # broker, FILLED, or is STILL RESTING (the cancel route can
                    # miss a live GTC order). Never free the slot on a guess:
                    # check the broker's ACTIVE orders first.
                    still_live = self._order_still_live(broker, trade_id, info.get("order_id"))
                    if still_live:
                        info["cancel_retry_at"] = time.time() + 30
                        logger.warning(
                            f"[{tag}] Limit order {trade_id}: cancel returned 404 but the "
                            f"order is STILL LIVE at the broker — keeping it tracked "
                            f"(retrying cancel in 30s)")
                        return False
                    cancel_gone = True
                    logger.info(f"[{tag}] Limit order {trade_id}: cancel returned 404 — "
                                f"order no longer exists as an order (filled or purged) — "
                                f"checking live positions...")
                else:
                    logger.warning(f"[{tag}] Limit cancel failed ({trade_id}): {e} — "
                                   f"retrying next cycle (order stays tracked)")
                    return False
        if trade is None:
            logger.info(f"[{tag}] Limit cancel for {trade_id}: no tracked trade "
                        f"(already filled/adopted or discarded) — pending entry dropped")
            return True
        if trade.position_id is not None:
            # The limit FILLED and the reconciler adopted the position live —
            # never discard a managed, open position
            logger.info(f"[{tag}] Limit order {trade_id} FILLED before cancel "
                        f"(position_id={trade.position_id}) — keeping live trade, "
                        f"pending entry dropped")
            return True
        # Disambiguate the 404: a live position with this trade's strategyId means
        # the limit FILLED (late, after the wait) — the position must be kept and
        # the slot NOT freed (freeing it lets a new signal double-book the account)
        if cancel_gone and broker is not None:
            filled_pid = self._find_position_by_trade_id(broker, tag, trade_id)
            if filled_pid is not None:
                trade.position_id = filled_pid
                trade.status = "open"
                if self.trade_store:
                    self.trade_store.save_trade(trade)
                logger.info(
                    f"[{tag}] Limit {trade_id} FILLED before expiry (late fill) — "
                    f"position {filled_pid} adopted live, entry KEPT (row restored "
                    f"to open), slot NOT freed — max_concurrent preserved")
                return True
        # The limit genuinely never filled: log the full picture
        broker_px = (broker or {}).get("current_price", 0.0)
        side_txt = "BUY (pullback below)" if trade.direction == 1 else "SELL (pullback above)"
        logger.info(
            f"[{tag}] Limit {trade_id} EXPIRED UNFILLED after {runner.config.get('limit_wait', 3)} "
            f"bars | limit was {side_txt} at {trade.entry:.2f}, price at expiry "
            f"{broker_px:.2f} — the market never reached the limit. Entry cancelled, "
            f"row marked failed (no trade occurred).")
        self._discard_failed_entry(
            runner, trade,
            f"expired_unfilled: limit {side_txt} @ {trade.entry:.2f} never reached "
            f"(price at expiry {broker_px:.2f}, waited "
            f"{runner.config.get('limit_wait', 3)} bars)")
        return True

    def _poll_quotes(self):
        first = True
        for bkey, broker in self.brokers.items():
            if not first:
                time.sleep(0.5)
            first = False
            client = broker["client"]
            try:
                d = client.quotes(broker["instrument_id"], broker["route_id"])
            except TradeLockerError:
                continue
            bid = _to_float(_row_get(d, "bid", "bp"))
            ask = _to_float(_row_get(d, "ask", "ap"))
            price = _to_float(_row_get(d, "price", "mid", "midPrice"))
            if bid is not None and ask is not None:
                broker["last_bid"] = bid
                broker["last_ask"] = ask
                mid = (bid + ask) / 2.0
            elif price is not None:
                mid = price
                broker["last_bid"] = price
                broker["last_ask"] = price
            else:
                continue
            broker["current_price"] = round(mid, 2)
            broker["last_poll_time"] = time.time()
            self._on_quote(mid, bkey)

    def _log_heartbeat(self):
        for bkey, broker in self.brokers.items():
            stale = ""
            if broker["last_poll_time"] is not None:
                age = time.time() - broker["last_poll_time"]
                if age > STALE_SPOT_SECONDS:
                    stale = f" (feed stale {age:.0f}s)"
            logger.info(f"[{bkey}] [HEARTBEAT] price={broker['current_price']:.2f} "
                        f"pending_orders={len(self._pending_orders)}{stale}")

    # ── Bar pipeline ──────────────────────────────────────────────────────
    def _on_quote(self, mid: float, broker_key: str):
        if not xau_market_open(datetime.now(timezone.utc)):
            return
        broker = self.brokers[broker_key]
        now = int(time.time())
        for period in sorted(broker["buffers"].keys()):
            bar_period = period * 60
            bar_start = (now // bar_period) * bar_period
            cb = broker["current_bars"][period]
            lbt = broker["last_bar_times"][period]

            if cb is None or bar_start != lbt:
                if cb is not None:
                    self._close_bar(period, cb, broker_key)
                new_bar = {
                    "open": mid, "high": mid, "low": mid,
                    "close": mid, "volume": 1, "time": bar_start,
                }
                # No history bar was available for this hour (history fetch
                # happened just after the boundary, or the endpoint omitted the
                # in-progress bar): the bar starts mid-hour and its OHLC is not
                # comparable to the backtest's full bar. Mark it so the close
                # path drops it instead of generating a signal from a partial.
                if now - bar_start > 60:
                    new_bar["_partial"] = True
                broker["current_bars"][period] = new_bar
                broker["last_bar_times"][period] = bar_start
                # Consume pending delayed-entry signals for this broker's runners
                for name, runner in self.runners.items():
                    if runner._broker_key != broker_key:
                        continue
                    consume_dt = datetime.fromtimestamp(bar_start, tz=timezone.utc)
                    if (isinstance(runner.config, dict)
                            and runner.config.get("friday_flat")
                            and runner.is_weekend_flat_window(consume_dt)):
                        dropped = self._pending_signals.pop(name, None)
                        if dropped:
                            tag = runner.config.get("tag", name)
                            logger.info(f"[{tag}] Weekend flat window — dropping "
                                        f"queued signal(s) at bar open")
                        continue
                    pending_map = self._pending_signals.pop(name, None)
                    if not pending_map:
                        continue
                    for ch_num, pending in list(pending_map.items()):
                        sig, atr_val, pattern_time, pattern_close = pending
                        ch = None
                        for c in runner.state.challenges:
                            if c.challenge_num == ch_num:
                                ch = c
                                break
                        if ch is None:
                            continue
                        target_tf = runner.config["eval_tf"] if ch.phase < 3 else runner.config["funded_tf"]
                        if period != target_tf:
                            self._pending_signals.setdefault(name, {})[ch_num] = pending
                            continue
                        # Backtest parity: the pattern closes bar p; the kernel's
                        # SHIFTED signal fires at p+1, anchoring the limit at
                        # C[p+1] with fills on p+2..p+4. O[p+2] ≈ C[p+1], so
                        # consume only at the start of bar p+2 (pattern + 2 bars).
                        # pattern_time == 0 marks a session-rejected re-queue
                        # (already gated at storage) → ready immediately.
                        delay = int(runner.config.get("entry_delay_bars", 2))
                        if pattern_time and bar_start < pattern_time + delay * bar_period:
                            self._pending_signals.setdefault(name, {})[ch_num] = pending
                            continue
                        if time.time() < broker.get("session_cooldown_until", 0):
                            # Session cooldown — keep the signal pending, retry next bar
                            self._pending_signals.setdefault(name, {})[ch_num] = pending
                            continue
                        if sig == 0 or ch.blown or len(ch.open_trades) >= runner._max_concurrent():
                            continue
                        # Add-on gate (script parity): with a position already
                        # open, a new entry requires the best open trade to be
                        # >= addon_min_r (script: best_profit_R >= 0.5).
                        addon_min_r = runner.config.get("addon_min_r")
                        if addon_min_r is not None and ch.open_trades:
                            best_r = runner._best_open_r(ch, pattern_close)
                            if best_r < addon_min_r:
                                tag = runner.config.get('tag', name)
                                logger.info(f"[{tag}] Add-on skipped: best open R={best_r:.2f} "
                                            f"< {addon_min_r:.2f}")
                                continue
                        # Entry ATR = the PATTERN bar's ATR (kernel exact):
                        # scripts build ATRV[0]=nan, ATRV[1:]=A[:-1] and fire
                        # at the shifted index i=p+1, so atrv[i] = A[p] — the
                        # tuple's stored pattern ATR. (_last_bar_atr = A[p+1]
                        # is one bar too new; it is only correct for trailing.)
                        entry_atr = atr_val
                        if runner.config.get("stop_mode") != "fixed":
                            if not entry_atr or np.isnan(entry_atr) or entry_atr <= 0:
                                continue
                        trade = runner._compute_entry(sig, {"open": mid, "time": bar_start},
                                                      entry_atr, ch, ref_price=pattern_close)
                        if trade is not None:
                            self._place_order_sync(trade, runner, ch, entry_atr)
            else:
                b = broker["current_bars"][period]
                b["high"] = max(b["high"], mid)
                b["low"] = min(b["low"], mid)
                b["close"] = mid
                b["volume"] += 1

    def _close_bar(self, period: int, bar: dict, broker_key: str):
        if bar.get("_partial"):
            dt_str = datetime.fromtimestamp(bar["time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            logger.warning(f"[{broker_key}] {period}m bar at {dt_str} started mid-hour "
                           f"after boot — partial bar dropped (no signal/buffer append)")
            return
        broker = self.brokers[broker_key]
        buf = broker["buffers"][period]
        buf["opens"].append(bar["open"])
        buf["highs"].append(bar["high"])
        buf["lows"].append(bar["low"])
        buf["closes"].append(bar["close"])
        buf["volumes"].append(bar["volume"])
        buf["times"].append(bar["time"])
        for k in buf:
            buf[k] = buf[k][-300:]

        r = build_indicator_dict(buf["opens"], buf["highs"], buf["lows"],
                                 buf["closes"], buf["volumes"], buf["times"])
        _atr_last = r["ATR14"][-1]
        if hasattr(_atr_last, "item"):
            _atr_last = _atr_last.item()  # numpy scalar -> python float (numpy 2.5+ safe)
        atr_val = 0.0 if np.isnan(_atr_last) else float(_atr_last)

        dt_str = datetime.fromtimestamp(bar["time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        bar_dt = datetime.fromtimestamp(bar["time"], tz=timezone.utc)
        weekday = bar_dt.weekday()
        hour = bar_dt.hour
        minute = bar_dt.minute
        is_weekend = (
            (weekday == 4 and (hour > 21 or (hour == 21 and minute >= 5)))
            or weekday == 5
            or (weekday == 6 and (hour < 22 or (hour == 22 and minute < 10)))
        )
        if is_weekend:
            logger.debug(f"[{broker_key}] Weekend guard — skipping bar at {dt_str}")
            return
        logger.info(f"[{broker_key}] [Bar] {period}m at {dt_str} "
                    f"O={bar['open']:.2f} H={bar['high']:.2f} L={bar['low']:.2f} C={bar['close']:.2f} "
                    f"buf_len={len(buf['closes'])}")

        # Only process runners assigned to this broker
        for name, runner in self.runners.items():
            if runner._broker_key != broker_key:
                continue
            try:
                self._process_runner_bar(name, runner, bar, atr_val, period, r)
            except Exception as e:
                logger.error(f"[{name}] bar-close processing failed: {e}", exc_info=True)

    def _process_runner_bar(self, name, runner, bar, atr_val, period, r):
        # Seed trailing's previous-bar ATR on the first bar after a restart:
        # on_bar_close assigns _last_bar_atr AFTER the trailing update, so the
        # first processed bar would otherwise fall back to the current bar's
        # ATR (backtest trailing uses A[j-1]).
        if getattr(runner, "_last_bar_atr", None) is None:
            a14 = r.get("ATR14")
            if a14 is not None and len(a14) >= 2:
                try:
                    prev = float(a14[-2])
                    if not np.isnan(prev) and prev > 0:
                        runner._last_bar_atr = prev
                except (TypeError, ValueError):
                    pass

        runner.on_bar_close(bar, atr_val, period)

        bar_dt = datetime.fromtimestamp(bar["time"], tz=timezone.utc)
        if (isinstance(runner.config, dict) and runner.config.get("friday_flat")
                and runner.is_weekend_flat_window(bar_dt)):
            dropped = self._pending_signals.pop(name, None)
            if dropped:
                tag = runner.config.get("tag", name)
                logger.info(f"[{tag}] Weekend flat window — dropping delayed "
                            f"signal(s); no entry before/through weekend")
            return

        max_conc = runner.config.get("max_concurrent", 1)
        for ch in list(runner.state.active_challenges):
            if ch.blown or len(ch.open_trades) >= max_conc:
                continue
            target_tf = runner.config["eval_tf"] if ch.phase < 3 else runner.config["funded_tf"]
            if period != target_tf:
                continue
            strat_name = runner.config["eval_strat"] if ch.phase < 3 else runner.config["funded_strat"]
            sig_fn = SIGNAL_MAP[strat_name]
            sig_idx = len(r["C"]) - 1
            sig = sig_fn(r, sig_idx)
            eff = ch.equity
            tag = runner.config.get('tag', name)
            logger.info(f"[{tag}] Ch#{ch.challenge_num} P{ch.phase} | "
                        f"sig={sig:+d} ATR={atr_val:.2f} eq={eff:.2f} "
                        f"open={len(ch.open_trades)}/{max_conc} pnl={ch.ch_trade_pnl:+.2f}")

            if runner.config.get("use_delayed_signal"):
                if sig != 0:
                    pmap = self._pending_signals.setdefault(name, {})
                    if ch.challenge_num not in pmap:
                        # (sig, pattern-bar ATR A[p], pattern bar start) —
                        # consumed at the start of bar pattern+2: the scripts
                        # shift(1) the signal, so backtest_core_limit fires at
                        # i = p+1, anchors the pullback limit at C[p+1] ≈ O[p+2],
                        # uses atrv[i] = A[p] (ATRV[1:] = A[:-1]), and fills on
                        # w ∈ [p+2..p+4]
                        pmap[ch.challenge_num] = (sig, atr_val, bar["time"], bar["close"])
                    else:
                        # Keep the older pending signal (the kernel resolves the
                        # earlier attempt first); the newer pattern is dropped —
                        # a broker-side overlap of two live limits is impossible
                        # with max_concurrent=1
                        logger.debug(f"[{tag}] Pending signal already queued — dropping newer pattern")
                continue
            if sig == 0:
                continue
            trade = runner._compute_entry(sig, bar, atr_val, ch)
            if trade is not None:
                self._place_order_sync(trade, runner, ch, atr_val)

    # ── Order execution ───────────────────────────────────────────────────
    def _place_order_sync(self, trade, runner, ch=None, atr_val=0.0):
        if ch is None:
            ch = runner.state.current_challenge
        if (isinstance(runner.config, dict) and runner.config.get("friday_flat")
                and runner.is_weekend_flat_window(
                    datetime.now(timezone.utc), live=True)):
            tag = runner.config.get("tag", trade.strategy)
            logger.info(f"[{tag}] Weekend flat window — entry blocked for {trade.id}")
            self._discard_failed_entry(runner, trade, "weekend_flat_window")
            return
        broker = runner._broker
        if broker is None:
            logger.error(f"[{trade.strategy}] TRADE FAILED: no broker assigned")
            self._discard_failed_entry(runner, trade, "no_broker: runner has no broker connection")
            return
        if time.time() < broker.get("session_cooldown_until", 0):
            tag = runner.config.get('tag', trade.strategy)
            remaining = int(broker["session_cooldown_until"] - time.time())
            logger.info(f"[{tag}] Skipping order — session cooldown ({remaining}s left)")
            self._discard_failed_entry(runner, trade,
                                       f"session_cooldown: broker in 30min session cooldown, "
                                       f"{remaining}s remaining")
            return
        client = broker["client"]
        instrument_id = broker["instrument_id"]
        trade_route_id = broker["trade_route_id"]
        route_id = broker["route_id"]
        account_id = broker["account_id"]
        price_digits = broker["price_digits"]

        if instrument_id is None:
            logger.error(f"[{trade.strategy}] TRADE FAILED: symbol not resolved")
            self._discard_failed_entry(runner, trade, "symbol_not_resolved")
            return
        side = "buy" if trade.direction == 1 else "sell"
        use_limit = bool(runner.config.get("use_limit_entry")) if isinstance(runner.config, dict) else False
        strategy_id = f"{runner.config.get('tag') or trade.strategy}|{trade.id}|ch_{ch.challenge_num}"
        if len(strategy_id) > 31:
            logger.error(f"[{trade.strategy}] TRADE FAILED: strategyId {len(strategy_id)} chars "
                         f"exceeds 31-char cap: {strategy_id}")
            self._discard_failed_entry(runner, trade,
                                       f"strategyId too long ({len(strategy_id)} > 31 chars)")
            return
        # The broker lot step is 0.01 — a float artifact (0.7000000000000001)
        # is rejected ("order amount is not multiple of lot step") and the
        # signal is lost. Round here too (defensive: entries are rounded at
        # sizing) and keep the row in sync with the qty actually sent.
        qty = round(float(trade.lots), 2)
        if qty < 0.01:
            self._discard_failed_entry(runner, trade,
                                       f"invalid_lots: qty={trade.lots} rounds to 0.00")
            return
        trade.lots = qty
        body = {
            "tradableInstrumentId": instrument_id,
            "routeId": trade_route_id or route_id,
            "type": "limit" if use_limit else "market",
            "side": side,
            "qty": qty,
            "validity": "GTC" if use_limit else "IOC",
            "price": round(trade.entry, price_digits) if use_limit else 0,
            "stopLoss": round(trade.sl, price_digits),
            "stopLossType": "absolute",
            "takeProfit": round(trade.tp, price_digits),
            "takeProfitType": "absolute",
            "strategyId": strategy_id,
        }
        mode = "DEMO" if not broker.get("live") else "LIVE"
        logger.info(f"[{trade.strategy}] Order body: side={side} qty={qty} "
                    f"sl={body['stopLoss']} tp={body['takeProfit']} strategyId='{strategy_id}'")
        try:
            order_id = client.place_order(account_id, body)
            # The order was placed WITH this SL — a limit carries it into the
            # position on fill (the reconciler re-syncs it on adoption).
            self._remember_broker_sl(trade.id, body["stopLoss"])
            logger.info(f"[{trade.strategy}] Order sent ({mode}): {trade.id} "
                        f"{'BUY' if trade.direction == 1 else 'SELL'} {trade.lots}L "
                        f"@ ~{trade.entry:.3f} SL:{trade.sl:.3f} TP:{trade.tp:.3f} order_id={order_id}")
            if use_limit:
                # Limit orders legitimately wait limit_wait bars before fill —
                # exact kernel window: placed at p+2 start, fills through p+4,
                # expires when p+5 opens (no wall-clock padding).
                lim_wait_bars = runner.config.get("limit_wait", 3)
                tf = runner.config.get("eval_tf", 60)
                limit_timeout = lim_wait_bars * tf * 60
                logger.info(f"[{trade.strategy}] Limit order {trade.id} waiting up to "
                            f"{lim_wait_bars} bars for maker fill")
            else:
                limit_timeout = None
            self._pending_orders[trade.id] = {
                "runner_name": runner.name,
                "kind": "limit_open" if use_limit else "open",
                "sent_at": time.time(),
                "order_id": order_id,
                "timeout": limit_timeout,
            }
        except TradeLockerError as e:
            err_msg = str(e)
            if "forbidden" in err_msg.lower() or "session" in err_msg.lower():
                broker["session_cooldown_until"] = time.time() + 1800
                tag = runner.config.get('tag', trade.strategy)
                logger.info(f"[{tag}] Session closed — cooldown 30min ({trade.id} rejected)")
                # Re-queue the signal so the entry is re-attempted at the next bar
                # start after the cooldown instead of being lost. pattern_time=0
                # marks it ready immediately (already past its original gates).
                self._pending_signals.setdefault(runner.name, {})[ch.challenge_num] = (
                    trade.direction, atr_val, 0, trade.entry)
                self._discard_failed_entry(
                    runner, trade,
                    f"order_rejected_session: broker session closed ({err_msg[:120]}) — "
                    f"signal re-queued for retry after cooldown")
            else:
                logger.error(f"[{trade.strategy}] Order send FAILED: {e}")
                logger.info(
                    f"[{trade.strategy}] Trade {trade.id} did NOT reach the broker | "
                    f"details: {trade.direction:+d} {trade.lots}L @ {trade.entry:.2f} "
                    f"SL {trade.sl:.2f} TP {trade.tp:.2f} | error: {err_msg[:200]}")
                traceback.print_exc()
                self._discard_failed_entry(
                    runner, trade,
                    f"order_send_failed: {err_msg[:200]}")

    def _discard_failed_entry(self, runner, trade, reason: str = None):
        """Remove a failed entry from open_trades and mark its DB row 'failed'
        (with the reason) so stale open rows don't pollute history/winrate."""
        for c in runner.state.challenges:
            if any(t.id == trade.id for t in c.open_trades):
                c.open_trades = [t for t in c.open_trades if t.id != trade.id]
                break
        if self.trade_store:
            self.trade_store.update_trade_pnl(trade.id, None, "failed", reason)
        tag = runner.config.get('tag', runner.name) if isinstance(runner.config, dict) else runner.name
        logger.info(
            f"[{tag}] Trade {trade.id} marked FAILED and removed from open_trades | "
            f"reason: {reason or 'unspecified'} | "
            f"details: {trade.direction:+d} {trade.lots}L @ {trade.entry:.2f} "
            f"SL {trade.sl:.2f} TP {trade.tp:.2f} (open_time {trade.open_time}) — "
            f"no trade occurred, PnL impact: none")

    def _close_guard(self):
        """The close check-and-send must be atomic: the 5s poll thread and the 1s
        SL watchdog both call it, and a duplicated DELETE races the first close."""
        lock = getattr(self, "_close_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._close_lock = lock
        return lock

    def _remember_broker_sl(self, trade_id, sl):
        """Record the SL the broker is known to carry for this trade."""
        if not hasattr(self, "_broker_sl"):
            self._broker_sl = {}
        if sl is None:
            return
        try:
            self._broker_sl[trade_id] = float(sl)
        except (TypeError, ValueError):
            pass

    def _forget_trade_stop(self, trade_id):
        if trade_id is None:
            return
        getattr(self, "_broker_sl", {}).pop(trade_id, None)
        getattr(self, "_last_trail_ts", {}).pop(trade_id, None)

    def _stop_is_armed(self, trade):
        """True when the broker's stop was successfully set to the trade's CURRENT
        SL (trail). Then the broker fills at the trail (kernel stop-fill model) and
        a market close at detection would only make the exit worse."""
        armed = getattr(self, "_broker_sl", {}).get(trade.id)
        try:
            return armed is not None and abs(float(armed) - float(trade.sl)) <= 0.01
        except (TypeError, ValueError):
            return False

    def _close_trade_at_market(self, trade, runner, ch=None):
        if ch is None:
            ch = runner.state.current_challenge
        broker = runner._broker
        if broker is None or not ch.open_trades:
            return
        client = broker["client"]
        account_id = broker["account_id"]
        with self._close_guard():
            if f"{trade.id}_close" in self._pending_orders:
                logger.debug(f"[{trade.strategy}] Close already requested for {trade.id} — skipping")
                return
            # Per-trade close-retry cooldown: the watchdog (1s thread) must not hammer
            # close_position when an attempt already succeeded/failed — the pending
            # dedupe above only covers *successful* requests until their 45s timeout
            last_attempt = self._close_attempt_ts.get(trade.id, 0)
            if time.time() - last_attempt < 15:
                return
            self._close_attempt_ts[trade.id] = time.time()
            pos_id = trade.position_id
            if pos_id is None:
                # Never run a full reconcile from the watchdog thread (it would
                # race the poll thread's reconcile). One targeted lookup instead.
                tag = runner.config.get('tag', runner.name) if isinstance(runner.config, dict) else runner.name
                pos_id = self._find_position_by_trade_id(broker, tag, trade.id)
                if pos_id is None:
                    logger.warning(f"[{trade.strategy}] Close requested but position_id unknown "
                                   f"(no live broker position for {trade.id})")
                    return
                trade.position_id = pos_id
                trade.status = "open"
                if self.trade_store:
                    self.trade_store.save_trade(trade)
            try:
                client.close_position(pos_id, qty=trade.lots)
                logger.info(f"[{trade.strategy}] Close order sent: pos_id={pos_id} qty={trade.lots}")
                self._pending_orders[f"{trade.id}_close"] = {
                    "runner_name": runner.name,
                    "kind": "close",
                    "sent_at": time.time(),
                }
                # Settle the fill-price PnL on the next poll instead of waiting
                # for the 5-minute orders-history cadence.
                self._orders_history_due = time.time()
            except TradeLockerError as e:
                self._handle_close_error(trade, runner, ch, pos_id, e)

    def _handle_close_error(self, trade, runner, ch, pos_id, e):
        """Classify a failed close: quantity mismatch (retry qty=0), already gone
        (settle as reconciled_unknown immediately — no retry spam), session
        closed (30min cooldown), anything else (log)."""
        client = runner._broker["client"]
        msg = str(e)
        low = msg.lower()
        if "quantity" in low:
            # A previous close partially filled / the position is already
            # (partly) closed — retry once with qty=0 to close the remainder
            try:
                client.close_position(pos_id, qty=0)
                logger.info(f"[{trade.strategy}] Close retried with qty=0 after "
                            f"quantity mismatch ({trade.id}): {e}")
                self._pending_orders[f"{trade.id}_close"] = {
                    "runner_name": runner.name,
                    "kind": "close",
                    "sent_at": time.time(),
                }
                self._orders_history_due = time.time()
            except TradeLockerError as e2:
                logger.warning(f"[{trade.strategy}] Close retry (qty=0) also failed "
                               f"({trade.id}): {e2} — position likely already closed")
            return
        if "no such position" in low or "position not found" in low:
            # The broker position is already gone (stop filled / manual close):
            # book it as a placeholder for fill-price settlement and stop the
            # watchdog from hammering close_position every 15s.
            self._pending_orders.pop(f"{trade.id}_close", None)
            self._close_attempt_ts.pop(trade.id, None)
            self._forget_trade_stop(trade.id)
            self._orders_history_due = time.time()
            trade.pnl = None
            trade.status = "reconciled_unknown"
            trade.close_time = datetime.now(timezone.utc).isoformat()
            runner.on_trade_closed(trade, datetime.now(timezone.utc), ch)
            if self.trade_store:
                self.trade_store.save_trade(trade)
            logger.info(f"[{trade.strategy}] {trade.id} position already gone at the "
                        f"broker — placeholder close, PnL pending fill-price settlement")
            return
        if "forbidden" in low or "session" in low:
            runner._broker["session_cooldown_until"] = time.time() + 1800
            logger.info(f"[{trade.strategy}] Close order rejected — session closed, "
                        f"cooldown 30min ({trade.id}): {e}")
            return
        logger.error(f"[{trade.strategy}] Close order FAILED ({trade.id}, "
                     f"pos_id={pos_id}): {e}")

    def push_trail_sl_to_broker(self, runner, ot):
        """Push the trailed SL to the broker (order-modify) so the stop fills at the
        trail level, matching the backtest's stop-fill model. Without this the trail
        is memory-only and exits book the market price at detection."""
        broker = runner._broker
        if broker is None or ot.position_id is None:
            return
        client = broker["client"]
        try:
            client.modify_position(ot.position_id, stop_loss=ot.sl)
            self._remember_broker_sl(ot.id, ot.sl)
            logger.info(f"[{ot.strategy}] Broker SL updated to {ot.sl:.2f} ({ot.id})")
        except TradeLockerError as e:
            msg = str(e)
            if "forbidden" in msg.lower() or "session" in msg.lower():
                broker["session_cooldown_until"] = time.time() + 1800
                logger.info(f"[{ot.strategy}] Broker SL update skipped — session closed ({ot.id})")
            else:
                logger.warning(f"[{ot.strategy}] Broker SL update failed ({ot.id}): {e}")

    # ── Uploaded credentials (1:1 model) ──────────────────────────────────

    def _get_credentials_store(self):
        if self._credentials_store is None:
            try:
                from config.credentials import CredentialsStore
                self._credentials_store = CredentialsStore(
                    db_path=os.path.join(self.data_dir, "credentials.db"))
            except Exception as e:
                logger.error(f"[credentials] store init failed: {e}")
                return None
        return self._credentials_store

    def _apply_uploaded_credentials(self, all_configs: dict):
        """Build one broker config per strategy tag that has uploaded credentials
        (the 1:1 model). Dormant until credentials are uploaded via the dashboard."""
        store = self._get_credentials_store()
        if store is None:
            return
        try:
            uploaded = store.load_all()
        except Exception as e:
            logger.error(f"[credentials] load failed: {e}")
            return
        self._uploaded_credentials = uploaded
        if not uploaded:
            return
        active_tags = {f.get("tag") for f in FARMS}
        for tag, cred in uploaded.items():
            if tag not in active_tags:
                # Retired strategy: keep the stored credentials, but do not
                # build a broker (the 1:1 model only trades active FARMS tags).
                logger.info(f"[credentials] {tag} is not an active farm — skipped")
                continue
            bkey = f"cred_{tag}"
            if bkey in all_configs:
                continue
            all_configs[bkey] = {
                "email": cred.get("email"),
                "password": cred.get("password"),
                "server": cred.get("server"),
                "account": cred.get("account", 1),
                "env_prefix": tag,
                "live": False,
                "broker": f"cred_{tag}",
            }
            logger.info(f"[credentials] broker config ready for {tag} "
                        f"(account index {cred.get('account', 1)})")

    def _assign_runners_to_brokers(self):
        """Assign broker refs to runners. Uploaded credentials override FARMS/env:
        a strategy tag with stored credentials routes to its own cred_{tag} broker."""
        uploaded_tags = set(self._uploaded_credentials.keys())
        for name, runner in self.runners.items():
            tag = runner.config.get("tag") or name
            if tag in uploaded_tags:
                if runner.config.get("broker_key") != f"cred_{tag}":
                    logger.info(f"[{name}] routed to uploaded-credentials broker "
                                f"(cred_{tag}) — 1:1 model")
                runner.config["broker_key"] = f"cred_{tag}"
            bkey = runner.config.get("broker_key")
            if bkey and bkey in self.brokers:
                runner._broker = self.brokers[bkey]
                runner._broker_key = bkey
            else:
                logger.warning(f"[{name}] No broker found for broker_key={bkey}")
            self._apply_uploaded_balance(runner, tag)

    def _apply_uploaded_balance(self, runner, tag):
        """Per-credential balance override (1:1 model). The profile is a
        per-runner copy, so the FARMS defaults stay untouched. A blank
        challenge cost scales the profile fee by balance / base balance."""
        cred = self._uploaded_credentials.get(tag) or {}
        balance = cred.get("balance")
        if balance in (None, ""):
            return
        try:
            balance = float(balance)
        except (TypeError, ValueError):
            return
        if balance <= 0:
            return
        base_ib = runner.profile.get("ib", balance)
        base_cc = runner.profile.get("cc", 0.0)
        runner.profile["ib"] = balance
        cost = cred.get("challenge_cost")
        if cost not in (None, ""):
            try:
                runner.profile["cc"] = float(cost)
            except (TypeError, ValueError):
                runner.profile["cc"] = base_cc
        elif base_ib > 0:
            runner.profile["cc"] = round(base_cc * (balance / base_ib), 2)

    def _check_account_change(self, runner):
        """If a runner's configured account differs from its stored one, reset it
        fresh (new challenge = new account; dormant while the account is unchanged)."""
        if not self.treasury_store:
            return
        broker = runner._broker
        configured = str(broker.get("account_id")) if broker and broker.get("account_id") else None
        if configured is None:
            return
        stored = self.treasury_store.get_account_id(runner.name)
        reset = False
        if stored is None:
            logger.info(f"[{runner.name}] stored account persisted ({configured})")
            self.treasury_store.set_account_id(runner.name, configured)
        elif str(stored) != configured:
            logger.warning(f"[{runner.name}] Account changed ({stored} -> {configured}) "
                           f"— resetting runner state fresh")
            reset = True
            self.treasury_store.set_account_id(runner.name, configured)
        # Balance change (uploaded credential) also starts the runner fresh —
        # the challenge's DD/targets are sized off the account balance.
        tag = runner.config.get("tag") or runner.name
        cred = self._uploaded_credentials.get(tag) or {}
        configured_bal = cred.get("balance")
        if configured_bal not in (None, ""):
            try:
                configured_bal = float(configured_bal)
            except (TypeError, ValueError):
                configured_bal = None
        if configured_bal and configured_bal > 0:
            stored_bal = self.treasury_store.get_account_balance(runner.name)
            ch = runner.state.current_challenge
            if stored_bal is not None and abs(float(stored_bal) - configured_bal) > 0.01:
                logger.warning(f"[{runner.name}] Balance changed ({stored_bal} -> "
                               f"{configured_bal}) — resetting runner state fresh")
                reset = True
            elif stored_bal is None and ch and abs(ch.phase_start_bal - configured_bal) > 0.01:
                logger.warning(f"[{runner.name}] Balance differs from challenge "
                               f"({ch.phase_start_bal} -> {configured_bal}) — resetting")
                reset = True
            self.treasury_store.set_account_balance(runner.name, configured_bal)
        if reset:
            self._reset_runner_state(runner)

    def _reset_runner_state(self, runner):
        """Fresh state for a runner: new challenge, treasury/stats/history zeroed.
        Old trade rows stay in trades.db as history; the trade-id counter is
        already reseeded from all rows at boot (no id collisions)."""
        for ch in runner.state.challenges:
            ch.open_trades = []
        runner.state.challenges = []
        runner.state.trade_pnl = 0.0
        runner.state.blown_pnl = 0.0
        runner.state.total_spent = 0.0
        runner.state.total_earned = 0.0
        runner.state.monthly_stats = {}
        runner.state.daily_pnl = {}
        runner._add_new_challenge()

    def _graceful_restart(self):
        """Dashboard-requested restart: save state, stop cleanly, exit so the
        container (restart: unless-stopped) brings the bot back with the new config."""
        logger.info("Restart requested via dashboard — saving state and exiting")
        try:
            self._save_treasuries()
            self.close()
        except Exception as e:
            logger.error(f"[restart] cleanup failed: {e}")
        os._exit(0)

    def _find_position_by_trade_id(self, broker, tag: str, trade_id: str):
        """Find a live broker position whose strategyId carries this trade id.
        Returns the position id, or None."""
        if broker is None:
            return None
        client = broker["client"]
        account_id = broker["account_id"]
        try:
            if not client.columns.get("positions"):
                client.config()
            positions = client.positions(account_id) or []
        except TradeLockerError as e:
            logger.warning(f"[{tag}] position lookup failed (fill disambiguation): {e}")
            return None
        for p in positions:
            sid = str(p.get("comment") or p.get("strategyId") or "")
            if trade_id in sid:
                return p.get("id") or p.get("positionId")
        return None

    def _enforce_max_concurrent(self):
        """Safety net: a challenge holding MORE than max_concurrent trades gets its
        extras force-closed (the newest first — entries that only exist because a
        slot was wrongly freed). Pending extras are cancelled, filled extras are
        closed at market. Runs every reconcile cycle."""
        for name, runner in self.runners.items():
            if not isinstance(runner.config, dict):
                continue
            max_conc = runner.config.get("max_concurrent", 1)
            for ch in runner.state.challenges:
                if ch.blown or len(ch.open_trades) <= max_conc:
                    continue
                # Sort by open_time DESC: the newest entries are the extras
                extras = sorted(ch.open_trades,
                                key=lambda t: t.open_time or "",
                                reverse=True)[:len(ch.open_trades) - max_conc]
                for ot in extras:
                    if ot.status == "pending":
                        logger.warning(
                            f"[{runner.name}] Ch#{ch.challenge_num} over concurrent limit "
                            f"({len(ch.open_trades)}/{max_conc}) — cancelling newest PENDING "
                            f"limit {ot.id} (entered {ot.open_time})")
                        info = self._pending_orders.pop(ot.id, None)
                        if info:
                            self._cancel_limit_order(ot.id, info)
                        else:
                            self._discard_failed_entry(
                                runner, ot, "over_concurrent: cancelled as newest pending extra")
                    else:
                        logger.warning(
                            f"[{runner.name}] Ch#{ch.challenge_num} over concurrent limit "
                            f"({len(ch.open_trades)}/{max_conc}) — force-closing newest "
                            f"extra {ot.id} (entered {ot.open_time}, "
                            f"uPnL tracks live) — restoring 1-concurrent state")
                        self._close_trade_at_market(ot, runner, ch)

    def _cancel_pending_for_challenge(self, runner, ch):
        """Cancel any live limit orders for this challenge before a force-close
        (news blackout / DD blow): a pending limit must not open exposure the
        force-close is trying to remove."""
        ids = [tid for tid, info in self._pending_orders.items()
               if info.get("runner_name") == runner.name
               and info.get("kind") == "limit_open"
               and any(t.id == tid for t in ch.open_trades)]
        for tid in ids:
            info = self._pending_orders.pop(tid)
            logger.info(f"[{runner.name}] Force-close: cancelling pending limit {tid}")
            self._cancel_limit_order(tid, info)

    def _on_limit_fill(self, trade_id, broker_sl=None):
        """A pending limit was adopted: drop the bot's pending-order entry and
        make sure the broker's SL carries the trade's SL. The resting order was
        placed with the INITIAL SL; memory may hold a different one (trail from
        the persisted DB row), and the reconciler clamps memory to the trail —
        comparing memory to itself would skip the push and leave a stale stop
        armed (2026-10-16: GOAT5-ch1-218 market-closed at 4176.57 while the
        broker stop sat 33 points away)."""
        self._pending_orders.pop(trade_id, None)
        if broker_sl is not None:
            self._remember_broker_sl(trade_id, broker_sl)
        for runner in self.runners.values():
            if not isinstance(runner.config, dict):
                continue
            for ch in runner.state.challenges:
                for t in ch.open_trades:
                    if t.id != trade_id:
                        continue
                    if t.position_id is None:
                        return
                    want = float(t.sl)
                    armed = getattr(self, "_broker_sl", {}).get(t.id)
                    if armed is None or abs(want - float(armed)) > 0.01:
                        logger.info(f"[{runner.name}] Limit fill {trade_id}: pushing "
                                    f"SL {want:.2f} to broker (order SL was "
                                    f"{'unknown' if armed is None else f'{armed:.2f}'})")
                        self.push_trail_sl_to_broker(runner, t)
                    return

    def _adopt_filled_limits(self):
        """Adopt limit fills every poll (not only on the 5-min reconcile): match
        live broker positions by strategyId, set position_id/status, persist, and
        sync the stop. Cheap: one positions call per broker that has a pending
        limit; nothing at all when no limit is outstanding."""
        pending = [(tid, info) for tid, info in list(self._pending_orders.items())
                   if info.get("kind") == "limit_open"]
        if not pending or not getattr(self, "brokers", None) or not getattr(self, "runners", None):
            return
        by_broker = {}
        for tid, info in pending:
            runner = self.runners.get(info.get("runner_name"))
            if runner is None or runner._broker is None:
                continue
            by_broker.setdefault(runner._broker_key, []).append((tid, runner))
        for bkey, items in by_broker.items():
            broker = self.brokers.get(bkey)
            if broker is None:
                continue
            client = broker["client"]
            try:
                if not getattr(client, "columns", {}).get("positions"):
                    try:
                        client.config()
                    except TradeLockerError:
                        pass
                positions = client.positions(broker["account_id"]) or []
            except TradeLockerError:
                continue
            for tid, runner in items:
                match = None
                for p in positions:
                    sid = str(p.get("comment") or p.get("strategyId") or "")
                    if tid in sid:
                        match = p
                        break
                if match is None:
                    continue
                trade = None
                for ch in runner.state.challenges:
                    for t in ch.open_trades:
                        if t.id == tid:
                            trade = t
                            break
                    if trade is not None:
                        break
                if trade is None:
                    continue
                pid = match.get("id") or match.get("positionId")
                if trade.position_id is None:
                    trade.position_id = pid
                trade.status = "open"
                if self.trade_store:
                    self.trade_store.save_trade(trade)
                tag = runner.config.get('tag', runner.name) if isinstance(runner.config, dict) else runner.name
                logger.info(f"[{tag}] Limit {tid} FILLED — position {pid} adopted "
                            f"(fast adoption, slot stays occupied)")
                self._on_limit_fill(tid, match.get("stopLoss") or match.get("sl"))

    # ── Reconcile ─────────────────────────────────────────────────────────
    def reconcile(self):
        first = True
        attempted_pids = set()  # legacy-close dedupe (herofx_aqua/herofx_goat share one account)
        for bkey, broker in self.brokers.items():
            if not first:
                time.sleep(1.5)
            first = False
            client = broker["client"]
            account_id = broker["account_id"]
            if not client.columns.get("positions"):
                try:
                    client.config()
                except TradeLockerError:
                    pass
            try:
                positions = client.positions(account_id)
            except TradeLockerError as e:
                logger.warning(f"[{bkey}] Reconcile positions fetch failed: {e}")
                continue
            # Only reconcile runners assigned to this broker
            broker_runners = {n: r for n, r in self.runners.items()
                              if r._broker_key == bkey}
            reconciled, settled, closed_stale, pmap = self.reconciler.reconcile(
                positions or [], broker_runners,
                pending_ids=set(self._pending_orders.keys()),
                on_fill=self._on_limit_fill,
            )
            logger.info(f"[{bkey}] Reconcile: {reconciled} new, {settled} settled, {closed_stale} stale-closed")
            if closed_stale:
                # A vanished position is settled from fill prices; fetch next poll
                # rather than waiting for the next 5-minute orders-history slot.
                self._orders_history_due = time.time()
            # Legacy cleanup: close bot-tagged positions whose strategy no longer
            # resolves to a current runner (old-strategy leftovers after a
            # replacement; manual/untagged positions are left alone)
            legacy = []
            for p in (positions or []):
                comment = str(p.get("comment") or p.get("strategyId") or "").strip()
                if "|" not in comment:
                    continue
                strat = comment.split("|")[0]
                if self._resolve_runner_name(strat) not in self.runners:
                    if str(p.get("id") or p.get("positionId")) not in attempted_pids:
                        legacy.append(p)
            if legacy:
                self._close_legacy_positions(bkey, legacy, attempted_pids)
        self._prune_settled_close_pending()
        self._cleanup_orphan_rows()
        self._enforce_max_concurrent()
        self._save_treasuries()

    def _cleanup_orphan_rows(self):
        """Mark stale 'open' trade rows that never reached the broker (no position_id,
        older than 2h) as 'failed' so they stop polluting history/winrate."""
        if not self.trade_store:
            return
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        try:
            cur = self.trade_store._conn.execute(
                "SELECT id FROM trades WHERE status='open' AND position_id IS NULL AND open_time < ?",
                (cutoff,))
            stale = [r[0] for r in cur.fetchall()]
            pending_ids = set(self._pending_orders.keys())
            # Memory is authoritative for live trades: never mark an id a runner
            # still tracks as open, even if its DB row lacks a position_id.
            mem_ids = set()
            for runner in self.runners.values():
                for ch in runner.state.challenges:
                    for ot in ch.open_trades:
                        mem_ids.add(ot.id)
            to_fail = [tid for tid in stale
                       if tid not in pending_ids and tid not in mem_ids]
            if to_fail:
                marked = self.trade_store.mark_stale_rows_failed(
                    to_fail, "orphan_row: status open but never reached the broker "
                    "(no position_id, older than 2h, not tracked as open)")
                logger.info(f"[cleanup] Marked {marked} orphan open row(s) as failed: "
                            f"{to_fail[:5]}{'...' if len(to_fail) > 5 else ''}")
        except Exception as e:
            logger.warning(f"[cleanup] Orphan row cleanup failed: {e}")

    def _close_legacy_positions(self, broker_key: str, legacy_positions: list, attempted_pids: set):
        """Legacy close (runs every reconcile cycle): close bot-tagged positions
        whose strategy no longer resolves to a current runner (old-strategy
        leftovers after a replacement). Manual/untagged positions are untouched."""
        broker = self.brokers[broker_key]
        client = broker["client"]
        for p in legacy_positions:
            pid = p.get("id") or p.get("positionId")
            comment = str(p.get("comment") or p.get("strategyId") or "").strip()
            if pid is None:
                continue
            attempted_pids.add(str(pid))
            qty = p.get("qty") or p.get("openQty") or p.get("quantity") or 0.0
            try:
                client.close_position(pid, qty=qty)
                logger.info(f"[{broker_key}] Closed legacy PP position {pid} ({comment})")
            except TradeLockerError as e:
                err = str(e).lower()
                if "session" in err or "forbidden" in err:
                    logger.info(f"[{broker_key}] Legacy PP close deferred (session closed): {pid}")
                else:
                    logger.warning(f"[{broker_key}] Legacy PP close failed {pid}: {e}")

    def _pending_close_broker_keys(self):
        """Brokers that have at least one in-flight close order — only these
        need the fast fill-price settlement fetch."""
        keys = set()
        for info in self._pending_orders.values():
            if info.get("kind") != "close":
                continue
            runner = self.runners.get(info.get("runner_name"))
            bkey = getattr(runner, "_broker_key", None) if runner else None
            if bkey in self.brokers:
                keys.add(bkey)
        return keys

    def _prune_settled_close_pending(self):
        """Drop close markers once the trade is no longer tracked as open.
        The reconciler can settle a close without the orders-history path, so
        the marker must not linger until its 45s timeout and force a second
        reconcile."""
        open_ids = set()
        for runner in self.runners.values():
            for ch in runner.state.challenges:
                for ot in ch.open_trades:
                    open_ids.add(ot.id)
        for key, info in list(self._pending_orders.items()):
            if info.get("kind") != "close" or not key.endswith("_close"):
                continue
            tid = key[:-len("_close")]
            if tid not in open_ids:
                self._pending_orders.pop(key, None)
                self._close_attempt_ts.pop(tid, None)

    # ── Orders history (PnL settlement) ──────────────────────────────────
    def _fetch_orders_history(self, broker_keys=None):
        keys = list(broker_keys) if broker_keys is not None else list(self.brokers)
        first = True
        for bkey in keys:
            broker = self.brokers.get(bkey)
            if broker is None:
                continue
            if not first:
                time.sleep(1.5)
            first = False
            client = broker["client"]
            account_id = broker["account_id"]
            if not client.columns.get("ordersHistory"):
                try:
                    client.config()
                except TradeLockerError:
                    pass
            try:
                rows = client.orders_history(account_id)
            except TradeLockerError as e:
                logger.warning(f"[{bkey}] OrdersHistory fetch failed: {e}")
                continue
            self._on_orders_history(rows or [], bkey)

    def _on_orders_history(self, rows, broker_key: str):
        updated = 0
        fills_by_pos = {}
        for row in rows:
            pos_id = _row_get(row, "positionId", "id")
            if pos_id is None:
                continue
            status = str(_row_get(row, "status") or "").lower()
            if status != "filled":
                continue
            qty = _to_float(_row_get(row, "filledQty", "qty", "quantity", "volume"))
            if not qty:
                continue
            price = _to_float(_row_get(row, "avgPrice", "price"))
            if not price:
                continue
            side_raw = str(_row_get(row, "side") or "").lower()
            fills_by_pos.setdefault(pos_id, []).append({
                "side": side_raw,
                "qty": qty,
                "price": price,
                "ts": _parse_ts(_row_get(row, "createdDate", "lastModified", "createdAt")),
                "strategyId": str(_row_get(row, "strategyId", "comment", "label") or ""),
            })

        blown_after_settle = []  # (runner_name, ch) — acted on after the scan
        for pos_id, fills in fills_by_pos.items():
            fills.sort(key=lambda f: (f["ts"] or datetime.min.replace(tzinfo=timezone.utc)))
            comment = next((f["strategyId"] for f in fills if f["strategyId"]), "")

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

            matched = False
            # Only check runners for this broker
            for name, runner in self.runners.items():
                if runner._broker_key != broker_key:
                    continue
                for ch in runner.state.challenges:
                    if ch.blown and not ch.open_trades:
                        continue
                    for ot in list(ch.open_trades):
                        if str(ot.position_id) == str(pos_id):
                            t = ot
                            gross, last_ts, exit_px = _realized(t.entry, t.direction)
                            if gross is None:
                                break
                            close_dt = last_ts or datetime.now(timezone.utc)
                            t.pnl = round(gross, 2)
                            t.close_time = close_dt.isoformat()
                            t.status = "closed"
                            if exit_px is not None:
                                t.exit_price = exit_px
                                t.exit_reason = classify_exit(t, exit_px)
                            runner.on_trade_closed(t, close_dt, ch)
                            self._pending_orders.pop(f"{t.id}_close", None)
                            self._close_attempt_ts.pop(t.id, None)
                            self._forget_trade_stop(t.id)
                            self.trade_store.save_trade(t)
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
                            # Swap joins at exit here too (kernel numba:1733),
                            # same as the path-(a) on_trade_closed booking
                            swap = runner._swap_usd(t.get("direction", 1), lots,
                                                    t.get("open_time"), close_dt,
                                                    runner.profile.get("swap_rate", 0.0))
                            net_pnl = round(gross - COMMISSION * lots + swap, 2)
                            old_pnl = t.get("pnl") or 0
                            delta = net_pnl - (float(old_pnl) if old_pnl else 0.0)
                            already_settled = delta == 0 and str(t.get("status")) == "closed"
                            t["pnl"] = net_pnl
                            t["status"] = "closed"
                            self._pending_orders.pop(f"{t.get('id')}_close", None)
                            self._close_attempt_ts.pop(t.get("id"), None)
                            self._forget_trade_stop(t.get("id"))
                            if not already_settled:
                                # Entry balance (bookings only at close; this
                                # vanished row already left open_trades)
                                start_bal = ch.equity - delta
                                ch.equity += delta
                                ch.ch_trade_pnl += delta
                                runner.state.trade_pnl += delta
                                cm = close_dt.strftime("%Y-%m")
                                runner.state.monthly_stats.setdefault(cm, {"spent": 0, "earned": 0, "trade_pnl": 0, "withdrawn": 0})
                                runner.state.monthly_stats[cm]["trade_pnl"] = runner.state.monthly_stats[cm].get("trade_pnl", 0) + delta
                                cd = close_dt.strftime("%Y-%m-%d")
                                runner.state.daily_pnl[cd] = runner.state.daily_pnl.get(cd, 0.0) + delta
                                self.trade_store.update_trade_pnl(
                                    t.get("id"), net_pnl, "closed",
                                    exit_price=exit_px,
                                    exit_reason=classify_exit(t, exit_px))
                                runner._refresh_qualifying_days(ch)
                                runner._apply_daily_profit_cap(ch, int(close_dt.timestamp() / 86400))
                                # Blow checks mirror on_trade_closed (path (a)):
                                # the placeholder close booked nothing, so the
                                # floor / per-trade cap must fire here.
                                if not ch.blown:
                                    mlpt = runner.profile.get("max_loss_per_trade")
                                    if runner.profile.get('daily_dd_mode') == 'fixed':
                                        daily_floor = ch.day_equity - runner.profile["dl"] * runner.profile["ib"]
                                    else:
                                        daily_floor = ch.day_equity * (1 - runner.profile["dl"])
                                    pertrade = bool(mlpt and delta <= -mlpt * start_bal)
                                    if ch.equity <= ch.dd_floor or ch.equity <= daily_floor or pertrade:
                                        ch.blown = True
                                        if pertrade:
                                            why = (f"pertrade mlpt: delta={delta:+.2f} <= "
                                                   f"-{mlpt:.4f} × start={start_bal:.2f}")
                                        else:
                                            why = (f"settlement DD: eq={ch.equity:.2f} "
                                                   f"dd_floor={ch.dd_floor:.2f} daily_floor={daily_floor:.2f}")
                                        logger.warning(f"[{name}] Ch#{ch.challenge_num} BLOWN ({why})")
                                        blown_after_settle.append((name, ch))
                                updated += 1
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break

        if blown_after_settle:
            # Deferred: cancel/close outside the scan to avoid mutating
            # runner state mid-iteration (mirrors the intraday-DD blow path)
            for bname, bch in blown_after_settle:
                brunner = self.runners[bname]
                self._cancel_pending_for_challenge(brunner, bch)
                for ot in list(bch.open_trades):
                    self._close_trade_at_market(ot, brunner, bch)
            self._save_treasuries()
            self._recycle()

        if updated:
            self._save_treasuries()
            self._broadcast()
            logger.info(f"[{broker_key}] OrdersHistory: updated {updated} trades from history")

    # ── Treasury & broadcast ──────────────────────────────────────────────
    def process_payout(self, runner_name: str, payout: float):
        runner = self.runners[runner_name]
        runner.state.treasury += payout
        logger.info(f"[Treasury] {runner_name} payout ${payout:.2f} -> treasury: ${runner.state.treasury:.2f}")
        self._save_treasuries()
        self._recycle()
        min_cost = min((r.profile["cc"] for r in self.runners.values()), default=17.0)
        if self._total_active() < MAX_CAP and payout >= min_cost:
            target = min(self.runners.values(), key=lambda r: r.state.treasury)
            cost = target.profile["cc"]
            if runner.state.treasury >= cost:
                runner.state.treasury -= cost
                target._add_new_challenge()
                self._save_treasuries()
                logger.info(f"[Treasury] Growth: {runner_name}'s payout bought a new "
                            f"challenge on {target.name} (cost ${cost:.2f} from "
                            f"{runner_name}'s treasury ${runner.state.treasury:.2f}, "
                            f"active challenges now {self._total_active()}/{MAX_CAP})")
                self._broadcast()

    def _recycle(self):
        """Auto-rebuy: replace each blown (not yet recycled) challenge with a new
        one, funded from the treasury. The `recycled` marker prevents re-recycling
        the same blown challenge on every payout/close (compounding creation)."""
        recycled = 0
        for runner in self.runners.values():
            blown_chs = [c for c in runner.state.challenges if c.blown]
            for ch in blown_chs:
                if getattr(ch, "recycled", False):
                    continue  # already replaced — no compounding
                if self._total_active() >= MAX_CAP:
                    logger.warning(f"[Treasury] {runner.name}: MAX_CAP {MAX_CAP} active "
                                   f"challenges reached — blown challenge #{ch.challenge_num} "
                                   f"NOT recycled (treasury ${runner.state.treasury:.2f})")
                    break
                cost = runner.profile["cc"]
                if runner.state.treasury >= cost:
                    runner.state.treasury -= cost
                    runner._add_new_challenge()
                    ch.recycled = True
                    recycled += 1
                    logger.info(f"[Treasury] {runner.name}: blown challenge "
                                f"#{ch.challenge_num} recycled -> new challenge "
                                f"(cost ${cost:.2f}, treasury ${runner.state.treasury:.2f}, "
                                f"total_spent ${runner.state.total_spent:.2f})")
                else:
                    logger.info(f"[Treasury] {runner.name}: treasury ${runner.state.treasury:.2f} "
                                f"below challenge cost ${cost:.2f} — blown challenge "
                                f"#{ch.challenge_num} awaits funds")
        if recycled:
            self._save_treasuries()
            logger.info(f"[Treasury] Recycled {recycled} blown challenge(s) from treasuries")
            self._broadcast()

    def _total_active(self) -> int:
        count = 0
        for runner in self.runners.values():
            count += len(runner.state.active_challenges)
        return count

    def _save_treasuries(self):
        self.treasury_store.save(self.runners)

    # ── Parity checker (live vs archive/lastly backtest) ───────────────────
    def _trigger_parity(self):
        threading.Thread(target=self._run_parity_all, daemon=True).start()

    def _collect_live_trades(self, runner, since_ts):
        out = []
        for ch in runner.state.challenges:
            for t in ch.trade_history:
                d = t if isinstance(t, dict) else asdict(t)
                if str(d.get("status") or "").lower() == "closed" and d.get("pnl") is not None:
                    out.append(d)
            for ot in ch.open_trades:
                if str(getattr(ot, "status", "") or "").lower() == "closed":
                    out.append(asdict(ot))
        live = []
        for d in out:
            try:
                ts = datetime.fromisoformat(str(d.get("open_time")).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts.timestamp() >= since_ts:
                    live.append(d)
            except (TypeError, ValueError):
                continue
        return live

    def _run_parity_all(self):
        if not self._parity_lock.acquire(blocking=False):
            return
        try:
            self._last_parity_run = time.time()
            bars_cache = {}
            for name, runner in list(self.runners.items()):
                try:
                    self._run_parity_check(runner, bars_cache)
                except Exception as e:
                    logger.error(f"[parity] {name} check failed: {e}", exc_info=True)
                    self.parity_reports[name] = build_report(
                        name, runner.config.get("tag", ""), {}, error=str(e))
        finally:
            self._parity_lock.release()
        self._broadcast()

    def _fetch_parity_bars(self, runner, bars_cache):
        broker = runner._broker
        if not broker or not broker.get("client"):
            raise RuntimeError("no broker assigned")
        bkey = runner._broker_key
        if bkey in bars_cache:
            return bars_cache[bkey]
        now_ms = int(time.time() * 1000)
        from_ms = now_ms - PARITY_LOOKBACK_DAYS * 86400 * 1000
        raw = broker["client"].history(broker["instrument_id"], broker["route_id"],
                                       "1H", from_ms, now_ms)
        bars = {"open": [], "high": [], "low": [], "close": [], "volume": [], "time": []}
        seen = set()
        for b in raw or []:
            o = _to_float(_row_get(b, "open", "o"))
            h = _to_float(_row_get(b, "high", "h"))
            l = _to_float(_row_get(b, "low", "l"))
            c = _to_float(_row_get(b, "close", "c"))
            ts = _parse_epoch(_row_get(b, "time", "t") or b.get("timestamp"))
            if None in (o, h, l, c, ts) or ts in seen:
                continue
            seen.add(ts)
            bars["open"].append(o); bars["high"].append(h)
            bars["low"].append(l); bars["close"].append(c)
            bars["volume"].append(_to_float(_row_get(b, "volume", "v")) or 1)
            bars["time"].append(ts)
        order = sorted(range(len(bars["time"])), key=lambda i: bars["time"][i])
        for k in bars:
            bars[k] = [bars[k][i] for i in order]
        bars_cache[bkey] = bars
        return bars

    def _run_parity_check(self, runner, bars_cache):
        name = runner.name
        tag = runner.config.get("tag", name)
        bars = self._fetch_parity_bars(runner, bars_cache)
        if len(bars["time"]) < 50:
            raise RuntimeError(f"insufficient history ({len(bars['time'])} bars)")
        since_ts = bars["time"][0]
        live = self._collect_live_trades(runner, since_ts)
        cfg = runner.config
        live_norm = normalize_live(live)
        first = min((t["open_time"] for t in live_norm), default=None)
        # Align the backtest with the live account's start: run the trade loop
        # from the first live bar with a FLAT state, while the indicators still
        # use the full lookback warmup. The signal from the prior bar can still
        # fill at that bar's open. Without this, positions the backtest carried
        # into the live start can skip/alter entries the freshly reset live
        # account takes (state divergence, not signal drift).
        start_index = 1
        first_bar = None
        if first is not None:
            tf_sec = int(cfg.get("eval_tf", 60)) * 60
            first_bar = int(first // tf_sec) * tf_sec
            idx = next((i for i, ts in enumerate(bars["time"]) if ts >= first_bar),
                       len(bars["time"]) - 1)
            start_index = max(1, idx)
        sim = run_trades(
            bars, cfg.get("eval_strat", "Supertrend RSI"),
            rr=cfg.get("eval_rr", 3.0),
            atr_mult=cfg.get("sl_atr_mult", 3.0),
            stop_mode=cfg.get("stop_mode", "atr"),
            stop_value=cfg.get("stop_value", 25.0),
            risk_pct=cfg.get("eval_rpt", runner.profile.get("rpt", 0.01)),
            max_pos=runner._max_concurrent(),
            addon_min_r=cfg.get("addon_min_r"),
            initial=runner.profile.get("ib", 5000.0),
            skip_below_min=True,
            start_index=start_index,
        )
        bt_trades = sim["trades"]
        if first_bar is not None:
            bt_trades = [t for t in bt_trades if t["entry_time"] >= first_bar]
        cmp = compare_trades(live, bt_trades, r_tol=PARITY_R_TOL)
        window = [
            datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ]
        self.parity_reports[name] = build_report(
            name, tag, cmp, window=window, bars=len(bars["time"]))
        logger.info(f"[parity] {tag}: {self.parity_reports[name]['status']} "
                    f"live={cmp['live_trades']} bt={cmp['bt_trades']} "
                    f"matched={cmp['matched']} live_only={cmp['live_only']} "
                    f"bt_only={cmp['bt_only']} dR={cmp['mean_r_delta']}")

    def _broadcast(self, name: str = None):
        self.ws_broadcaster.broadcast_state(self.get_full_snapshot())

    def get_full_snapshot(self):
        from engine.market import SYMBOL
        combined = {}
        for name, runner in self.runners.items():
            snap = runner.snapshot()
            snap["broker"] = runner._broker_key or ""
            snap["broker_price"] = (runner._broker or {}).get("current_price", 0.0)
            combined[name] = snap
        total_payouts = sum(
            c.total_payouts
            for runner in self.runners.values()
            for c in runner.state.challenges
        )
        farm_daily_pnl = {}
        for name, runner in self.runners.items():
            for day, pnl in runner.state.daily_pnl.items():
                farm_daily_pnl[day] = round(farm_daily_pnl.get(day, 0.0) + pnl, 2)
        # Use first broker's price as the display price
        primary_price = 0.0
        if self.brokers:
            primary_price = next(iter(self.brokers.values()))["current_price"]
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategies": combined,
            "symbol": SYMBOL,
            "price": primary_price,
            "farm_treasury": round(sum(r.state.treasury for r in self.runners.values()), 2),
            "farm_total_payouts": round(total_payouts, 2),
            "farm_daily_pnl": farm_daily_pnl,
            "parity": getattr(self, "parity_reports", {}),
        }

    def close(self):
        self._poll_running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        self.ws_broadcaster.close()
        self.trade_store.close()
        self.treasury_store.close()


def initialize_components(data_dir: str = "bot/data"):
    bot = Bot(data_dir=data_dir)
    logger.info("All components initialized")
    return bot


def main():
    import signal
    from config.loader import load_all_configs
    bot = initialize_components()
    logger.info("Starting trading engine...")

    def _shutdown(sig=None, frame=None):
        signame = signal.Signals(sig).name if sig else "KeyboardInterrupt"
        logger.info(f"Received {signame} — shutting down...")
        bot._save_treasuries()
        bot.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        all_configs = load_all_configs()
        bot.connect(all_configs)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    main()
