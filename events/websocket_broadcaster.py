"""WebSocket broadcaster for the dashboard — with authentication + credentials API.

Runs a FastAPI + uvicorn server in a daemon thread with its own asyncio loop.
Thread-safe `broadcast_state()` / `broadcast_log()` bridge the poll thread
→ asyncio loop via `run_coroutine_threadsafe()`.

Auth: single password (DASHBOARD_PASSWORD env = bootstrap; the hash lives in the
credentials DB). Auth is DISABLED until a password is set. Session cookie:
`expiry + hmac_sha256(session_secret)`, HttpOnly, 30-day expiry. Password
recover/change via one-time emailed codes (SMTP) with a docker-logs fallback.
"""
import asyncio
import collections
import hashlib
import hmac
import json
import logging
import os
import pathlib
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

BROADCAST_INTERVAL = 1.0
LOG_DRAIN_INTERVAL = 0.5
SESSION_COOKIE = "gpf_session"
SESSION_TTL_SECONDS = 30 * 86400
RECOVER_RATE_LIMIT_SECONDS = 300

DASHBOARD_PATH = pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "index.html"

LOGIN_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Trading Engine — Login</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: #0d1117;
         color: #e6edf3; display: flex; align-items: center; justify-content: center;
         min-height: 100vh; margin: 0; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px;
          padding: 32px; width: 320px; }}
  h1 {{ font-size: 18px; margin: 0 0 18px; }}
  input {{ width: 100%; box-sizing: border-box; padding: 10px; margin: 6px 0 14px;
          background: #0d1117; border: 1px solid #30363d; border-radius: 6px; color: #e6edf3; }}
  button {{ width: 100%; padding: 10px; background: #238636; color: white; border: none;
           border-radius: 6px; cursor: pointer; font-size: 14px; }}
  button:hover {{ background: #2ea043; }}
  .err {{ color: #f85149; font-size: 13px; margin-bottom: 10px; }}
  .hint {{ color: #8b949e; font-size: 12px; margin-top: 14px; }}
</style></head>
<body><div class="card">
  <h1>Trading Engine</h1>
  {error_html}
  <form method="post" action="/login">
    <label for="password">Password</label>
    <input type="password" id="password" name="password" autofocus required>
    <button type="submit">Sign in</button>
  </form>
  <div class="hint">Forgot your password? POST to /api/auth/recover — a one-time
  code will be emailed to the configured recovery address (or logged on the server).</div>
</div></body></html>"""


class WebSocketBroadcaster:
    """Real WS broadcaster — FastAPI on a background thread, with auth + API."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._snapshot: dict = {}
        self._snapshot_lock = threading.Lock()
        self._log_queue: collections.deque = collections.deque(maxlen=500)
        self._started = False
        self._auth_store = None
        self._credentials_store = None
        self._last_recover_ts = 0.0
        self._last_reset_attempt = {}
        self.restart_fn = None
        self._parity_provider = None
        self._parity_trigger = None

    # ── Auth helpers ───────────────────────────────────────────────────────

    def _get_auth_store(self):
        if self._auth_store is None:
            try:
                from config.credentials import CredentialsStore
                self._auth_store = CredentialsStore()
            except Exception as e:
                logger.error(f"[auth] store init failed: {e}")
        return self._auth_store

    def auth_enabled(self) -> bool:
        store = self._get_auth_store()
        return bool(store and store.has_password())

    def bootstrap_password(self):
        """Seed the DB password hash from DASHBOARD_PASSWORD on first boot."""
        pw = os.environ.get("DASHBOARD_PASSWORD")
        if not pw:
            pw = self._env_file_password()
        if not pw:
            return
        store = self._get_auth_store()
        if store and not store.has_password():
            store.set_password(pw)
            logger.info("[auth] dashboard password initialized from DASHBOARD_PASSWORD")

    def _env_file_password(self):
        try:
            from config.loader import _find_env_file, _parse_env_file
            env_path = _find_env_file()
            if env_path:
                return _parse_env_file(env_path).get("DASHBOARD_PASSWORD")
        except Exception:
            pass
        return None

    def _make_session_cookie(self) -> str:
        store = self._get_auth_store()
        secret = store.get_session_secret()
        expiry = str(int(time.time()) + SESSION_TTL_SECONDS)
        sig = hmac.new(secret.encode("utf-8"), expiry.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{expiry}.{sig}"

    def _session_cookie_valid(self, cookie_header: Optional[str]) -> bool:
        if not self.auth_enabled():
            return True
        if not cookie_header:
            return False
        raw = None
        for part in cookie_header.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                if k.strip() == SESSION_COOKIE:
                    raw = v.strip()
                    break
        if not raw or "." not in raw:
            return False
        expiry_str, sig = raw.rsplit(".", 1)
        if not expiry_str.isdigit():
            return False
        if int(expiry_str) < time.time():
            return False
        store = self._get_auth_store()
        if not store:
            return False
        secret = store.get_session_secret()
        expected = hmac.new(secret.encode("utf-8"), expiry_str.encode("utf-8"),
                            hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        if self._started:
            return
        self._started = True
        self.bootstrap_password()
        self._thread = threading.Thread(target=self._run_server, daemon=True, name="ws-server")
        self._thread.start()
        logger.info(f"WS server starting on {self.host}:{self.port}")

    def build_app(self):
        """Build (and return) the FastAPI app. Extracted for testability."""
        import uvicorn  # noqa: F401  (availability check)
        from fastapi import FastAPI, Request, WebSocket as FastWS, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

        app = FastAPI()
        broadcaster = self

        def cookie_of(request: Request):
            return request.headers.get("cookie")

        def denied_json():
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        @app.get("/")
        async def index(request: Request):
            if not broadcaster._session_cookie_valid(cookie_of(request)):
                return RedirectResponse("/login", status_code=307)
            try:
                html = DASHBOARD_PATH.read_text(encoding="utf-8")
                return HTMLResponse(html)
            except FileNotFoundError:
                return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)

        @app.get("/login")
        async def login_page(request: Request):
            if not broadcaster.auth_enabled():
                return RedirectResponse("/", status_code=307)
            if broadcaster._session_cookie_valid(cookie_of(request)):
                return RedirectResponse("/", status_code=307)
            err = request.query_params.get("error")
            error_html = '<div class="err">Wrong password — try again.</div>' if err else ""
            return HTMLResponse(LOGIN_PAGE.format(error_html=error_html))

        @app.post("/login")
        async def login_submit(request: Request):
            if not broadcaster.auth_enabled():
                return RedirectResponse("/", status_code=307)
            from urllib.parse import parse_qs
            form = parse_qs((await request.body()).decode("utf-8"))
            password = (form.get("password") or [""])[0]
            store = broadcaster._get_auth_store()
            if store and store.verify_password(password):
                logger.info("[auth] dashboard login ok")
                resp = RedirectResponse("/", status_code=303)
                resp.set_cookie(SESSION_COOKIE, broadcaster._make_session_cookie(),
                                max_age=SESSION_TTL_SECONDS, httponly=True,
                                samesite="lax")
                return resp
            logger.warning("[auth] FAILED dashboard login attempt")
            return RedirectResponse("/login?error=1", status_code=303)

        @app.post("/logout")
        async def logout():
            # Rotate the session secret so a captured cookie value is dead, not
            # just unlinked from this browser
            store = broadcaster._get_auth_store()
            if store and store.has_password():
                store.rotate_session_secret()
                logger.info("[auth] logout — session secret rotated")
            resp = RedirectResponse("/login", status_code=303)
            resp.delete_cookie(SESSION_COOKIE)
            return resp

        @app.get("/auth-check")
        async def auth_check(request: Request):
            if broadcaster._session_cookie_valid(cookie_of(request)):
                return JSONResponse({"ok": True})
            return denied_json()

        @app.websocket("/ws/")
        async def ws_endpoint(ws: FastWS):
            if not broadcaster._session_cookie_valid(ws.headers.get("cookie")):
                logger.warning("[auth] WS handshake rejected (missing/invalid session cookie)")
                await ws.close(code=1008)
                return
            await ws.accept()
            broadcaster._clients.add(ws)
            try:
                with broadcaster._snapshot_lock:
                    snap = dict(broadcaster._snapshot)
                if snap:
                    await ws.send_json({"type": "snapshot", "data": snap})
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.receive_text(), timeout=30)
                        await broadcaster._handle_command(ws, raw)
                    except asyncio.TimeoutError:
                        pass
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                broadcaster._clients.discard(ws)

        @app.get("/health")
        async def health():
            return {"status": "ok", "clients": len(broadcaster._clients)}

        # ── Auth API ──────────────────────────────────────────────────────

        @app.post("/api/auth/change")
        async def auth_change(request: Request):
            if not broadcaster._session_cookie_valid(cookie_of(request)):
                return denied_json()
            body = await request.json()
            store = broadcaster._get_auth_store()
            if not store:
                return JSONResponse({"error": "auth store unavailable"}, status_code=500)
            current = str(body.get("current_password") or "")
            new = str(body.get("new_password") or "")
            if len(new) < 8:
                return JSONResponse({"error": "new password must be at least 8 chars"}, status_code=400)
            if not store.verify_password(current):
                return JSONResponse({"error": "current password is wrong"}, status_code=400)
            store.set_password(new)
            logger.info("[auth] dashboard password changed via dashboard")
            return JSONResponse({"ok": True})

        @app.post("/api/auth/recover")
        async def auth_recover(request: Request):
            now = time.time()
            if now - broadcaster._last_recover_ts < RECOVER_RATE_LIMIT_SECONDS:
                logger.warning("[auth] recovery request rate-limited")
                return JSONResponse(
                    {"error": "rate limited — try again later"}, status_code=429)
            broadcaster._last_recover_ts = now
            logger.info("[auth] recovery code requested")
            store = broadcaster._get_auth_store()
            if not store or not store.has_password():
                return JSONResponse({"error": "auth not enabled"}, status_code=400)
            from config import mailer
            to_email = store.get_recovery_email() or mailer.recovery_email()
            code = store.create_recovery_code()
            sent = False
            if to_email:
                sent = mailer.send_recovery_email(to_email, code)
            if not sent:
                logger.warning(
                    f"[auth] SMTP not configured or failed — RECOVERY CODE (expires 15min): {code}")
                return JSONResponse({"ok": True, "sent": False,
                                     "message": "code logged on the server"})
            return JSONResponse({"ok": True, "sent": True,
                                 "message": f"code sent to {to_email}"})

        @app.post("/api/auth/reset")
        async def auth_reset(request: Request):
            body = await request.json()
            store = broadcaster._get_auth_store()
            if not store:
                return JSONResponse({"error": "auth store unavailable"}, status_code=500)
            code = str(body.get("code") or "")
            new = str(body.get("new_password") or "")
            # Rate-limit attempts (3 per 5 min per client) — a 32-bit code with
            # unlimited attempts is brute-forceable inside the 15-min window
            now = time.time()
            window_start = now - 300
            attempts = [t for t in broadcaster._last_reset_attempt.values() if t > window_start]
            if len(attempts) >= 3:
                logger.warning("[auth] reset attempts rate-limited")
                return JSONResponse({"error": "too many attempts — try later"}, status_code=429)
            broadcaster._last_reset_attempt[str(now)] = now
            if len(new) < 8:
                return JSONResponse({"error": "new password must be at least 8 chars"}, status_code=400)
            if not store.consume_recovery_code(code):
                logger.warning("[auth] reset rejected: invalid or expired code")
                return JSONResponse({"error": "invalid or expired code"}, status_code=400)
            store.set_password(new)
            logger.info("[auth] dashboard password reset via recovery code")
            return JSONResponse({"ok": True})

        @app.post("/api/auth/recovery-email")
        async def auth_recovery_email(request: Request):
            if not broadcaster._session_cookie_valid(cookie_of(request)):
                return denied_json()
            body = await request.json()
            email = str(body.get("recovery_email") or "").strip()
            if "@" not in email or "." not in email:
                return JSONResponse({"error": "invalid email"}, status_code=400)
            store = broadcaster._get_auth_store()
            if not store:
                return JSONResponse({"error": "auth store unavailable"}, status_code=500)
            store.set_recovery_email(email)
            logger.info(f"[auth] recovery email updated to {email}")
            return JSONResponse({"ok": True})

        # ── Credentials API (uploaded challenge credentials, 1:1 model) ───

        def _credentials_store():
            # Cached singleton — a per-request store leaks a SQLite connection
            # (fd exhaustion over the process lifetime)
            if broadcaster._credentials_store is None:
                try:
                    from config.credentials import CredentialsStore
                    broadcaster._credentials_store = CredentialsStore()
                except Exception as e:
                    logger.error(f"[credentials] store init failed: {e}")
                    return None
            return broadcaster._credentials_store

        @app.get("/api/credentials")
        async def credentials_list(request: Request):
            if not broadcaster._session_cookie_valid(cookie_of(request)):
                return denied_json()
            store = _credentials_store()
            if not store:
                return JSONResponse({"error": "store unavailable"}, status_code=500)
            from config import mailer
            auth_store = broadcaster._get_auth_store()
            recovery = (auth_store.get_recovery_email() if auth_store else None) or mailer.recovery_email()
            return JSONResponse({"credentials": store.list_all(),
                                 "recovery_email": recovery})

        @app.post("/api/credentials")
        async def credentials_upload(request: Request):
            if not broadcaster._session_cookie_valid(cookie_of(request)):
                return denied_json()
            body = await request.json()
            raw = str(body.get("strategy") or "").strip()
            email = str(body.get("email") or "").strip()
            password = str(body.get("password") or "")
            server = str(body.get("server") or "").strip()
            account = body.get("account")
            balance = body.get("balance")
            challenge_cost = body.get("challenge_cost")
            if not raw:
                return JSONResponse({"error": "strategy is required"}, status_code=400)
            from engine.market import FARMS

            def _resolve_tag(s):
                for farm in FARMS:
                    if farm["tag"] == s or farm["name"] == s:
                        return farm["tag"]
                for farm in FARMS:
                    if s in farm.get("aliases", []):
                        return farm["tag"]
                return None

            tag = _resolve_tag(raw)
            if tag is None:
                return JSONResponse({"error": f"unknown strategy: {raw}"}, status_code=400)
            if not (email and password and server):
                return JSONResponse({"error": "email, password and server are required"}, status_code=400)
            from config.credentials import normalize_account
            try:
                acct_idx = normalize_account(account)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            store = _credentials_store()
            if not store:
                return JSONResponse({"error": "store unavailable"}, status_code=500)
            try:
                store.save_credentials(tag, email, password, server, acct_idx,
                                       balance=balance, challenge_cost=challenge_cost)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            logger.info(f"[credentials] uploaded credentials for {tag} "
                        f"(account index {acct_idx}) — restart the bot to apply")
            return JSONResponse({"ok": True, "strategy": tag, "account": acct_idx,
                                 "message": "saved — use Save & Restart to apply"})

        @app.delete("/api/credentials/{tag}")
        async def credentials_delete(tag: str, request: Request):
            if not broadcaster._session_cookie_valid(cookie_of(request)):
                return denied_json()
            store = _credentials_store()
            if not store:
                return JSONResponse({"error": "store unavailable"}, status_code=500)
            removed = store.delete_credentials(tag)
            logger.info(f"[credentials] deleted credentials for {tag} (removed={removed})")
            return JSONResponse({"ok": True, "removed": removed})

        @app.get("/api/parity")
        async def parity_get(request: Request):
            if not broadcaster._session_cookie_valid(cookie_of(request)):
                return denied_json()
            provider = broadcaster._parity_provider
            try:
                reports = provider() if provider else {}
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
            return JSONResponse({"parity": reports or {}})

        @app.post("/api/parity/run")
        async def parity_run(request: Request):
            if not broadcaster._session_cookie_valid(cookie_of(request)):
                return denied_json()
            trigger = broadcaster._parity_trigger
            if not trigger:
                return JSONResponse({"error": "parity checker unavailable"}, status_code=503)
            import threading
            threading.Thread(target=trigger, daemon=True).start()
            return JSONResponse({"ok": True, "message": "parity check started"})

        @app.post("/api/restart")
        async def restart(request: Request):
            if not broadcaster._session_cookie_valid(cookie_of(request)):
                return denied_json()
            logger.info("[restart] requested via dashboard — exiting (container restarts)")
            asyncio.get_running_loop().create_task(broadcaster._do_restart())
            return JSONResponse({"ok": True, "message": "restarting"})

        return app

    async def _do_restart(self):
        await asyncio.sleep(0.5)  # let the response flush
        fn = self.restart_fn
        if fn:
            fn()
        os._exit(0)

    def _run_server(self):
        from fastapi import FastAPI  # noqa: F401
        app = self.build_app()
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._periodic_broadcast())
        self._loop.create_task(self._periodic_log_drain())

        import uvicorn
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        self._loop.run_until_complete(server.serve())

    # ── Command handling ───────────────────────────────────────────────────

    async def _handle_command(self, ws, raw: str):
        try:
            cmd = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        action = cmd.get("action")
        if action == "request_snapshot":
            with self._snapshot_lock:
                snap = dict(self._snapshot)
            await ws.send_json({"type": "snapshot", "data": snap})

    # ── Periodic broadcast ─────────────────────────────────────────────────

    async def _periodic_broadcast(self):
        while True:
            try:
                await asyncio.sleep(BROADCAST_INTERVAL)
                if not self._clients:
                    continue
                with self._snapshot_lock:
                    snap = dict(self._snapshot)
                if snap:
                    payload = {"type": "update", "data": snap}
                    dead = []
                    for ws in self._clients:
                        try:
                            await ws.send_json(payload)
                        except Exception:
                            dead.append(ws)
                    for ws in dead:
                        self._clients.discard(ws)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WS broadcast] error: {e}")

    async def _periodic_log_drain(self):
        while True:
            try:
                await asyncio.sleep(LOG_DRAIN_INTERVAL)
                if not self._clients:
                    continue
                while self._log_queue:
                    try:
                        entry = self._log_queue.popleft()
                    except IndexError:
                        break
                    dead = []
                    for ws in self._clients:
                        try:
                            await ws.send_json(entry)
                        except Exception:
                            dead.append(ws)
                    for ws in dead:
                        self._clients.discard(ws)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WS log-drain] error: {e}")

    # ── Thread-safe API (called from poll thread) ──────────────────────────

    def broadcast_state(self, snapshot: dict):
        with self._snapshot_lock:
            self._snapshot = snapshot
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._force_broadcast(), self._loop)

    async def _force_broadcast(self):
        if not self._clients:
            return
        with self._snapshot_lock:
            snap = dict(self._snapshot)
        if not snap:
            return
        payload = {"type": "update", "data": snap}
        dead = []
        for ws in self._clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def push_log(self, level: str, text: str, ts: str = None):
        entry = {
            "type": "log",
            "ts": ts or time.strftime("%H:%M:%S", time.gmtime()),
            "level": level,
            "text": text,
        }
        self._log_queue.append(entry)

    def close(self):
        self._started = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("WebSocket broadcaster stopped")

    @property
    def client_count(self):
        return len(self._clients)
