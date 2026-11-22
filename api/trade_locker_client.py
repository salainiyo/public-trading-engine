"""Thin synchronous client for the TradeLocker Public REST API.

All methods are blocking (requests) — call them from a worker thread, or via
asyncio.to_thread from the bot's event loop so the WebSocket broadcast never
stalls.

Auth model (self-healing, no manual token updates):
  - On startup the bot logs in with email/password/server -> JWT pair.
  - The access token expires in ~1 hour: any 401 automatically triggers a
    refresh via /auth/jwt/refresh and a retry.
  - Refresh tokens expire in ~7 days: if a refresh fails, the client falls
    back to a full email/password/server login.
  - New tokens are persisted to bot/data/tokens.json via the `on_tokens` callback.

TradeLocker API conventions handled here:
  - Base URL: https://demo.tradelocker.com/backend-api (or live).
  - `Authorization: Bearer <accessToken>` on every /trade and /auth request
    (except the token login/refresh calls themselves).
  - `accNum` header on /trade/* + /auth/jwt/all-accounts.
  - Responses are wrapped in `{"d": ..., "s": "ok"}` envelopes.
  - Tabular endpoints (positions/orders/ordersHistory/state) return rows whose
    column order is defined by /trade/config — we fetch it once and map rows to
    dicts by column id, so renamed/reordered columns don't break us.
"""
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import requests

log = logging.getLogger("tradelocker_api")

DEMO_BASE_URL = "https://demo.tradelocker.com/backend-api"
LIVE_BASE_URL = "https://live.tradelocker.com/backend-api"

RESOLUTIONS = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1H", 240: "4H", 1440: "1D"}


class TradeLockerError(Exception):
    """Base error raised for HTTP/API-level failures (incl. business errors)."""


class TradeLockerClient:
    def __init__(
        self,
        live: bool = False,
        email: Optional[str] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        on_tokens: Optional[Callable[[str, Optional[str]], None]] = None,
    ):
        self.base_url = LIVE_BASE_URL if live else DEMO_BASE_URL
        self.email = email
        self.password = password
        self.server = server
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.on_tokens = on_tokens
        self.acc_num: Optional[int] = None
        self.account_id: Optional[int] = None
        self.columns: Dict[str, List[str]] = {}
        self._auth_lock = threading.Lock()

    def _headers(self) -> Dict[str, str]:
        h = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        if self.access_token:
            h["Authorization"] = f"Bearer {self.access_token}"
        if self.acc_num is not None:
            h["accNum"] = str(self.acc_num)
        return h

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        raw: bool = False,
        allow_auth: bool = True,
    ) -> Any:
        url = self.base_url + path
        for attempt in range(4):
            try:
                resp = requests.request(
                    method, url, params=params, json=body,
                    headers=self._headers(), timeout=20,
                )
            except requests.RequestException as e:
                if attempt == 0:
                    log.warning(f"[tl_api] {method} {path} network error: {e} — retrying")
                    time.sleep(0.5)
                    continue
                raise TradeLockerError(f"{method} {path}: network error: {e}") from e

            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, 10)
                log.warning(f"[tl_api] {method} {path} rate-limited (429) — retrying in {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code == 401 and allow_auth and attempt == 0 and self.access_token:
                if self._try_restore_auth():
                    continue
            if resp.status_code >= 400:
                raise self._raise_api_error(method, path, resp)

            data = resp.json() if resp.content else {}
            if raw:
                return data
            if isinstance(data, dict):
                s = data.get("s")
                if s not in (None, "ok"):
                    msg = data.get("d") or data.get("message") or data.get("errmsg") or "unknown"
                    log.warning(f"[tl_api] {method} {path}: full response={data}")
                    raise TradeLockerError(f"{method} {path}: status={s} {msg}")
                return data["d"] if "d" in data else data
            return data
        raise TradeLockerError(f"{method} {path}: exhausted retries")

    @staticmethod
    def _raise_api_error(method: str, path: str, resp: requests.Response):
        detail = resp.text[:500] if resp.text else "no body"
        if resp.status_code == 401:
            return TradeLockerError(f"{method} {path}: 401 unauthorized {detail}")
        if resp.status_code == 404:
            return TradeLockerError(f"{method} {path}: 404 not found {detail}")
        return TradeLockerError(f"{method} {path}: HTTP {resp.status_code} {detail}")

    # ── Auth ──────────────────────────────────────────────────────────────
    def _set_tokens(self, data: dict):
        new_access = data.get("accessToken")
        if new_access:
            self.access_token = new_access
            self.refresh_token = data.get("refreshToken", self.refresh_token)
            if self.on_tokens:
                try:
                    self.on_tokens(self.access_token, self.refresh_token)
                except Exception as e:
                    log.warning(f"[tl_api] on_tokens callback failed: {e}")
        else:
            raise TradeLockerError("auth response missing accessToken")

    def login(self):
        if not (self.email and self.password and self.server):
            raise TradeLockerError("login() requires email, password and server")
        body = {"email": self.email, "password": self.password, "server": self.server}
        data = self._request("POST", "/auth/jwt/token", body=body, raw=True, allow_auth=False)
        self._set_tokens(data)
        log.info("[tl_api] Logged in (email/password/server)")

    def refresh(self):
        if not (self.refresh_token):
            raise TradeLockerError("refresh() requires a refresh token")
        body = {"refreshToken": self.refresh_token}
        data = self._request("POST", "/auth/jwt/refresh", body=body, raw=True, allow_auth=False)
        self._set_tokens(data)
        log.info("[tl_api] Access token refreshed")

    def _try_restore_auth(self) -> bool:
        with self._auth_lock:
            try:
                self.refresh()
                return True
            except TradeLockerError as e:
                log.warning(f"[tl_api] token refresh failed ({e}) — attempting re-login")
            if self.email and self.password and self.server:
                try:
                    self.login()
                    return True
                except TradeLockerError as e:
                    log.error(f"[tl_api] re-login failed: {e}")
            return False

    def ensure_token(self) -> bool:
        if self.access_token:
            try:
                self.all_accounts()
                return True
            except TradeLockerError:
                pass
        if self.refresh_token:
            try:
                self.refresh()
                return True
            except TradeLockerError:
                pass
        if self.email and self.password and self.server:
            try:
                self.login()
                return True
            except TradeLockerError as e:
                log.error(f"[tl_api] login failed: {e}")
        return False

    # ── Accounts / config ─────────────────────────────────────────────────
    def all_accounts(self) -> List[dict]:
        d = self._request("GET", "/auth/jwt/all-accounts")
        return (d or {}).get("accounts", [])

    def config(self) -> dict:
        d = self._request("GET", "/trade/config")
        if d:
            self.columns = {
                key: [c.get("id") for c in (d.get(cfg, {}) or {}).get("columns", [])]
                for key, cfg in {
                    "positions": "positionsConfig",
                    "orders": "ordersConfig",
                    "ordersHistory": "ordersHistoryConfig",
                    "filledOrders": "filledOrdersConfig",
                    "accountDetails": "accountDetailsConfig",
                }.items()
            }
        return d or {}

    def _map(self, key: str, rows: List[list]) -> List[dict]:
        cols = self.columns.get(key, [])
        out = []
        for row in rows or []:
            out.append({c: (row[i] if i < len(row) else None) for i, c in enumerate(cols)})
        return out

    # ── Instruments / market data ─────────────────────────────────────────
    def instruments(self, account_id: int) -> List[dict]:
        d = self._request("GET", f"/trade/accounts/{account_id}/instruments")
        return (d or {}).get("instruments", [])

    def instrument_details(self, tradable_instrument_id: int, route_id: int) -> dict:
        d = self._request(
            "GET", f"/trade/instruments/{tradable_instrument_id}",
            params={"routeId": route_id},
        )
        return d or {}

    def history(
        self,
        tradable_instrument_id: int,
        route_id: int,
        resolution: str,
        from_ms: int,
        to_ms: int,
    ) -> List[dict]:
        params = {
            "tradableInstrumentId": tradable_instrument_id,
            "routeId": route_id,
            "resolution": resolution,
            "from": int(from_ms),
            "to": int(to_ms),
        }
        d = self._request("GET", "/trade/history", params=params)
        return (d or {}).get("barDetails", [])

    def quotes(self, tradable_instrument_id: int, route_id: int) -> dict:
        params = {"tradableInstrumentId": tradable_instrument_id, "routeId": route_id}
        d = self._request("GET", "/trade/quotes", params=params)
        return d or {}

    # ── Account trading data ──────────────────────────────────────────────
    def positions(self, account_id: int) -> List[dict]:
        d = self._request("GET", f"/trade/accounts/{account_id}/positions")
        return self._map("positions", (d or {}).get("positions", []))

    def orders(self, account_id: int) -> List[dict]:
        d = self._request("GET", f"/trade/accounts/{account_id}/orders")
        return self._map("orders", (d or {}).get("orders", []))

    def orders_history(
        self,
        account_id: int,
        from_ms: Optional[int] = None,
        to_ms: Optional[int] = None,
        tradable_instrument_id: Optional[int] = None,
    ) -> List[dict]:
        params = {}
        if from_ms is not None:
            params["from"] = int(from_ms)
        if to_ms is not None:
            params["to"] = int(to_ms)
        if tradable_instrument_id is not None:
            params["tradableInstrumentId"] = tradable_instrument_id
        d = self._request("GET", f"/trade/accounts/{account_id}/ordersHistory", params=params)
        return self._map("ordersHistory", (d or {}).get("ordersHistory", []))

    def state(self, account_id: int) -> dict:
        d = self._request("GET", f"/trade/accounts/{account_id}/state")
        rows = self._map("accountDetails", [(d or {}).get("accountDetailsData", [])])
        return rows[0] if rows else {}

    # ── Trading ───────────────────────────────────────────────────────────
    def place_order(self, account_id: int, body: dict) -> Optional[str]:
        d = self._request("POST", f"/trade/accounts/{account_id}/orders", body=body)
        return (d or {}).get("orderId")

    def cancel_order(self, account_id: int, order_id) -> dict:
        """Cancel an unfilled (limit) order."""
        d = self._request("DELETE", f"/trade/accounts/{account_id}/orders/{order_id}")
        return d or {}

    def modify_position(self, position_id: int, stop_loss=None, take_profit=None) -> dict:
        body = {}
        if stop_loss is not None:
            body["stopLoss"] = stop_loss
        if take_profit is not None:
            body["takeProfit"] = take_profit
        # TradeLocker API v2.11: modify position is PATCH (PUT → HTTP 405)
        d = self._request("PATCH", f"/trade/positions/{position_id}", body=body)
        return d or {}

    def close_position(self, position_id, qty: float = 0, strategy_id: Optional[str] = None) -> dict:
        params = {}
        if strategy_id:
            params["strategyId"] = strategy_id
        d = self._request(
            "DELETE", f"/trade/positions/{position_id}",
            params=params, body={"qty": qty},
        )
        return d or {}
