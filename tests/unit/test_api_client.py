import unittest
from unittest.mock import patch, MagicMock
from api.trade_locker_client import TradeLockerClient, TradeLockerError, RESOLUTIONS


class TestAuth(unittest.TestCase):
    def test_init(self):
        c = TradeLockerClient(live=False, email="a@b.com", password="pw", server="s1")
        self.assertEqual(c.base_url, "https://demo.tradelocker.com/backend-api")
        self.assertEqual(c.email, "a@b.com")
        self.assertIsNone(c.access_token)
        self.assertIsNone(c.acc_num)

    def test_init_live(self):
        c = TradeLockerClient(live=True)
        self.assertEqual(c.base_url, "https://live.tradelocker.com/backend-api")

    def test_ensure_token_no_creds(self):
        c = TradeLockerClient()
        self.assertFalse(c.ensure_token())

    def test_ensure_token_with_existing(self):
        c = TradeLockerClient(access_token="tok", refresh_token="ref")
        with patch.object(c, "all_accounts", return_value=[{"id": 1}]):
            self.assertTrue(c.ensure_token())

    def test_login_success(self):
        c = TradeLockerClient(email="a@b.com", password="pw", server="s1")
        with patch.object(c, "_request", return_value={"accessToken": "new_acc", "refreshToken": "new_ref"}):
            c.login()
            self.assertEqual(c.access_token, "new_acc")
            self.assertEqual(c.refresh_token, "new_ref")

    def test_login_missing(self):
        c = TradeLockerClient()
        with self.assertRaises(TradeLockerError):
            c.login()

    def test_refresh(self):
        c = TradeLockerClient(refresh_token="old_ref")
        with patch.object(c, "_request", return_value={"accessToken": "acc2", "refreshToken": "ref2"}):
            c.refresh()
            self.assertEqual(c.access_token, "acc2")

    def test_refresh_no_token(self):
        c = TradeLockerClient()
        with self.assertRaises(TradeLockerError):
            c.refresh()

    def test_set_tokens_callback(self):
        cb = MagicMock()
        c = TradeLockerClient(on_tokens=cb)
        c._set_tokens({"accessToken": "a1", "refreshToken": "r1"})
        cb.assert_called_once_with("a1", "r1")
        self.assertEqual(c.access_token, "a1")

    def test_set_tokens_no_access_raises(self):
        c = TradeLockerClient()
        with self.assertRaises(TradeLockerError):
            c._set_tokens({})

    def test_try_restore_refresh_ok(self):
        c = TradeLockerClient(refresh_token="ref")
        with patch.object(c, "refresh", return_value=True):
            self.assertTrue(c._try_restore_auth())

    def test_try_restore_relogin_fallback(self):
        c = TradeLockerClient(email="a@b.com", password="pw", server="s1", refresh_token="ref")
        with patch.object(c, "refresh", side_effect=TradeLockerError("fail")), \
             patch.object(c, "login", return_value=True):
            self.assertTrue(c._try_restore_auth())


class TestRequests(unittest.TestCase):
    def test_headers(self):
        c = TradeLockerClient(access_token="abc123")
        h = c._headers()
        self.assertEqual(h["Authorization"], "Bearer abc123")
        self.assertNotIn("accNum", h)

    def test_headers_acc_num(self):
        c = TradeLockerClient(access_token="abc")
        c.acc_num = 42
        h = c._headers()
        self.assertEqual(h["accNum"], "42")

    @patch("api.trade_locker_client.requests.request")
    def test_401_triggers_refresh(self, mock_req):
        resp_401 = MagicMock(status_code=401, content=b'{}', text="unauthorized")
        resp_ok = MagicMock(status_code=200, content=b'{"s":"ok","d":{"accounts":[]}}', json=lambda: {"s": "ok", "d": {"accounts": []}})
        mock_req.side_effect = [resp_401, resp_ok]
        c = TradeLockerClient(access_token="exp", refresh_token="ref")
        with patch.object(c, "_try_restore_auth", return_value=True):
            result = c.all_accounts()
        self.assertEqual(mock_req.call_count, 2)

    @patch("api.trade_locker_client.requests.request")
    def test_all_accounts(self, mock_req):
        mock_req.return_value = MagicMock(
            status_code=200, content=b'ok',
            json=lambda: {"s": "ok", "d": {"accounts": [{"id": 1}]}}
        )
        c = TradeLockerClient(access_token="tok")
        accounts = c.all_accounts()
        self.assertEqual(len(accounts), 1)

    @patch("api.trade_locker_client.requests.request")
    def test_close_position(self, mock_req):
        mock_req.return_value = MagicMock(
            status_code=200, content=b'ok',
            json=lambda: {"s": "ok", "d": {"result": "closed"}}
        )
        c = TradeLockerClient(access_token="tok")
        result = c.close_position(123, qty=1.0)
        self.assertIn("result", result)

    @patch("api.trade_locker_client.requests.request")
    def test_history(self, mock_req):
        mock_req.return_value = MagicMock(
            status_code=200, content=b'ok',
            json=lambda: {"s": "ok", "d": {"barDetails": [{"o": 1}]}}
        )
        c = TradeLockerClient(access_token="tok")
        bars = c.history(1, 2, "1H", 0, 1000)
        self.assertEqual(len(bars), 1)

    @patch("api.trade_locker_client.requests.request")
    def test_modify_position(self, mock_req):
        mock_req.return_value = MagicMock(
            status_code=200, content=b'ok',
            json=lambda: {"s": "ok", "d": {"modified": True}}
        )
        c = TradeLockerClient(access_token="tok")
        result = c.modify_position(99, stop_loss=1.1, take_profit=1.3)
        self.assertTrue(result.get("modified"))
        # TradeLocker API v2.11 uses PATCH — PUT returns HTTP 405
        self.assertEqual(mock_req.call_args[0][0], "PATCH")
        self.assertIn("/trade/positions/99", mock_req.call_args[0][1])

    @patch("api.trade_locker_client.requests.request")
    def test_place_order(self, mock_req):
        mock_req.return_value = MagicMock(
            status_code=200, content=b'ok',
            json=lambda: {"s": "ok", "d": {"orderId": "ord-123"}}
        )
        c = TradeLockerClient(access_token="tok")
        oid = c.place_order(1, {"side": "buy"})
        self.assertEqual(oid, "ord-123")

    @patch("api.trade_locker_client.requests.request")
    def test_positions(self, mock_req):
        mock_req.return_value = MagicMock(
            status_code=200, content=b'ok',
            json=lambda: {"s": "ok", "d": {"positions": []}}
        )
        c = TradeLockerClient(access_token="tok")
        c.columns = {"positions": []}
        result = c.positions(1)
        self.assertIsInstance(result, list)

    @patch("api.trade_locker_client.requests.request")
    def test_quotes(self, mock_req):
        mock_req.return_value = MagicMock(
            status_code=200, content=b'ok',
            json=lambda: {"s": "ok", "d": {"bid": 100, "ask": 101}}
        )
        c = TradeLockerClient(access_token="tok")
        q = c.quotes(1, 2)
        self.assertEqual(q["bid"], 100)

    def test_resolutions(self):
        self.assertEqual(RESOLUTIONS[1], "1m")
        self.assertEqual(RESOLUTIONS[60], "1H")
        self.assertEqual(RESOLUTIONS[1440], "1D")


if __name__ == "__main__":
    unittest.main()
