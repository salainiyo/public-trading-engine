import asyncio
import json
import logging
import os
import time
import unittest
import threading

from events.websocket_broadcaster import WebSocketBroadcaster


class TestWebSocketBroadcaster(unittest.TestCase):
    def setUp(self):
        self.port = 18769
        self.broadcaster = WebSocketBroadcaster(port=self.port)

    def tearDown(self):
        try:
            self.broadcaster.close()
        except Exception:
            pass

    def test_init_state(self):
        b = WebSocketBroadcaster(port=18770)
        self.assertFalse(b._started)
        self.assertEqual(len(b._clients), 0)

    def test_start_is_idempotent(self):
        self.broadcaster.start()
        self.assertTrue(self.broadcaster._started)
        first_thread = self.broadcaster._thread
        self.broadcaster.start()
        self.assertIs(self.broadcaster._thread, first_thread)

    def test_broadcast_state_stores_snapshot(self):
        snap = {"test": "data", "value": 42}
        self.broadcaster.broadcast_state(snap)
        with self.broadcaster._snapshot_lock:
            stored = dict(self.broadcaster._snapshot)
        self.assertEqual(stored["test"], "data")
        self.assertEqual(stored["value"], 42)

    def test_broadcast_state_overwrites(self):
        self.broadcaster.broadcast_state({"a": 1})
        self.broadcaster.broadcast_state({"b": 2})
        with self.broadcaster._snapshot_lock:
            stored = dict(self.broadcaster._snapshot)
        self.assertNotIn("a", stored)
        self.assertEqual(stored["b"], 2)

    def test_push_log(self):
        self.broadcaster.push_log("INFO", "test message")
        self.assertEqual(len(self.broadcaster._log_queue), 1)
        entry = self.broadcaster._log_queue[0]
        self.assertEqual(entry["level"], "INFO")
        self.assertEqual(entry["text"], "test message")

    def test_push_log_custom_ts(self):
        self.broadcaster.push_log("ERROR", "err", ts="12:34:56")
        entry = self.broadcaster._log_queue[0]
        self.assertEqual(entry["ts"], "12:34:56")

    def test_log_queue_bounded(self):
        for i in range(600):
            self.broadcaster.push_log("INFO", f"msg {i}")
        self.assertEqual(len(self.broadcaster._log_queue), 500)

    def test_close_is_idempotent(self):
        self.broadcaster.close()
        self.broadcaster.close()


class TestWSHealthEndpoint(unittest.TestCase):
    def test_health_endpoint(self):
        port = 18771
        broadcaster = WebSocketBroadcaster(port=port)
        broadcaster.start()
        time.sleep(1.0)
        try:
            import requests
            resp = requests.get(f"http://localhost:{port}/health", timeout=5)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "ok")
        finally:
            broadcaster.close()


class TestDashboardAuth(unittest.TestCase):
    """Real-server tests: login, session cookie, credentials API, recovery."""

    PORT = 18781

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmpdir = tempfile.mkdtemp()
        cls._old_data_dir = os.environ.get("GOATBOT_DATA_DIR")
        os.environ["GOATBOT_DATA_DIR"] = cls._tmpdir
        os.environ["DASHBOARD_PASSWORD"] = "test-password-123"
        os.environ.pop("SMTP_EMAIL", None)
        os.environ.pop("SMTP_APP_PASSWORD", None)
        os.environ.pop("RECOVERY_EMAIL", None)
        cls.broadcaster = WebSocketBroadcaster(port=cls.PORT)
        cls.broadcaster.start()
        cls._wait_health()

    @classmethod
    def _wait_health(cls, timeout=10):
        import requests
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if requests.get(f"http://localhost:{cls.PORT}/health", timeout=2).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.2)
        raise RuntimeError("server did not start")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.broadcaster.close()
        except Exception:
            pass
        if cls._old_data_dir is None:
            os.environ.pop("GOATBOT_DATA_DIR", None)
        else:
            os.environ["GOATBOT_DATA_DIR"] = cls._old_data_dir
        for k in ("DASHBOARD_PASSWORD", "SMTP_EMAIL", "SMTP_APP_PASSWORD", "RECOVERY_EMAIL"):
            os.environ.pop(k, None)
        import shutil
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    @classmethod
    def _base(cls):
        return f"http://localhost:{cls.PORT}"

    def _login_session(self, password="test-password-123"):
        import requests
        s = requests.Session()
        resp = s.post(f"{self._base()}/login", data={"password": password}, timeout=5,
                      allow_redirects=False)
        return s, resp

    def test_health_open_without_auth(self):
        import requests
        resp = requests.get(f"{self._base()}/health", timeout=5)
        self.assertEqual(resp.status_code, 200)

    def test_index_redirects_without_cookie(self):
        import requests
        resp = requests.get(self._base() + "/", timeout=5, allow_redirects=False)
        self.assertEqual(resp.status_code, 307)
        self.assertIn("/login", resp.headers.get("location", ""))

    def test_login_page_renders(self):
        import requests
        resp = requests.get(f"{self._base()}/login", timeout=5)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Sign in", resp.text)

    def test_login_wrong_password(self):
        _, resp = self._login_session("wrong-password")
        self.assertEqual(resp.status_code, 303)
        self.assertIn("/login?error=1", resp.headers.get("location", ""))

    def test_login_correct_password_sets_cookie(self):
        s, resp = self._login_session()
        self.assertEqual(resp.status_code, 303)
        self.assertIn("/", resp.headers.get("location", ""))
        self.assertIn("gpf_session", s.cookies.get_dict())

    def test_index_loads_with_cookie(self):
        s, _ = self._login_session()
        resp = s.get(self._base() + "/", timeout=5, allow_redirects=False)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("GOAT PROP FARM", resp.text)

    def test_auth_check_without_cookie(self):
        import requests
        resp = requests.get(f"{self._base()}/auth-check", timeout=5)
        self.assertEqual(resp.status_code, 401)

    def test_auth_check_with_cookie(self):
        s, _ = self._login_session()
        resp = s.get(f"{self._base()}/auth-check", timeout=5)
        self.assertEqual(resp.status_code, 200)

    def test_credentials_upload_and_redaction(self):
        s, _ = self._login_session()
        resp = s.post(f"{self._base()}/api/credentials", timeout=5, json={
            "strategy": "GOAT5", "email": "me@gmail.com",
            "password": "my-broker-secret", "server": "srv.example.com",
            "account": "second"})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json().get("ok"))
        listing = s.get(f"{self._base()}/api/credentials", timeout=5).json()
        entry = listing["credentials"]["GOAT5"]
        self.assertEqual(entry["email"], "me@gmail.com")
        self.assertEqual(entry["account"], 2)
        self.assertNotIn("password", entry)
        self.assertIn("••••", entry["password_hint"])
        # no plaintext in the DB file
        import glob
        dbs = glob.glob(os.path.join(self.__class__._tmpdir, "*.db"))
        raw = b"".join(open(p, "rb").read() for p in dbs)
        self.assertNotIn(b"my-broker-secret", raw)

    def test_credentials_requires_auth(self):
        import requests
        resp = requests.post(f"{self._base()}/api/credentials", timeout=5, json={})
        self.assertEqual(resp.status_code, 401)

    def test_credentials_unknown_strategy(self):
        s, _ = self._login_session()
        resp = s.post(f"{self._base()}/api/credentials", timeout=5, json={
            "strategy": "NOPE", "email": "a@b.com", "password": "x",
            "server": "s", "account": "1"})
        self.assertEqual(resp.status_code, 400)

    def test_credentials_invalid_account(self):
        s, _ = self._login_session()
        resp = s.post(f"{self._base()}/api/credentials", timeout=5, json={
            "strategy": "GOAT5", "email": "a@b.com", "password": "x",
            "server": "s", "account": "banana"})
        self.assertEqual(resp.status_code, 400)

    def test_credentials_delete(self):
        s, _ = self._login_session()
        s.post(f"{self._base()}/api/credentials", timeout=5, json={
            "strategy": "AQUA5", "email": "a@b.com", "password": "x",
            "server": "s", "account": "first"})
        resp = s.delete(f"{self._base()}/api/credentials/AQUA5", timeout=5)
        self.assertTrue(resp.json().get("removed"))
        listing = s.get(f"{self._base()}/api/credentials", timeout=5).json()
        self.assertNotIn("AQUA5", listing["credentials"])

    def test_change_password_invalidates_old_sessions(self):
        s, _ = self._login_session()
        resp = s.post(f"{self._base()}/api/auth/change", timeout=5, json={
            "current_password": "test-password-123", "new_password": "brand-new-pw-1"})
        self.assertEqual(resp.status_code, 200)
        # the old session cookie is invalidated (session_secret regenerated)
        resp = s.get(f"{self._base()}/auth-check", timeout=5)
        self.assertEqual(resp.status_code, 401)
        # login with the new password works
        s2, resp2 = self._login_session("brand-new-pw-1")
        self.assertEqual(resp2.status_code, 303)
        self.assertEqual(s2.get(f"{self._base()}/auth-check", timeout=5).status_code, 200)
        # restore the bootstrap password for the other tests
        s2.post(f"{self._base()}/api/auth/change", timeout=5, json={
            "current_password": "brand-new-pw-1", "new_password": "test-password-123"})

    def test_change_password_wrong_current(self):
        s, _ = self._login_session()
        resp = s.post(f"{self._base()}/api/auth/change", timeout=5, json={
            "current_password": "totally-wrong", "new_password": "brand-new-pw-2"})
        self.assertEqual(resp.status_code, 400)

    def test_recover_and_reset_via_logged_code(self):
        import requests
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Capture()
        logging.getLogger().addHandler(handler)
        try:
            resp = requests.post(f"{self._base()}/api/auth/recover", timeout=5)
        finally:
            logging.getLogger().removeHandler(handler)
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json().get("sent"))  # SMTP unconfigured -> logged
        code_lines = [m for m in records if "RECOVERY CODE" in m]
        self.assertTrue(code_lines)
        code = code_lines[-1].rsplit(": ", 1)[-1].strip()
        # reset with the code
        resp = requests.post(f"{self._base()}/api/auth/reset", timeout=5,
                             json={"code": code, "new_password": "recovered-pw-9"})
        self.assertEqual(resp.status_code, 200)
        # the code is one-time use
        resp = requests.post(f"{self._base()}/api/auth/reset", timeout=5,
                             json={"code": code, "new_password": "again-pw-999999"})
        self.assertEqual(resp.status_code, 400)
        # login with the recovered password works
        s, resp = self._login_session("recovered-pw-9")
        self.assertEqual(resp.status_code, 303)
        # restore the bootstrap password for the other tests
        s.post(f"{self._base()}/api/auth/change", timeout=5, json={
            "current_password": "recovered-pw-9", "new_password": "test-password-123"})

    def test_recover_rate_limited(self):
        import requests
        resp = requests.post(f"{self._base()}/api/auth/recover", timeout=5)
        self.assertEqual(resp.status_code, 429)

    def test_recovery_email_update(self):
        s, _ = self._login_session()
        resp = s.post(f"{self._base()}/api/auth/recovery-email", timeout=5,
                      json={"recovery_email": "new-recovery@gmail.com"})
        self.assertEqual(resp.status_code, 200)
        listing = s.get(f"{self._base()}/api/credentials", timeout=5).json()
        self.assertEqual(listing.get("recovery_email"), "new-recovery@gmail.com")

    def test_ws_rejected_without_cookie(self):
        import websockets

        async def attempt():
            uri = f"ws://localhost:{self.PORT}/ws/"
            async with websockets.connect(uri, open_timeout=5) as ws:
                await ws.recv()

        with self.assertRaises(Exception):
            asyncio.run(attempt())

    def test_ws_works_with_cookie(self):
        import websockets
        s, _ = self._login_session()
        cookie = s.cookies.get_dict().get("gpf_session")

        async def attempt():
            uri = f"ws://localhost:{self.PORT}/ws/"
            async with websockets.connect(uri, open_timeout=5,
                                          additional_headers={"Cookie": f"gpf_session={cookie}"}) as ws:
                await ws.send(json.dumps({"action": "request_snapshot"}))
                return await asyncio.wait_for(ws.recv(), timeout=5)

        msg = json.loads(asyncio.run(attempt()))
        self.assertEqual(msg.get("type"), "snapshot")


class TestAuthDisabled(unittest.TestCase):
    """No DASHBOARD_PASSWORD -> auth disabled, everything open (current behavior)."""

    PORT = 18782

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmpdir = tempfile.mkdtemp()
        cls._old_data_dir = os.environ.get("GOATBOT_DATA_DIR")
        os.environ["GOATBOT_DATA_DIR"] = cls._tmpdir
        os.environ.pop("DASHBOARD_PASSWORD", None)
        cls.broadcaster = WebSocketBroadcaster(port=cls.PORT)
        cls.broadcaster.start()
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                import requests
                if requests.get(f"http://localhost:{cls.PORT}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.broadcaster.close()
        except Exception:
            pass
        if cls._old_data_dir is None:
            os.environ.pop("GOATBOT_DATA_DIR", None)
        else:
            os.environ["GOATBOT_DATA_DIR"] = cls._old_data_dir
        import shutil
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_index_open_without_auth(self):
        import requests
        resp = requests.get(f"http://localhost:{self.PORT}/", timeout=5, allow_redirects=False)
        self.assertEqual(resp.status_code, 200)

    def test_auth_check_ok_when_disabled(self):
        import requests
        resp = requests.get(f"http://localhost:{self.PORT}/auth-check", timeout=5)
        self.assertEqual(resp.status_code, 200)

    def test_login_redirects_to_dashboard_when_disabled(self):
        import requests
        resp = requests.get(f"http://localhost:{self.PORT}/login", timeout=5, allow_redirects=False)
        self.assertEqual(resp.status_code, 307)
        self.assertIn("/", resp.headers.get("location", ""))


if __name__ == "__main__":
    unittest.main()
