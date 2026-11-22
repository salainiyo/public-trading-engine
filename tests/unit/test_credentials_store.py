import os
import stat
import tempfile
import unittest

from config.credentials import CredentialsStore, normalize_account


class TestNormalizeAccount(unittest.TestCase):
    def test_words(self):
        self.assertEqual(normalize_account("first"), 1)
        self.assertEqual(normalize_account("Second"), 2)
        self.assertEqual(normalize_account("third"), 3)

    def test_numeric(self):
        self.assertEqual(normalize_account("1"), 1)
        self.assertEqual(normalize_account(2), 2)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            normalize_account(None)
        with self.assertRaises(ValueError):
            normalize_account("banana")
        with self.assertRaises(ValueError):
            normalize_account("0")
        with self.assertRaises(ValueError):
            normalize_account("-1")


class TestCredentialsStore(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "credentials.db")
        self._old_data_dir = os.environ.get("GOATBOT_DATA_DIR")
        os.environ["GOATBOT_DATA_DIR"] = self._tmpdir
        self.store = CredentialsStore(db_path=self.db_path)
        os.environ.pop("CREDENTIALS_KEY", None)

    def tearDown(self):
        self.store.close()
        if self._old_data_dir is None:
            os.environ.pop("GOATBOT_DATA_DIR", None)
        else:
            os.environ["GOATBOT_DATA_DIR"] = self._old_data_dir
        os.environ.pop("CREDENTIALS_KEY", None)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_and_load_roundtrip(self):
        self.store.save_credentials("TCH", "a@b.com", "secret-pw", "srv.example.com", "first")
        cred = self.store.load_credentials("TCH")
        self.assertEqual(cred["email"], "a@b.com")
        self.assertEqual(cred["password"], "secret-pw")
        self.assertEqual(cred["server"], "srv.example.com")
        self.assertEqual(cred["account"], 1)

    def test_no_plaintext_in_db(self):
        self.store.save_credentials("TCH", "a@b.com", "super-secret-password-123", "srv", 2)
        with open(self.db_path, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"super-secret-password-123", raw)
        self.assertNotIn(b"a@b.com", raw)

    def test_key_file_generated_0600(self):
        self.store.save_credentials("TCH", "a@b.com", "pw", "srv", 1)
        key_path = os.path.join(self._tmpdir, "credentials.key")
        self.assertTrue(os.path.exists(key_path))
        with open(key_path, "rb") as f:
            self.assertEqual(len(f.read()), 32)
        mode = stat.S_IMODE(os.stat(key_path).st_mode)
        self.assertEqual(mode, 0o600)

    def test_key_roundtrip_across_instances(self):
        self.store.save_credentials("TCH", "a@b.com", "pw", "srv", 1)
        store2 = CredentialsStore(db_path=self.db_path)
        try:
            cred = store2.load_credentials("TCH")
            self.assertEqual(cred["password"], "pw")
        finally:
            store2.close()

    def test_wrong_env_key_fails_decrypt(self):
        self.store.save_credentials("TCH", "a@b.com", "pw", "srv", 1)
        os.environ["CREDENTIALS_KEY"] = os.urandom(32).hex()
        self.assertIsNone(self.store.load_credentials("TCH"))

    def test_list_all_redacts_password(self):
        self.store.save_credentials("TCH", "a@b.com", "mypassword", "srv", 2)
        entries = self.store.list_all()
        e = entries["TCH"]
        self.assertEqual(e["email"], "a@b.com")
        self.assertEqual(e["password_hint"], "••••rd")
        self.assertNotIn("password", e)

    def test_delete(self):
        self.store.save_credentials("TCH", "a@b.com", "pw", "srv", 1)
        self.assertTrue(self.store.delete_credentials("TCH"))
        self.assertIsNone(self.store.load_credentials("TCH"))
        self.assertFalse(self.store.delete_credentials("TCH"))

    def test_save_preserves_created_at(self):
        self.store.save_credentials("TCH", "a@b.com", "pw", "srv", 1)
        cur = self.store._conn.execute(
            "SELECT created_at FROM broker_credentials WHERE tag='TCH'")
        created1 = cur.fetchone()[0]
        self.store.save_credentials("TCH", "a@b.com", "pw2", "srv", 1)
        cur = self.store._conn.execute(
            "SELECT created_at, updated_at FROM broker_credentials WHERE tag='TCH'")
        created2, updated2 = cur.fetchone()
        self.assertEqual(created1, created2)
        self.assertEqual(self.store.load_credentials("TCH")["password"], "pw2")


class TestDashboardAuth(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "credentials.db")
        self.store = CredentialsStore(db_path=self.db_path)

    def tearDown(self):
        self.store.close()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_password_hash_verify(self):
        self.assertFalse(self.store.has_password())
        self.store.set_password("hunter2secret")
        self.assertTrue(self.store.has_password())
        self.assertTrue(self.store.verify_password("hunter2secret"))
        self.assertFalse(self.store.verify_password("wrong"))

    def test_password_change_invalidates_sessions(self):
        self.store.set_password("first-password")
        secret1 = self.store.get_session_secret()
        self.store.set_password("second-password")
        secret2 = self.store.get_session_secret()
        self.assertNotEqual(secret1, secret2)
        self.assertTrue(self.store.verify_password("second-password"))
        self.assertFalse(self.store.verify_password("first-password"))

    def test_recovery_email(self):
        self.assertIsNone(self.store.get_recovery_email())
        self.store.set_recovery_email("me@gmail.com")
        self.assertEqual(self.store.get_recovery_email(), "me@gmail.com")
        self.store.set_password("pw123456")
        self.assertEqual(self.store.get_recovery_email(), "me@gmail.com")

    def test_recovery_code_one_time_use(self):
        code = self.store.create_recovery_code(ttl_seconds=60)
        self.assertTrue(self.store.consume_recovery_code(code))
        self.assertFalse(self.store.consume_recovery_code(code))  # single use

    def test_recovery_code_expiry(self):
        code = self.store.create_recovery_code(ttl_seconds=-1)  # already expired
        self.assertFalse(self.store.consume_recovery_code(code))

    def test_recovery_code_invalid(self):
        self.assertFalse(self.store.consume_recovery_code("NOPE"))


if __name__ == "__main__":
    unittest.main()
