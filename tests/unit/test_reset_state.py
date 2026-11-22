"""One-shot fresh-start flag (RESET_STATE)."""
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import main


def _make_db(path, table="t"):
    con = sqlite3.connect(path)
    try:
        con.execute(f"CREATE TABLE {table} (x INTEGER)")
        con.execute(f"INSERT INTO {table} VALUES (42)")
        con.commit()
    finally:
        con.close()


def _rows(path):
    con = sqlite3.connect(path)
    try:
        return con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    finally:
        con.close()


class TestResetState(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.trades = os.path.join(self.tmp, "trades.db")
        self.treas = os.path.join(self.tmp, "treasuries.db")
        _make_db(self.trades)
        _make_db(self.treas)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_flag_is_noop(self):
        with patch.dict(os.environ, {"RESET_STATE": ""}, clear=False):
            self.assertFalse(main._maybe_reset_state(self.tmp))
        self.assertTrue(os.path.exists(self.trades))

    def test_flag_archives_and_clears(self):
        with patch.dict(os.environ, {"RESET_STATE": "1"}, clear=False):
            self.assertTrue(main._maybe_reset_state(self.tmp))
        self.assertFalse(os.path.exists(self.trades))
        self.assertFalse(os.path.exists(self.treas))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, ".reset_state_done")))
        archives = [d for d in os.listdir(self.tmp) if d.startswith("archive_reset_")]
        self.assertEqual(len(archives), 1)
        arch = os.path.join(self.tmp, archives[0])
        self.assertEqual(_rows(os.path.join(arch, "trades.db")), 1)
        self.assertEqual(_rows(os.path.join(arch, "treasuries.db")), 1)

    def test_same_flag_does_not_reset_twice(self):
        with patch.dict(os.environ, {"RESET_STATE": "1"}, clear=False):
            self.assertTrue(main._maybe_reset_state(self.tmp))
            _make_db(self.trades)  # new data since the reset
            self.assertFalse(main._maybe_reset_state(self.tmp))
        self.assertTrue(os.path.exists(self.trades))

    def test_new_value_resets_again(self):
        with patch.dict(os.environ, {"RESET_STATE": "1"}, clear=False):
            self.assertTrue(main._maybe_reset_state(self.tmp))
        _make_db(self.trades)
        with patch.dict(os.environ, {"RESET_STATE": "2"}, clear=False):
            self.assertTrue(main._maybe_reset_state(self.tmp))
        self.assertFalse(os.path.exists(self.trades))
        archives = [d for d in os.listdir(self.tmp) if d.startswith("archive_reset_")]
        self.assertEqual(len(archives), 2)

    def test_credentials_and_tokens_are_kept(self):
        cred = os.path.join(self.tmp, "credentials.db")
        tok = os.path.join(self.tmp, "tokens.json")
        open(cred, "w").close()
        open(tok, "w").close()
        with patch.dict(os.environ, {"RESET_STATE": "1"}, clear=False):
            main._maybe_reset_state(self.tmp)
        self.assertTrue(os.path.exists(cred))
        self.assertTrue(os.path.exists(tok))


if __name__ == "__main__":
    unittest.main()
