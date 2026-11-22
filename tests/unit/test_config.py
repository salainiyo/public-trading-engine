import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from config.loader import (
    load_config, _to_int_or_none, _parse_env_file,
    load_all_configs,
)
from config.tokens import (
    make_on_tokens, _persist_tokens, load_tokens, _write_atomic,
)


class TestToIntOrNull(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(_to_int_or_none("123"), 123)

    def test_none(self):
        self.assertIsNone(_to_int_or_none(None))

    def test_no_digits(self):
        self.assertIsNone(_to_int_or_none("abc"))

    def test_mixed(self):
        self.assertEqual(_to_int_or_none("abc123xyz"), 123)


class TestParseEnvFile(unittest.TestCase):
    def test_basic_parse(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("KEY1=val1\nKEY2=val2\n")
            f.flush()
            result = _parse_env_file(Path(f.name))
        os.unlink(f.name)
        self.assertEqual(result["KEY1"], "val1")
        self.assertEqual(result["KEY2"], "val2")

    def test_quoted_values(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write('KEY1="double_quoted"\nKEY2=\'single_quoted\'\n')
            f.flush()
            result = _parse_env_file(Path(f.name))
        os.unlink(f.name)
        self.assertEqual(result["KEY1"], "double_quoted")
        self.assertEqual(result["KEY2"], "single_quoted")

    def test_comments_skipped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("# comment\nKEY1=val1\n")
            f.flush()
            result = _parse_env_file(Path(f.name))
        os.unlink(f.name)
        self.assertNotIn("# comment", result)
        self.assertEqual(result["KEY1"], "val1")

    def test_nonexistent_file(self):
        result = _parse_env_file(Path("/nonexistent/.env"))
        self.assertEqual(result, {})


class TestLoadConfig(unittest.TestCase):
    def test_load_config_from_env(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("TRADELOCKER_EMAIL=test@example.com\n")
            f.write("TRADELOCKER_PASSWORD=pass123\n")
            f.write("TRADELOCKER_SERVER=s1\n")
            f.flush()
            cfg = load_config(env_file=f.name)
        os.unlink(f.name)
        self.assertEqual(cfg["email"], "test@example.com")
        self.assertEqual(cfg["password"], "pass123")
        self.assertEqual(cfg["broker"], "gfttl")

    def test_config_live_flag(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("TRADELOCKER_EMAIL=e@b.com\nTRADELOCKER_PASSWORD=p\nTRADELOCKER_SERVER=s\nTRADELOCKER_LIVE=true\n")
            f.flush()
            cfg = load_config(env_file=f.name)
        os.unlink(f.name)
        self.assertTrue(cfg["live"])

    def test_load_config_missing_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("SOME_RANDOM_KEY=value\n")
            f.flush()
            with self.assertRaises(RuntimeError):
                load_config(env_file=f.name)
        os.unlink(f.name)

    def test_load_config_no_file(self):
        with self.assertRaises(RuntimeError):
            load_config(env_file="/nonexistent/.env")

    def test_load_config_with_tokens(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("TRADELOCKER_EMAIL=e@b.com\nTRADELOCKER_PASSWORD=p\nTRADELOCKER_SERVER=s\n")
            f.write("TRADELOCKER_ACCESS_TOKEN=acc_tok\nTRADELOCKER_REFRESH_TOKEN=ref_tok\n")
            f.flush()
            cfg = load_config(env_file=f.name)
        os.unlink(f.name)
        self.assertEqual(cfg["access_token"], "acc_tok")
        self.assertEqual(cfg["refresh_token"], "ref_tok")

    def test_load_config_aquafunded_fallback(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("AQUAFUNDED_EMAIL=a@b.com\n")
            f.write("AQUAFUNDED_PASSWORD=ap\n")
            f.write("AQUAFUNDED_SERVER=aquaserver\n")
            f.flush()
            cfg = load_config(env_file=f.name)
        os.unlink(f.name)
        self.assertEqual(cfg["email"], "a@b.com")
        self.assertEqual(cfg["password"], "ap")
        self.assertEqual(cfg["server"], "aquaserver")
        self.assertEqual(cfg["broker"], "aquafunded")
        self.assertEqual(cfg["env_prefix"], "AQUAFUNDED")

    def test_load_config_aquafunded_account(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("AQUAFUNDED_EMAIL=a@b.com\n")
            f.write("AQUAFUNDED_PASSWORD=ap\n")
            f.write("AQUAFUNDED_SERVER=aquaserver\n")
            f.write("AQUAFUNDED_ACCOUNT_ID=12345\n")
            f.write("AQUAFUNDED_ACC_NUM=2\n")
            f.flush()
            cfg = load_config(env_file=f.name)
        os.unlink(f.name)
        self.assertEqual(cfg["account_id"], 12345)
        self.assertEqual(cfg["acc_num"], 2)

    def test_load_config_aquafunded_live(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("AQUAFUNDED_EMAIL=a@b.com\n")
            f.write("AQUAFUNDED_PASSWORD=ap\n")
            f.write("AQUAFUNDED_SERVER=aquaserver\n")
            f.write("AQUAFUNDED_LIVE=true\n")
            f.flush()
            cfg = load_config(env_file=f.name)
        os.unlink(f.name)
        self.assertTrue(cfg["live"])

    def test_load_config_goat100k_fallback(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("GOAT100K_EMAIL=g@b.com\n")
            f.write("GOAT100K_PASSWORD=gp\n")
            f.write("GOAT100K_SERVER=goatserver\n")
            f.flush()
            cfg = load_config(env_file=f.name)
        os.unlink(f.name)
        self.assertEqual(cfg["email"], "g@b.com")
        self.assertEqual(cfg["password"], "gp")
        self.assertEqual(cfg["server"], "goatserver")
        self.assertEqual(cfg["broker"], "goat100k")
        self.assertEqual(cfg["env_prefix"], "GOAT100K")

    def test_load_config_goat100k_account(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("GOAT100K_EMAIL=g@b.com\n")
            f.write("GOAT100K_PASSWORD=gp\n")
            f.write("GOAT100K_SERVER=goatserver\n")
            f.write("GOAT100K_ACCOUNT_ID=99999\n")
            f.write("GOAT100K_ACC_NUM=3\n")
            f.flush()
            cfg = load_config(env_file=f.name)
        os.unlink(f.name)
        self.assertEqual(cfg["account_id"], 99999)
        self.assertEqual(cfg["acc_num"], 3)

    def test_load_config_goat100k_live(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("GOAT100K_EMAIL=g@b.com\n")
            f.write("GOAT100K_PASSWORD=gp\n")
            f.write("GOAT100K_SERVER=goatserver\n")
            f.write("GOAT100K_LIVE=true\n")
            f.flush()
            cfg = load_config(env_file=f.name)
        os.unlink(f.name)
        self.assertTrue(cfg["live"])


class TestLoadAllConfigs(unittest.TestCase):
    def test_loads_multiple_brokers(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("TRADELOCKER_EMAIL=tl@b.com\nTRADELOCKER_PASSWORD=tlp\nTRADELOCKER_SERVER=tlserver\n")
            f.write("AQUAFUNDED_EMAIL=aq@b.com\nAQUAFUNDED_PASSWORD=aqp\nAQUAFUNDED_SERVER=aqserver\n")
            f.flush()
            configs = load_all_configs(env_file=f.name)
        os.unlink(f.name)
        self.assertIn("tradeler", configs)
        self.assertIn("herofx_aqua", configs)
        self.assertEqual(configs["tradeler"]["email"], "tl@b.com")
        self.assertEqual(configs["herofx_aqua"]["email"], "aq@b.com")


class TestTokenPersistence(unittest.TestCase):
    def test_make_on_tokens_callback(self):
        cb = make_on_tokens("TRADELOCKER")
        self.assertTrue(callable(cb))

    def test_write_atomic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            _write_atomic(path, {"key": "value"})
            self.assertTrue(path.exists())
            data = json.loads(path.read_text())
            self.assertEqual(data["key"], "value")

    def test_persist_tokens_creates_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from config import tokens
            orig_dir = tokens.TOKENS_DIR
            orig_file = tokens.TOKENS_FILE
            tokens.TOKENS_DIR = Path(tmpdir)
            tokens.TOKENS_FILE = Path(tmpdir) / "tokens.json"
            try:
                _persist_tokens("TL", "new_access", "new_refresh")
                data = json.loads(tokens.TOKENS_FILE.read_text())
                self.assertEqual(data["TL"]["access_token"], "new_access")
                self.assertEqual(data["TL"]["refresh_token"], "new_refresh")
            finally:
                tokens.TOKENS_DIR = orig_dir
                tokens.TOKENS_FILE = orig_file

    def test_persist_tokens_updates_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from config import tokens
            orig_dir = tokens.TOKENS_DIR
            orig_file = tokens.TOKENS_FILE
            tokens.TOKENS_DIR = Path(tmpdir)
            tokens.TOKENS_FILE = Path(tmpdir) / "tokens.json"
            try:
                tokens.TOKENS_FILE.write_text(json.dumps({"OLD": {"access_token": "old", "refresh_token": "old_r"}}))
                _persist_tokens("TL", "updated_access", "updated_refresh")
                data = json.loads(tokens.TOKENS_FILE.read_text())
                self.assertEqual(data["TL"]["access_token"], "updated_access")
                self.assertEqual(data["OLD"]["access_token"], "old")
            finally:
                tokens.TOKENS_DIR = orig_dir
                tokens.TOKENS_FILE = orig_file


if __name__ == "__main__":
    unittest.main()
