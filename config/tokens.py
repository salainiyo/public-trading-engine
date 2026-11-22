"""Token persistence — save/load JWT tokens to bot/data/tokens.json.

Uses atomic writes (temp file + os.replace) and file locking to prevent
race conditions when multiple brokers refresh tokens concurrently.
"""
import fcntl
import json
import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

TOKENS_DIR = Path(__file__).resolve().parent.parent / "bot" / "data"
TOKENS_FILE = TOKENS_DIR / "tokens.json"


def make_on_tokens(env_prefix: str):
    """Return a callback(access_token, refresh_token) that persists tokens to JSON."""
    def on_tokens(access_token: str, refresh_token: str):
        try:
            _persist_tokens(env_prefix, access_token, refresh_token)
        except Exception as e:
            log.warning(f"Token persistence failed: {e}")
    return on_tokens


def _persist_tokens(env_prefix: str, access_token: str, refresh_token: str):
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = TOKENS_FILE.with_suffix(".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            data = _load_all()
            data[env_prefix] = {
                "access_token": access_token,
                "refresh_token": refresh_token,
            }
            _write_atomic(TOKENS_FILE, data)
            log.info(f"Tokens persisted to {TOKENS_FILE}")
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _write_atomic(path: Path, data: dict):
    """Write JSON data atomically via temp file + os.replace."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_tokens(env_prefix: str) -> dict:
    """Load saved tokens for a given broker prefix. Returns dict with access_token/refresh_token or empty dict."""
    data = _load_all()
    return data.get(env_prefix, {})


def _load_all() -> dict:
    if TOKENS_FILE.exists():
        try:
            return json.loads(TOKENS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning(f"Could not read {TOKENS_FILE} — starting fresh")
    return {}
