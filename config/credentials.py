"""Encrypted credentials store — uploaded challenge credentials in a SQLite DB.

blobs are AES-GCM encrypted with a 32-byte key. The key lives in
bot/data/credentials.key (auto-generated on first use, chmod 600) or the
CREDENTIALS_KEY env override (hex). The key file and DB are on the same host,
so this protects the DB leaking alone (backups, accidental exposure) — not a
full server compromise.
"""
import fcntl
import json
import logging
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

log = logging.getLogger(__name__)

def _default_db_path() -> str:
    data_dir = Path(os.environ.get("GOATBOT_DATA_DIR", "bot/data"))
    return str(data_dir / "credentials.db")

_ACCOUNT_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4,
                  "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}


def normalize_account(account) -> int:
    """Normalize the account field: 'first'/'second'/... or '1'/'2'/... -> 1-based index."""
    if account is None:
        raise ValueError("account field is required")
    s = str(account).strip().lower()
    if s in _ACCOUNT_WORDS:
        return _ACCOUNT_WORDS[s]
    try:
        n = int(s)
        if n < 1:
            raise ValueError
        return n
    except ValueError:
        raise ValueError(f"invalid account value: {account!r} (use first/second/... or 1/2/...)")


def _key_path() -> Path:
    """Dynamic key path (env read at call time — tests override GOATBOT_DATA_DIR)."""
    data_dir = Path(os.environ.get("GOATBOT_DATA_DIR", "bot/data"))
    return data_dir / "credentials.key"


def _load_key() -> bytes:
    """Load or create the encryption key."""
    env_key = os.environ.get("CREDENTIALS_KEY")
    if env_key:
        try:
            return bytes.fromhex(env_key)
        except ValueError:
            raise ValueError("CREDENTIALS_KEY must be a hex string")
    key_file = _key_path()
    key_file.parent.mkdir(parents=True, exist_ok=True)
    if key_file.exists():
        key = key_file.read_bytes()
        if len(key) == 32:
            return key
        log.warning(f"[credentials] key file has wrong size ({len(key)}) — regenerating")
    key = secrets.token_bytes(32)
    tmp = key_file.with_suffix(".key.tmp")
    with open(tmp, "wb") as f:
        f.write(key)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o600)  # perms set BEFORE the rename — no default-perm window
    os.replace(tmp, key_file)
    log.info(f"[credentials] generated new encryption key at {key_file}")
    return key


def _encrypt(data: dict, key: bytes) -> bytes:
    nonce = secrets.token_bytes(12)
    aes = AESGCM(key)
    return nonce + aes.encrypt(nonce, json.dumps(data).encode("utf-8"), None)


def _decrypt(blob: bytes, key: bytes) -> dict:
    nonce, ct = blob[:12], blob[12:]
    aes = AESGCM(key)
    return json.loads(aes.decrypt(nonce, ct, None).decode("utf-8"))


class CredentialsStore:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or _default_db_path())
        self._conn = None
        self._key = None
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS broker_credentials ("
            "tag TEXT PRIMARY KEY, blob BLOB NOT NULL, "
            "created_at TEXT, updated_at TEXT)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS dashboard_auth ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), "
            "password_hash TEXT, salt TEXT, "
            "session_secret TEXT, recovery_email TEXT)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS recovery_codes ("
            "code_hash TEXT PRIMARY KEY, expires_at TEXT, used INTEGER DEFAULT 0)"
        )
        self._conn.commit()

    @property
    def _lock_fd(self):
        lock_path = str(self.db_path) + ".lock"
        # "a+" — non-truncating; the with-block closes the handle exactly once
        return open(lock_path, "a+")

    def _ensure_key(self) -> bytes:
        """The CREDENTIALS_KEY env override governs whenever it is set."""
        env_key = os.environ.get("CREDENTIALS_KEY")
        if env_key:
            key = bytes.fromhex(env_key)
            self._key = key
            return key
        if self._key is None:
            self._key = _load_key()
        return self._key

    def save_credentials(self, tag: str, email: str, password: str,
                         server: str, account, balance=None,
                         challenge_cost=None) -> int:
        """Encrypt + store credentials for a strategy tag. Returns the account index."""
        acct_idx = normalize_account(account)
        payload = {"email": email, "password": password,
                   "server": server, "account": acct_idx}
        if balance not in (None, ""):
            try:
                payload["balance"] = float(balance)
            except (TypeError, ValueError):
                raise ValueError(f"invalid balance: {balance!r}")
        if challenge_cost not in (None, ""):
            try:
                payload["challenge_cost"] = float(challenge_cost)
            except (TypeError, ValueError):
                raise ValueError(f"invalid challenge_cost: {challenge_cost!r}")
        key = self._ensure_key()
        blob = _encrypt(payload, key)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock_fd as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            self._conn.execute(
                "INSERT OR REPLACE INTO broker_credentials "
                "(tag, blob, created_at, updated_at) VALUES (?, ?, "
                "COALESCE((SELECT created_at FROM broker_credentials WHERE tag=?), ?), ?)",
                (tag, blob, tag, now, now),
            )
            self._conn.commit()
        return acct_idx

    def load_credentials(self, tag: str):
        """Decrypt + return {email, password, server, account} or None."""
        cur = self._conn.execute(
            "SELECT blob FROM broker_credentials WHERE tag=?", (tag,))
        row = cur.fetchone()
        if not row:
            return None
        try:
            return _decrypt(row[0], self._ensure_key())
        except Exception as e:
            log.error(f"[credentials] failed to decrypt {tag}: {e}")
            return None

    def load_all(self) -> dict:
        """All stored credentials as {tag: {email, password, server, account}}."""
        cur = self._conn.execute("SELECT tag, blob FROM broker_credentials")
        out = {}
        for tag, blob in cur.fetchall():
            try:
                out[tag] = _decrypt(blob, self._ensure_key())
            except Exception as e:
                log.error(f"[credentials] failed to decrypt {tag}: {e}")
        return out

    def list_all(self) -> dict:
        """All entries with REDACTED passwords (for the dashboard)."""
        cur = self._conn.execute(
            "SELECT tag, blob, updated_at FROM broker_credentials ORDER BY tag")
        out = {}
        for tag, blob, updated in cur.fetchall():
            entry = {"tag": tag, "updated_at": updated}
            try:
                data = _decrypt(blob, self._ensure_key())
                pw = data.get("password") or ""
                entry["email"] = data.get("email")
                entry["server"] = data.get("server")
                entry["account"] = data.get("account")
                entry["balance"] = data.get("balance")
                entry["challenge_cost"] = data.get("challenge_cost")
                entry["password_hint"] = ("••••" + pw[-2:]) if len(pw) >= 2 else "••••"
            except Exception:
                entry["error"] = "decryption failed"
            out[tag] = entry
        return out

    def delete_credentials(self, tag: str) -> bool:
        with self._lock_fd as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            cur = self._conn.execute(
                "DELETE FROM broker_credentials WHERE tag=?", (tag,))
            self._conn.commit()
            return cur.rowcount > 0

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Dashboard auth (password hash, session secret, recovery) ──────────

    _PBKDF2_ITERATIONS = 100_000

    def _hash_password(self, password: str, salt: bytes) -> str:
        import hashlib
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt,
                                 self._PBKDF2_ITERATIONS)
        return dk.hex()

    def has_password(self) -> bool:
        try:
            row = self._conn.execute(
                "SELECT password_hash FROM dashboard_auth WHERE id=1").fetchone()
            return bool(row and row[0])
        except Exception:
            return False

    def set_password(self, password: str):
        """Store (or update) the password hash. Regenerates the session secret so
        all existing sessions are invalidated."""
        import secrets as _secrets
        salt = _secrets.token_bytes(16)
        pw_hash = self._hash_password(password, salt)
        session_secret = _secrets.token_hex(32)
        self._conn.execute(
            "INSERT INTO dashboard_auth (id, password_hash, salt, session_secret, recovery_email) "
            "VALUES (1, ?, ?, ?, NULL) "
            "ON CONFLICT(id) DO UPDATE SET password_hash=excluded.password_hash, "
            "salt=excluded.salt, session_secret=excluded.session_secret",
            (pw_hash, salt.hex(), session_secret))
        self._conn.commit()

    def verify_password(self, password: str) -> bool:
        try:
            row = self._conn.execute(
                "SELECT password_hash, salt FROM dashboard_auth WHERE id=1").fetchone()
            if not row or not row[0]:
                return False
            import hmac as _hmac
            expected = row[0]
            actual = self._hash_password(password, bytes.fromhex(row[1]))
            return _hmac.compare_digest(expected, actual)
        except Exception:
            return False

    def rotate_session_secret(self):
        """Regenerate the session secret — invalidates ALL existing sessions."""
        import secrets as _secrets
        secret = _secrets.token_hex(32)
        self._conn.execute(
            "UPDATE dashboard_auth SET session_secret=? WHERE id=1", (secret,))
        self._conn.commit()
        return secret

    def get_session_secret(self) -> str:
        row = self._conn.execute(
            "SELECT session_secret FROM dashboard_auth WHERE id=1").fetchone()
        if row and row[0]:
            return row[0]
        import secrets as _secrets
        secret = _secrets.token_hex(32)
        self._conn.execute(
            "UPDATE dashboard_auth SET session_secret=? WHERE id=1", (secret,))
        self._conn.commit()
        return secret

    def get_recovery_email(self):
        try:
            row = self._conn.execute(
                "SELECT recovery_email FROM dashboard_auth WHERE id=1").fetchone()
            return row[0] if row and row[0] else None
        except Exception:
            return None

    def set_recovery_email(self, email):
        self._conn.execute(
            "INSERT INTO dashboard_auth (id, recovery_email) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET recovery_email=excluded.recovery_email",
            (email,))
        self._conn.commit()

    def create_recovery_code(self, ttl_seconds: int = 900) -> str:
        """Generate a one-time recovery code; stores only its hash. Returns the code."""
        import hashlib
        import secrets as _secrets
        code = _secrets.token_hex(4).upper()
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        expires = (datetime.now(timezone.utc).timestamp() + ttl_seconds)
        self._conn.execute("DELETE FROM recovery_codes")
        self._conn.execute(
            "INSERT INTO recovery_codes (code_hash, expires_at, used) VALUES (?, ?, 0)",
            (code_hash, str(expires)))
        self._conn.commit()
        return code

    def consume_recovery_code(self, code: str) -> bool:
        """Validate a one-time code (not expired, not used) and consume it."""
        import hashlib
        code_hash = hashlib.sha256(str(code).strip().upper().encode("utf-8")).hexdigest()
        row = self._conn.execute(
            "SELECT expires_at, used FROM recovery_codes WHERE code_hash=?",
            (code_hash,)).fetchone()
        if not row or row[1]:
            return False
        try:
            if datetime.now(timezone.utc).timestamp() > float(row[0]):
                return False
        except ValueError:
            return False
        self._conn.execute(
            "UPDATE recovery_codes SET used=1 WHERE code_hash=?", (code_hash,))
        self._conn.commit()
        return True
