"""Config loader — reads .env and returns broker configuration dicts."""
import logging
import os
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _find_env_file() -> Optional[Path]:
    d = Path.cwd()
    for _ in range(5):
        candidate = d / ".env"
        if candidate.exists():
            return candidate
        parent = d.parent
        if parent == d:
            break
        d = parent
    return None


def _to_int_or_none(val: str) -> Optional[int]:
    if not val:
        return None
    digits = re.sub(r"[^\d]", "", val)
    return int(digits) if digits else None


def _parse_env_file(path: Path) -> dict:
    """Parse a .env file into a dict of key=value pairs (no dotenv dependency)."""
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        result[key] = value
    return result


def load_config(env_file: Optional[str] = None) -> dict:
    """Load broker configuration — tries GFTTL first, falls back to HEROFX.

    Args:
        env_file: Explicit path to .env file. If None, searches parent dirs.

    Returns dict with keys: email, password, server, access_token, refresh_token,
    account_id, acc_num, live, env_prefix, ws_port, broker.
    Raises RuntimeError if no credentials found for any broker.
    """
    if env_file:
        env_vars = _parse_env_file(Path(env_file))
    else:
        found = _find_env_file()
        if found:
            env_vars = _parse_env_file(found)
            log.info(f"Loaded .env from {found}")
        else:
            env_vars = {}

    def _get(key, fallback=None):
        return env_vars.get(key) or os.environ.get(key, fallback)

    # Try GFTTL (primary) first
    email = _get("TRADELOCKER_EMAIL") or _get("TL_EMAIL")
    password = _get("TRADELOCKER_PASSWORD") or _get("TL_PASSWORD")
    server = _get("TRADELOCKER_SERVER") or _get("TL_SERVER")
    access_token = _get("TRADELOCKER_ACCESS_TOKEN") or _get("TL_ACCESS_TOKEN")
    refresh_token = _get("TRADELOCKER_REFRESH_TOKEN") or _get("TL_REFRESH_TOKEN")
    account_id = _to_int_or_none(_get("TRADELOCKER_ACCOUNT_ID") or _get("TL_ACCOUNT_ID") or "")
    acc_num = _to_int_or_none(_get("TRADELOCKER_ACC_NUM") or _get("TL_ACC_NUM") or "")
    live = _get("TRADELOCKER_LIVE", "false").lower() in ("true", "1", "yes")
    ws_port = int(_get("WS_PORT", "8765"))
    env_prefix = "TRADELOCKER"
    broker = "gfttl"

    # Fall back to HEROFX if GFTTL creds absent
    if not email and not (access_token and refresh_token):
        h_email = _get("HEROFX_EMAIL")
        h_password = _get("HEROFX_PASSWORD")
        h_server = _get("HEROFX_SERVER")
        if h_email and h_password and h_server:
            log.info("No GFTTL credentials — using HEROFX config")
            email = h_email
            password = h_password
            server = h_server
            access_token = _get("HEROFX_ACCESS_TOKEN")
            refresh_token = _get("HEROFX_REFRESH_TOKEN")
            account_id = _to_int_or_none(_get("HEROFX_ACCOUNT_ID") or "")
            acc_num = _to_int_or_none(_get("HEROFX_ACC_NUM") or "")
            live = _get("HEROFX_LIVE", "false").lower() in ("true", "1", "yes")
            env_prefix = "HEROFX"
            broker = "herofx"
        else:
            # Fall back to AQUAFUNDED
            a_email = _get("AQUAFUNDED_EMAIL")
            a_password = _get("AQUAFUNDED_PASSWORD")
            a_server = _get("AQUAFUNDED_SERVER")
            if a_email and a_password and a_server:
                log.info("No GFTTL/HEROFX credentials — using AQUAFUNDED config")
                email = a_email
                password = a_password
                server = a_server
                access_token = _get("AQUAFUNDED_ACCESS_TOKEN")
                refresh_token = _get("AQUAFUNDED_REFRESH_TOKEN")
                account_id = _to_int_or_none(_get("AQUAFUNDED_ACCOUNT_ID") or "")
                acc_num = _to_int_or_none(_get("AQUAFUNDED_ACC_NUM") or "")
                live = _get("AQUAFUNDED_LIVE", "false").lower() in ("true", "1", "yes")
                env_prefix = "AQUAFUNDED"
                broker = "aquafunded"
            else:
                # Fall back to GOAT100K
                g_email = _get("GOAT100K_EMAIL")
                g_password = _get("GOAT100K_PASSWORD")
                g_server = _get("GOAT100K_SERVER")
                if g_email and g_password and g_server:
                    log.info("No GFTTL/HEROFX/AQUAFUNDED credentials — using GOAT100K config")
                    email = g_email
                    password = g_password
                    server = g_server
                    access_token = _get("GOAT100K_ACCESS_TOKEN")
                    refresh_token = _get("GOAT100K_REFRESH_TOKEN")
                    account_id = _to_int_or_none(_get("GOAT100K_ACCOUNT_ID") or "")
                    acc_num = _to_int_or_none(_get("GOAT100K_ACC_NUM") or "")
                    live = _get("GOAT100K_LIVE", "false").lower() in ("true", "1", "yes")
                    env_prefix = "GOAT100K"
                    broker = "goat100k"
                else:
                    raise RuntimeError(
                        "No credentials found: set TRADELOCKER_EMAIL + TRADELOCKER_PASSWORD "
                        "or HEROFX_EMAIL + HEROFX_PASSWORD "
                        "or AQUAFUNDED_EMAIL + AQUAFUNDED_PASSWORD "
                        "or GOAT100K_EMAIL + GOAT100K_PASSWORD in .env"
                    )

    from config.tokens import load_tokens
    saved = load_tokens(env_prefix)
    if saved.get("access_token") and not access_token:
        access_token = saved["access_token"]
        log.info(f"Loaded saved access_token from tokens.json for {env_prefix}")
    if saved.get("refresh_token") and not refresh_token:
        refresh_token = saved["refresh_token"]
        log.info(f"Loaded saved refresh_token from tokens.json for {env_prefix}")

    return {
        "email": email,
        "password": password,
        "server": server,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "account_id": account_id,
        "acc_num": acc_num,
        "live": live,
        "env_prefix": env_prefix,
        "ws_port": ws_port,
        "broker": broker,
    }


def _load_one_broker(env_vars: dict, prefix: str, broker_key: str, broker_name: str) -> Optional[dict]:
    """Load a single broker config from env_vars by prefix. Returns None if missing."""
    def _get(key, default=None):
        return env_vars.get(key) or os.environ.get(key) or default

    email = _get(f"{prefix}_EMAIL")
    password = _get(f"{prefix}_PASSWORD")
    server = _get(f"{prefix}_SERVER")
    if not all([email, password, server]):
        return None

    from config.tokens import load_tokens
    access_token = _get(f"{prefix}_ACCESS_TOKEN")
    refresh_token = _get(f"{prefix}_REFRESH_TOKEN")
    saved = load_tokens(prefix)
    if saved.get("access_token") and not access_token:
        access_token = saved["access_token"]
        log.info(f"Loaded saved access_token from tokens.json for {prefix}")
    if saved.get("refresh_token") and not refresh_token:
        refresh_token = saved["refresh_token"]
        log.info(f"Loaded saved refresh_token from tokens.json for {prefix}")

    return {
        "email": email,
        "password": password,
        "server": server,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "account_id": _to_int_or_none(_get(f"{prefix}_ACCOUNT_ID") or ""),
        "acc_num": _to_int_or_none(_get(f"{prefix}_ACC_NUM") or ""),
        "live": _get(f"{prefix}_LIVE", "false").lower() in ("true", "1", "yes"),
        "env_prefix": prefix,
        "ws_port": int(_get("WS_PORT", "8765")),
        "broker": broker_name,
        "broker_key": broker_key,
    }


def load_all_configs(env_file: Optional[str] = None) -> dict:
    """Load ALL broker configs from .env. Returns {broker_key: config_dict}.

    Tries each prefix independently (not sequential fallback).
    Supports: TRADELOCKER, AQUAFUNDED, GOAT (with GOAT100K fallback).
    """
    if env_file:
        env_vars = _parse_env_file(Path(env_file))
    else:
        found = _find_env_file()
        if found:
            env_vars = _parse_env_file(found)
            log.info(f"Loaded .env from {found}")
        else:
            env_vars = {}

    configs = {}

    # TRADELOCKER (GFTTL)
    cfg = _load_one_broker(env_vars, "TRADELOCKER", "tradeler", "gfttl")
    if cfg:
        configs["tradeler"] = cfg

    # AQUAFUNDED
    cfg = _load_one_broker(env_vars, "AQUAFUNDED", "herofx_aqua", "aquafunded")
    if cfg:
        configs["herofx_aqua"] = cfg

    # GOAT — try GOAT first, fall back to GOAT100K
    cfg = _load_one_broker(env_vars, "GOAT", "herofx_goat", "goat100k")
    if not cfg:
        cfg = _load_one_broker(env_vars, "GOAT100K", "herofx_goat", "goat100k")
        if cfg:
            cfg["env_prefix"] = "GOAT100K"
    if cfg:
        configs["herofx_goat"] = cfg

    if not configs:
        raise RuntimeError(
            "No broker credentials found in .env. Need at least one of: "
            "TRADELOCKER_*, AQUAFUNDED_*, GOAT_* (or GOAT100K_*)"
        )

    log.info(f"Loaded {len(configs)} broker config(s): {list(configs.keys())}")
    return configs
