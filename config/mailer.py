"""Recovery email sender — SMTP via Gmail App Password (stdlib smtplib).

Config (from .env / os.environ):
  SMTP_HOST (default smtp.gmail.com), SMTP_PORT (default 587),
  SMTP_EMAIL, SMTP_APP_PASSWORD, RECOVERY_EMAIL
If SMTP is not configured, the caller falls back to logging the code.
"""
import logging
import os
import smtplib
from email.mime.text import MIMEText

log = logging.getLogger(__name__)


def _get(key: str, default=None):
    return os.environ.get(key, default)


def smtp_configured() -> bool:
    return bool(_get("SMTP_EMAIL") and _get("SMTP_APP_PASSWORD"))


def recovery_email() -> str:
    return _get("RECOVERY_EMAIL") or _get("SMTP_EMAIL") or ""


def send_recovery_email(to_email: str, code: str) -> bool:
    """Send a password-reset code. Returns True on success."""
    host = _get("SMTP_HOST", "smtp.gmail.com")
    port = int(_get("SMTP_PORT", "587"))
    user = _get("SMTP_EMAIL")
    app_password = _get("SMTP_APP_PASSWORD")
    if not (user and app_password and to_email):
        return False
    try:
        msg = MIMEText(
            f"Your Trading Engine dashboard password reset code is:\n\n"
            f"    {code}\n\n"
            f"This code expires in 15 minutes and can be used once.\n"
            f"If you did not request this, ignore this email.",
        )
        msg["Subject"] = "Dashboard password reset code"
        msg["From"] = user
        msg["To"] = to_email
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(user, app_password)
            smtp.send_message(msg)
        log.info(f"[mailer] recovery code sent to {to_email}")
        return True
    except Exception as e:
        log.warning(f"[mailer] recovery email failed: {e}")
        return False
