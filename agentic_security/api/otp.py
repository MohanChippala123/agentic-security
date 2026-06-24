"""OTP — 6-digit codes sent via Gmail for 2FA.

Setup required (add to .env or server environment):
    GMAIL_USER=your@gmail.com
    GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx   # Gmail App Password (not your real password)

How to get a Gmail App Password:
    1. Enable 2-Step Verification on your Google account
    2. Go to myaccount.google.com/apppasswords
    3. Create an app password for "Mail"
    4. Paste the 16-char code as GMAIL_APP_PASSWORD
"""

from __future__ import annotations

import os
import random
import secrets
import smtplib
import string
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

# ── In-memory OTP store ───────────────────────────────────────────────────────
# Structure: temp_token → {email, otp, expires}
_OTP_TTL = 10 * 60  # 10 minutes
_store: dict[str, dict] = {}


def _purge_expired() -> None:
    now = time.time()
    expired = [k for k, v in _store.items() if v["expires"] < now]
    for k in expired:
        del _store[k]


def generate(email: str) -> tuple[str, str]:
    """Generate a 6-digit OTP and a temp token. Returns (temp_token, otp)."""
    _purge_expired()
    otp = "".join(random.choices(string.digits, k=6))
    temp_token = secrets.token_urlsafe(32)
    _store[temp_token] = {
        "email": email.lower().strip(),
        "otp": otp,
        "expires": time.time() + _OTP_TTL,
    }
    return temp_token, otp


def verify(temp_token: str, otp: str) -> Optional[str]:
    """Verify OTP. Returns email on success, None on failure. Consumes the token."""
    _purge_expired()
    entry = _store.get(temp_token)
    if not entry:
        return None
    if entry["expires"] < time.time():
        del _store[temp_token]
        return None
    # Constant-time comparison to prevent timing attacks
    import hmac
    if not hmac.compare_digest(entry["otp"], otp.strip()):
        return None
    email = entry["email"]
    del _store[temp_token]  # one-time use
    return email


# ── Gmail sender ──────────────────────────────────────────────────────────────

def _gmail_user() -> str:
    # DB takes priority over env var
    try:
        from . import db
        val = db.config_get("gmail_user")
        if val:
            return val
    except Exception:
        pass
    return os.environ.get("GMAIL_USER", "")


def _gmail_password() -> str:
    try:
        from . import db
        enc = db.config_get("gmail_password_enc")
        if enc:
            from .db import _xor_decrypt, _db_secret
            return _xor_decrypt(enc, _db_secret())
    except Exception:
        pass
    return os.environ.get("GMAIL_APP_PASSWORD", "")


def send_otp(to_email: str, otp: str) -> dict:
    """Send OTP to email via Gmail SMTP. Returns {ok, error}."""
    gmail_user = _gmail_user()
    gmail_pass = _gmail_password()

    if not gmail_user or not gmail_pass:
        # Dev fallback: print to console instead of failing
        print(f"\n[2FA OTP] Email: {to_email} | Code: {otp} | (GMAIL_USER/GMAIL_APP_PASSWORD not set)\n")
        return {"ok": True, "dev_mode": True}

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = f"AgentShield <{gmail_user}>"
        msg["To"] = to_email
        msg["Subject"] = f"AgentShield — Your login code: {otp}"

        plain = f"""Your AgentShield verification code is:

  {otp}

This code expires in 10 minutes.
If you did not request this code, you can safely ignore this email.

— AgentShield Security"""

        html = f"""<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#0a0a0a;font-family:Inter,system-ui,sans-serif">
<div style="max-width:480px;margin:48px auto;padding:0 20px">
  <div style="margin-bottom:32px">
    <span style="font-weight:700;font-size:14px;color:#e8e6df">AgentShield</span>
  </div>
  <div style="background:#111;border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:36px">
    <h1 style="margin:0 0 8px;font-size:22px;font-weight:700;color:#e8e6df;letter-spacing:-.02em">Your login code</h1>
    <p style="margin:0 0 28px;font-size:14px;color:rgba(232,230,223,.5);line-height:1.6">Enter this code to complete sign-in to AgentShield.</p>
    <div style="background:#0a0a0a;border:1px solid rgba(196,241,53,.2);border-radius:8px;padding:24px;text-align:center;margin-bottom:28px">
      <span style="font-family:JetBrains Mono,monospace;font-size:36px;font-weight:700;letter-spacing:.2em;color:#c4f135">{otp}</span>
    </div>
    <p style="margin:0;font-size:12px;color:rgba(232,230,223,.35);line-height:1.6">This code expires in <strong style="color:rgba(232,230,223,.5)">10 minutes</strong>. If you didn't request it, ignore this email.</p>
  </div>
  <p style="margin:24px 0 0;font-size:11px;color:rgba(232,230,223,.25);text-align:center">AgentShield Security Platform</p>
</div>
</body>
</html>"""

        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, to_email, msg.as_string())

        return {"ok": True}

    except smtplib.SMTPAuthenticationError:
        return {"ok": False, "error": "Gmail authentication failed. Check GMAIL_USER and GMAIL_APP_PASSWORD."}
    except smtplib.SMTPException as e:
        return {"ok": False, "error": f"Failed to send email: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def is_configured() -> bool:
    return bool(_gmail_user() and _gmail_password())


def save_gmail_config(gmail_user: str, app_password: str) -> None:
    """Save Gmail credentials to DB (encrypted at rest)."""
    from . import db
    from .db import _xor_key, _db_secret
    db.config_set("gmail_user", gmail_user.strip())
    db.config_set("gmail_password_enc", _xor_key(app_password.strip(), _db_secret()))


def clear_gmail_config() -> None:
    from . import db
    db.config_set("gmail_user", "")
    db.config_set("gmail_password_enc", "")
