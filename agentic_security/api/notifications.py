"""Email notifications - budget alerts and critical attack alerts.

Uses the same Gmail config as 2FA OTPs.
Fires in a background thread so it never blocks a gateway request.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import urllib.request
import urllib.error
from typing import Any


# ── Email alerts ──────────────────────────────────────────────────────────────

def _send_alert(to_email: str, subject: str, body_plain: str, body_html: str) -> None:
    try:
        from .otp import _gmail_user, _gmail_password
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        gmail_user = _gmail_user()
        gmail_pass = _gmail_password()
        if not gmail_user or not gmail_pass:
            return

        msg = MIMEMultipart("alternative")
        msg["From"] = f"AgentShield <{gmail_user}>"
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body_plain, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, to_email, msg.as_string())
    except Exception:
        pass  # never let notification failures break the gateway


def notify_budget_warning(user_email: str, key_name: str, pct_used: float, spent: float, budget: float) -> None:
    subject = f"AgentShield - Key '{key_name}' is at {pct_used:.0f}% of its budget"
    plain = f"""Your AgentShield virtual key '{key_name}' has used {pct_used:.0f}% of its budget.

Spent: ${spent:.4f} / ${budget:.2f}

If requests continue at the current rate, this key will be suspended when it hits 100%.
Log in to top up the budget or create a new key: http://localhost:8000/app

- AgentShield"""
    html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:#0a0a0a;font-family:Inter,system-ui,sans-serif">
<div style="max-width:480px;margin:48px auto;padding:0 20px">
  <div style="margin-bottom:24px"><span style="font-weight:700;font-size:14px;color:#e8e6df">AgentShield</span></div>
  <div style="background:#111;border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:32px">
    <h1 style="margin:0 0 8px;font-size:20px;font-weight:700;color:#f5a64a">Budget Warning</h1>
    <p style="margin:0 0 20px;font-size:13.5px;color:rgba(232,230,223,.6);line-height:1.65">
      Virtual key <strong style="color:#e8e6df">{key_name}</strong> has used <strong style="color:#f5a64a">{pct_used:.0f}%</strong> of its budget.
    </p>
    <div style="background:#0a0a0a;border:1px solid rgba(245,166,74,.2);border-radius:8px;padding:16px 20px;margin-bottom:24px">
      <div style="font-size:13px;color:rgba(232,230,223,.5);margin-bottom:4px">Spent</div>
      <div style="font-size:22px;font-weight:700;color:#f5a64a">${spent:.4f} <span style="font-size:13px;font-weight:400;color:rgba(232,230,223,.4)">/ ${budget:.2f}</span></div>
    </div>
    <a href="http://localhost:8000/app" style="display:block;text-align:center;background:#c4f135;color:#000;border-radius:7px;padding:12px;font-weight:700;font-size:12px;text-decoration:none">Manage Keys →</a>
  </div>
</div></body></html>"""
    threading.Thread(target=_send_alert, args=(user_email, subject, plain, html), daemon=True).start()


def notify_critical_attack(user_email: str, key_name: str, threat: str, message: str, risk_score: int) -> None:
    subject = f"AgentShield - Critical attack blocked on '{key_name}'"
    plain = f"""AgentShield blocked a critical attack on your virtual key '{key_name}'.

Threat: {threat}
Risk score: {risk_score}/100
Message: "{message}"

Log in to review the full event log: http://localhost:8000/app

- AgentShield"""
    html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:#0a0a0a;font-family:Inter,system-ui,sans-serif">
<div style="max-width:480px;margin:48px auto;padding:0 20px">
  <div style="margin-bottom:24px"><span style="font-weight:700;font-size:14px;color:#e8e6df">AgentShield</span></div>
  <div style="background:#111;border:1px solid rgba(240,106,130,.2);border-radius:12px;padding:32px">
    <h1 style="margin:0 0 8px;font-size:20px;font-weight:700;color:#f06a82">Critical Attack Blocked</h1>
    <p style="margin:0 0 20px;font-size:13.5px;color:rgba(232,230,223,.6);line-height:1.65">
      A critical-severity attack was blocked on key <strong style="color:#e8e6df">{key_name}</strong>.
    </p>
    <div style="background:#0a0a0a;border:1px solid rgba(240,106,130,.15);border-radius:8px;padding:16px 20px;margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;margin-bottom:8px">
        <span style="font-size:11px;color:rgba(232,230,223,.4);text-transform:uppercase;letter-spacing:.1em">Threat</span>
        <span style="font-size:12px;color:#f06a82;font-weight:600">{threat}</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:8px">
        <span style="font-size:11px;color:rgba(232,230,223,.4);text-transform:uppercase;letter-spacing:.1em">Risk Score</span>
        <span style="font-size:12px;color:#f06a82;font-weight:600">{risk_score}/100</span>
      </div>
      <div style="border-top:1px solid rgba(255,255,255,.06);padding-top:10px;margin-top:4px">
        <span style="font-size:11px;color:rgba(232,230,223,.4)">Message</span>
        <div style="font-size:12px;color:rgba(232,230,223,.7);margin-top:4px;font-style:italic">"{message[:100]}{'...' if len(message)>100 else ''}"</div>
      </div>
    </div>
    <a href="http://localhost:8000/app" style="display:block;text-align:center;background:#c4f135;color:#000;border-radius:7px;padding:12px;font-weight:700;font-size:12px;text-decoration:none">View Event Log →</a>
  </div>
</div></body></html>"""
    threading.Thread(target=_send_alert, args=(user_email, subject, plain, html), daemon=True).start()


# ── Webhooks ──────────────────────────────────────────────────────────────────

def _fire_webhook(url: str, secret: str, payload: dict, wh_id: str) -> None:
    try:
        body = json.dumps(payload).encode()
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        req = urllib.request.Request(
            url, data=body,
            headers={
                "Content-Type": "application/json",
                "X-AgentShield-Signature": f"sha256={sig}",
                "X-AgentShield-Event": payload.get("event", "block"),
                "User-Agent": "AgentShield/1.0",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        from . import db
        db.webhook_record_fire(wh_id)
    except Exception:
        pass


def fire_webhooks(user_email: str, event: dict) -> None:
    """Fire all enabled webhooks for a user. Runs in background threads."""
    try:
        from . import db
        webhooks = db.webhook_list(user_email)
        for wh in webhooks:
            if not wh.get("enabled"):
                continue
            payload = {
                "event": "attack_blocked" if event.get("outcome") == "blocked" else "request_passed",
                "timestamp": event.get("timestamp", time.time()),
                "key_name": event.get("key_name", ""),
                "threat": event.get("threat", ""),
                "risk_score": event.get("risk_score", 0),
                "severity": event.get("severity", ""),
                "message_preview": event.get("message", "")[:120],
                "cost_saved_usd": event.get("cost_saved_usd", 0),
            }
            threading.Thread(
                target=_fire_webhook,
                args=(wh["url"], wh["secret"], payload, wh["id"]),
                daemon=True,
            ).start()
    except Exception:
        pass
