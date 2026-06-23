"""Authentication — PBKDF2-HMAC-SHA256 passwords, HMAC-signed session cookies.

User records now live in SQLite (via db.py) instead of users.json.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

from . import db

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SECRET_FILE = _DATA_DIR / ".session_secret"

COOKIE = "agsec_session"
_TOKEN_TTL = 7 * 24 * 3600
_PBKDF2_ROUNDS = 200_000


# ── session secret ───────────────────────────────────────────────────────────

def _secret() -> bytes:
    env = os.environ.get("AGSEC_SECRET")
    if env:
        return env.encode()
    _DATA_DIR.mkdir(exist_ok=True)
    if not _SECRET_FILE.exists():
        _SECRET_FILE.write_text(secrets.token_hex(32), encoding="utf-8")
    return _SECRET_FILE.read_text(encoding="utf-8").strip().encode()


# ── password hashing ─────────────────────────────────────────────────────────

def _hash_pw(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ROUNDS)
    return dk.hex()


# ── user management ──────────────────────────────────────────────────────────

def create_user(email: str, password: str, name: str) -> dict:
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("A valid email is required.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if db.user_get(email):
        raise ValueError("An account with that email already exists.")
    salt = secrets.token_hex(16)
    db.user_create(email, name.strip() or email.split("@")[0], salt, _hash_pw(password, salt))
    return {"email": email, "name": name.strip() or email.split("@")[0]}


def verify_user(email: str, password: str) -> dict:
    email = email.strip().lower()
    user = db.user_get(email)
    if not user:
        raise ValueError("Invalid email or password.")
    if not hmac.compare_digest(user["hash"], _hash_pw(password, user["salt"])):
        raise ValueError("Invalid email or password.")
    db.user_record_login(email)
    return {"email": email, "name": user["name"]}


# ── session tokens ───────────────────────────────────────────────────────────

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue_token(email: str) -> str:
    payload = json.dumps({"e": email, "x": int(time.time()) + _TOKEN_TTL}).encode()
    body = _b64(payload)
    sig = hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def read_token(token: str | None) -> dict | None:
    if not token or "." not in token:
        return None
    body, _, sig = token.partition(".")
    expected = hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    try:
        data = json.loads(_unb64(body))
    except Exception:
        return None
    if data.get("x", 0) < time.time():
        return None
    email = data.get("e")
    user = db.user_get(email)
    if not user:
        return None
    return {"email": email, "name": user["name"]}


# ── bootstrap ────────────────────────────────────────────────────────────────

def seed_demo_account() -> None:
    """Migrate any legacy users.json on first run only.
    Never creates fake or demo accounts — real users sign up themselves."""
    migrated = db.migrate_legacy_users()
    if migrated:
        import logging
        logging.getLogger("agentic_security").info("Migrated %d user(s) from users.json → SQLite", migrated)
