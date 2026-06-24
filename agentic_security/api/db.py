"""SQLite persistence layer.

Single file database at data/agentic.db. Tables:
  users         — email, hashed password, salt, 2FA, lockout
  scans         — one row per completed scan, linked to a user
  findings      — one row per de-duplicated finding, linked to a scan
  virtual_keys  — gateway virtual keys with live spend/blocked counters
  gateway_events— full activity log per user (capped at 200/user)
  upstream_keys — encrypted provider API key + provider name per user
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_DB_PATH = _DATA_DIR / "agentic.db"

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    email      TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    salt       TEXT NOT NULL,
    hash       TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS scans (
    id          TEXT PRIMARY KEY,
    user_email  TEXT NOT NULL REFERENCES users(email) ON DELETE CASCADE,
    target      TEXT NOT NULL,
    risk_score  REAL NOT NULL DEFAULT 0,
    grade       TEXT NOT NULL DEFAULT 'A',
    probes_run  TEXT NOT NULL DEFAULT '[]',
    started_at  REAL NOT NULL,
    finished_at REAL,
    duration_s  REAL
);

CREATE TABLE IF NOT EXISTS findings (
    id          TEXT NOT NULL,
    scan_id     TEXT NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    probe       TEXT NOT NULL,
    title       TEXT NOT NULL,
    severity    TEXT NOT NULL,
    severity_score INTEGER NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    payload     TEXT NOT NULL DEFAULT '',
    response    TEXT NOT NULL DEFAULT '',
    confidence  REAL NOT NULL DEFAULT 0.5,
    tags        TEXT NOT NULL DEFAULT '[]',
    timestamp   REAL NOT NULL,
    PRIMARY KEY (id, scan_id)
);

CREATE INDEX IF NOT EXISTS idx_scans_user   ON scans(user_email, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_findings_scan ON findings(scan_id);

CREATE TABLE IF NOT EXISTS virtual_keys (
    key                TEXT PRIMARY KEY,
    user_email         TEXT NOT NULL,
    name               TEXT NOT NULL,
    budget_usd         REAL NOT NULL,
    spent_usd          REAL NOT NULL DEFAULT 0,
    rate_limit_per_min INTEGER NOT NULL DEFAULT 30,
    enabled            INTEGER NOT NULL DEFAULT 1,
    request_count      INTEGER NOT NULL DEFAULT 0,
    blocked_count      INTEGER NOT NULL DEFAULT 0,
    cost_saved_usd     REAL NOT NULL DEFAULT 0,
    created_at         REAL NOT NULL,
    recent_blocks      TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS gateway_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_email TEXT NOT NULL,
    timestamp  REAL NOT NULL,
    data       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS upstream_keys (
    user_email   TEXT PRIMARY KEY,
    enc_key      TEXT NOT NULL,
    provider     TEXT NOT NULL DEFAULT 'openai',
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS server_config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS webhooks (
    id         TEXT PRIMARY KEY,
    user_email TEXT NOT NULL,
    name       TEXT NOT NULL,
    url        TEXT NOT NULL,
    secret     TEXT NOT NULL,
    enabled    INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    last_fired REAL,
    fire_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_webhooks_user ON webhooks(user_email);

CREATE INDEX IF NOT EXISTS idx_vk_user   ON virtual_keys(user_email, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evt_user  ON gateway_events(user_email, timestamp DESC);
"""


def connect() -> sqlite3.Connection:
    _DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    # Add new columns to existing tables (ALTER TABLE IF NOT EXISTS is not supported in SQLite)
    for sql in [
        "ALTER TABLE users ADD COLUMN last_login_at REAL",
        "ALTER TABLE users ADD COLUMN login_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN twofa_enabled INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN failed_attempts INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN locked_until REAL",
    ]:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


# module-level connection (FastAPI runs single-process; WAL handles concurrent reads fine)
_conn: sqlite3.Connection | None = None


def get() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = connect()
    return _conn


# ── users ────────────────────────────────────────────────────────────────────

def user_get(email: str) -> dict | None:
    row = get().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    return dict(row) if row else None


def user_create(email: str, name: str, salt: str, hash_: str) -> None:
    get().execute(
        "INSERT INTO users (email, name, salt, hash, created_at) VALUES (?,?,?,?,?)",
        (email, name, salt, hash_, time.time()),
    )
    get().commit()


def user_delete(email: str) -> None:
    get().execute("DELETE FROM users WHERE email=?", (email,))
    get().commit()


def purge_fake_users() -> None:
    """Remove any seeded / test / demo accounts that were never real users.
    Also deletes the legacy users.json to prevent re-importing fake accounts."""
    _FAKE = {
        "olivia@acme.com",
        "anonymous",
    }
    # Delete any account whose email looks like a generated test user
    rows = get().execute("SELECT email FROM users").fetchall()
    for row in rows:
        email = row["email"]
        if email in _FAKE or email.startswith("user-") and "@example.com" in email:
            get().execute("DELETE FROM users WHERE email=?", (email,))
    get().commit()

    # Wipe the legacy JSON so it can never re-import fake accounts
    legacy = Path(__file__).resolve().parents[2] / "data" / "users.json"
    if legacy.exists():
        legacy.unlink(missing_ok=True)


def user_record_login(email: str) -> None:
    get().execute(
        "UPDATE users SET last_login_at=?, login_count=COALESCE(login_count,0)+1 WHERE email=?",
        (time.time(), email),
    )
    get().commit()


def user_record_failed_login(email: str) -> int:
    """Increment failed attempt count. Returns new count."""
    get().execute(
        "UPDATE users SET failed_attempts=COALESCE(failed_attempts,0)+1 WHERE email=?",
        (email,)
    )
    get().commit()
    row = get().execute("SELECT failed_attempts FROM users WHERE email=?", (email,)).fetchone()
    return row["failed_attempts"] if row else 1


def user_lock(email: str, until: float) -> None:
    get().execute(
        "UPDATE users SET locked_until=? WHERE email=?", (until, email)
    )
    get().commit()


def user_reset_failed(email: str) -> None:
    get().execute(
        "UPDATE users SET failed_attempts=0, locked_until=NULL WHERE email=?", (email,)
    )
    get().commit()


def user_set_twofa(email: str, enabled: bool) -> None:
    get().execute(
        "UPDATE users SET twofa_enabled=? WHERE email=?", (1 if enabled else 0, email)
    )
    get().commit()


def user_all() -> list[dict]:
    return [dict(r) for r in get().execute("SELECT * FROM users ORDER BY created_at")]


# ── virtual keys ─────────────────────────────────────────────────────────────

def vk_upsert(user_email: str, vk: dict) -> None:
    get().execute(
        """INSERT INTO virtual_keys
           (key, user_email, name, budget_usd, spent_usd, rate_limit_per_min,
            enabled, request_count, blocked_count, cost_saved_usd, created_at, recent_blocks)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(key) DO UPDATE SET
             spent_usd=excluded.spent_usd,
             enabled=excluded.enabled,
             request_count=excluded.request_count,
             blocked_count=excluded.blocked_count,
             cost_saved_usd=excluded.cost_saved_usd,
             recent_blocks=excluded.recent_blocks""",
        (vk["key"], user_email, vk["name"], vk["budget_usd"], vk["spent_usd"],
         vk["rate_limit_per_min"], 1 if vk["enabled"] else 0,
         vk["request_count"], vk["blocked_count"], vk["cost_saved_usd"],
         vk["created_at"], json.dumps(vk.get("recent_blocks", []))),
    )
    get().commit()


def vk_list(user_email: str) -> list[dict]:
    rows = get().execute(
        "SELECT * FROM virtual_keys WHERE user_email=? ORDER BY created_at DESC",
        (user_email,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["recent_blocks"] = json.loads(d.get("recent_blocks") or "[]")
        d["enabled"] = bool(d["enabled"])
        out.append(d)
    return out


def vk_get(key: str) -> dict | None:
    row = get().execute("SELECT * FROM virtual_keys WHERE key=?", (key,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["recent_blocks"] = json.loads(d.get("recent_blocks") or "[]")
    d["enabled"] = bool(d["enabled"])
    return d


# ── gateway events ────────────────────────────────────────────────────────────

_MAX_EVENTS_DB = 200


def evt_append(user_email: str, event: dict) -> None:
    get().execute(
        "INSERT INTO gateway_events (user_email, timestamp, data) VALUES (?,?,?)",
        (user_email, event.get("timestamp", time.time()), json.dumps(event)),
    )
    # Keep only the most recent _MAX_EVENTS_DB rows per user
    get().execute(
        """DELETE FROM gateway_events WHERE user_email=? AND id NOT IN (
             SELECT id FROM gateway_events WHERE user_email=?
             ORDER BY timestamp DESC LIMIT ?)""",
        (user_email, user_email, _MAX_EVENTS_DB),
    )
    get().commit()


def evt_list(user_email: str, limit: int = 50) -> list[dict]:
    rows = get().execute(
        "SELECT data FROM gateway_events WHERE user_email=? ORDER BY timestamp DESC LIMIT ?",
        (user_email, limit),
    ).fetchall()
    return [json.loads(r["data"]) for r in rows]


# ── upstream keys (encrypted at rest) ────────────────────────────────────────

def _xor_key(plaintext: str, secret: bytes) -> str:
    """Simple XOR cipher with repeating secret. Good enough for at-rest protection."""
    import base64
    enc = bytes(b ^ secret[i % len(secret)] for i, b in enumerate(plaintext.encode()))
    return base64.urlsafe_b64encode(enc).decode()


def _xor_decrypt(ciphertext: str, secret: bytes) -> str:
    import base64
    enc = base64.urlsafe_b64decode(ciphertext)
    return bytes(b ^ secret[i % len(secret)] for i, b in enumerate(enc)).decode()


def _db_secret() -> bytes:
    import os
    s = os.environ.get("AGSEC_SECRET", "")
    if s:
        return s.encode()
    from pathlib import Path
    sf = Path(__file__).resolve().parents[2] / "data" / ".session_secret"
    return sf.read_text(encoding="utf-8").strip().encode() if sf.exists() else b"agentshield-default-secret"


def upstream_save(user_email: str, api_key: str, provider: str) -> None:
    enc = _xor_key(api_key, _db_secret())
    get().execute(
        """INSERT INTO upstream_keys (user_email, enc_key, provider, updated_at)
           VALUES (?,?,?,?)
           ON CONFLICT(user_email) DO UPDATE SET enc_key=excluded.enc_key,
             provider=excluded.provider, updated_at=excluded.updated_at""",
        (user_email, enc, provider, time.time()),
    )
    get().commit()


def upstream_load(user_email: str) -> tuple[str, str] | None:
    """Returns (api_key, provider) or None."""
    row = get().execute(
        "SELECT enc_key, provider FROM upstream_keys WHERE user_email=?", (user_email,)
    ).fetchone()
    if not row:
        return None
    try:
        return _xor_decrypt(row["enc_key"], _db_secret()), row["provider"]
    except Exception:
        return None


def upstream_delete(user_email: str) -> None:
    get().execute("DELETE FROM upstream_keys WHERE user_email=?", (user_email,))
    get().commit()


# ── server config (global key-value store) ────────────────────────────────────

def config_set(key: str, value: str) -> None:
    get().execute(
        "INSERT INTO server_config (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    get().commit()


def config_get(key: str) -> str | None:
    row = get().execute("SELECT value FROM server_config WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


# ── webhooks ──────────────────────────────────────────────────────────────────

def webhook_create(user_email: str, wh_id: str, name: str, url: str, secret: str) -> None:
    get().execute(
        "INSERT INTO webhooks (id, user_email, name, url, secret, enabled, created_at) VALUES (?,?,?,?,?,1,?)",
        (wh_id, user_email, name, url, secret, time.time()),
    )
    get().commit()


def webhook_list(user_email: str) -> list[dict]:
    rows = get().execute(
        "SELECT * FROM webhooks WHERE user_email=? ORDER BY created_at DESC", (user_email,)
    ).fetchall()
    return [dict(r) for r in rows]


def webhook_delete(user_email: str, wh_id: str) -> bool:
    cur = get().execute("DELETE FROM webhooks WHERE id=? AND user_email=?", (wh_id, user_email))
    get().commit()
    return cur.rowcount > 0


def webhook_record_fire(wh_id: str) -> None:
    get().execute(
        "UPDATE webhooks SET last_fired=?, fire_count=fire_count+1 WHERE id=?",
        (time.time(), wh_id),
    )
    get().commit()


# ── chart data ────────────────────────────────────────────────────────────────

def chart_data(user_email: str, days: int = 7) -> dict:
    """Return per-day attack/request/spend counts for the last N days."""
    since = time.time() - days * 86400
    rows = get().execute(
        "SELECT data FROM gateway_events WHERE user_email=? AND timestamp>? ORDER BY timestamp ASC",
        (user_email, since),
    ).fetchall()

    import datetime
    buckets: dict[str, dict] = {}
    for i in range(days):
        d = (datetime.date.today() - datetime.timedelta(days=days - 1 - i)).isoformat()
        buckets[d] = {"attacks": 0, "passed": 0, "spend": 0.0, "saved": 0.0}

    for row in rows:
        try:
            ev = json.loads(row["data"])
            day = datetime.datetime.fromtimestamp(ev["timestamp"]).date().isoformat()
            if day not in buckets:
                continue
            if ev.get("outcome") == "blocked":
                buckets[day]["attacks"] += 1
                buckets[day]["saved"] += ev.get("cost_saved_usd", 0.0)
            else:
                buckets[day]["passed"] += 1
                buckets[day]["spend"] += ev.get("cost_usd", 0.0)
        except Exception:
            pass

    labels = list(buckets.keys())
    return {
        "labels": [l[5:] for l in labels],  # MM-DD format
        "attacks": [buckets[l]["attacks"] for l in labels],
        "passed":  [buckets[l]["passed"]  for l in labels],
        "spend":   [round(buckets[l]["spend"], 5)  for l in labels],
        "saved":   [round(buckets[l]["saved"], 5)  for l in labels],
    }


# ── password change ───────────────────────────────────────────────────────────

def user_update_password(email: str, new_salt: str, new_hash: str) -> None:
    get().execute(
        "UPDATE users SET salt=?, hash=? WHERE email=?", (new_salt, new_hash, email)
    )
    get().commit()


# ── scans ────────────────────────────────────────────────────────────────────

def scan_create(
    scan_id: str,
    user_email: str,
    target: str,
    risk_score: float,
    grade: str,
    probes_run: list[str],
    started_at: float,
    finished_at: float,
) -> None:
    duration = round(finished_at - started_at, 2)
    get().execute(
        """INSERT INTO scans
           (id, user_email, target, risk_score, grade, probes_run, started_at, finished_at, duration_s)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (scan_id, user_email, target, risk_score, grade,
         json.dumps(probes_run), started_at, finished_at, duration),
    )
    get().commit()


def scan_get(scan_id: str) -> dict | None:
    row = get().execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["probes_run"] = json.loads(d["probes_run"])
    d["findings"] = findings_for_scan(scan_id)
    return d


def scans_for_user(user_email: str, limit: int = 50) -> list[dict]:
    rows = get().execute(
        "SELECT * FROM scans WHERE user_email=? ORDER BY started_at DESC LIMIT ?",
        (user_email, limit),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["probes_run"] = json.loads(d["probes_run"])
        d["finding_count"] = get().execute(
            "SELECT COUNT(*) FROM findings WHERE scan_id=?", (d["id"],)
        ).fetchone()[0]
        d["severity_counts"] = _severity_counts(d["id"])
        out.append(d)
    return out


def _severity_counts(scan_id: str) -> dict[str, int]:
    rows = get().execute(
        "SELECT severity, COUNT(*) as n FROM findings WHERE scan_id=? GROUP BY severity",
        (scan_id,),
    ).fetchall()
    counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Info": 0}
    for r in rows:
        counts[r["severity"]] = r["n"]
    return counts


# ── findings ─────────────────────────────────────────────────────────────────

def findings_insert_batch(scan_id: str, findings: list[dict]) -> None:
    db = get()
    db.executemany(
        """INSERT OR IGNORE INTO findings
           (id, scan_id, probe, title, severity, severity_score,
            description, payload, response, confidence, tags, timestamp)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                f["id"], scan_id, f["probe"], f["title"],
                f["severity"], f["severity_score"],
                f.get("description", ""), f.get("payload", ""), f.get("response", ""),
                f.get("confidence", 0.5), json.dumps(f.get("tags", [])), f.get("timestamp", time.time()),
            )
            for f in findings
        ],
    )
    db.commit()


def findings_for_scan(scan_id: str) -> list[dict]:
    rows = get().execute(
        "SELECT * FROM findings WHERE scan_id=? ORDER BY severity_score DESC, confidence DESC",
        (scan_id,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["tags"] = json.loads(d["tags"])
        out.append(d)
    return out


# ── migration: import legacy users.json if DB is empty ──────────────────────

def migrate_legacy_users() -> int:
    legacy = Path(__file__).resolve().parents[2] / "data" / "users.json"
    if not legacy.exists():
        return 0
    if get().execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
        return 0  # already have users, don't clobber
    try:
        data = json.loads(legacy.read_text(encoding="utf-8"))
    except Exception:
        return 0
    count = 0
    for email, u in data.items():
        try:
            user_create(email, u.get("name", email.split("@")[0]), u["salt"], u["hash"])
            count += 1
        except Exception:
            pass
    return count
