"""SQLite persistence layer.

Single file database at data/agentic.db. Three tables:
  users    — email, hashed password, salt
  scans    — one row per completed scan, linked to a user
  findings — one row per de-duplicated finding, linked to a scan
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
