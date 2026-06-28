"""API Key Guard - multi-provider LLM gateway with active attack blocking.

Per-user isolation: every user's virtual keys, events, spend, and upstream
key are stored separately. One user can never see or touch another's data.

Providers supported: OpenAI, Anthropic (Claude), Groq, Google Gemini,
Mistral, Together AI, Cohere.
"""

from __future__ import annotations

import os
import re
import time
import secrets
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .detector import full_scan
from .sanitizer import redact_pii
from .providers import (
    call_provider, detect_provider, get_cost,
    list_models, all_providers, DEFAULT_MODELS, PROVIDER_PRICING,
)

DEFAULT_MODEL = "gpt-4o-mini"

# ── Indirect injection patterns (hidden in external content) ──────────────────
_INDIRECT_PATTERNS = [
    re.compile(p, re.I | re.S) for p in [
        r"<!--\s*(?:ai|llm|assistant|ignore|system|instruction)",
        r"<\s*(?:script|iframe|object)[^>]*>",
        r"\[\s*system\s*\]",
        r"\{\s*[\"']?(?:role|system|instruction)[\"']?\s*:",
        r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
        r"new\s+(?:system\s+)?prompt\s*:",
        r"<\|(?:im_start|im_end|endoftext)\|>",
        r"\[INST\]|\[/?SYS\]|<<SYS>>|</?(user|system|assistant)>",
        r"---\s*(?:END\s+OF\s+)?(?:SYSTEM\s+)?PROMPT\s*---",
        r"#{3,}\s*OVERRIDE",
        r"you\s+are\s+now\s+(?:in\s+)?(?:jailbreak|unrestricted|dan|evil)",
    ]
]

# ── Data exfiltration patterns (in LLM output - attacker trying to leak) ──────
_EXFIL_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"(?:system\s+prompt|hidden\s+instruction|secret\s+instruction)\s*(?:is|says|reads|contains)\s*[:\"]",
        r"(?:my|the)\s+(?:system\s+prompt|instructions)\s+(?:are|say|tell)",
        r"(?:api[_\s]key|secret[_\s]key|access[_\s]token)\s*[:=]\s*\S+",
        r"(?:password|passwd|pwd)\s*[:=]\s*\S+",
        r"(?:sk-|gsk_|AIzaSy|sk-ant-)[a-zA-Z0-9\-_]{8,}",
        r"Bearer\s+[a-zA-Z0-9\-._~+/]{20,}",
        r"(?:here\s+is|here\s+are)\s+(?:the|your|all)\s+(?:user|retrieved|database|private)",
    ]
]


@dataclass
class VirtualKey:
    key: str
    name: str
    budget_usd: float
    spent_usd: float = 0.0
    rate_limit_per_min: int = 30
    enabled: bool = True
    request_count: int = 0
    blocked_count: int = 0
    cost_saved_usd: float = 0.0
    created_at: float = field(default_factory=time.time)
    hits: list[float] = field(default_factory=list)
    recent_blocks: list[float] = field(default_factory=list)
    # ── New security controls ──────────────────────────────────────────────────
    expires_at: float | None = None               # Unix ts; None = never expires
    allowed_models: list[str] = field(default_factory=list)   # empty = all allowed
    allowed_hours: list[int] = field(default_factory=list)    # UTC hours 0-23; empty = all hours
    velocity_window_sec: int = 60                 # rolling window for spend velocity
    velocity_max_usd: float = 0.0                 # max spend in window (0 = disabled)
    velocity_spent: list[tuple] = field(default_factory=list) # [(ts, cost), ...]
    allowed_ips: list[str] = field(default_factory=list)       # empty = all IPs allowed

    def public(self, reveal: bool = False) -> dict[str, Any]:
        shown = self.key if reveal else (self.key[:8] + "..." + self.key[-4:])
        return {
            "key": shown,
            "name": self.name,
            "budget_usd": round(self.budget_usd, 4),
            "spent_usd": round(self.spent_usd, 6),
            "remaining_usd": round(max(0.0, self.budget_usd - self.spent_usd), 6),
            "pct_used": round(min(100.0, self.spent_usd / self.budget_usd * 100) if self.budget_usd else 0, 1),
            "rate_limit_per_min": self.rate_limit_per_min,
            "enabled": self.enabled,
            "request_count": self.request_count,
            "blocked_count": self.blocked_count,
            "cost_saved_usd": round(self.cost_saved_usd, 6),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "allowed_models": self.allowed_models,
            "allowed_hours": self.allowed_hours,
            "velocity_window_sec": self.velocity_window_sec,
            "velocity_max_usd": self.velocity_max_usd,
            "is_expired": bool(self.expires_at and time.time() > self.expires_at),
            "allowed_ips": self.allowed_ips,
        }


# ── Per-user state (write-through cache → SQLite) ─────────────────────────────
# In-memory layer for hot-path reads; every mutation is also written to DB.
_user_keys:     dict[str, dict[str, VirtualKey]] = {}
_user_events:   dict[str, list[dict]] = {}
_user_upstream: dict[str, str] = {}          # email -> real provider key
_user_provider: dict[str, str] = {}          # email -> provider name
_MAX_EVENTS = 120
_user_loaded:   set[str] = set()             # emails whose data has been loaded from DB

# Persistent attacker escalation: track blocks per virtual key
# If a key blocks ≥ 3 times within 5 minutes → auto-suspend that key
_ATTACK_WINDOW_SEC = 300   # 5 minutes
_ATTACK_THRESHOLD  = 3     # blocks before auto-suspend


def _ensure_loaded(user: str) -> None:
    """Load a user's gateway data from DB into memory on first access."""
    if user in _user_loaded:
        return
    from ..api import db
    # Load virtual keys
    if user not in _user_keys:
        _user_keys[user] = {}
    for row in db.vk_list(user):
        vk = VirtualKey(
            key=row["key"], name=row["name"],
            budget_usd=row["budget_usd"], spent_usd=row["spent_usd"],
            rate_limit_per_min=row["rate_limit_per_min"], enabled=row["enabled"],
            request_count=row["request_count"], blocked_count=row["blocked_count"],
            cost_saved_usd=row["cost_saved_usd"], created_at=row["created_at"],
            recent_blocks=row.get("recent_blocks", []),
            expires_at=row.get("expires_at"),
            allowed_models=row.get("allowed_models") or [],
            allowed_hours=row.get("allowed_hours") or [],
            velocity_window_sec=row.get("velocity_window_sec") or 60,
            velocity_max_usd=row.get("velocity_max_usd") or 0.0,
            velocity_spent=row.get("velocity_spent") or [],
            allowed_ips=row.get("allowed_ips") or [],
        )
        _user_keys[user][vk.key] = vk
    # Load upstream key
    result = db.upstream_load(user)
    if result and user not in _user_upstream:
        _user_upstream[user], _user_provider[user] = result
    # Load recent events into memory cache
    if user not in _user_events:
        _user_events[user] = list(reversed(db.evt_list(user, 120)))
    _user_loaded.add(user)


def _keys(user: str) -> dict[str, VirtualKey]:
    _ensure_loaded(user)
    if user not in _user_keys:
        _user_keys[user] = {}
    return _user_keys[user]


def _events(user: str) -> list[dict]:
    _ensure_loaded(user)
    if user not in _user_events:
        _user_events[user] = []
    return _user_events[user]


def _persist_key(user: str, vk: VirtualKey) -> None:
    """Write a virtual key to the DB."""
    try:
        from ..api import db
        db.vk_upsert(user, {
            "key": vk.key, "name": vk.name, "budget_usd": vk.budget_usd,
            "spent_usd": vk.spent_usd, "rate_limit_per_min": vk.rate_limit_per_min,
            "enabled": vk.enabled, "request_count": vk.request_count,
            "blocked_count": vk.blocked_count, "cost_saved_usd": vk.cost_saved_usd,
            "created_at": vk.created_at, "recent_blocks": vk.recent_blocks,
            "expires_at": vk.expires_at,
            "allowed_models": vk.allowed_models,
            "allowed_hours": vk.allowed_hours,
            "velocity_window_sec": vk.velocity_window_sec,
            "velocity_max_usd": vk.velocity_max_usd,
            "velocity_spent": vk.velocity_spent,
            "allowed_ips": vk.allowed_ips,
        })
    except Exception:
        pass  # never let DB errors break a request


def _log_event(user: str, **event: Any) -> None:
    event["timestamp"] = time.time()
    # In-memory cache
    ev = _events(user)
    ev.append(event)
    if len(ev) > _MAX_EVENTS:
        ev.pop(0)
    # Persist to DB
    try:
        from ..api import db
        db.evt_append(user, event)
    except Exception:
        pass


def recent_events(user: str, limit: int = 50) -> list[dict]:
    return list(reversed(_events(user)[-limit:]))


# ── Key management ─────────────────────────────────────────────────────────────

def create_key(
    user: str, name: str, budget_usd: float = 5.0, rate_limit_per_min: int = 30,
    expires_in_days: float | None = None,
    allowed_models: list[str] | None = None,
    allowed_hours: list[int] | None = None,
    velocity_window_sec: int = 60,
    velocity_max_usd: float = 0.0,
) -> dict:
    key = "agk-" + secrets.token_hex(16)
    expires_at = time.time() + expires_in_days * 86400 if expires_in_days else None
    vk = VirtualKey(
        key=key, name=name or "unnamed",
        budget_usd=max(0.0, budget_usd),
        rate_limit_per_min=max(1, rate_limit_per_min),
        expires_at=expires_at,
        allowed_models=allowed_models or [],
        allowed_hours=[h for h in (allowed_hours or []) if 0 <= h <= 23],
        velocity_window_sec=max(10, velocity_window_sec),
        velocity_max_usd=max(0.0, velocity_max_usd),
    )
    _keys(user)[key] = vk
    _persist_key(user, vk)
    return vk.public(reveal=True)


def update_key(
    user: str, key: str,
    expires_in_days: float | None = None,
    clear_expiry: bool = False,
    allowed_models: list[str] | None = None,
    allowed_hours: list[int] | None = None,
    velocity_window_sec: int | None = None,
    velocity_max_usd: float | None = None,
) -> dict:
    """Update security controls on an existing virtual key."""
    ku = _keys(user)
    vk = ku.get(key)
    if not vk:
        for k, v in ku.items():
            if k.startswith(key.split("...")[0]):
                vk = v; break
    if not vk:
        return {"error": "Key not found"}
    if clear_expiry:
        vk.expires_at = None
    elif expires_in_days is not None:
        vk.expires_at = time.time() + expires_in_days * 86400
    if allowed_models is not None:
        vk.allowed_models = allowed_models
    if allowed_hours is not None:
        vk.allowed_hours = [h for h in allowed_hours if 0 <= h <= 23]
    if velocity_window_sec is not None:
        vk.velocity_window_sec = max(10, velocity_window_sec)
    if velocity_max_usd is not None:
        vk.velocity_max_usd = max(0.0, velocity_max_usd)
    _persist_key(user, vk)
    return {"ok": True, "key": vk.public()}


def list_keys(user: str) -> list[dict]:
    return [vk.public() for vk in sorted(_keys(user).values(), key=lambda k: -k.created_at)]


def revoke_key(user: str, key: str) -> bool:
    ku = _keys(user)
    vk = ku.get(key)
    if not vk:
        for k, v in ku.items():
            if k.startswith(key.split("...")[0]):
                vk = v
                break
    if vk:
        vk.enabled = False
        _persist_key(user, vk)
        return True
    return False


def enable_key(user: str, key: str) -> bool:
    """Re-enable a suspended or revoked virtual key and clear its block history."""
    ku = _keys(user)
    vk = ku.get(key)
    if not vk:
        for k, v in ku.items():
            if k.startswith(key.split("...")[0]):
                vk = v
                break
    if vk:
        vk.enabled = True
        vk.recent_blocks = []
        _persist_key(user, vk)
        return True
    return False


def upstream_status(user: str) -> dict:
    real = _user_upstream.get(user, "") or os.environ.get("OPENAI_API_KEY", "")
    has_real = bool(real)
    provider = _user_provider.get(user, "")
    if not provider and real:
        provider = detect_provider(real)
    if not provider and os.environ.get("OPENAI_API_KEY"):
        provider = "openai"

    provider_models = list_models(provider) if provider else list(PROVIDER_PRICING.get("openai", {}).keys())

    return {
        "upstream": provider if has_real else "local",
        "provider": provider,
        "real_key_configured": has_real,
        "key_hint": (real[:3] + "..." + real[-4:]) if len(real) > 8 else ("set" if has_real else ""),
        "note": (
            f"Forwarding to {provider.title() if provider else 'provider'}. Your real key stays server-side."
            if has_real else
            "Demo mode: forwarding to the AgentShield Security LLM (no real key set). "
            "Connect a provider key to protect a real upstream."
        ),
        "models": provider_models,
        "all_providers": all_providers(),
    }


def set_upstream_key(user: str, key: str, provider: str | None = None) -> dict:
    key = (key or "").strip()
    if not key:
        return {"ok": False, "error": "empty key"}
    _ensure_loaded(user)
    _user_upstream[user] = key
    detected = provider or detect_provider(key)
    _user_provider[user] = detected
    try:
        from ..api import db
        db.upstream_save(user, key, detected)
    except Exception:
        pass
    return {"ok": True, **upstream_status(user)}


def clear_upstream_key(user: str) -> dict:
    _ensure_loaded(user)
    _user_upstream.pop(user, None)
    _user_provider.pop(user, None)
    try:
        from ..api import db
        db.upstream_delete(user)
    except Exception:
        pass
    return {"ok": True, **upstream_status(user)}


def stats(user: str) -> dict:
    keys = list(_keys(user).values())
    return {
        "total_keys": len(keys),
        "active_keys": sum(1 for k in keys if k.enabled),
        "total_requests": sum(k.request_count for k in keys),
        "total_blocked": sum(k.blocked_count for k in keys),
        "total_spent_usd": round(sum(k.spent_usd for k in keys), 6),
        "total_saved_usd": round(sum(k.cost_saved_usd for k in keys), 6),
    }


# ── Rate limiting ──────────────────────────────────────────────────────────────

def _rate_ok(vk: VirtualKey) -> bool:
    now = time.time()
    vk.hits = [t for t in vk.hits if now - t < 60]
    if len(vk.hits) >= vk.rate_limit_per_min:
        return False
    vk.hits.append(now)
    return True


# ── Persistent attacker detection ─────────────────────────────────────────────

def _trigger_notifications(user: str, vk: VirtualKey, event: dict) -> None:
    """Fire webhooks and send email alerts for block events. Always background."""
    try:
        from ..api.notifications import fire_webhooks, notify_critical_attack, notify_budget_warning
        fire_webhooks(user, event)
        # Critical attack alert
        if event.get("severity") in ("critical", "high") and event.get("risk_score", 0) >= 80:
            notify_critical_attack(
                user, vk.name, event.get("threat", ""), event.get("message", ""),
                event.get("risk_score", 0),
            )
        # Budget warning at 80%
        if vk.budget_usd > 0:
            pct = vk.spent_usd / vk.budget_usd * 100
            if 79 <= pct <= 82:  # narrow band so we don't spam on every request
                notify_budget_warning(user, vk.name, pct, vk.spent_usd, vk.budget_usd)
    except Exception:
        pass


def _record_block(vk: VirtualKey) -> bool:
    """Record a block hit. Returns True if key should be auto-suspended."""
    now = time.time()
    vk.recent_blocks = [t for t in vk.recent_blocks if now - t < _ATTACK_WINDOW_SEC]
    vk.recent_blocks.append(now)
    if len(vk.recent_blocks) >= _ATTACK_THRESHOLD:
        vk.enabled = False  # auto-suspend
        return True
    return False


# ── Multi-turn conversation memory (per session) ──────────────────────────────
# Stores the last N user messages per session so we can detect jailbreaks that
# spread across multiple turns (each innocent alone, dangerous together).
_CONV_STORE: dict[str, list[str]] = {}  # session_id -> [user messages]
_CONV_WINDOW = 6       # how many past turns to concatenate for the scan
_CONV_MAX_SESSIONS = 5000  # evict oldest when limit reached


def _record_turn(session_id: str, user_text: str) -> str:
    """Append this turn and return a concatenated context window for scanning."""
    if session_id not in _CONV_STORE:
        if len(_CONV_STORE) >= _CONV_MAX_SESSIONS:
            # evict oldest entry
            oldest = next(iter(_CONV_STORE))
            del _CONV_STORE[oldest]
        _CONV_STORE[session_id] = []
    _CONV_STORE[session_id].append(user_text)
    # Keep only the last N turns
    _CONV_STORE[session_id] = _CONV_STORE[session_id][-_CONV_WINDOW:]
    return " |TURN| ".join(_CONV_STORE[session_id])


# ── Replay attack detection ───────────────────────────────────────────────────
# Same exact payload sent twice within 30s from the same key = flag as replay
_REPLAY_STORE: dict[str, dict[str, float]] = {}  # key -> {payload_hash: ts}
_REPLAY_WINDOW_SEC = 30


def _is_replay(api_key: str, text: str) -> bool:
    import hashlib
    h = hashlib.sha256(text.strip().encode()).hexdigest()[:16]
    now = time.time()
    store = _REPLAY_STORE.setdefault(api_key, {})
    # Prune expired entries
    expired = [k for k, ts in store.items() if now - ts > _REPLAY_WINDOW_SEC]
    for k in expired:
        del store[k]
    if h in store:
        return True
    store[h] = now
    return False


# ── Session anomaly: auto-suspend if >60% block rate after 10+ requests ──────

def _check_anomaly_suspend(vk: VirtualKey) -> bool:
    """Auto-suspend if >60% of recent requests were blocked (active attack)."""
    if vk.request_count < 10:
        return False
    block_rate = vk.blocked_count / vk.request_count
    if block_rate >= 0.60 and vk.enabled:
        vk.enabled = False
        return True
    return False


# ── Spend velocity guard ───────────────────────────────────────────────────────

def _velocity_ok(vk: VirtualKey, cost: float) -> bool:
    """Return False if this spend would exceed the velocity cap in the rolling window."""
    if vk.velocity_max_usd <= 0:
        return True  # disabled
    now = time.time()
    window_start = now - vk.velocity_window_sec
    # Prune old entries
    vk.velocity_spent = [(ts, c) for ts, c in vk.velocity_spent if ts >= window_start]
    window_sum = sum(c for _, c in vk.velocity_spent) + cost
    if window_sum > vk.velocity_max_usd:
        return False
    vk.velocity_spent.append((now, cost))
    return True


# ── Indirect injection scanner ────────────────────────────────────────────────

def _scan_indirect_injection(text: str) -> dict | None:
    """Scan text (e.g. from external content, tool results) for hidden instructions."""
    for pat in _INDIRECT_PATTERNS:
        m = pat.search(text)
        if m:
            return {
                "detected": True,
                "pattern": m.group(0)[:80],
                "threat": "indirect_prompt_injection",
                "severity": "high",
            }
    return None


# ── Output / data exfiltration scanner ───────────────────────────────────────

def _scan_output_exfil(output: str) -> str:
    """Scan LLM output for data exfiltration attempts. Returns cleaned output."""
    for pat in _EXFIL_PATTERNS:
        if pat.search(output):
            # Redact the sensitive segment
            output = pat.sub("[REDACTED BY AGENTSHIELD]", output)
    return output


# ── Resolve virtual key across all users (for proxy endpoint) ─────────────────

def resolve_virtual_key(api_key: str) -> tuple[str | None, VirtualKey | None]:
    """Find which user owns a virtual key. Used by the drop-in proxy."""
    for user, keys in _user_keys.items():
        if api_key in keys:
            return user, keys[api_key]
    return None, None


# ── Token estimator ───────────────────────────────────────────────────────────

def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── The protected call ────────────────────────────────────────────────────────

def gateway_chat(
    user: str,
    api_key: str | None,
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    source: str = "user",
    client_ip: str | None = None,
) -> dict:
    """Screen, budget-check, and forward a chat request. Isolated to `user`."""
    ku = _keys(user)
    vk = ku.get(api_key or "")
    if not vk:
        return {"error": "Invalid API key.", "status": 401, "blocked": True}
    if not vk.enabled:
        suspended_msg = (
            "This key has been auto-suspended after repeated attack attempts. "
            "Contact your admin to restore access."
            if vk.recent_blocks else
            "This key has been revoked."
        )
        return {"error": suspended_msg, "status": 403, "blocked": True}

    if not _rate_ok(vk):
        return {"error": f"Rate limit exceeded ({vk.rate_limit_per_min}/min).",
                "status": 429, "blocked": True, "key": vk.public()}

    # ── Security check 0a: IP allowlist ──────────────────────────────────────
    if vk.allowed_ips and client_ip:
        if client_ip not in vk.allowed_ips:
            vk.blocked_count += 1
            _persist_key(user, vk)
            _log_event(user, outcome="blocked", layer="ip-allowlist",
                       key_name=vk.name, threat="unauthorized_ip",
                       explanation=f"Request from {client_ip} not in allowlist.",
                       risk_score=85, severity="high")
            return {"error": f"IP {client_ip} is not in this key's allowlist.",
                    "status": 403, "blocked": True, "key": vk.public()}

    # ── Security check 1: Key expiry ─────────────────────────────────────────
    if vk.expires_at and time.time() > vk.expires_at:
        vk.enabled = False
        _persist_key(user, vk)
        import datetime as _dt
        exp = _dt.datetime.utcfromtimestamp(vk.expires_at).strftime("%Y-%m-%d %H:%M UTC")
        return {"error": f"This key expired on {exp} and has been automatically disabled.",
                "status": 403, "blocked": True, "key": vk.public()}

    # ── Security check 2: Allowed hours (UTC) ────────────────────────────────
    if vk.allowed_hours:
        import datetime as _dt
        current_hour = _dt.datetime.utcnow().hour
        if current_hour not in vk.allowed_hours:
            return {"error": f"This key is restricted to UTC hours {sorted(vk.allowed_hours)}. "
                             f"Current UTC hour: {current_hour}.",
                    "status": 403, "blocked": True, "key": vk.public()}

    # ── Security check 3: Model allowlist ────────────────────────────────────
    if vk.allowed_models and model not in vk.allowed_models:
        return {"error": f"Model '{model}' is not in this key's allowlist {vk.allowed_models}.",
                "status": 403, "blocked": True, "key": vk.public()}

    vk.request_count += 1
    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    msg_preview = (user_text[:120] + "…") if len(user_text) > 120 else user_text

    # ── Layer 1: Indirect injection scan (catches hidden instructions in content) ──
    full_text = " ".join(m.get("content", "") for m in messages)
    indirect = _scan_indirect_injection(full_text)
    if indirect and source in ("external_content", "tool_result", "document"):
        vk.blocked_count += 1
        in_tok = _est_tokens(full_text)
        provider = _user_provider.get(user, "openai")
        saved = get_cost(provider, model, in_tok, max_tokens)
        vk.cost_saved_usd += saved
        auto_suspended = _record_block(vk)
        _log_event(user, outcome="blocked", layer="indirect-injection-scanner",
                   key_name=vk.name, message=msg_preview,
                   threat="indirect_prompt_injection",
                   explanation=f"Hidden instruction detected in external content: {indirect['pattern']}",
                   risk_score=90, severity="high", auto_suspended=auto_suspended,
                   cost_saved_usd=round(saved, 6))
        return {
            "blocked": True, "status": 200,
            "threat": "indirect_prompt_injection",
            "explanation": f"AgentShield blocked a hidden instruction in external content: \"{indirect['pattern']}\"",
            "severity": "high", "risk_score": 90,
            "auto_suspended": auto_suspended,
            "message": "Blocked: indirect prompt injection detected in external content.",
        }

    # ── Security check 4: Multi-turn jailbreak detection ─────────────────────
    # Build a rolling window of the last N user turns (keyed by virtual key so
    # each key gets its own conversation memory). A jailbreak spread across
    # multiple innocent-looking turns is caught by scanning the concatenation.
    session_id = f"{user}:{api_key}"
    multi_turn_ctx = _record_turn(session_id, user_text)

    # ── Security check 4b: Replay attack detection ───────────────────────────
    if _is_replay(api_key or "", user_text):
        vk.blocked_count += 1
        auto_suspended = _record_block(vk)
        _check_anomaly_suspend(vk)
        _persist_key(user, vk)
        _log_event(user, outcome="blocked", layer="replay-detector",
                   key_name=vk.name, threat="replay_attack",
                   explanation="Identical payload seen twice within 30s.",
                   risk_score=70, severity="high", auto_suspended=auto_suspended)
        return {"blocked": True, "status": 200, "threat": "replay_attack",
                "explanation": "AgentShield blocked a replay attack (identical payload repeated within 30s).",
                "risk_score": 70, "severity": "high",
                "message": "Blocked: replay attack detected.", "key": vk.public()}

    # ── Security check 4c: Session anomaly auto-suspend ──────────────────────
    if _check_anomaly_suspend(vk):
        _persist_key(user, vk)
        _log_event(user, outcome="blocked", layer="anomaly-detector",
                   key_name=vk.name, threat="high_block_rate",
                   explanation=f"Key auto-suspended: {vk.blocked_count}/{vk.request_count} requests blocked (>{60}%).",
                   risk_score=90, severity="critical")
        return {"blocked": True, "status": 403, "threat": "high_block_rate",
                "explanation": "Key auto-suspended due to abnormally high attack rate (>60% of requests blocked).",
                "message": "Key suspended: active attack pattern detected.", "key": vk.public()}

    # ── Layer 2: Full Security LLM analysis ───────────────────────────────────
    from ..agentshield import analyze_threat, record_action
    # Scan the multi-turn context window; fall back to single-turn if the
    # concatenated scan raises the risk score above single-turn.
    report = analyze_threat(multi_turn_ctx, source=source)
    if report["risk_score"] < 40 and multi_turn_ctx != user_text:
        # No cross-turn escalation detected; use single-turn report for latency
        report = analyze_threat(user_text, source=source)

    if report["decision"] in ("block", "review"):
        vk.blocked_count += 1
        in_tok = _est_tokens(user_text)
        provider = _user_provider.get(user, "openai")
        saved = get_cost(provider, model, in_tok, max_tokens)
        vk.cost_saved_usd += saved
        auto_suspended = _record_block(vk)
        record_action(vk.name, action="blocked", blocked=True)
        _persist_key(user, vk)

        # Fire webhooks + send alerts in background
        _trigger_notifications(user, vk, {
            "outcome": "blocked", "key_name": vk.name,
            "threat": report["attack_type"], "risk_score": report["risk_score"],
            "severity": report["severity"], "message": msg_preview,
            "cost_saved_usd": round(saved, 6),
            "timestamp": time.time(),
        })

        suspension_note = (
            f" Key auto-suspended after {_ATTACK_THRESHOLD} attacks in {_ATTACK_WINDOW_SEC//60} minutes."
            if auto_suspended else ""
        )

        _log_event(
            user,
            outcome="blocked",
            layer=report["signals"][0]["layer"] if report["signals"] else "security-llm",
            key_name=vk.name, message=msg_preview,
            threat=report["attack_type"],
            explanation=report["reason"],
            risk_score=report["risk_score"],
            severity=report["severity"],
            attack_chain=report["attack_chain"],
            confidence=report["confidence"],
            decision=report["decision"],
            analyst_report=report,
            cost_saved_usd=round(saved, 6),
            auto_suspended=auto_suspended,
        )
        return {
            "blocked": True, "status": 200,
            "threat": report["attack_type"],
            "explanation": report["reason"] + suspension_note,
            "risk_score": report["risk_score"],
            "severity": report["severity"],
            "decision": report["decision"],
            "attack_chain": report["attack_chain"],
            "confidence": report["confidence"],
            "recommended_action": report["recommended_action"],
            "analyst_report": report,
            "layer": "agentshield-security-llm",
            "cost_saved_usd": round(saved, 6),
            "auto_suspended": auto_suspended,
            "message": f"Blocked by AgentShield. Risk: {report['risk_score']}/100 ({report['severity']}).{suspension_note}",
            "key": vk.public(),
        }

    # ── Security check 5: Spend velocity guard ───────────────────────────────
    est_cost = get_cost(_user_provider.get(user, "openai"), model,
                        _est_tokens(user_text), max_tokens)
    if not _velocity_ok(vk, est_cost):
        return {"error": f"Spend velocity limit hit: max ${vk.velocity_max_usd:.4f} "
                         f"per {vk.velocity_window_sec}s. Slow down or raise the limit.",
                "status": 429, "blocked": True, "key": vk.public()}

    # ── Budget check ──────────────────────────────────────────────────────────
    if vk.spent_usd >= vk.budget_usd:
        return {"error": f"Budget exhausted (${vk.budget_usd:.2f}). Request blocked to prevent overspend.",
                "status": 402, "blocked": True, "key": vk.public()}

    # ── Forward to provider ───────────────────────────────────────────────────
    real_key = _user_upstream.get(user, "") or os.environ.get("OPENAI_API_KEY", "")
    provider = _user_provider.get(user, "")
    if not provider:
        provider = detect_provider(real_key) if real_key else "local"

    if real_key:
        result = call_provider(
            messages=messages, model=model, api_key=real_key,
            provider=provider, max_tokens=max_tokens,
        )
        if "error" in result:
            return {"error": f"upstream error: {result['error']}", "status": 502,
                    "blocked": False, "key": vk.public()}
        output = result["output"]
        in_tok = result["in_tok"]
        out_tok = result["out_tok"]
        upstream = result["provider"]
        model = result["model"]
    else:
        from ..llm import engine
        r = engine.chat(messages[-1].get("content", "") if messages else "")
        output = r.get("response", "")
        in_tok = _est_tokens(user_text)
        out_tok = _est_tokens(output)
        upstream = "local"

    # ── Layer 3: Output scanning (data exfiltration / prompt leak detection) ──
    raw_output = output
    output = _scan_output_exfil(output)
    output = redact_pii(output)
    was_redacted = output != raw_output

    cost = get_cost(provider if real_key else "openai", model, in_tok, out_tok)
    vk.spent_usd += cost

    record_action(vk.name, action="allowed", blocked=False, cost_usd=cost)
    _persist_key(user, vk)
    _log_event(
        user,
        outcome="forwarded", layer="security-llm-passed",
        key_name=vk.name, message=msg_preview,
        upstream=upstream, model=model,
        cost_usd=round(cost, 6),
        risk_score=report["risk_score"],
        severity=report["severity"],
        judge_latency_ms=report.get("latency_ms", 0),
        response_preview=(output[:140] + "…") if len(output) > 140 else output,
        output_redacted=was_redacted,
    )

    return {
        "blocked": False, "status": 200,
        "response": output,
        "upstream": upstream,
        "provider": upstream,
        "model": model,
        "passed_layers": ["agentshield-security-llm"],
        "risk_score": report["risk_score"],
        "severity": report["severity"],
        "judge_latency_ms": report.get("latency_ms", 0),
        "analyst_report": report,
        "cost_usd": round(cost, 6),
        "tokens": {"input": in_tok, "output": out_tok},
        "output_redacted": was_redacted,
        "key": vk.public(),
    }
