"""API Key Guard - an LLM gateway that protects your real API key.

Per-user isolation: every user's virtual keys, events, spend, and upstream
key are stored separately. One user can never see or touch another's data.
"""

from __future__ import annotations

import os
import time
import secrets
from dataclasses import dataclass, field
from typing import Any

from .detector import full_scan
from .sanitizer import redact_pii

# Approx provider pricing, USD per 1K tokens: (input, output)
PRICING = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
    "gpt-3.5-turbo": (0.00050, 0.00150),
}
DEFAULT_MODEL = "gpt-4o-mini"


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
        }


# ── Per-user state ────────────────────────────────────────────────────────────
# All gateway data is scoped to the authenticated user's email. One user can
# never access, spend from, or see another user's keys or events.

_user_keys:   dict[str, dict[str, VirtualKey]] = {}  # email -> {key -> VirtualKey}
_user_events: dict[str, list[dict]] = {}             # email -> event list
_user_upstream: dict[str, str] = {}                  # email -> real provider key
_MAX_EVENTS = 80


def _keys(user: str) -> dict[str, VirtualKey]:
    if user not in _user_keys:
        _user_keys[user] = {}
    return _user_keys[user]


def _events(user: str) -> list[dict]:
    if user not in _user_events:
        _user_events[user] = []
    return _user_events[user]


def _log_event(user: str, **event: Any) -> None:
    ev = _events(user)
    event["timestamp"] = time.time()
    ev.append(event)
    if len(ev) > _MAX_EVENTS:
        ev.pop(0)


def recent_events(user: str, limit: int = 50) -> list[dict]:
    return list(reversed(_events(user)[-limit:]))


# ── Key management ────────────────────────────────────────────────────────────

def create_key(user: str, name: str, budget_usd: float = 5.0, rate_limit_per_min: int = 30) -> dict:
    key = "agk-" + secrets.token_hex(16)
    vk = VirtualKey(key=key, name=name or "unnamed",
                    budget_usd=max(0.0, budget_usd),
                    rate_limit_per_min=max(1, rate_limit_per_min))
    _keys(user)[key] = vk
    return vk.public(reveal=True)


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
        return True
    return False


def upstream_status(user: str) -> dict:
    real = _user_upstream.get(user, "") or os.environ.get("OPENAI_API_KEY", "")
    has_real = bool(real)
    return {
        "upstream": "openai" if has_real else "local",
        "real_key_configured": has_real,
        "key_hint": (real[:3] + "..." + real[-4:]) if len(real) > 8 else ("set" if has_real else ""),
        "note": ("Forwarding to OpenAI. Your real key stays server-side and is never exposed."
                 if has_real else
                 "Demo mode: forwarding to the AgentShield Security LLM (no real key set). "
                 "Connect a provider key to protect a real upstream."),
        "models": list(PRICING.keys()),
    }


def set_upstream_key(user: str, key: str) -> dict:
    key = (key or "").strip()
    if not key:
        return {"ok": False, "error": "empty key"}
    _user_upstream[user] = key
    return {"ok": True, **upstream_status(user)}


def clear_upstream_key(user: str) -> dict:
    _user_upstream.pop(user, None)
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


# ── Cost helpers ──────────────────────────────────────────────────────────────

def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = PRICING.get(model, PRICING[DEFAULT_MODEL])
    return (in_tok / 1000.0) * pin + (out_tok / 1000.0) * pout


def _rate_ok(vk: VirtualKey) -> bool:
    now = time.time()
    vk.hits = [t for t in vk.hits if now - t < 60]
    if len(vk.hits) >= vk.rate_limit_per_min:
        return False
    vk.hits.append(now)
    return True


# ── The protected call ────────────────────────────────────────────────────────

def gateway_chat(
    user: str,
    api_key: str | None,
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
) -> dict:
    """Screen, budget-check, and forward a chat request. Isolated to `user`."""
    ku = _keys(user)
    vk = ku.get(api_key or "")
    if not vk:
        return {"error": "Invalid API key.", "status": 401, "blocked": True}
    if not vk.enabled:
        return {"error": "This key has been revoked.", "status": 403, "blocked": True}

    if not _rate_ok(vk):
        return {"error": f"Rate limit exceeded ({vk.rate_limit_per_min}/min).",
                "status": 429, "blocked": True, "key": vk.public()}

    vk.request_count += 1
    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    msg_preview = (user_text[:120] + "…") if len(user_text) > 120 else user_text

    from ..agentshield import analyze_threat, record_action
    report = analyze_threat(user_text, source="user")

    if report["decision"] in ("block", "review"):
        vk.blocked_count += 1
        in_tok = _est_tokens(user_text)
        saved = _cost(model, in_tok, max_tokens)
        vk.cost_saved_usd += saved
        record_action(vk.name, action="blocked", blocked=True)
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
        )
        return {
            "blocked": True, "status": 200,
            "threat": report["attack_type"],
            "explanation": report["reason"],
            "risk_score": report["risk_score"],
            "severity": report["severity"],
            "decision": report["decision"],
            "attack_chain": report["attack_chain"],
            "confidence": report["confidence"],
            "recommended_action": report["recommended_action"],
            "analyst_report": report,
            "layer": "agentshield-security-llm",
            "cost_saved_usd": round(saved, 6),
            "message": f"Blocked by AgentShield Security LLM. Risk: {report['risk_score']}/100 ({report['severity']}). Cost: $0.",
            "key": vk.public(),
        }

    if vk.spent_usd >= vk.budget_usd:
        return {"error": f"Budget exhausted (${vk.budget_usd:.2f}). Request blocked to prevent overspend.",
                "status": 402, "blocked": True, "key": vk.public()}

    real_key = _user_upstream.get(user, "") or os.environ.get("OPENAI_API_KEY", "")
    if real_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=real_key)
            resp = client.chat.completions.create(model=model, messages=messages,
                                                  max_tokens=max_tokens, temperature=0.7)
            output = resp.choices[0].message.content or ""
            in_tok = resp.usage.prompt_tokens if resp.usage else _est_tokens(user_text)
            out_tok = resp.usage.completion_tokens if resp.usage else _est_tokens(output)
            upstream = "openai"
        except Exception as e:
            return {"error": f"upstream error: {e}", "status": 502, "blocked": False, "key": vk.public()}
    else:
        from ..llm import engine
        r = engine.chat(messages[-1].get("content", "") if messages else "")
        output = r.get("response", "")
        in_tok = _est_tokens(user_text)
        out_tok = _est_tokens(output)
        upstream = "local"

    output = redact_pii(output)
    cost = _cost(model, in_tok, out_tok)
    vk.spent_usd += cost

    record_action(vk.name, action="allowed", blocked=False, cost_usd=cost)
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
    )

    return {
        "blocked": False, "status": 200,
        "response": output,
        "upstream": upstream,
        "model": model,
        "passed_layers": ["agentshield-security-llm"],
        "risk_score": report["risk_score"],
        "severity": report["severity"],
        "judge_latency_ms": report.get("latency_ms", 0),
        "analyst_report": report,
        "cost_usd": round(cost, 6),
        "tokens": {"input": in_tok, "output": out_tok},
        "key": vk.public(),
    }
