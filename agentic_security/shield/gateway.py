"""API Key Guard - an LLM gateway that protects your real API key.

The problem this solves:
  - You have ONE real provider key (OpenAI/Anthropic). If it leaks, or an app
    gets prompt-injected, or traffic spikes, you get a huge bill.

How the gateway protects you:
  1. Your REAL key lives only on the server (env var). It is NEVER sent to
     clients and NEVER returned by any endpoint.
  2. You issue "virtual keys" (agk-...) to your apps. Each has its own budget
     and rate limit and can be revoked instantly.
  3. Every request is screened by the prompt-injection firewall BEFORE it is
     forwarded - blocked attacks cost you $0.
  4. Per-key USD budgets and rate limits stop cost runaway.
  5. Output is scrubbed for PII/secrets.

If no real key is configured, the gateway forwards to the local agentic-1
model so the whole thing is demoable with zero real keys / zero spend.
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


_keys: dict[str, VirtualKey] = {}

# Recent gateway events so the UI can show Mohan's LLM in action live.
_events: list[dict] = []
_MAX_EVENTS = 80


def _log_event(**event: Any) -> None:
    event["timestamp"] = time.time()
    _events.append(event)
    if len(_events) > _MAX_EVENTS:
        _events.pop(0)


def recent_events(limit: int = 50) -> list[dict]:
    return list(reversed(_events[-limit:]))


# ── Key management ──
def create_key(name: str, budget_usd: float = 5.0, rate_limit_per_min: int = 30) -> dict:
    key = "agk-" + secrets.token_hex(16)
    vk = VirtualKey(key=key, name=name or "unnamed",
                    budget_usd=max(0.0, budget_usd),
                    rate_limit_per_min=max(1, rate_limit_per_min))
    _keys[key] = vk
    # reveal the full key exactly once, on creation
    return vk.public(reveal=True)


def list_keys() -> list[dict]:
    return [vk.public() for vk in sorted(_keys.values(), key=lambda k: -k.created_at)]


def revoke_key(key: str) -> bool:
    vk = _keys.get(key)
    if not vk:
        # allow revoking by masked prefix from the UI
        for k, v in _keys.items():
            if k.startswith(key.split("...")[0]):
                vk = v
                break
    if vk:
        vk.enabled = False
        return True
    return False


def delete_key(key: str) -> bool:
    return _keys.pop(key, None) is not None


def upstream_status() -> dict:
    real = os.environ.get("OPENAI_API_KEY", "")
    has_real = bool(real)
    return {
        "upstream": "openai" if has_real else "local",
        "real_key_configured": has_real,
        # masked hint only - the real key is never returned in full
        "key_hint": (real[:3] + "..." + real[-4:]) if len(real) > 8 else ("set" if has_real else ""),
        "note": ("Forwarding to OpenAI. Your real key stays server-side and is never exposed."
                 if has_real else
                 "Demo mode: forwarding to the AgentShield Security LLM (no real key set). "
                 "Connect a provider key to protect a real upstream."),
        "models": list(PRICING.keys()),
    }


def set_upstream_key(key: str) -> dict:
    """Store the real provider key server-side (in-process). Never returned in full."""
    key = (key or "").strip()
    if not key:
        return {"ok": False, "error": "empty key"}
    os.environ["OPENAI_API_KEY"] = key
    return {"ok": True, **upstream_status()}


def clear_upstream_key() -> dict:
    os.environ.pop("OPENAI_API_KEY", None)
    return {"ok": True, **upstream_status()}


def stats() -> dict:
    keys = list(_keys.values())
    return {
        "total_keys": len(keys),
        "active_keys": sum(1 for k in keys if k.enabled),
        "total_requests": sum(k.request_count for k in keys),
        "total_blocked": sum(k.blocked_count for k in keys),
        "total_spent_usd": round(sum(k.spent_usd for k in keys), 6),
        "total_saved_usd": round(sum(k.cost_saved_usd for k in keys), 6),
    }


# ── Cost helpers ──
def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # ~4 chars per token


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


# ── The protected call ──
def gateway_chat(
    api_key: str | None,
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
) -> dict:
    """Screen, budget-check, and forward a chat request using the server's real key."""
    vk = _keys.get(api_key or "")
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

    # ── AgentShield Security LLM: full threat analysis ──
    # The Security LLM produces a rich threat report (risk score, severity,
    # attack chain, reasoning) - not just SAFE/BLOCK. The gateway uses that
    # verdict to decide whether to forward the request.
    from ..agentshield import analyze_threat, record_action
    report = analyze_threat(user_text, source="user")

    if report["decision"] in ("block", "review"):
        vk.blocked_count += 1
        in_tok = _est_tokens(user_text)
        saved = _cost(model, in_tok, max_tokens)
        vk.cost_saved_usd += saved
        record_action(vk.name, action="blocked", blocked=True)
        _log_event(
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
            "blocked": True,
            "status": 200,
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

    # ── Budget check ──
    if vk.spent_usd >= vk.budget_usd:
        return {"error": f"Budget exhausted (${vk.budget_usd:.2f}). Request blocked to prevent overspend.",
                "status": 402, "blocked": True, "key": vk.public()}

    # ── Forward using the REAL key (server-side only) or local model ──
    real_key = os.environ.get("OPENAI_API_KEY")
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
        # demo upstream: our local agentic-1
        from ..llm import engine
        r = engine.chat(messages[-1].get("content", "") if messages else "")
        output = r.get("response", "")
        in_tok = _est_tokens(user_text)
        out_tok = _est_tokens(output)
        upstream = "local"

    # ── Scrub output, account for cost ──
    output = redact_pii(output)
    cost = _cost(model, in_tok, out_tok)
    vk.spent_usd += cost

    record_action(vk.name, action="allowed", blocked=False, cost_usd=cost)
    _log_event(
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
        "blocked": False,
        "status": 200,
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
