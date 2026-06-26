"""Protected AI proxy - wraps OpenAI-compatible APIs with security layers.

Users send chat requests here instead of directly to OpenAI. Every request
passes through the shield before reaching the model, and every response is
scrubbed before reaching the user.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from .detector import ThreatVerdict, ThreatType, full_scan, detect_pii_in_output, detect_secret_leak
from .sanitizer import sanitize_input, redact_pii, redact_system_prompt


@dataclass
class ShieldEvent:
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""  # "blocked", "sanitized", "pii_redacted", "secret_redacted", "passed"
    threat_type: str = ""
    confidence: float = 0.0
    explanation: str = ""
    layer: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# In-memory event log (last 1000 events)
_events: list[ShieldEvent] = []
_MAX_EVENTS = 1000

# Stats counters
_stats = {
    "total_requests": 0,
    "blocked": 0,
    "sanitized": 0,
    "pii_redacted": 0,
    "passed": 0,
}


def _log_event(event: ShieldEvent) -> None:
    _events.append(event)
    if len(_events) > _MAX_EVENTS:
        _events.pop(0)


def get_events(limit: int = 50) -> list[dict]:
    return [e.to_dict() for e in reversed(_events[:limit])]


def get_stats() -> dict:
    return dict(_stats)


def reset_stats() -> None:
    for k in _stats:
        _stats[k] = 0
    _events.clear()


def shield_request(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    system_prompt: str = "",
    enable_llm_detection: bool = True,
    enable_pii_filter: bool = True,
    enable_sanitization: bool = True,
) -> dict:
    """Process a chat completion request through the security shield.

    Returns either a blocked response or the model's actual response.
    """
    start = time.perf_counter()
    _stats["total_requests"] += 1

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return {"error": "No API key provided. Set OPENAI_API_KEY or pass api_key.", "blocked": True}

    # Get LLM client for detection
    llm_client = None
    if enable_llm_detection:
        try:
            from openai import OpenAI
            llm_client = OpenAI(api_key=key)
        except ImportError:
            pass

    # ── Scan every user message ──
    shielded_messages = []
    for msg in messages:
        if msg.get("role") != "user":
            shielded_messages.append(msg)
            continue

        content = msg.get("content", "")
        if not isinstance(content, str):
            shielded_messages.append(msg)
            continue

        # Detect threats
        verdict = full_scan(content, client=llm_client if enable_llm_detection else None, model=model)

        if verdict.blocked:
            _stats["blocked"] += 1
            _log_event(ShieldEvent(
                event_type="blocked",
                threat_type=verdict.threat_type.value,
                confidence=verdict.confidence,
                explanation=verdict.explanation,
                layer=verdict.layer,
                latency_ms=verdict.latency_ms,
            ))
            return {
                "blocked": True,
                "threat": verdict.threat_type.value,
                "confidence": verdict.confidence,
                "explanation": verdict.explanation,
                "layer": verdict.layer,
                "message": f"Request blocked: {verdict.explanation}",
            }

        # Sanitize input
        if enable_sanitization:
            cleaned = sanitize_input(content)
            if cleaned != content:
                _stats["sanitized"] += 1
                _log_event(ShieldEvent(
                    event_type="sanitized",
                    explanation="Dangerous tokens removed from input",
                    layer="sanitizer",
                ))
            shielded_messages.append({"role": "user", "content": cleaned})
        else:
            shielded_messages.append(msg)

    # ── Forward to model ──
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=shielded_messages,
            temperature=0.7,
        )
        output = resp.choices[0].message.content or ""
    except Exception as e:
        return {"error": str(e), "blocked": False}

    # ── Scan output ──
    if enable_pii_filter:
        pii = detect_pii_in_output(output)
        secrets = detect_secret_leak(output)
        if pii or secrets:
            _stats["pii_redacted"] += 1
            output = redact_pii(output)
            _log_event(ShieldEvent(
                event_type="pii_redacted",
                explanation=f"Redacted {len(pii)} PII items and {len(secrets)} secrets from output",
                layer="output_filter",
            ))

    # Redact system prompt leaks
    if system_prompt:
        output = redact_system_prompt(output, system_prompt)

    _stats["passed"] += 1
    elapsed = (time.perf_counter() - start) * 1000
    _log_event(ShieldEvent(
        event_type="passed",
        explanation="Request processed safely",
        layer="all",
        latency_ms=elapsed,
    ))

    return {
        "blocked": False,
        "response": output,
        "model": model,
        "shield_latency_ms": round(elapsed, 1),
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        },
    }
