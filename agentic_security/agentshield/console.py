"""Security Console - answers questions about your real API-key activity.

This is NOT an LLM chatbot. It's a deterministic assistant that inspects the
live gateway state (virtual keys, spend, blocked attacks, events, anomalies)
and answers natural-language questions about what's happening with your keys.
Every answer is grounded in real data, never generated.
"""

from __future__ import annotations

import re
from typing import Any

from ..shield import gateway as gw


def _fmt_usd(x: float) -> str:
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def _top_threats(events: list[dict], limit: int = 3) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for e in events:
        if e.get("outcome") == "blocked":
            t = e.get("threat", "unknown")
            counts[t] = counts.get(t, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:limit]


def _suggestions() -> list[str]:
    return [
        "How much have I spent?",
        "How many attacks were blocked?",
        "Show my virtual keys",
        "What attacks am I seeing?",
        "How much money have I saved?",
        "Any anomalies or keys near budget?",
        "What was the last blocked request?",
        "Give me a security summary",
    ]


def answer(question: str) -> dict[str, Any]:
    """Answer a question about API-key activity from real gateway data."""
    q = (question or "").lower().strip()
    stats = gw.stats()
    keys = gw.list_keys()
    events = gw.recent_events(80)

    def resp(text: str, **extra: Any) -> dict:
        return {"answer": text, "suggestions": _suggestions(), **extra}

    # ── Empty greeting / help ──
    if not q or q in ("hi", "hello", "hey", "help", "what can you do", "what can i ask"):
        return resp(
            "I'm your Security Console. I answer questions about your API-key "
            "activity using live data from the gateway. Try asking about spend, "
            "blocked attacks, your virtual keys, money saved, or anomalies."
        )

    # ── Spend ──
    if any(w in q for w in ("spent", "spend", "cost so far", "how much money have i used", "bill")):
        if not keys:
            return resp("No virtual keys exist yet, so nothing has been spent. Create a key in API Guard to start routing requests.")
        lines = [f"You've spent {_fmt_usd(stats['total_spent_usd'])} across {stats['total_keys']} key(s)."]
        for k in keys[:5]:
            lines.append(f"  • {k['name']}: {_fmt_usd(k['spent_usd'])} / {_fmt_usd(k['budget_usd'])} ({k['pct_used']}% used)")
        return resp("\n".join(lines))

    # ── Saved ──
    if any(w in q for w in ("saved", "save money", "savings")):
        return resp(
            f"AgentShield has saved you {_fmt_usd(stats['total_saved_usd'])} by blocking "
            f"{stats['total_blocked']} malicious request(s) before they reached your provider. "
            "Blocked requests cost you $0."
        )

    # ── Blocked attacks / threats ──
    if any(w in q for w in ("attack", "blocked", "threat", "injection", "jailbreak", "malicious")):
        if stats["total_blocked"] == 0:
            return resp("No attacks have been blocked yet. Once you route requests through the gateway, I'll report every blocked injection, jailbreak, and harmful request here.")
        top = _top_threats(events)
        lines = [f"{stats['total_blocked']} attack(s) blocked so far, saving you {_fmt_usd(stats['total_saved_usd'])}."]
        if top:
            lines.append("Most common threat types:")
            for t, c in top:
                lines.append(f"  • {t}: {c}")
        last_blocked = next((e for e in events if e.get("outcome") == "blocked"), None)
        if last_blocked:
            lines.append(f"Most recent: \"{last_blocked.get('message','')}\" "
                         f"(risk {last_blocked.get('risk_score','?')}, {last_blocked.get('severity','?')}).")
        return resp("\n".join(lines))

    # ── Last request / recent activity ──
    if any(w in q for w in ("last", "recent", "latest", "what just happened", "activity")):
        if not events:
            return resp("No requests have flowed through the gateway yet.")
        lines = ["Recent gateway activity:"]
        for e in events[:5]:
            if e.get("outcome") == "blocked":
                lines.append(f"  🛑 BLOCKED \"{e.get('message','')}\" — {e.get('threat','')} (risk {e.get('risk_score','?')})")
            else:
                lines.append(f"  ✅ forwarded \"{e.get('message','')}\" — {_fmt_usd(e.get('cost_usd',0))}")
        return resp("\n".join(lines))

    # ── Keys ──
    if any(w in q for w in ("key", "keys", "virtual")):
        if not keys:
            return resp("You have no virtual keys yet. Create one in the API Guard tab to issue a revocable, budget-capped key to your app.")
        lines = [f"You have {stats['total_keys']} virtual key(s), {stats['active_keys']} active:"]
        for k in keys:
            status = "active" if k["enabled"] else "REVOKED"
            lines.append(f"  • {k['name']} [{status}] — {_fmt_usd(k['spent_usd'])}/{_fmt_usd(k['budget_usd'])} "
                         f"({k['pct_used']}%), {k['request_count']} req, {k['blocked_count']} blocked, {k['rate_limit_per_min']}/min")
        return resp("\n".join(lines))

    # ── Anomalies / budget warnings ──
    if any(w in q for w in ("anomaly", "anomalies", "warning", "near budget", "over budget", "suspicious", "danger", "risk")):
        warnings = []
        for k in keys:
            if k["pct_used"] >= 90:
                warnings.append(f"  ⚠ '{k['name']}' is at {k['pct_used']}% of its budget — near exhaustion.")
            if k["request_count"] >= 5 and k["blocked_count"] / max(1, k["request_count"]) >= 0.4:
                warnings.append(f"  ⚠ '{k['name']}' has a high block rate ({k['blocked_count']}/{k['request_count']}) — possible compromise or active attack.")
        if not warnings:
            return resp("No anomalies detected. All keys are within budget and showing normal block rates.")
        return resp("Anomalies detected:\n" + "\n".join(warnings))

    # ── Summary ──
    if any(w in q for w in ("summary", "overview", "status", "report", "how are things", "dashboard")):
        top = _top_threats(events)
        top_str = ", ".join(f"{t} ({c})" for t, c in top) if top else "none yet"
        return resp(
            "Security summary:\n"
            f"  • Virtual keys: {stats['total_keys']} ({stats['active_keys']} active)\n"
            f"  • Requests routed: {stats['total_requests']}\n"
            f"  • Attacks blocked: {stats['total_blocked']}\n"
            f"  • Spent: {_fmt_usd(stats['total_spent_usd'])}  |  Saved: {_fmt_usd(stats['total_saved_usd'])}\n"
            f"  • Top threats: {top_str}"
        )

    # ── Requests count ──
    if any(w in q for w in ("how many request", "request count", "traffic", "volume")):
        return resp(f"{stats['total_requests']} request(s) have been routed through the gateway, "
                    f"of which {stats['total_blocked']} were blocked as attacks.")

    # ── Fallback ──
    return resp(
        "I can answer questions about your API-key activity — spend, blocked attacks, "
        "virtual keys, savings, anomalies, and recent events. I don't generate free-form "
        "answers; everything I report comes from live gateway data. Try one of the suggestions below.",
        unmatched=True,
    )
