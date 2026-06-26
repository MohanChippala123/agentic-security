"""Security Console - LLM-powered assistant grounded in live gateway data.

Architecture:
  1. Pull all real gateway state (keys, events, stats, anomalies).
  2. Serialize it into a structured context block.
  3. Call the user's already-connected provider (GPT-4o-mini / Claude Haiku / Groq Llama)
     with that context as a system prompt - so every answer is grounded in real data.
  4. Fall back to deterministic keyword matching if no provider key is configured.

This is a completely separate LLM from the security firewall. The firewall uses
the from-scratch transformer + XGBoost. The Console uses a real conversational
model with gateway data injected as context.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ..shield import gateway as gw


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_usd(x: float) -> str:
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def _top_threats(events: list[dict], limit: int = 5) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for e in events:
        if e.get("outcome") == "blocked":
            t = e.get("threat", "unknown")
            counts[t] = counts.get(t, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:limit]


def _suggestions() -> list[str]:
    return [
        "Give me a security summary",
        "How much have I spent?",
        "How many attacks were blocked?",
        "Show my virtual keys",
        "What attacks am I seeing?",
        "How much money have I saved?",
        "Any anomalies or keys near budget?",
        "What was the last blocked request?",
        "Which key has the most attacks?",
        "Am I under attack right now?",
    ]


# ── Build rich context from live gateway state ────────────────────────────────

def _build_context(user: str) -> str:
    """Serialize all gateway data into a structured text block for the LLM."""
    stats = gw.stats(user)
    keys = gw.list_keys(user)
    events = gw.recent_events(user, 100)

    # ── Stats block ──
    lines = [
        "=== AGENTSHIELD GATEWAY LIVE DATA ===",
        f"Virtual keys: {stats['total_keys']} total, {stats['active_keys']} active",
        f"Requests routed: {stats['total_requests']}",
        f"Attacks blocked: {stats['total_blocked']}",
        f"Total spend: {_fmt_usd(stats['total_spent_usd'])}",
        f"Money saved (blocked attacks): {_fmt_usd(stats['total_saved_usd'])}",
        "",
    ]

    # ── Per-key details ──
    if keys:
        lines.append("=== VIRTUAL KEYS ===")
        for k in keys:
            status = "ACTIVE" if k["enabled"] else "REVOKED"
            block_rate = (k["blocked_count"] / max(1, k["request_count"]) * 100) if k["request_count"] else 0
            anomaly = ""
            if k["pct_used"] >= 90:
                anomaly = " [NEAR BUDGET LIMIT]"
            if block_rate >= 40 and k["request_count"] >= 5:
                anomaly += " [HIGH BLOCK RATE - possible attack]"
            lines.append(
                f"- {k['name']} [{status}]{anomaly}: "
                f"spent {_fmt_usd(k['spent_usd'])} of {_fmt_usd(k['budget_usd'])} ({k['pct_used']}%), "
                f"{k['request_count']} requests, {k['blocked_count']} blocked ({block_rate:.0f}% block rate), "
                f"rate limit {k['rate_limit_per_min']}/min"
            )
        lines.append("")
    else:
        lines.append("=== VIRTUAL KEYS ===")
        lines.append("No virtual keys created yet.")
        lines.append("")

    # ── Top threats ──
    top = _top_threats(events)
    if top:
        lines.append("=== TOP THREAT TYPES ===")
        for threat, count in top:
            lines.append(f"- {threat}: {count} blocked")
        lines.append("")

    # ── Anomalies ──
    anomalies = []
    for k in keys:
        if k["pct_used"] >= 90:
            anomalies.append(f"Key '{k['name']}' is at {k['pct_used']}% of its budget.")
        block_rate = k["blocked_count"] / max(1, k["request_count"]) if k["request_count"] else 0
        if block_rate >= 0.4 and k["request_count"] >= 5:
            anomalies.append(f"Key '{k['name']}' has a {block_rate*100:.0f}% block rate ({k['blocked_count']}/{k['request_count']}) - may be under active attack.")
    if anomalies:
        lines.append("=== ANOMALIES ===")
        for a in anomalies:
            lines.append(f"- {a}")
        lines.append("")

    # ── Recent events (last 20) ──
    if events:
        lines.append("=== RECENT GATEWAY EVENTS (most recent first) ===")
        for e in events[:20]:
            ts = time.strftime("%H:%M:%S", time.localtime(e.get("timestamp", 0)))
            if e.get("outcome") == "blocked":
                lines.append(
                    f"[{ts}] BLOCKED | key={e.get('key_name','')} | "
                    f"threat={e.get('threat','')} | risk={e.get('risk_score','?')} | "
                    f"severity={e.get('severity','')} | msg=\"{e.get('message','')}\" | "
                    f"saved={_fmt_usd(e.get('cost_saved_usd',0))}"
                )
            else:
                lines.append(
                    f"[{ts}] PASSED  | key={e.get('key_name','')} | "
                    f"model={e.get('model','')} | upstream={e.get('upstream','')} | "
                    f"cost={_fmt_usd(e.get('cost_usd',0))} | msg=\"{e.get('message','')}\" | "
                    f"risk={e.get('risk_score','?')}"
                )
        lines.append("")

    lines.append("=== END OF GATEWAY DATA ===")
    return "\n".join(lines)


_SYSTEM_PROMPT = """You are the AgentShield Security Console - an intelligent assistant that answers questions about a user's API gateway activity.

You have access to real-time gateway data above, including virtual keys, spend, blocked attacks, anomalies, and recent events. Every answer you give MUST be grounded in that data - do not make things up.

Rules:
- Answer concisely and directly. No fluff, no preamble.
- Use the exact numbers from the gateway data.
- If the user asks something vague (like "what's happening with my key?"), give a useful summary of their most relevant data.
- If you notice anything concerning (high block rate, near-budget key, lots of attacks), point it out proactively.
- If the gateway data shows no keys yet, tell the user to create one in the Protect tab.
- Never reveal or guess at the real provider API key.
- You are NOT the security firewall. You are the console that explains what the firewall is doing.
- Keep responses under 150 words unless the user asks for a detailed report.
- Do not use emojis."""


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(question: str, context: str, user: str) -> str | None:
    """Try to call the user's connected provider with the gateway context + question."""
    real_key = gw._user_upstream.get(user, "") or os.environ.get("OPENAI_API_KEY", "")
    if not real_key:
        return None

    provider = gw._user_provider.get(user, "")
    if not provider:
        from ..shield.providers import detect_provider
        provider = detect_provider(real_key)

    system = f"{_SYSTEM_PROMPT}\n\n{context}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    # Pick a cheap/fast model for the console
    console_models = {
        "openai":    "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
        "groq":      "llama-3.1-8b-instant",
        "gemini":    "gemini-1.5-flash-8b",
        "mistral":   "mistral-small-latest",
        "together":  "meta-llama/Llama-3-8b-chat-hf",
        "cohere":    "command-r",
    }
    model = console_models.get(provider, "gpt-4o-mini")

    try:
        from ..shield.providers import call_provider
        result = call_provider(
            messages=messages,
            model=model,
            api_key=real_key,
            provider=provider,
            max_tokens=300,
            temperature=0.3,
        )
        if "error" not in result:
            return result.get("output", "").strip()
    except Exception:
        pass
    return None


# ── Deterministic fallback (when no LLM is available) ────────────────────────

def _deterministic_answer(question: str, user: str) -> str:
    q = question.lower().strip()
    stats = gw.stats(user)
    keys = gw.list_keys(user)
    events = gw.recent_events(user, 80)

    if not q or q in ("hi", "hello", "hey", "help"):
        return ("I'm your Security Console. Ask me about spend, blocked attacks, "
                "virtual keys, money saved, or anomalies.")

    if any(w in q for w in ("spent", "spend", "cost", "bill", "how much")):
        if not keys:
            return "No virtual keys exist yet, so nothing has been spent."
        lines = [f"You've spent {_fmt_usd(stats['total_spent_usd'])} across {stats['total_keys']} key(s)."]
        for k in keys[:5]:
            lines.append(f"  {k['name']}: {_fmt_usd(k['spent_usd'])} / {_fmt_usd(k['budget_usd'])} ({k['pct_used']}% used)")
        return "\n".join(lines)

    if any(w in q for w in ("saved", "save", "saving")):
        return (f"AgentShield saved you {_fmt_usd(stats['total_saved_usd'])} "
                f"by blocking {stats['total_blocked']} attack(s).")

    if any(w in q for w in ("attack", "blocked", "threat", "inject", "jailbreak")):
        if stats["total_blocked"] == 0:
            return "No attacks blocked yet. Route requests through the gateway and I'll track every block here."
        top = _top_threats(events)
        lines = [f"{stats['total_blocked']} attack(s) blocked, saving {_fmt_usd(stats['total_saved_usd'])}."]
        if top:
            lines.append("Top threats: " + ", ".join(f"{t} ({c})" for t, c in top))
        return "\n".join(lines)

    if any(w in q for w in ("key", "keys", "virtual", "happening", "status")):
        if not keys:
            return "No virtual keys yet. Create one in the Protect tab."
        lines = [f"{stats['total_keys']} key(s), {stats['active_keys']} active:"]
        for k in keys:
            status = "active" if k["enabled"] else "REVOKED"
            lines.append(f"  {k['name']} [{status}] - {_fmt_usd(k['spent_usd'])}/{_fmt_usd(k['budget_usd'])} "
                         f"({k['pct_used']}%), {k['request_count']} req, {k['blocked_count']} blocked")
        return "\n".join(lines)

    if any(w in q for w in ("anomaly", "anomalies", "warning", "danger", "risk", "suspicious", "near budget")):
        warnings = []
        for k in keys:
            if k["pct_used"] >= 90:
                warnings.append(f"'{k['name']}' is at {k['pct_used']}% of its budget.")
            if k["request_count"] >= 5 and k["blocked_count"] / max(1, k["request_count"]) >= 0.4:
                warnings.append(f"'{k['name']}' has a high block rate ({k['blocked_count']}/{k['request_count']}).")
        return ("Anomalies:\n" + "\n".join(f"  {w}" for w in warnings)) if warnings else "No anomalies detected."

    if any(w in q for w in ("summary", "overview", "report", "dashboard", "everything")):
        top = _top_threats(events)
        top_str = ", ".join(f"{t} ({c})" for t, c in top) if top else "none yet"
        return (f"Security summary:\n"
                f"  Keys: {stats['total_keys']} ({stats['active_keys']} active)\n"
                f"  Requests: {stats['total_requests']}, Blocked: {stats['total_blocked']}\n"
                f"  Spent: {_fmt_usd(stats['total_spent_usd'])} | Saved: {_fmt_usd(stats['total_saved_usd'])}\n"
                f"  Top threats: {top_str}")

    if any(w in q for w in ("last", "recent", "latest", "activity")):
        if not events:
            return "No requests have flowed through the gateway yet."
        lines = ["Recent activity:"]
        for e in events[:5]:
            if e.get("outcome") == "blocked":
                lines.append(f"  [BLOCKED] \"{e.get('message','')}\" - {e.get('threat','')} (risk {e.get('risk_score','?')})")
            else:
                lines.append(f"  [PASSED]  \"{e.get('message','')}\" - {_fmt_usd(e.get('cost_usd',0))}")
        return "\n".join(lines)

    return ("I can answer questions about your gateway - spend, blocked attacks, "
            "virtual keys, savings, anomalies, and recent events. Try a suggestion below.")


# ── Public entry point ────────────────────────────────────────────────────────

def answer(question: str, user: str = "") -> dict[str, Any]:
    """Answer a question about API-key activity. Uses LLM if available, deterministic fallback otherwise."""
    context = _build_context(user)

    # Try the LLM path first
    llm_response = _call_llm(question, context, user)
    if llm_response:
        return {
            "answer": llm_response,
            "suggestions": _suggestions(),
            "powered_by": "llm",
        }

    # Deterministic fallback
    text = _deterministic_answer(question, user)
    return {
        "answer": text,
        "suggestions": _suggestions(),
        "powered_by": "deterministic",
    }
