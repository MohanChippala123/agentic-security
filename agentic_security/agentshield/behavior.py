"""Per-agent behavioral monitoring: build a profile, flag anomalies."""

from __future__ import annotations

import time
from collections import defaultdict, Counter
from typing import Any


_profiles: dict[str, dict[str, Any]] = defaultdict(lambda: {
    "first_seen": time.time(),
    "last_seen": time.time(),
    "tools": Counter(),
    "actions": Counter(),
    "spend_usd": 0.0,
    "requests": 0,
    "blocked": 0,
    "anomalies": [],
})


def record_action(agent_id: str, *, action: str, tool: str | None = None, blocked: bool = False, cost_usd: float = 0.0) -> None:
    p = _profiles[agent_id]
    p["last_seen"] = time.time()
    p["actions"][action] += 1
    if tool:
        p["tools"][tool] += 1
    p["requests"] += 1
    if blocked:
        p["blocked"] += 1
    p["spend_usd"] += cost_usd


def get_agent_profile(agent_id: str) -> dict[str, Any]:
    if agent_id not in _profiles:
        return {"agent_id": agent_id, "exists": False}
    p = _profiles[agent_id]
    return {
        "agent_id": agent_id,
        "exists": True,
        "first_seen": p["first_seen"],
        "last_seen": p["last_seen"],
        "requests": p["requests"],
        "blocked": p["blocked"],
        "block_rate": round(p["blocked"] / max(1, p["requests"]), 3),
        "spend_usd": round(p["spend_usd"], 6),
        "top_tools": p["tools"].most_common(5),
        "top_actions": p["actions"].most_common(5),
        "anomalies": p["anomalies"][-10:],
    }


def anomaly_report(agent_id: str) -> dict[str, Any]:
    """Flag suspicious patterns in this agent's behavior."""
    if agent_id not in _profiles:
        return {"agent_id": agent_id, "anomalies": [], "risk": "unknown"}
    p = _profiles[agent_id]
    findings = []

    # High block rate = likely under attack / compromised
    block_rate = p["blocked"] / max(1, p["requests"])
    if p["requests"] >= 10 and block_rate >= 0.3:
        findings.append({
            "type": "high_block_rate",
            "severity": "high" if block_rate >= 0.5 else "medium",
            "detail": f"Block rate {block_rate:.0%} over {p['requests']} requests - this agent may be under active attack or compromised.",
        })

    # Sudden destructive-tool spike
    destructive_calls = sum(
        c for t, c in p["tools"].items()
        if any(w in t.lower() for w in ("delete", "drop", "transfer", "exec", "shell", "wipe"))
    )
    if destructive_calls >= 3:
        findings.append({
            "type": "destructive_tool_burst",
            "severity": "critical" if destructive_calls >= 5 else "high",
            "detail": f"{destructive_calls} destructive tool calls observed. Investigate for compromise / runaway autonomy.",
        })

    # Spend anomaly (>$10 in one session for this demo)
    if p["spend_usd"] > 10.0:
        findings.append({
            "type": "spend_anomaly",
            "severity": "high",
            "detail": f"Spend ${p['spend_usd']:.2f} exceeded normal envelope. Possible cost-runaway / abuse.",
        })

    risk = "critical" if any(f["severity"] == "critical" for f in findings) \
        else "high" if any(f["severity"] == "high" for f in findings) \
        else "medium" if findings else "low"
    p["anomalies"].extend(findings)
    return {"agent_id": agent_id, "anomalies": findings, "risk": risk}


def list_profiles() -> list[dict]:
    return [get_agent_profile(aid) for aid in _profiles]
