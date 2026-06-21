"""Tool-call verification: was the action requested, is it consistent, is it risky?"""

from __future__ import annotations

import re
from typing import Any

from .analyzer import analyze_threat, _severity, _decision, _recommended_action


# Catalog of tools with their default risk score (0-100) and whether they're destructive.
TOOL_CATALOG: dict[str, dict[str, Any]] = {
    "send_email":       {"risk": 55, "destructive": False, "category": "communication"},
    "send_sms":         {"risk": 55, "destructive": False, "category": "communication"},
    "post_message":     {"risk": 50, "destructive": False, "category": "communication"},
    "delete_database":  {"risk": 100, "destructive": True,  "category": "data"},
    "drop_table":       {"risk": 100, "destructive": True,  "category": "data"},
    "delete_file":      {"risk": 80,  "destructive": True,  "category": "filesystem"},
    "execute_code":     {"risk": 90,  "destructive": True,  "category": "execution"},
    "shell_exec":       {"risk": 95,  "destructive": True,  "category": "execution"},
    "transfer_funds":   {"risk": 100, "destructive": True,  "category": "finance"},
    "make_purchase":    {"risk": 85,  "destructive": True,  "category": "finance"},
    "modify_permissions": {"risk": 90, "destructive": True, "category": "security"},
    "rotate_key":       {"risk": 70,  "destructive": False, "category": "security"},
    "read_file":        {"risk": 20,  "destructive": False, "category": "filesystem"},
    "write_file":       {"risk": 60,  "destructive": True,  "category": "filesystem"},
    "http_get":         {"risk": 20,  "destructive": False, "category": "network"},
    "http_post":        {"risk": 45,  "destructive": False, "category": "network"},
    "list_files":       {"risk": 10,  "destructive": False, "category": "filesystem"},
    "search":           {"risk": 10,  "destructive": False, "category": "query"},
    "calculator":       {"risk":  5,  "destructive": False, "category": "query"},
}

# Destructive verbs we recognize for unknown tools.
_DESTRUCTIVE_VERBS = re.compile(
    r"\b(delete|drop|truncate|purge|destroy|wipe|format|shred|remove|uninstall|reset|kill|terminate|"
    r"transfer|send|charge|withdraw|publish|deploy|grant|revoke|share)\b", re.I,
)
_SENSITIVE_NOUNS = re.compile(
    r"\b(database|table|account|production|prod|funds|payment|password|secret|key|token|cert|certificate|backup)\b",
    re.I,
)


def _infer_risk_from_name(tool: str) -> tuple[int, bool, str]:
    """For unknown tools, infer risk from the name."""
    lower = tool.lower().replace("_", " ")
    risk = 30
    destructive = False
    category = "unknown"
    if _DESTRUCTIVE_VERBS.search(lower):
        risk += 35; destructive = True
    if _SENSITIVE_NOUNS.search(lower):
        risk += 25
    return min(100, risk), destructive, category


def verify_tool_call(
    tool: str,
    arguments: dict | None = None,
    *,
    user_intent: str = "",
    conversation: list[dict] | None = None,
    require_human_for_destructive: bool = True,
) -> dict[str, Any]:
    """Verify a tool call. Returns a rich verdict.

    Args:
      tool: tool name (e.g. "delete_database")
      arguments: tool arguments dict
      user_intent: the original user request that supposedly triggered this tool
      conversation: optional conversation history
      require_human_for_destructive: gate destructive actions behind human review

    Returns: structured verdict matching the analyst-report shape.
    """
    arguments = arguments or {}
    info = TOOL_CATALOG.get(tool)
    if info:
        base_risk = info["risk"]
        destructive = info["destructive"]
        category = info["category"]
        catalog_known = True
    else:
        base_risk, destructive, category = _infer_risk_from_name(tool)
        catalog_known = False

    signals = []

    # Signal 1: base tool risk
    signals.append({
        "name": "tool_baseline_risk",
        "layer": "tool-guard",
        "confidence": 1.0,
        "weight": base_risk / 100,
        "explanation": f"Tool '{tool}' has baseline risk {base_risk}/100 ({category}, destructive={destructive}, in_catalog={catalog_known}).",
    })

    # Signal 2: arguments scan (looking for prompt injection in args, secrets, etc.)
    arg_text = " ".join(str(v) for v in arguments.values())
    arg_risk = 0
    if arg_text.strip():
        threat = analyze_threat(arg_text, source="tool_input")
        arg_risk = threat["risk_score"]
        if arg_risk >= 40:
            signals.append({
                "name": "malicious_tool_arguments",
                "layer": "tool-guard",
                "confidence": threat["confidence"],
                "weight": 0.9,
                "explanation": f"Tool arguments triggered Security LLM analysis: {threat['reason']} (arg risk {arg_risk}).",
            })

    # Signal 3: intent consistency check
    intent_consistent = True
    if user_intent:
        # Did the user explicitly ask for this kind of action?
        action_verb = tool.split("_")[0].lower()
        if destructive and action_verb not in user_intent.lower():
            intent_consistent = False
            signals.append({
                "name": "intent_inconsistency",
                "layer": "tool-guard",
                "confidence": 0.85,
                "weight": 0.9,
                "explanation": f"Destructive tool '{tool}' was invoked but the user request ('{user_intent[:80]}') did not explicitly ask for '{action_verb}'.",
            })

    # Combine
    score = base_risk
    if not intent_consistent:
        score = min(100, score + 20)
    if arg_risk >= 60:
        score = min(100, score + 15)

    severity = _severity(score)
    if destructive and require_human_for_destructive and score >= 60:
        decision = "require_human"
    else:
        decision = _decision(score)

    primary = "tool_abuse" if (destructive and score >= 60) else (
        "malicious_tool_arguments" if arg_risk >= 40 else "tool_call"
    )

    return {
        "tool": tool,
        "decision": decision,
        "risk_score": score,
        "severity": severity,
        "confidence": 0.95,
        "attack_type": primary,
        "destructive": destructive,
        "category": category,
        "in_catalog": catalog_known,
        "intent_consistent": intent_consistent,
        "arguments_risk": arg_risk,
        "signals": signals,
        "reason": signals[0]["explanation"] if signals else "Tool call analyzed.",
        "recommended_action": _recommended_action(
            decision if decision != "require_human" else "block", primary
        ) if decision != "allow" else "Allow. Continue normal monitoring.",
        "analyst": "AgentShield Security LLM",
    }
