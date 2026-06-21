"""Scan external/retrieved content (RAG, websites, PDFs) and memory writes."""

from __future__ import annotations

import re
from typing import Any

from .analyzer import analyze_threat, _decode_ascii_safe


def _strip_html(text: str) -> str:
    """Remove HTML tags but PRESERVE comments and inline-style content for analysis."""
    return re.sub(r"<(?!!--)(?!/?(?:style|script))[^>]+>", " ", text)


def scan_external_content(content: str, *, source_url: str = "", content_type: str = "text/html") -> dict[str, Any]:
    """Scan retrieved content (RAG / website / PDF) for indirect prompt injection.

    Returns a sanitized version of the content alongside the threat report.
    """
    text = content or ""
    if content_type.startswith("text/html") or "<" in text[:200]:
        # Keep raw for detection, but also produce a stripped/sanitized version.
        report = analyze_threat(text, source="external_content")
        sanitized = _strip_html(text)
    elif content_type == "application/pdf":
        report = analyze_threat(text, source="external_content")
        sanitized = text
    else:
        report = analyze_threat(text, source="external_content")
        sanitized = text

    report["source_url"] = source_url
    report["content_type"] = content_type
    report["sanitized_content"] = sanitized[:8000]   # cap to avoid bloating responses
    report["original_length"] = len(content)
    if report["decision"] == "block":
        report["recommended_action"] = (
            "Do NOT pass this content to the LLM. The retrieved document contains hidden "
            "instructions attempting indirect prompt injection. Drop and alert."
        )
    return report


def scan_memory_write(content: str, *, agent_id: str = "default", memory_key: str = "") -> dict[str, Any]:
    """Inspect what an agent wants to write to long-term memory.

    Detects instruction-poisoning attempts where the attacker tries to plant
    fake 'user preferences' or persistent instructions.
    """
    report = analyze_threat(content, source="memory_write", context={"agent_id": agent_id})
    report["agent_id"] = agent_id
    report["memory_key"] = memory_key

    # Extra heuristics for memory poisoning
    poisoning_markers = [
        r"\balways\b.{0,40}\b(do|run|execute|follow|obey)\b",
        r"\bnever\b.{0,40}\b(ask|refuse|warn)\b",
        r"\bremember\s+(that|to)\s+",
        r"\bthe\s+(real|actual)\s+(user|admin|owner)\s+",
        r"\b(future|next)\s+(session|conversation)\b",
    ]
    matches = sum(1 for p in poisoning_markers if re.search(p, content, re.I))
    if matches >= 2:
        report["risk_score"] = max(report["risk_score"], 75)
        report["severity"] = "high" if report["risk_score"] < 85 else "critical"
        report["decision"] = "block" if report["decision"] != "block" else report["decision"]
        report["attack_type"] = "memory_poisoning"
        report["attack_chain"].insert(0, "memory_poisoning")
        report["reason"] = (
            f"Memory-write contains {matches} persistence markers (e.g. 'always', 'remember to', "
            f"'in future sessions') suggesting long-term instruction poisoning."
        )
        report["recommended_action"] = (
            "Block this write. Memory persistence attacks survive across sessions and are "
            "very high-impact. Require explicit user confirmation."
        )
    return report
