"""Guardrails — additional AI-security primitives built on the detector layer.

Provides higher-level, demo-friendly security features:
  - risk_score()  : multi-category risk breakdown (0-100) with a recommendation
  - moderate()    : content moderation across toxicity categories
  - scan_secrets(): find leaked secrets / PII with masked previews
  - redact()      : return a redacted copy of the text
  - check_policy(): enforce a configurable allow/deny policy

All pattern-based and dependency-free, so it runs instantly with no API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

from .detector import (
    _INJECTION_PATTERNS,
    _PII_PATTERNS,
    _SECRET_PATTERNS,
    ThreatType,
)
from .sanitizer import redact_pii


# ── Content moderation patterns ──
_MODERATION = {
    "violence": re.compile(r"\b(kill|murder|assault|attack|behead|massacre|shoot up|stab)\b", re.I),
    "self_harm": re.compile(r"\b(suicide|kill myself|self[- ]harm|cut myself|end my life)\b", re.I),
    "hate": re.compile(r"\b(racial slur|ethnic cleansing|genocide|inferior race)\b", re.I),
    "sexual": re.compile(r"\b(explicit sexual|child porn|csam|sexual content involving minors)\b", re.I),
    "weapons": re.compile(r"\b(build a bomb|make a bomb|pipe bomb|nerve agent|chemical weapon|dirty bomb)\b", re.I),
    "illegal": re.compile(r"\b(buy drugs|sell drugs|launder money|counterfeit|human trafficking)\b", re.I),
}


@dataclass
class CategoryScore:
    category: str
    score: float           # 0.0 - 1.0
    hits: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _injection_scores(text: str) -> dict[str, CategoryScore]:
    """Score injection-family threats by ThreatType using the detector patterns."""
    buckets: dict[str, list[float]] = {}
    for pattern, threat_type, confidence, _expl in _INJECTION_PATTERNS:
        key = threat_type.value
        buckets.setdefault(key, [])
        if pattern.search(text):
            buckets[key].append(confidence)
    out: dict[str, CategoryScore] = {}
    for key, scores in buckets.items():
        out[key] = CategoryScore(
            category=key,
            score=max(scores) if scores else 0.0,
            hits=len(scores),
        )
    return out


def risk_score(text: str) -> dict:
    """Return a multi-category risk breakdown with an overall score 0-100."""
    cats: dict[str, CategoryScore] = _injection_scores(text)

    # PII
    pii_hits = sum(len(p.findall(text)) for p, _ in _PII_PATTERNS)
    cats["pii_leak"] = CategoryScore("pii_leak", min(1.0, pii_hits * 0.5), pii_hits)

    # secrets
    sec_hits = sum(len(p.findall(text)) for p in _SECRET_PATTERNS)
    cats["secret_leak"] = CategoryScore("secret_leak", min(1.0, sec_hits * 0.6), sec_hits)

    # moderation contributes to harmful_content
    mod_hits = sum(1 for p in _MODERATION.values() if p.search(text))
    if mod_hits:
        existing = cats.get("harmful_content")
        base = existing.score if existing else 0.0
        cats["harmful_content"] = CategoryScore(
            "harmful_content", max(base, min(1.0, 0.6 + mod_hits * 0.15)),
            (existing.hits if existing else 0) + mod_hits,
        )

    overall = max((c.score for c in cats.values()), default=0.0)
    if overall >= 0.85:
        rec = "BLOCK"
    elif overall >= 0.5:
        rec = "REVIEW"
    else:
        rec = "ALLOW"

    return {
        "overall_score": round(overall * 100),
        "recommendation": rec,
        "categories": [c.to_dict() for c in sorted(cats.values(), key=lambda c: -c.score)],
        "text_length": len(text),
    }


def moderate(text: str) -> dict:
    """Content moderation across toxicity categories."""
    flagged = {}
    for category, pattern in _MODERATION.items():
        if pattern.search(text):
            flagged[category] = True
    return {
        "flagged": bool(flagged),
        "categories": flagged,
        "action": "block" if flagged else "allow",
    }


def _mask(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def scan_secrets(text: str) -> dict:
    """Find leaked secrets and PII, returning masked previews and positions."""
    findings = []
    for pattern, pii_type in _PII_PATTERNS:
        for m in pattern.finditer(text):
            val = m.group(0)
            findings.append({
                "type": pii_type,
                "category": "pii",
                "preview": _mask(val),
                "start": m.start(),
                "end": m.end(),
            })
    for pattern in _SECRET_PATTERNS:
        for m in pattern.finditer(text):
            val = m.group(0)
            findings.append({
                "type": "secret",
                "category": "secret",
                "preview": _mask(val),
                "start": m.start(),
                "end": m.end(),
            })
    return {"count": len(findings), "findings": findings}


def redact(text: str) -> dict:
    """Return a redacted copy of the text plus how many items were removed."""
    import re as _re
    cleaned = redact_pii(text)
    items_redacted = len(_re.findall(r"\[REDACTED-[A-Z-]+\]", cleaned))
    return {"redacted": cleaned, "items_redacted": items_redacted}


# ── Configurable policy engine ──
@dataclass
class Policy:
    max_length: int = 8000
    denied_keywords: list[str] = field(default_factory=list)
    block_pii: bool = True
    block_secrets: bool = True
    block_on_risk: int = 85   # block if overall risk >= this


def check_policy(text: str, policy: Policy | None = None) -> dict:
    """Enforce a policy against the text. Returns allow/deny with reasons."""
    policy = policy or Policy()
    reasons = []

    if len(text) > policy.max_length:
        reasons.append(f"exceeds max length ({len(text)} > {policy.max_length})")

    low = text.lower()
    for kw in policy.denied_keywords:
        if kw.lower() in low:
            reasons.append(f"contains denied keyword: {kw!r}")

    secrets = scan_secrets(text)
    if policy.block_pii and any(f["category"] == "pii" for f in secrets["findings"]):
        reasons.append("contains PII")
    if policy.block_secrets and any(f["category"] == "secret" for f in secrets["findings"]):
        reasons.append("contains secrets")

    risk = risk_score(text)
    if risk["overall_score"] >= policy.block_on_risk:
        reasons.append(f"risk score {risk['overall_score']} >= {policy.block_on_risk}")

    return {
        "allowed": not reasons,
        "action": "deny" if reasons else "allow",
        "reasons": reasons,
        "risk_score": risk["overall_score"],
    }
