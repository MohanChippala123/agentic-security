"""AgentShield Security LLM - structured threat analysis.

The analyzer combines multiple signals (regex patterns, attack-similarity,
hidden-instruction detection, obfuscation checks, and the trained Security LLM's
refusal output) into a single rich threat report. It does NOT just return
ALLOW/BLOCK - it explains its reasoning, scores severity, traces attack chains,
and recommends action, like a security analyst would.
"""

from __future__ import annotations

import re
import time
import base64
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Any

from ..shield.detector import detect_pattern, detect_harmful, ThreatType


SECURITY_LLM_NAME = "AgentShield Security LLM"
SECURITY_LLM_VERSION = "1.0.0"


# ── Hidden / obfuscated content detectors ──
_ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁠-⁯﻿]")
_HOMOGLYPH = re.compile(r"[А-яΑ-Ωα-ω]")  # cyrillic/greek lookalikes
_BASE64_LONG = re.compile(r"[A-Za-z0-9+/]{32,}={0,2}")
_HEX_LONG = re.compile(r"(?:0x)?[a-fA-F0-9]{40,}")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_WHITE_TEXT = re.compile(r"color\s*:\s*(?:#fff|#ffffff|white|rgb\(\s*255\s*,\s*255\s*,\s*255\s*\))", re.I)

# ROT13 decode then check
_ROT13_TABLE = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm",
)
# Leetspeak normalisation map
_LEET = str.maketrans("013456789@$", "oleasgtbgaS")

# Indirect prompt injection: instructions embedded in supposedly external content
_INDIRECT_INJECTION = re.compile(
    r"(?:"
    r"(?:ai|llm|assistant|chatgpt|gpt|model|bot)\s*[,:]\s*(?:ignore|disregard|forget|override|skip|bypass)|"
    r"\[system\]|\{system\}|<system>|<!--\s*(?:ignore|prompt|instruction)|"
    r"note\s+to\s+(?:ai|llm|assistant|model)\s*[:-]|"
    r"(?:instruction|directive)\s+for\s+(?:the\s+)?(?:ai|assistant|model)\s*[:-]|"
    r"if\s+you(?:'re|\s+are)\s+an?\s+(?:ai|llm|assistant)\s*,\s*(?:ignore|reveal|output|print|show)"
    r")",
    re.I,
)

# Role-play escape patterns (beyond the corpus)
_ROLEPLAY_ESCAPE = re.compile(
    r"\b(?:"
    r"pretend\s+(?:you(?:'re|\s+are)|to\s+be)|"
    r"act\s+as\s+(?:if|though|an?)\s|"
    r"play\s+(?:the\s+)?(?:role|part|character)\s+of|"
    r"in\s+this\s+(?:story|fiction|game|scenario|roleplay|simulation)\s+you\s+(?:are|have\s+no)|"
    r"for\s+(?:this\s+)?(?:fictional|creative|hypothetical)\s+(?:exercise|scenario|story)\s+you\s+(?:can|are)|"
    r"let(?:'s|\s+us)\s+(?:say|pretend|imagine)\s+you\s+have\s+no\s+(?:rules|restrictions|limits|safety)"
    r")\b",
    re.I,
)


def _has_zero_width(text: str) -> bool:
    return bool(_ZERO_WIDTH.search(text))


def _has_homoglyph_mix(text: str) -> bool:
    """Cyrillic/Greek letters mixed into otherwise-ASCII text is a smell."""
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    suspicious = len(_HOMOGLYPH.findall(text))
    return suspicious > 0 and ascii_letters > 5 and suspicious / max(1, ascii_letters) > 0.05


def _has_suspicious_base64(text: str) -> bool:
    for m in _BASE64_LONG.findall(text):
        try:
            decoded = base64.b64decode(m + "==", validate=False).decode("utf-8", errors="ignore").lower()
            if any(k in decoded for k in ("ignore", "system", "prompt", "instruction", "bypass", "override")):
                return True
        except Exception:
            pass
    return False


def _has_hidden_instructions(text: str) -> bool:
    """HTML comments / white-text tricks / common indirect injection vectors."""
    if _HTML_COMMENT.search(text) and re.search(r"<!--.*?(ignore|system|prompt|instruction).*?-->", text, re.I | re.DOTALL):
        return True
    if _WHITE_TEXT.search(text):
        return True
    return False


def _decode_ascii_safe(text: str) -> str:
    """Strip non-printable obfuscation to expose what the model would actually see."""
    text = _ZERO_WIDTH.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    return text


# ── Severity / decision thresholds ──
def _severity(score: int) -> str:
    if score >= 85:
        return "critical"
    if score >= 65:
        return "high"
    if score >= 40:
        return "medium"
    if score >= 15:
        return "low"
    return "info"


def _decision(score: int) -> str:
    if score >= 70:
        return "block"
    if score >= 40:
        return "review"
    return "allow"


# ── Attack-type catalog (judge UI uses these labels) ──
ATTACK_CATALOG = {
    "prompt_injection":          "Direct instruction override attempt.",
    "indirect_prompt_injection": "Hidden instructions in external content.",
    "jailbreak":                 "Persona / mode-switch jailbreak.",
    "system_prompt_leak":        "Attempt to extract system prompt.",
    "encoding_attack":           "Encoded or obfuscated payload (base64, unicode tricks, delimiters).",
    "harmful_content":           "Operational request for harmful or illegal content.",
    "data_exfiltration":         "Attempt to leak secrets or sensitive data.",
    "hidden_instructions":       "Hidden text (HTML comments / white text / zero-width) targeting the LLM.",
    "homoglyph_obfuscation":     "Non-ASCII letters substituted for ASCII to bypass filters.",
    "role_confusion":            "Conflicting role / authority claims.",
    "goal_hijacking":            "Attempt to redirect the agent's objective.",
    "context_manipulation":      "Manipulating prior turns or memory to change behavior.",
    "tool_abuse":                "Calling a destructive or unauthorized tool.",
    "cost_bombing":              "Oversized / repetitive prompt aimed at inflating API cost.",
    "clean":                     "No threats detected.",
}


@dataclass
class ThreatSignal:
    name: str
    confidence: float
    weight: float
    explanation: str
    layer: str  # which detector found it


@dataclass
class AnalystVerdict:
    risk_score: int
    severity: str
    decision: str
    attack_type: str
    attack_chain: list[str]
    confidence: float
    reason: str
    explanation: str
    signals: list[dict] = field(default_factory=list)
    recommended_action: str = ""
    analyst: str = SECURITY_LLM_NAME
    analyst_version: str = SECURITY_LLM_VERSION
    latency_ms: float = 0.0
    sanitized_input: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _recommended_action(decision: str, attack_type: str) -> str:
    if decision == "block":
        if attack_type == "tool_abuse":
            return "Block the tool call and require explicit user re-authorization with a fresh prompt."
        if attack_type in ("system_prompt_leak", "data_exfiltration"):
            return "Block, alert security team, rotate any potentially exposed credentials."
        if attack_type == "harmful_content":
            return "Block, do not log full payload, surface a refusal to the user."
        return "Block the request and respond with a refusal. Increment attacker score for this session."
    if decision == "review":
        return "Flag for human review or require step-up auth before proceeding."
    return "Allow. Continue normal monitoring."


def _narrate(signals: list[ThreatSignal], score: int, decision: str, attack_type: str) -> str:
    if not signals:
        return ("The Security LLM analyzed this request and found no indicators of prompt injection, "
                "jailbreak, encoding tricks, or harmful intent. Decision: allow.")
    parts = [
        f"The Security LLM identified {len(signals)} threat signal(s), combined into a risk score of "
        f"{score}/100 ({_severity(score)}). Decision: {decision.upper()}."
    ]
    for s in signals:
        parts.append(f"  - [{s.layer}] {s.name} (confidence {s.confidence:.2f}): {s.explanation}")
    parts.append(f"Primary attack vector: {attack_type}. {ATTACK_CATALOG.get(attack_type, '')}")
    return "\n".join(parts)


def _attack_similarity_signal(text: str) -> ThreatSignal | None:
    """Use the trained Security LLM's attack-question knowledge."""
    try:
        from ..llm.engine import _attack_match_score, _question_match_score  # type: ignore
        score = _attack_match_score(text)
    except Exception:
        return None
    if score < 0.22:
        return None
    # Suppress signal if the input strongly matches a known benign question
    # (educational security terms like "sql injection" or "prompt injection"
    # have the same keywords as attack phrases after stopword removal).
    try:
        benign = _question_match_score(text)
        if benign >= 0.40:
            return None
    except Exception:
        pass
    # Weight proportional to similarity so weak signals don't dominate
    weight = min(0.95, 0.4 + score * 0.6)
    return ThreatSignal(
        name="known_attack_pattern",
        confidence=min(1.0, score),
        weight=round(weight, 2),
        explanation=f"Input matches known attack phrasings the Security LLM was trained to refuse (similarity {score:.0%}).",
        layer="security-llm",
    )


def _llm_refusal_signal(text: str) -> ThreatSignal | None:
    """Run the Security LLM directly and check if it would refuse this request."""
    try:
        from ..llm.engine import judge_message
        verdict = judge_message(text)
    except Exception:
        return None
    if verdict.get("safe"):
        return None
    return ThreatSignal(
        name=verdict.get("threat", "prompt_injection_llm"),
        confidence=0.92,
        weight=1.0,
        explanation=f"Security LLM refused this request. Model said: \"{(verdict.get('model_said') or '')[:200]}\"",
        layer="security-llm",
    )


def _hidden_signal(text: str) -> ThreatSignal | None:
    if _has_hidden_instructions(text):
        return ThreatSignal(
            name="hidden_instructions",
            confidence=0.95,
            weight=1.0,
            explanation="Found hidden instructions (HTML comment or invisible text) targeting the LLM.",
            layer="content-analyzer",
        )
    if _has_zero_width(text):
        return ThreatSignal(
            name="hidden_instructions",
            confidence=0.85,
            weight=0.8,
            explanation="Found zero-width / direction-override unicode characters used to hide payload.",
            layer="content-analyzer",
        )
    return None


def _homoglyph_signal(text: str) -> ThreatSignal | None:
    if _has_homoglyph_mix(text):
        return ThreatSignal(
            name="homoglyph_obfuscation",
            confidence=0.88,
            weight=0.85,
            explanation="Cyrillic/Greek letters substituted for ASCII to bypass keyword filters.",
            layer="content-analyzer",
        )
    return None


def _encoding_signal(text: str) -> ThreatSignal | None:
    if _has_suspicious_base64(text):
        return ThreatSignal(
            name="encoding_attack",
            confidence=0.90,
            weight=0.95,
            explanation="Base64-encoded payload contains attack keywords (ignore/system/prompt/bypass).",
            layer="content-analyzer",
        )
    # ROT13 decode and check
    decoded_rot13 = text.translate(_ROT13_TABLE).lower()
    if any(k in decoded_rot13 for k in ("ignore", "system prompt", "jailbreak", "bypass", "override")):
        return ThreatSignal(
            name="encoding_attack",
            confidence=0.88,
            weight=0.92,
            explanation="ROT13-encoded payload contains attack keywords after decoding.",
            layer="content-analyzer",
        )
    # Leetspeak normalisation
    leet_decoded = text.translate(_LEET).lower()
    if any(k in leet_decoded for k in ("ignore", "system", "jailbreak", "bypass", "override")):
        return ThreatSignal(
            name="encoding_attack",
            confidence=0.75,
            weight=0.80,
            explanation="Leetspeak-encoded payload contains attack keywords after normalisation.",
            layer="content-analyzer",
        )
    return None


def _indirect_injection_signal(text: str, source: str) -> ThreatSignal | None:
    if _INDIRECT_INJECTION.search(text):
        return ThreatSignal(
            name="indirect_prompt_injection",
            confidence=0.91,
            weight=1.0,
            explanation=(
                "Indirect prompt injection detected: instructions embedded inside external/retrieved content "
                "targeting the AI model."
            ),
            layer="content-analyzer",
        )
    return None


def _roleplay_escape_signal(text: str) -> ThreatSignal | None:
    if _ROLEPLAY_ESCAPE.search(text):
        return ThreatSignal(
            name="jailbreak",
            confidence=0.84,
            weight=0.90,
            explanation="Role-play escape attempt: asking the model to pretend/act as an entity with no restrictions.",
            layer="content-analyzer",
        )
    return None


def _guard_graph_signal(text: str) -> ThreatSignal | None:
    """Run the LangGraph hybrid firewall (XGBoost+MiniLM + Security LLM) as a unified signal."""
    try:
        from ..llm.guard_graph import run_firewall, is_available
        if not is_available():
            return None
        state = run_firewall(text)
        if state.get("verdict") != "attack":
            return None
        score = state.get("score", 0.0)
        prob = state.get("keep_jailbreak", 0.0)
        notes = state.get("notes", "")
        return ThreatSignal(
            name="prompt_injection",
            confidence=round(min(1.0, score / 100), 3),
            weight=1.05,  # hybrid layer is high-precision
            explanation=f"LangGraph hybrid firewall verdict: ATTACK. {notes}",
            layer="hybrid-layer",
        )
    except Exception:
        return None


# ── API-key-specific attack vectors ──
# Attempts to extract the protected upstream key / secrets.
# A "secret thing" - the object an exfil verb might target. Allows qualifier
# words in between (e.g. "the openai api key", "all your secret tokens").
_SECRET_OBJ = r"(?:[\w.]+\s+){0,3}(?:api[\s_-]?key|api[\s_-]?keys|openai[\s_-]?key|secret|secrets|token|tokens|credential|credentials|password|passwords|os\.environ|process\.env|environment\s+variables?|\.env|env\s+vars?)"
_KEY_EXFIL = re.compile(
    r"\b(?:"
    r"what(?:'s| is| are)\s+(?:the|your|our)\s+" + _SECRET_OBJ + r"|"
    r"(?:reveal|show|print|give|tell|expose|leak|exfiltrate|dump|export|email|send|fetch|get|read|display|output)\s+"
    r"(?:me\s+)?(?:the|your|our|all|every)?\s*" + _SECRET_OBJ +
    r")\b",
    re.I,
)


def _key_exfil_signal(text: str) -> ThreatSignal | None:
    if _KEY_EXFIL.search(text):
        return ThreatSignal(
            name="data_exfiltration",
            confidence=0.93,
            weight=1.0,
            explanation="Attempt to extract the protected API key, secrets, or environment variables.",
            layer="content-analyzer",
        )
    return None


def _cost_bomb_signal(text: str, truncated: bool) -> ThreatSignal | None:
    """Abnormally large input is a cost / token-bombing attack on the API key."""
    n = len(text)
    if truncated or n >= 12000:
        return ThreatSignal(
            name="cost_bombing",
            confidence=0.80,
            weight=0.85,
            explanation=f"Abnormally large prompt ({n}+ chars) - likely a token/cost-bombing attempt to run up the API bill.",
            layer="content-analyzer",
        )
    # Repetition flooding (e.g. "a"*5000 or repeated tokens)
    if n >= 600:
        words = text.split()
        if words and len(set(words)) / len(words) < 0.05:
            return ThreatSignal(
                name="cost_bombing",
                confidence=0.78,
                weight=0.8,
                explanation="Highly repetitive payload - token-flooding pattern aimed at inflating cost.",
                layer="content-analyzer",
            )
    return None


# Mapping from regex ThreatType to richer AgentShield categories
_TT_MAP = {
    ThreatType.PROMPT_INJECTION:   "prompt_injection",
    ThreatType.JAILBREAK:          "jailbreak",
    ThreatType.SYSTEM_PROMPT_LEAK: "system_prompt_leak",
    ThreatType.ENCODING_ATTACK:    "encoding_attack",
    ThreatType.HARMFUL_CONTENT:    "harmful_content",
    ThreatType.DATA_EXFIL:         "data_exfiltration",
    ThreatType.PII_LEAK:           "data_exfiltration",
    ThreatType.CLEAN:              "clean",
}


def _compose_risk_score(signals: list[ThreatSignal]) -> int:
    """Combine signal confidences into a 0-100 risk score (saturating add)."""
    if not signals:
        return 0
    # Use a probabilistic-OR style combination so multiple weak signals stack.
    fail = 1.0
    for s in signals:
        fail *= (1.0 - min(1.0, s.confidence * s.weight))
    score = (1.0 - fail) * 100
    # Bonus for diversity (attack chain across multiple layers).
    layers = {s.layer for s in signals}
    if len(layers) >= 2:
        score = min(100, score + 5)
    if len(signals) >= 3:
        score = min(100, score + 5)
    return int(round(score))


def analyze_threat(text: str, *, source: str = "user", context: dict | None = None) -> dict[str, Any]:
    """Run the AgentShield Security LLM analysis on `text`.

    Args:
      text: the input to analyze
      source: "user", "external_content" (RAG/website), "tool_input", "memory_write"
      context: optional dict with extra context (conversation history, agent_id, etc.)

    Returns a rich threat report (see AnalystVerdict).
    """
    t0 = time.perf_counter()

    # ── Input hardening: never crash, whatever comes in ──
    if text is None:
        text = ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            text = ""
    # Cap absurdly large inputs (DoS / memory) - analyze a generous prefix.
    MAX_LEN = 20000
    truncated = len(text) > MAX_LEN
    if truncated:
        text = text[:MAX_LEN]

    try:
        cleaned = _decode_ascii_safe(text)
    except Exception:
        cleaned = text
    signals: list[ThreatSignal] = []

    def _safe(fn, *args):
        """Run a detector; swallow any failure so one bad detector can't break analysis."""
        try:
            return fn(*args)
        except Exception:
            return None

    # Regex shield
    rv = _safe(detect_pattern, cleaned)
    if rv:
        signals.append(ThreatSignal(
            name=_TT_MAP.get(rv.threat_type, rv.threat_type.value),
            confidence=rv.confidence,
            weight=1.0,
            explanation=rv.explanation,
            layer="regex-shield",
        ))

    # Harmful intent (independent of regex)
    if not rv:
        hv = _safe(detect_harmful, cleaned)
        if hv:
            signals.append(ThreatSignal(
                name="harmful_content",
                confidence=hv.confidence,
                weight=1.0,
                explanation=hv.explanation,
                layer="regex-shield",
            ))

    # Content-analyzer - all 6 attack categories
    weight_bump = 1.0 if source == "user" else 1.15
    for sig in (
        # 1. Prompt injection / hidden instructions
        _safe(_hidden_signal, cleaned),
        # 2. Jailbreak / role-play escapes
        _safe(_roleplay_escape_signal, cleaned),
        # 5. Obfuscation & encoding attacks (homoglyph, base64, ROT13, leet)
        _safe(_homoglyph_signal, cleaned),
        _safe(_encoding_signal, text),
        # 3. System prompt extraction / 4. Indirect injection / key exfil
        _safe(_key_exfil_signal, cleaned),
        _safe(_indirect_injection_signal, cleaned, source),
        # Cost bombing
        _safe(_cost_bomb_signal, text, truncated),
    ):
        if sig:
            sig.weight = min(1.0, sig.weight * weight_bump)
            if source != "user" and sig.name == "hidden_instructions":
                sig.name = "indirect_prompt_injection"
            signals.append(sig)

    # Security LLM signals
    sim = _safe(_attack_similarity_signal, cleaned)
    if sim:
        signals.append(sim)

    # LLM judge + hybrid-layer graph - both run as CONFIRMERS (only when there
    # is pre-existing suspicion, to avoid over-triggering on benign queries).
    pre_score = _compose_risk_score(signals)
    if 25 <= pre_score < 80:
        llm = _safe(_llm_refusal_signal, cleaned)
        if llm:
            signals.append(llm)

    # Hybrid-layer (XGBoost+MiniLM via LangGraph) - only as a confirmer when
    # there is already some suspicion, to avoid false positives on trivially
    # benign inputs like short greetings ("hello", "hi") that the XGBoost
    # classifier was never trained on.
    if pre_score >= 25:
        graph_sig = _safe(_guard_graph_signal, cleaned)
        if graph_sig:
            signals.append(graph_sig)

    # Compose final report
    score = _compose_risk_score(signals)
    severity = _severity(score)
    decision = _decision(score)
    chain = [s.name for s in sorted(signals, key=lambda s: -s.confidence * s.weight)]
    primary = chain[0] if chain else "clean"
    confidence = max((s.confidence for s in signals), default=0.0)
    reason = signals[0].explanation if signals else "No indicators of attack."

    verdict = AnalystVerdict(
        risk_score=score,
        severity=severity,
        decision=decision,
        attack_type=primary,
        attack_chain=chain,
        confidence=round(confidence, 3),
        reason=reason,
        explanation=_narrate(signals, score, decision, primary),
        signals=[asdict(s) for s in signals],
        recommended_action=_recommended_action(decision, primary),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        sanitized_input=cleaned if cleaned != text else "",
    )
    return verdict.to_dict()
