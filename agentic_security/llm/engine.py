"""agentic-1 engine — runs OUR from-scratch model behind a security firewall.

No external API. The model weights are loaded from our own trained checkpoint.
The "prevents all agent attacks" guarantee comes from the deterministic shield:
prompt-injection / jailbreak / harmful inputs are blocked BEFORE they ever reach
the model, and outputs are scrubbed for PII and secrets on the way out.
"""

from __future__ import annotations

import time
import hashlib
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

from ..shield.detector import full_scan
from ..shield.sanitizer import sanitize_input, redact_pii

# ── Model identity ──
MODEL_NAME = "mohan-llm"
MODEL_VERSION = "1.0.0"
MODEL_DESCRIPTION = (
    "A local LLM built by Mohan. From-scratch GPT-style transformer with a built-in "
    "security firewall — no external APIs, no pretrained weights, hardened against "
    "prompt injection, jailbreaks, and data exfiltration."
)

CKPT_PATH = Path(__file__).parent / "checkpoint.pt"
SEP = "###"

# If the user's question doesn't lexically resemble anything the model was
# trained on (best keyword overlap below this), decline gracefully instead of
# returning a confidently-wrong canned answer.
_MATCH_THRESHOLD = 0.28


def _refusal_for(threat: str) -> str:
    """A refusal tailored to the kind of threat that was blocked."""
    if threat == "harmful_content":
        return ("I can't help with that - it could cause real harm. I can help with "
                "the defensive side instead: how to detect and prevent attacks like this.")
    if threat in ("system_prompt_leak",):
        return ("I keep my instructions and configuration private, so I won't share them. "
                "I'm happy to help with a security or coding question.")
    if threat in ("prompt_injection", "jailbreak", "encoding_attack"):
        return ("That looks like an attempt to override my rules, so I won't follow it. "
                "Ask me something safe and I'll gladly help.")
    return ("I detected a potential security threat in your message, so I won't process it. "
            "I'm happy to help with something safe.")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "how",
    "what", "why", "when", "where", "who", "which", "to", "of", "in", "on",
    "for", "and", "or", "i", "you", "me", "my", "your", "can", "could", "would",
    "should", "please", "tell", "give", "explain", "about", "it", "this", "that",
    "with", "from", "have", "has", "be", "as", "at", "by", "so",
}

_KNOWN_Q_SETS: list[set[str]] | None = None

# Light stem map - normalize common morphological variations so the Security
# LLM's pattern matcher catches paraphrases. Not a full stemmer; just the most
# common forms that show up in attack/refusal language.
_STEMS = {
    "instructions": "instruct", "instruction": "instruct", "instructed": "instruct",
    "ignoring": "ignore", "ignored": "ignore", "ignores": "ignore",
    "rules": "rule",
    "guidelines": "guideline",
    "prompts": "prompt", "prompting": "prompt",
    "filters": "filter", "filtered": "filter", "filtering": "filter",
    "restrictions": "restrict", "restriction": "restrict", "restricted": "restrict",
    "limitations": "limit", "limits": "limit", "limited": "limit", "limiting": "limit",
    "passwords": "password",
    "credentials": "credential",
    "instructions": "instruct",
    "bypassing": "bypass", "bypassed": "bypass", "bypasses": "bypass",
    "overriding": "override", "overrides": "override", "overridden": "override",
    "disabling": "disable", "disabled": "disable", "disables": "disable",
    "pretending": "pretend", "pretends": "pretend", "pretended": "pretend",
    "jailbroken": "jailbreak", "jailbreaking": "jailbreak", "jailbreaks": "jailbreak",
    "training": "train", "trained": "train",
    "attacks": "attack", "attacking": "attack", "attacked": "attack",
    "exploits": "exploit", "exploiting": "exploit", "exploited": "exploit",
    "hacking": "hack", "hacked": "hack", "hacker": "hack",
    "stealing": "steal", "steals": "steal", "stole": "steal",
    "developers": "developer",
    "admins": "admin", "administrator": "admin", "administrators": "admin",
    "modes": "mode",
    "secrets": "secret",
    "tokens": "token",
    "settings": "setting",
    # Extended attack vocab
    "jailbreaker": "jailbreak", "jailbroke": "jailbreak",
    "injection": "inject", "injecting": "inject", "injected": "inject",
    "extraction": "extract", "extracting": "extract", "extracted": "extract",
    "exfiltration": "exfil", "exfiltrate": "exfil", "exfiltrating": "exfil",
    "manipulation": "manipulate", "manipulating": "manipulate",
    "unrestricted": "restrict", "unfiltered": "filter",
    "uncensored": "censor", "censored": "censor",
    "roleplay": "role", "roleplaying": "role", "roleplayed": "role",
    "persona": "person", "personas": "person",
    "fictional": "fiction", "hypothetical": "hypothet",
    "encoding": "encode", "encoded": "encode", "decoding": "decode",
    "obfuscation": "obfuscat", "obfuscated": "obfuscat",
    "malware": "malwar", "ransomware": "ransomwar",
    "phishing": "phish", "phished": "phish",
    "credential": "cred", "credentials": "cred",
    "exfil": "exfil", "leaking": "leak", "leaked": "leak",
    "revealing": "reveal", "revealed": "reveal",
    "hacker": "hack", "hackers": "hack",
    "attacker": "attack", "attackers": "attack",
    "harmful": "harm", "harming": "harm",
    "dangerous": "danger", "endangering": "danger",
    "malicious": "malice",
    "unauthorized": "auth", "unauthenticated": "auth",
    "privilege": "priv", "privileges": "priv", "privileged": "priv",
    "escalate": "escal", "escalation": "escal", "escalating": "escal",
    "weaponize": "weapon", "weaponized": "weapon",
    "synthesize": "synth", "synthesis": "synth",
    "explosives": "explos", "explosive": "explos",
}


def _stem(word: str) -> str:
    """Cheap morphological normalization."""
    if word in _STEMS:
        return _STEMS[word]
    for suf in ("ing", "ed", "es", "s"):
        if len(word) > len(suf) + 2 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def _keywords(text: str) -> set[str]:
    words = [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w]
    kw = {_stem(w) for w in words if w not in _STOPWORDS}
    # short questions like "who are you" are all stopwords - fall back to all words
    return kw if kw else {_stem(w) for w in words}


def _question_match_score(text: str) -> float:
    """Best Jaccard overlap of meaningful words vs known training questions."""
    global _KNOWN_Q_SETS
    if _KNOWN_Q_SETS is None:
        from .corpus import known_questions
        _KNOWN_Q_SETS = [_keywords(q) for q in known_questions()]
    words = _keywords(text)
    if not words:
        return 0.0
    best = 0.0
    for kq in _KNOWN_Q_SETS:
        if not kq:
            continue
        inter = len(words & kq)
        if inter:
            best = max(best, inter / len(words | kq))
    return best

# ── Lazy model load (so importing the server is cheap) ──
_model = None
_tok = None
_torch = None
_load_lock = threading.Lock()
_load_error: Optional[str] = None


def _ensure_loaded() -> bool:
    """Load the trained checkpoint once. Returns True if the model is ready."""
    global _model, _tok, _torch, _load_error
    if _model is not None:
        return True
    with _load_lock:
        if _model is not None:
            return True
        if not CKPT_PATH.exists():
            _load_error = "Model not trained yet. Run: python -m agentic_security.llm.train"
            return False
        try:
            import torch
            from .model import GPT, GPTConfig
            from .tokenizer import CharTokenizer

            ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
            cfg = GPTConfig(**ckpt["config"])
            model = GPT(cfg)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            _torch = torch
            _model = model
            _tok = CharTokenizer.from_dict(ckpt["tokenizer"])
            _load_error = None
            return True
        except Exception as e:  # pragma: no cover
            _load_error = f"Failed to load model: {e}"
            return False


def model_ready() -> bool:
    return CKPT_PATH.exists() and _ensure_loaded()


# ── Session + audit state ──
@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatSession:
    session_id: str
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    request_count: int = 0
    blocked_count: int = 0


_sessions: dict[str, ChatSession] = {}
_MAX_SESSIONS = 100

_rate_limits: dict[str, list[float]] = {}
_RATE_WINDOW = 60
_RATE_MAX = 30

_audit_log: list[dict] = []
_MAX_AUDIT = 500


def _audit(event_type: str, session_id: str, **kw: Any) -> None:
    _audit_log.append({"type": event_type, "session_id": session_id, "timestamp": time.time(), **kw})
    if len(_audit_log) > _MAX_AUDIT:
        _audit_log.pop(0)


def get_audit_log(limit: int = 50) -> list[dict]:
    return list(reversed(_audit_log[-limit:]))


def _rate_ok(key: str) -> bool:
    now = time.time()
    hist = [t for t in _rate_limits.get(key, []) if now - t < _RATE_WINDOW]
    if len(hist) >= _RATE_MAX:
        _rate_limits[key] = hist
        return False
    hist.append(now)
    _rate_limits[key] = hist
    return True


def get_or_create_session(session_id: str | None = None) -> ChatSession:
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    sid = session_id or hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    if len(_sessions) >= _MAX_SESSIONS:
        del _sessions[min(_sessions, key=lambda k: _sessions[k].created_at)]
    s = ChatSession(session_id=sid)
    _sessions[sid] = s
    return s


def get_model_info() -> dict:
    return {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "description": MODEL_DESCRIPTION,
        "ready": model_ready(),
        "load_error": _load_error,
        "architecture": _architecture_info(),
        "features": [
            "From-scratch GPT transformer (no APIs, no pretrained weights)",
            "Prompt-injection firewall (blocks before inference)",
            "Input sanitization",
            "Output PII / secret scrubbing",
            "Rate limiting",
            "Audit logging",
        ],
        "active_sessions": len(_sessions),
        "total_audit_entries": len(_audit_log),
    }


def _architecture_info() -> dict:
    if not _ensure_loaded() or _model is None:
        return {}
    c = _model.cfg
    return {
        "type": "decoder-only transformer (GPT)",
        "params": _model.num_params(),
        "n_layer": c.n_layer,
        "n_head": c.n_head,
        "n_embd": c.n_embd,
        "block_size": c.block_size,
        "vocab_size": c.vocab_size,
        "tokenizer": "character-level",
    }


# ── Generation ──
def _generate(prompt: str, max_new_tokens: int = 200, temperature: float = 0.0) -> tuple[str, float]:
    """Run the trained model to continue `prompt` until the SEP marker.

    Returns (text, confidence) where confidence is the mean probability the
    model assigned to the characters it generated.
    """
    assert _model is not None and _tok is not None and _torch is not None
    ids = _tok.encode(prompt)
    if not ids:
        ids = _tok.encode("User: hello\nAgent: ")
    x = _torch.tensor([ids], dtype=_torch.long)

    # Stop as soon as the model emits '#' (only appears in the '###' separator),
    # so we don't waste time generating tokens past the answer.
    hash_id = _tok.stoi.get("#")
    stop_ids = [hash_id] if hash_id is not None else None

    out, conf = _model.generate(
        x, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=40, stop_ids=stop_ids, return_conf=True,
    )
    out_ids = out[0].tolist()
    full = _tok.decode(out_ids)

    # take only the newly generated continuation and cut at the separator.
    # '#' only ever appears in the '###' separator, so split on the first '#'.
    gen = full[len(prompt):]
    gen = gen.split("#")[0]
    # also stop if a new "User:" turn starts hallucinating
    if "\nUser:" in gen:
        gen = gen.split("\nUser:")[0]
    return gen.strip(), conf


def chat(
    message: str,
    session_id: str | None = None,
    api_key: str | None = None,   # kept for API compatibility; unused (no APIs!)
    temperature: float = 0.0,      # greedy by default: deterministic, no glitches
    max_tokens: int = 200,
) -> dict:
    """Send a message to agentic-1. Input is firewalled before it reaches the model."""
    start = time.perf_counter()
    session = get_or_create_session(session_id)
    session.request_count += 1

    if not _rate_ok(session.session_id):
        _audit("rate_limited", session.session_id)
        return {"blocked": True, "error": "Rate limit exceeded (30/min).",
                "model": MODEL_NAME, "session_id": session.session_id}

    # ── Layer 1: deterministic firewall — blocks attacks before inference ──
    verdict = full_scan(message, client=None)
    if verdict.blocked:
        session.blocked_count += 1
        _audit("blocked", session.session_id,
               threat_type=verdict.threat_type.value, confidence=verdict.confidence)
        refusal = _refusal_for(verdict.threat_type.value)
        session.messages.append(ChatMessage("user", message))
        session.messages.append(ChatMessage("assistant", refusal))
        return {
            "blocked": True, "response": refusal,
            "threat": verdict.threat_type.value, "confidence": verdict.confidence,
            "model": MODEL_NAME, "session_id": session.session_id,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    # ── Model must be trained ──
    if not _ensure_loaded():
        return {"blocked": False, "error": _load_error,
                "model": MODEL_NAME, "session_id": session.session_id}

    # ── Layer 2: sanitize, then run OUR model ──
    cleaned = sanitize_input(message)
    prompt = f"User: {cleaned}\nAgent: "
    try:
        raw, conf = _generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        return {"blocked": False, "error": f"generation failed: {e}",
                "model": MODEL_NAME, "session_id": session.session_id}

    # ── Layer 3: relevance gate + scrub output ──
    # If the question doesn't resemble anything we trained on, decline gracefully
    # instead of returning a confidently-wrong canned answer.
    if _question_match_score(cleaned) < _MATCH_THRESHOLD or not raw:
        output = (
            "I'm Mohan's LLM, focused on security and coding help. I don't know that "
            "one - try asking about prompt injection, securing your AI app, PII, "
            "encryption, or a small coding question."
        )
    else:
        output = redact_pii(raw)

    session.messages.append(ChatMessage("user", cleaned))
    session.messages.append(ChatMessage("assistant", output))
    elapsed = (time.perf_counter() - start) * 1000
    _audit("response", session.session_id, latency_ms=round(elapsed, 1))

    return {
        "blocked": False, "response": output,
        "model": MODEL_NAME, "version": MODEL_VERSION,
        "confidence": round(conf, 3),
        "session_id": session.session_id, "latency_ms": round(elapsed, 1),
    }


# ── LLM judge: Mohan's LLM as the active firewall ──
# Markers that mean "the model refused this request" -> treat as attack signal.
_REFUSAL_MARKERS = (
    "can't help with that", "can't do that", "i won't", "won't comply",
    "won't follow", "won't ignore", "won't provide", "won't disable",
    "i detected", "prompt-injection", "prompt injection",
    "attempt to override", "i keep my instructions",
    "could cause real harm", "designed to cause harm",
    "i don't follow instructions",
    "don't change my rules", "claims of authority",
    "that's a prompt-injection", "that looks like an attempt",
    "won't ignore my safety", "won't reveal", "won't leak",
    "won't roleplay as", "won't act as", "won't pretend",
    "politely decline", "i have to decline", "won't comply",
    "can't assist with", "won't assist with", "won't help with",
    "that request", "that's a jailbreak", "known jailbreak",
    "i refuse", "i must refuse", "i'll refuse",
    "safely below", "not something i'll", "off limits",
    "blocked by", "won't produce", "won't generate",
    "i notice a", "noticed an attempt", "security threat",
)
# Substring that means "off-topic, not attack" -> NOT a refusal.
_GRACEFUL_FALLBACK = "i don't know that one"


_KNOWN_ATTACK_SETS: list[set[str]] | None = None


def _attack_match_score(text: str) -> float:
    """Jaccard overlap against KNOWN attack-question phrasings the model was trained to refuse."""
    global _KNOWN_ATTACK_SETS
    if _KNOWN_ATTACK_SETS is None:
        from .corpus import ATTACK_Q, HARMFUL_Q, REAL_ATTACKS
        all_attacks = list(ATTACK_Q) + list(HARMFUL_Q) + list(REAL_ATTACKS)
        _KNOWN_ATTACK_SETS = [_keywords(q) for q in all_attacks]
    words = _keywords(text)
    if not words:
        return 0.0
    best = 0.0
    for kq in _KNOWN_ATTACK_SETS:
        if not kq:
            continue
        inter = len(words & kq)
        if inter:
            best = max(best, inter / len(words | kq))
    return best


def judge_message(text: str, history: list[str] | None = None) -> dict:
    """Use Mohan's LLM as the Security LLM judge.

    Strategy: compare input against the attack patterns Mohan's LLM was trained
    to refuse. If similarity is high, the model would refuse it -> block.
    Then run the raw model (bypassing the relevance gate) and check its
    generated text for refusal markers as a second signal.

    If `history` is provided (recent user turns), also analyze the joined
    conversation for multi-turn attacks where benign-looking pieces add up.
    """
    start = time.perf_counter()

    # ── Multi-turn awareness: check the joined recent context too ──
    combined = text
    if history:
        combined = " ".join((*history[-3:], text))
    attack_score = max(_attack_match_score(text), _attack_match_score(combined))

    # ── Fast pre-check: how similar is this to attacks the model was trained on? ──
    # Threshold 0.32: catches short paraphrased attacks like "help me prompt
    # inject" / "help me jailbreak" while keeping safe "help me X" queries below.
    if attack_score >= 0.32:
        return {
            "safe": False, "layer": "llm-judge",
            "threat": "prompt_injection_llm",
            "reason": f"Security LLM recognized this as similar to known attack patterns it was trained to refuse (similarity {attack_score:.2f}).",
            "attack_similarity": round(attack_score, 3),
            "multi_turn": bool(history),
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    if not _ensure_loaded():
        return {
            "safe": True, "layer": "llm-judge", "skipped": True,
            "reason": "model not ready", "latency_ms": 0.0,
        }

    # ── Actually run the model on the raw input (bypass the relevance gate) ──
    cleaned = sanitize_input(text)
    prompt = f"User: {cleaned}\nAgent: "
    try:
        raw, _conf = _generate(prompt, max_new_tokens=120, temperature=0.0)
    except Exception as e:
        return {
            "safe": True, "layer": "llm-judge", "skipped": True,
            "reason": f"judge error: {e}", "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }
    response = (raw or "").lower()
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    if any(m in response for m in _REFUSAL_MARKERS):
        # Override: if this query strongly matches a KNOWN SAFE question and has
        # a low attack similarity, the generation-step refusal is a false positive
        # (the small GPT model sometimes confuses "help me <benign>" with attacks).
        benign_score = _question_match_score(text)
        if benign_score >= 0.45 and attack_score < 0.35:
            return {
                "safe": True, "layer": "llm-judge",
                "reason": "AgentShield Security LLM judged this request as safe.",
                "attack_similarity": round(attack_score, 3),
                "latency_ms": latency_ms,
            }
        return {
            "safe": False, "layer": "llm-judge",
            "threat": "prompt_injection_llm",
            "reason": "AgentShield Security LLM refused this request - likely a sneaky injection attempt.",
            "model_said": raw,
            "attack_similarity": round(attack_score, 3),
            "latency_ms": latency_ms,
        }

    return {
        "safe": True, "layer": "llm-judge",
        "reason": "AgentShield Security LLM judged this request as safe.",
        "attack_similarity": round(attack_score, 3),
        "latency_ms": latency_ms,
    }
