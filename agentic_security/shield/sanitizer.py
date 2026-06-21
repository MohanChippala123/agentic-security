"""Input sanitizer and output filter.

Cleans dangerous patterns from inputs before they reach the model,
and scrubs sensitive data from outputs before they reach the user.
"""

from __future__ import annotations

import re


def sanitize_input(text: str) -> str:
    """Remove or neutralize known injection patterns from user input."""
    out = text
    # Strip token delimiters
    out = re.sub(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>", "", out)
    out = re.sub(r"<<SYS>>|<</SYS>>", "", out)
    out = re.sub(r"\[INST\]|\[/INST\]", "", out)
    out = re.sub(r"</?system>", "", out, flags=re.I)
    # Neutralize markdown code fence role injection
    out = re.sub(r"```\s*(system|assistant|function_call)", "``` \\1", out, flags=re.I)
    return out.strip()


def redact_pii(text: str) -> str:
    """Redact PII from model output."""
    out = text
    # SSN
    out = re.sub(r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b", "[REDACTED-SSN]", out)
    # Credit card
    out = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[REDACTED-CC]", out)
    # Email addresses
    out = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED-EMAIL]", out)
    # Phone numbers (US-style)
    out = re.sub(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED-PHONE]", out)
    # API keys
    out = re.sub(r"\bsk-[a-zA-Z0-9]{20,}\b", "[REDACTED-KEY]", out)
    out = re.sub(r"\b(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*\S{8,}",
                 "\\1=[REDACTED]", out, flags=re.I)
    # Private keys
    out = re.sub(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(RSA\s+)?PRIVATE\s+KEY-----",
                 "[REDACTED-PRIVATE-KEY]", out)
    # Passwords in output
    out = re.sub(r"(password|passwd|pwd)\s*[:=]\s*\S+", "\\1=[REDACTED]", out, flags=re.I)
    return out


def redact_system_prompt(text: str, system_prompt: str = "") -> str:
    """Remove any trace of the system prompt from model output."""
    if not system_prompt:
        return text
    # If the model literally echoed the system prompt, strip it
    if system_prompt in text:
        return text.replace(system_prompt, "[SYSTEM PROMPT REDACTED]")
    # Also check for partial leaks (first 50 chars)
    prefix = system_prompt[:50]
    if len(prefix) > 20 and prefix in text:
        return text.replace(prefix, "[SYSTEM PROMPT REDACTED]")
    return text
