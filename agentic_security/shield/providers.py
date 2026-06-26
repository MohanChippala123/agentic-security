"""Provider adapters - unified interface to OpenAI, Anthropic, Groq, Gemini, Mistral, etc."""

from __future__ import annotations

import re
import time
from typing import Any

# ── Pricing: USD per 1K tokens (input, output) ────────────────────────────────
PROVIDER_PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "openai": {
        "gpt-4o":           (0.00250, 0.01000),
        "gpt-4o-mini":      (0.00015, 0.00060),
        "gpt-4-turbo":      (0.01000, 0.03000),
        "gpt-3.5-turbo":    (0.00050, 0.00150),
        "o1-mini":          (0.00300, 0.01200),
    },
    "anthropic": {
        "claude-opus-4-8":          (0.01500, 0.07500),
        "claude-sonnet-4-6":        (0.00300, 0.01500),
        "claude-haiku-4-5-20251001":(0.00025, 0.00125),
        "claude-3-5-sonnet-20241022":(0.00300, 0.01500),
        "claude-3-5-haiku-20241022": (0.00025, 0.00125),
        "claude-3-opus-20240229":    (0.01500, 0.07500),
    },
    "groq": {
        "llama-3.3-70b-versatile":  (0.00059, 0.00079),
        "llama-3.1-8b-instant":     (0.00005, 0.00008),
        "mixtral-8x7b-32768":       (0.00024, 0.00024),
        "gemma2-9b-it":             (0.00020, 0.00020),
        "llama3-70b-8192":          (0.00059, 0.00079),
        "llama3-8b-8192":           (0.00005, 0.00008),
        "whisper-large-v3":         (0.00011, 0.00011),
    },
    "gemini": {
        "gemini-1.5-pro":           (0.00125, 0.00500),
        "gemini-1.5-flash":         (0.000075, 0.000300),
        "gemini-1.5-flash-8b":      (0.0000375, 0.000150),
        "gemini-2.0-flash":         (0.000075, 0.000300),
        "gemini-2.0-flash-lite":    (0.000075, 0.000300),
    },
    "mistral": {
        "mistral-large-latest":     (0.00200, 0.00600),
        "mistral-small-latest":     (0.00020, 0.00060),
        "codestral-latest":         (0.00020, 0.00060),
        "open-mixtral-8x7b":        (0.00070, 0.00070),
        "open-mistral-7b":          (0.00025, 0.00025),
    },
    "together": {
        "meta-llama/Llama-3-70b-chat-hf": (0.00090, 0.00090),
        "meta-llama/Llama-3-8b-chat-hf":  (0.00020, 0.00020),
        "mistralai/Mixtral-8x7B-Instruct-v0.1": (0.00060, 0.00060),
    },
    "cohere": {
        "command-r-plus":  (0.00300, 0.01500),
        "command-r":       (0.00050, 0.00150),
        "command":         (0.00100, 0.00200),
    },
}

# Flat lookup: model name → (provider, price_tuple)
_MODEL_TO_PROVIDER: dict[str, tuple[str, tuple[float, float]]] = {}
for _prov, _models in PROVIDER_PRICING.items():
    for _model, _price in _models.items():
        _MODEL_TO_PROVIDER[_model] = (_prov, _price)

DEFAULT_MODELS: dict[str, str] = {
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "groq":      "llama-3.3-70b-versatile",
    "gemini":    "gemini-1.5-flash",
    "mistral":   "mistral-small-latest",
    "together":  "meta-llama/Llama-3-8b-chat-hf",
    "cohere":    "command-r",
}


def detect_provider(api_key: str) -> str:
    """Guess provider from key prefix."""
    k = (api_key or "").strip()
    if k.startswith("sk-ant-"):
        return "anthropic"
    if k.startswith("sk-"):
        # Could be OpenAI or Mistral - check length/format
        if len(k) > 60:
            return "openai"
        return "openai"
    if k.startswith("gsk_"):
        return "groq"
    if k.startswith("AIzaSy"):
        return "gemini"
    if re.match(r"^[0-9a-f]{32}$", k):
        return "mistral"
    if k.startswith("together_"):
        return "together"
    # Cohere keys are long random strings
    return "openai"  # safe default


def get_cost(provider: str, model: str, in_tok: int, out_tok: int) -> float:
    models = PROVIDER_PRICING.get(provider, {})
    pin, pout = models.get(model, (0.0002, 0.0008))
    return (in_tok / 1000.0) * pin + (out_tok / 1000.0) * pout


def list_models(provider: str) -> list[str]:
    return list(PROVIDER_PRICING.get(provider, {}).keys())


def all_providers() -> list[str]:
    return list(PROVIDER_PRICING.keys())


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def call_provider(
    messages: list[dict],
    model: str,
    api_key: str,
    provider: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Unified provider call. Returns {output, in_tok, out_tok, provider, model, latency_ms}."""
    if not provider:
        provider = detect_provider(api_key)

    # Ensure model belongs to provider; fall back to default
    if model not in PROVIDER_PRICING.get(provider, {}):
        model = DEFAULT_MODELS.get(provider, model)

    t0 = time.time()

    try:
        if provider == "openai":
            return _call_openai(messages, model, api_key, max_tokens, temperature, t0)
        elif provider == "anthropic":
            return _call_anthropic(messages, model, api_key, max_tokens, temperature, t0)
        elif provider == "groq":
            return _call_groq(messages, model, api_key, max_tokens, temperature, t0)
        elif provider == "gemini":
            return _call_gemini(messages, model, api_key, max_tokens, temperature, t0)
        elif provider == "mistral":
            return _call_mistral(messages, model, api_key, max_tokens, temperature, t0)
        elif provider == "together":
            return _call_together(messages, model, api_key, max_tokens, temperature, t0)
        elif provider == "cohere":
            return _call_cohere(messages, model, api_key, max_tokens, temperature, t0)
        else:
            return _call_openai(messages, model, api_key, max_tokens, temperature, t0)
    except Exception as exc:
        return {"error": str(exc), "provider": provider, "model": model}


def _latency(t0: float) -> float:
    return round((time.time() - t0) * 1000, 1)


def _call_openai(messages, model, api_key, max_tokens, temperature, t0):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    output = resp.choices[0].message.content or ""
    in_tok = resp.usage.prompt_tokens if resp.usage else _est_tokens(str(messages))
    out_tok = resp.usage.completion_tokens if resp.usage else _est_tokens(output)
    return {"output": output, "in_tok": in_tok, "out_tok": out_tok,
            "provider": "openai", "model": model, "latency_ms": _latency(t0)}


def _call_anthropic(messages, model, api_key, max_tokens, temperature, t0):
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    # Separate system message from conversation
    system = ""
    conv = []
    for m in messages:
        if m.get("role") == "system":
            system = m.get("content", "")
        else:
            conv.append(m)
    kwargs: dict = dict(model=model, messages=conv, max_tokens=max_tokens)
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    output = resp.content[0].text if resp.content else ""
    in_tok = resp.usage.input_tokens if resp.usage else _est_tokens(str(messages))
    out_tok = resp.usage.output_tokens if resp.usage else _est_tokens(output)
    return {"output": output, "in_tok": in_tok, "out_tok": out_tok,
            "provider": "anthropic", "model": model, "latency_ms": _latency(t0)}


def _call_groq(messages, model, api_key, max_tokens, temperature, t0):
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    output = resp.choices[0].message.content or ""
    in_tok = resp.usage.prompt_tokens if resp.usage else _est_tokens(str(messages))
    out_tok = resp.usage.completion_tokens if resp.usage else _est_tokens(output)
    return {"output": output, "in_tok": in_tok, "out_tok": out_tok,
            "provider": "groq", "model": model, "latency_ms": _latency(t0)}


def _call_gemini(messages, model, api_key, max_tokens, temperature, t0):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model)
    # Convert messages to Gemini format
    parts = "\n".join(
        f"{m.get('role','user').upper()}: {m.get('content','')}"
        for m in messages if m.get("role") != "system"
    )
    resp = gmodel.generate_content(parts)
    output = resp.text or ""
    in_tok = _est_tokens(parts)
    out_tok = _est_tokens(output)
    return {"output": output, "in_tok": in_tok, "out_tok": out_tok,
            "provider": "gemini", "model": model, "latency_ms": _latency(t0)}


def _call_mistral(messages, model, api_key, max_tokens, temperature, t0):
    from mistralai import Mistral
    client = Mistral(api_key=api_key)
    resp = client.chat.complete(
        model=model, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    output = resp.choices[0].message.content or ""
    in_tok = resp.usage.prompt_tokens if resp.usage else _est_tokens(str(messages))
    out_tok = resp.usage.completion_tokens if resp.usage else _est_tokens(output)
    return {"output": output, "in_tok": in_tok, "out_tok": out_tok,
            "provider": "mistral", "model": model, "latency_ms": _latency(t0)}


def _call_together(messages, model, api_key, max_tokens, temperature, t0):
    from together import Together
    client = Together(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    output = resp.choices[0].message.content or ""
    in_tok = resp.usage.prompt_tokens if resp.usage else _est_tokens(str(messages))
    out_tok = resp.usage.completion_tokens if resp.usage else _est_tokens(output)
    return {"output": output, "in_tok": in_tok, "out_tok": out_tok,
            "provider": "together", "model": model, "latency_ms": _latency(t0)}


def _call_cohere(messages, model, api_key, max_tokens, temperature, t0):
    import cohere
    client = cohere.Client(api_key)
    # Convert to Cohere chat format
    history = []
    last_msg = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            last_msg = content
        elif role == "assistant":
            history.append({"role": "CHATBOT", "message": content})
    resp = client.chat(
        message=last_msg,
        chat_history=history,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    output = resp.text or ""
    in_tok = _est_tokens(last_msg)
    out_tok = _est_tokens(output)
    return {"output": output, "in_tok": in_tok, "out_tok": out_tok,
            "provider": "cohere", "model": model, "latency_ms": _latency(t0)}
