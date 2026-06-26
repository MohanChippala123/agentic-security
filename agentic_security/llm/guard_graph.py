"""LangGraph orchestration for the AgentShield prompt-injection firewall.

Architecture (mirrors the diagram):

  START
    ├──→ hybrid_Layer  (XGBoost + All-MiniLM-L6-v2)
    └──→ ml_layer      (AgentShield Security LLM - from-scratch)
              ↓ (both merge into)
         decision_layer (weighted combination → block / allow)
              ↓
             END

State schema:
  prompt          str      - the input to analyse
  verdict         str      - "safe" | "attack"
  keep_jailbreak  float    - attack probability from hybrid layer (0-1)
  score           float    - combined risk score 0-100
  notes           str      - human-readable decision explanation
  signals         list     - per-layer signal dicts (merged from both branches)
  hybrid_result   dict     - raw hybrid layer output
  ml_result       dict     - raw ML layer output
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, TypedDict


# ── State ─────────────────────────────────────────────────────────────────────

class GuardState(TypedDict, total=False):
    prompt: str
    verdict: str
    keep_jailbreak: float
    score: float
    notes: str
    # Annotated with operator.add so both parallel branches can append
    signals: Annotated[list, operator.add]
    hybrid_result: dict
    ml_result: dict


# ── Nodes ─────────────────────────────────────────────────────────────────────

def hybrid_layer_node(state: GuardState) -> dict:
    """Node: All-MiniLM-L6-v2 embeddings → XGBoost classifier."""
    try:
        from .hybrid_layer import predict, is_available
        if not is_available():
            return {"hybrid_result": {}, "signals": []}
        result = predict(state["prompt"])
    except Exception as exc:
        result = {"verdict": "safe", "attack_probability": 0.0, "confidence": 0.5,
                  "layer": "hybrid_layer", "error": str(exc)}

    return {
        "hybrid_result": result,
        "keep_jailbreak": result.get("attack_probability", 0.0),
        "signals": [result],
    }


def ml_layer_node(state: GuardState) -> dict:
    """Node: AgentShield Security LLM (from-scratch GPT judge)."""
    try:
        from .engine import judge_message
        verdict_raw = judge_message(state["prompt"])
        # attack_similarity is 0..1 where higher = more attack-like
        sim = verdict_raw.get("attack_similarity", 0.0)
        is_attack = not verdict_raw.get("safe", True)
        result = {
            "verdict": "attack" if is_attack else "safe",
            "attack_probability": round(sim, 4),
            "confidence": round(max(sim, 1 - sim), 4),
            "layer": "ml_layer",
            "model": "AgentShield Security LLM",
            "reason": verdict_raw.get("reason", ""),
            "threat": verdict_raw.get("threat", ""),
            "latency_ms": verdict_raw.get("latency_ms", 0),
        }
    except Exception as exc:
        result = {"verdict": "safe", "attack_probability": 0.0, "confidence": 0.5,
                  "layer": "ml_layer", "error": str(exc)}

    return {"ml_result": result, "signals": [result]}


def decision_layer_node(state: GuardState) -> dict:
    """Node: weighted combination of hybrid + ML layer verdicts."""
    h = state.get("hybrid_result") or {}
    m = state.get("ml_result") or {}

    h_prob = h.get("attack_probability", 0.0)
    m_prob = m.get("attack_probability", 0.0)

    # Weighted combination (hybrid layer more accurate at 98.4% vs 92.5%)
    combined = 0.62 * h_prob + 0.38 * m_prob
    score = round(combined * 100, 1)

    # Block if either layer is confident OR combined exceeds threshold.
    # Tuned for maximum recall: we would rather block a safe request
    # than let a single attack through.
    is_attack = (
        h_prob >= 0.40     # hybrid layer: lowered for higher recall
        or m_prob >= 0.55  # ML layer: lowered for higher recall
        or combined >= 0.35  # combined: aggressive threshold
    )
    verdict = "attack" if is_attack else "safe"

    h_pct = f"{h_prob:.1%}"
    m_pct = f"{m_prob:.1%}"
    notes = (
        f"Hybrid Layer (XGBoost+MiniLM): {h_pct} attack  ·  "
        f"ML Layer (Security LLM): {m_pct} attack  →  "
        f"Combined: {combined:.1%}  →  {'BLOCK' if verdict == 'attack' else 'ALLOW'}"
    )

    return {"verdict": verdict, "score": score, "notes": notes}


# ── Graph ──────────────────────────────────────────────────────────────────────

_graph = None


def _build_graph():
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(GuardState)

    # Add all nodes (matches the diagram)
    graph.add_node("hybrid_Layer", hybrid_layer_node)
    graph.add_node("ml_layer", ml_layer_node)
    graph.add_node("decision_layer", decision_layer_node)

    # Parallel fan-out from START to both analysis layers
    graph.add_edge(START, "hybrid_Layer")
    graph.add_edge(START, "ml_layer")

    # Merge both into the decision layer
    graph.add_edge("hybrid_Layer", "decision_layer")
    graph.add_edge("ml_layer", "decision_layer")

    # End after decision
    graph.add_edge("decision_layer", END)

    return graph.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def run_firewall(prompt: str) -> GuardState:
    """Run the full LangGraph firewall on a prompt. Returns the final GuardState.

    Falls back to a minimal safe state if langgraph isn't installed.
    """
    try:
        graph = get_graph()
        initial: GuardState = {
            "prompt": prompt,
            "signals": [],
            "hybrid_result": {},
            "ml_result": {},
        }
        result = graph.invoke(initial)
        return result
    except Exception:
        # Graceful degradation: return a neutral state so the rest of the
        # analyzer can still run without the graph
        return {
            "prompt": prompt,
            "verdict": "safe",
            "keep_jailbreak": 0.0,
            "score": 0.0,
            "notes": "Guard graph unavailable - falling back to regex + LLM signals.",
            "signals": [],
            "hybrid_result": {},
            "ml_result": {},
        }


def is_available() -> bool:
    """True if langgraph is installed."""
    try:
        import langgraph  # noqa: F401
        return True
    except ImportError:
        return False
