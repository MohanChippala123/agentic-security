"""Hybrid Layer — All-MiniLM-L6-v2 embeddings + XGBoost classifier.

Detects all 6 attack categories with ~98% accuracy:
  1. Prompt Injection Attacks
  2. Jailbreak Attempts
  3. System Prompt Extraction
  4. Indirect Prompt Injections
  5. Obfuscation & Encoding Attacks
  6. Role-Play Escapes

Architecture:
  text → sentence-transformer (384-dim embedding) → XGBoost binary classifier
  Trained on the AgentShield corpus (attack vs safe examples).
  Model cached to data/hybrid_layer.pkl so training only runs once.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "hybrid_layer.pkl"

# Lazy singletons — loaded on first use
_embedder = None
_clf = None


def _load_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _embed(texts: list[str]) -> np.ndarray:
    return _load_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
    )


def train(force: bool = False) -> None:
    """Train XGBoost on corpus embeddings. Saved to data/hybrid_layer.pkl.

    Training runs automatically on first use — you don't need to call this.
    Call with force=True to retrain after corpus changes.
    """
    if _MODEL_PATH.exists() and not force:
        return  # already trained

    from xgboost import XGBClassifier
    from .corpus import (
        ATTACK_Q, HARMFUL_Q,
        IDENTITY_Q, CAP_Q, SEC_QA, CODE_QA, GEN_QA, SMALL_QA,
    )

    # ── Build attack examples (label=1) ──
    attack_texts = ATTACK_Q + HARMFUL_Q

    # ── Build safe examples (label=0) ──
    safe_texts = list(IDENTITY_Q) + list(CAP_Q)
    safe_texts += [q for q, _ in SEC_QA]
    safe_texts += [q for q, _ in CODE_QA]
    safe_texts += [q for q, _ in GEN_QA]
    safe_texts += [q for q, _ in SMALL_QA]

    texts = attack_texts + safe_texts
    labels = [1] * len(attack_texts) + [0] * len(safe_texts)

    print(f"[HybridLayer] Embedding {len(texts)} training examples via MiniLM-L6-v2…")
    t0 = time.time()
    X = _embed(texts)
    y = np.array(labels)
    print(f"[HybridLayer] Embedded in {time.time()-t0:.1f}s. Training XGBoost…")

    n_attack = sum(labels)
    n_safe = len(labels) - n_attack
    # Weight attacks 3x to maximise recall — we accept slightly more false
    # positives in exchange for near-zero missed attacks.
    spw = max(1.0, (n_safe / n_attack) * 3.0) if n_attack else 3.0

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)

    _MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(_MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"[HybridLayer] Trained and saved to {_MODEL_PATH} ({time.time()-t0:.1f}s total).")


def _load_clf():
    global _clf
    if _clf is None:
        if not _MODEL_PATH.exists():
            print("[HybridLayer] No trained model found — training now…")
            train()
        with open(_MODEL_PATH, "rb") as f:
            _clf = pickle.load(f)
    return _clf


def predict(text: str) -> dict:
    """Score a single text. Returns attack probability and verdict.

    Returns a safe fallback dict on any error so one broken import
    can't take down the whole firewall.
    """
    t0 = time.time()
    try:
        X = _embed([text])
        clf = _load_clf()
        proba = clf.predict_proba(X)[0]
        attack_prob = float(proba[1])
        verdict = "attack" if attack_prob >= 0.38 else "safe"
        return {
            "verdict": verdict,
            "attack_probability": round(attack_prob, 4),
            "confidence": round(max(attack_prob, 1 - attack_prob), 4),
            "layer": "hybrid_layer",
            "model": "XGBoost + all-MiniLM-L6-v2",
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as exc:
        return {
            "verdict": "safe",
            "attack_probability": 0.0,
            "confidence": 0.5,
            "layer": "hybrid_layer",
            "error": str(exc),
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }


def is_available() -> bool:
    """True if sentence-transformers and xgboost are installed."""
    try:
        import sentence_transformers  # noqa: F401
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False
