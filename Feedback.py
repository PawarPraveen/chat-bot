"""
Feedback loop: recruiters mark answers correct/incorrect after each
response. Incorrect answers are persisted with corrections, and future
semantically-similar queries get warned via an injected prompt hint —
an LLM-native equivalent of in-context correction (not model retraining).
"""

import json
import time

import numpy as np

import Config as config
from Embedding import embed

_FEEDBACK_LOG: list[dict] = []
_FEEDBACK_EMBEDDINGS: np.ndarray | None = None


def load_feedback_log() -> None:
    """Read feedback_log.json (if present) and build the correction index."""
    global _FEEDBACK_LOG
    if config.FEEDBACK_PATH.exists():
        with open(config.FEEDBACK_PATH, encoding="utf-8") as f:
            _FEEDBACK_LOG = json.load(f)
    else:
        _FEEDBACK_LOG = []
    _rebuild_feedback_index()


def _rebuild_feedback_index() -> None:
    """Recompute embeddings for every logged query marked incorrect —
    those are the ones worth warning future queries about."""
    global _FEEDBACK_EMBEDDINGS

    incorrect_entries = [e for e in _FEEDBACK_LOG if not e["was_correct"]]
    if not incorrect_entries:
        _FEEDBACK_EMBEDDINGS = None
        return

    queries = [e["query"] for e in incorrect_entries]
    _FEEDBACK_EMBEDDINGS = embed(queries)


def log_feedback(query: str, answer: str, was_correct: bool, correction: str | None = None) -> None:
    """Record a recruiter's feedback and persist it to disk immediately."""
    entry = {
        "query": query,
        "answer": answer,
        "was_correct": was_correct,
        "correction": correction,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _FEEDBACK_LOG.append(entry)

    with open(config.FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(_FEEDBACK_LOG, f, indent=2)

    _rebuild_feedback_index()


def check_past_corrections(query: str) -> str | None:
    """Look up whether a semantically similar query was previously answered
    incorrectly. Returns a hint string to inject into the prompt, or None."""
    if _FEEDBACK_EMBEDDINGS is None:
        return None

    incorrect_entries = [e for e in _FEEDBACK_LOG if not e["was_correct"]]
    query_embedding = embed([query])[0]

    similarities = _FEEDBACK_EMBEDDINGS @ query_embedding
    best_idx = int(np.argmax(similarities))
    best_score = similarities[best_idx]

    if best_score < config.FEEDBACK_SIMILARITY_THRESHOLD:
        return None

    past = incorrect_entries[best_idx]
    hint = (
        f"Note: a previous, similarly-phrased question (\"{past['query']}\") "
        f"was answered incorrectly before: \"{past['answer']}\"."
    )
    if past.get("correction"):
        hint += f" The correct answer was: \"{past['correction']}\"."
    return hint


def entry_count() -> int:
    return len(_FEEDBACK_LOG)