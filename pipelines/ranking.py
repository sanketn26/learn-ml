"""One ranking policy for every list: score descending, then user_id ascending.

Tree models give thousands of customers the *same* score (every "signed up,
logged one event, never came back" account lands in one leaf). Cut a list
inside that plateau with `np.argsort(-scores)` and which customers make it
— and how many of them churn — depends on the sort algorithm, not the
model. Every list, metric, and bootstrap in the course ranks through here,
so the same scores always pick the same people.
"""

from __future__ import annotations

import numpy as np


def rank_order(scores, user_ids) -> np.ndarray:
    """Row positions, best first: highest score, ties broken by user_id."""
    scores = np.asarray(scores, dtype=float)
    return np.lexsort((np.asarray(user_ids), -scores))


def top_k(scores, user_ids, k: int) -> np.ndarray:
    """Positions of the k rows a capacity-k list ships."""
    return rank_order(scores, user_ids)[: min(k, len(scores))]


def precision_at_k(y, scores, user_ids, k: int) -> float:
    idx = top_k(scores, user_ids, k)
    return float(np.asarray(y)[idx].mean()) if len(idx) else 0.0
