"""Inference and expected free energy utilities."""

from __future__ import annotations

from typing import Dict, TypedDict

import numpy as np


class EfeTerms(TypedDict):
    """Expected free energy terms for a single action."""

    G: float
    pragmatic: float
    epistemic: float
    pred_obs_yes: float


EPS = 1e-12


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to sum to 1, fallback to uniform if needed."""
    s = vec.sum()
    if s <= 0:
        return np.ones_like(vec) / len(vec)
    return vec / s


def entropy(prob: np.ndarray) -> float:
    """Compute Shannon entropy for a probability vector."""
    p = np.clip(prob, EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def predict_height(
    q_height: np.ndarray,
    q_aff: np.ndarray,
    b_height: np.ndarray,
    action: int,
) -> np.ndarray:
    """Predict next height by marginalizing over affordance beliefs."""

    pred = np.zeros_like(q_height, dtype=float)
    for aff_idx, aff_prob in enumerate(q_aff):
        pred += aff_prob * (b_height[action, aff_idx] @ q_height)
    return normalize(pred)


def infer_posterior(
    obs_idx: int,
    action: int,
    prior_height: np.ndarray,
    prior_aff: np.ndarray,
    a: np.ndarray,
    b_height: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Update hidden-state beliefs from an observation and selected action."""

    likelihood_h = a[obs_idx, :]
    q_height = normalize(likelihood_h * prior_height)

    q_aff = prior_aff.copy()
    if action == 1:  # climb action carries affordance evidence.
        aff_like = np.zeros_like(prior_aff)
        for aff_idx in range(len(prior_aff)):
            pred_h_aff = b_height[action, aff_idx] @ prior_height
            aff_like[aff_idx] = np.sum(a[obs_idx, :] * pred_h_aff)
        q_aff = normalize(prior_aff * aff_like)

    return {"q_height": q_height, "q_aff": q_aff}


def expected_free_energy_terms(
    q_height: np.ndarray,
    q_aff: np.ndarray,
    action: int,
    a: np.ndarray,
    b_height: np.ndarray,
    c: np.ndarray,
) -> EfeTerms:
    """Compute pragmatic risk and epistemic value for one-step policy."""

    pred_h = predict_height(q_height, q_aff, b_height, action)
    q_o = normalize(a @ pred_h)

    # Pragmatic term: expected negative log preference.
    pragmatic = float(-np.sum(q_o * np.log(np.clip(c, EPS, 1.0))))

    # Epistemic value: expected information gain about affordance.
    prior_h_aff = entropy(q_aff)
    expected_post_h = 0.0
    for obs_idx, p_obs in enumerate(q_o):
        aff_like = np.zeros_like(q_aff)
        for aff_idx in range(len(q_aff)):
            pred_h_aff = b_height[action, aff_idx] @ q_height
            aff_like[aff_idx] = np.sum(a[obs_idx, :] * pred_h_aff)
        q_aff_post = normalize(q_aff * aff_like)
        expected_post_h += p_obs * entropy(q_aff_post)

    epistemic = float(prior_h_aff - expected_post_h)
    g_value = pragmatic - epistemic

    return {
        "G": g_value,
        "pragmatic": pragmatic,
        "epistemic": epistemic,
        "pred_obs_yes": float(q_o[1]),
    }


def softmax(logits: np.ndarray, precision: float = 4.0) -> np.ndarray:
    """Compute a precision-scaled softmax distribution."""
    x = precision * logits
    x = x - np.max(x)
    e = np.exp(x)
    return normalize(e)
