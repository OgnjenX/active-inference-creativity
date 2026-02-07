"""Inference and expected free energy utilities for Phase 1.5."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from agent.inference_utils import EPS, entropy, normalize, softmax
from env.actions import CLIMB_OBJECT_1, CLIMB_OBJECT_2


class EfeTermsTwoObject(TypedDict):
    """Expected free energy terms for a single action."""

    G: float
    pragmatic: float
    epistemic: float
    pred_obs_yes: float


def predict_height_two_objects(
    q_height: np.ndarray,
    q_aff_1: np.ndarray,
    q_aff_2: np.ndarray,
    b_height: np.ndarray,
    action: int,
) -> np.ndarray:
    """Predict next height by marginalizing across both affordance factors."""

    pred = np.zeros_like(q_height, dtype=float)
    for aff_1_idx, aff_1_prob in enumerate(q_aff_1):
        for aff_2_idx, aff_2_prob in enumerate(q_aff_2):
            pred += aff_1_prob * aff_2_prob * (b_height[action, aff_1_idx, aff_2_idx] @ q_height)
    return normalize(pred)


def _aff_likelihood(
    obs_idx: int,
    action: int,
    prior_height: np.ndarray,
    a: np.ndarray,
    b_height: np.ndarray,
    aff_1_idx: int,
    aff_2_idx: int,
) -> float:
    pred_h = b_height[action, aff_1_idx, aff_2_idx] @ prior_height
    return float(np.sum(a[obs_idx, :] * pred_h))


def infer_posterior_two_objects(
    obs_idx: int,
    action: int,
    prior_height: np.ndarray,
    prior_aff_1: np.ndarray,
    prior_aff_2: np.ndarray,
    a: np.ndarray,
    b_height: np.ndarray,
) -> dict:
    """Update height and object-indexed affordance beliefs from an observation.

    Affordances are modeled as static latent causes. Actions only control the
    height-state transition, while observations update beliefs over both
    affordance hypotheses.
    """

    assert len(prior_aff_1) == 3 and len(prior_aff_2) == 3, "Phase 1.5 assumes 3 affordance states."

    q_height = normalize(a[obs_idx, :] * prior_height)
    q_aff_1 = prior_aff_1.copy()
    q_aff_2 = prior_aff_2.copy()

    if action == CLIMB_OBJECT_1:
        like_1 = np.zeros_like(prior_aff_1)
        for aff_1_idx in range(len(prior_aff_1)):
            acc = 0.0
            for aff_2_idx, prob_aff_2 in enumerate(prior_aff_2):
                acc += prob_aff_2 * _aff_likelihood(
                    obs_idx,
                    action,
                    prior_height,
                    a,
                    b_height,
                    aff_1_idx,
                    aff_2_idx,
                )
            like_1[aff_1_idx] = acc
        q_aff_1 = normalize(prior_aff_1 * like_1)

    if action == CLIMB_OBJECT_2:
        like_2 = np.zeros_like(prior_aff_2)
        for aff_2_idx in range(len(prior_aff_2)):
            acc = 0.0
            for aff_1_idx, prob_aff_1 in enumerate(prior_aff_1):
                acc += prob_aff_1 * _aff_likelihood(
                    obs_idx,
                    action,
                    prior_height,
                    a,
                    b_height,
                    aff_1_idx,
                    aff_2_idx,
                )
            like_2[aff_2_idx] = acc
        q_aff_2 = normalize(prior_aff_2 * like_2)

    return {
        "q_height": q_height,
        "q_aff_1": q_aff_1,
        "q_aff_2": q_aff_2,
    }


def expected_free_energy_terms_two_objects(
    q_height: np.ndarray,
    q_aff_1: np.ndarray,
    q_aff_2: np.ndarray,
    action: int,
    a: np.ndarray,
    b_height: np.ndarray,
    c: np.ndarray,
) -> EfeTermsTwoObject:
    """Compute one-step EFE and decompose it into pragmatic and epistemic terms.

    Early in episodes, uncertainty over `q_aff_1` and `q_aff_2` drives the
    epistemic component. As beliefs collapse, pragmatic preference for
    `can_reach=yes` dominates action selection.
    """

    pred_h = predict_height_two_objects(
        q_height=q_height,
        q_aff_1=q_aff_1,
        q_aff_2=q_aff_2,
        b_height=b_height,
        action=action,
    )
    q_obs = normalize(a @ pred_h)

    pragmatic = float(-np.sum(q_obs * np.log(np.clip(c, EPS, 1.0))))

    prior_entropy = entropy(q_aff_1) + entropy(q_aff_2)
    expected_post_entropy = 0.0
    for obs_idx, p_obs in enumerate(q_obs):
        post = infer_posterior_two_objects(
            obs_idx=obs_idx,
            action=action,
            prior_height=pred_h,
            prior_aff_1=q_aff_1,
            prior_aff_2=q_aff_2,
            a=a,
            b_height=b_height,
        )
        expected_post_entropy += p_obs * (entropy(post["q_aff_1"]) + entropy(post["q_aff_2"]))

    epistemic = float(prior_entropy - expected_post_entropy)
    g_value = pragmatic - epistemic

    return {
        "G": g_value,
        "pragmatic": pragmatic,
        "epistemic": epistemic,
        "pred_obs_yes": float(q_obs[1]),
    }


__all__ = [
    "EfeTermsTwoObject",
    "expected_free_energy_terms_two_objects",
    "infer_posterior_two_objects",
    "normalize",
    "predict_height_two_objects",
    "softmax",
]
