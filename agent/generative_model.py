"""Generative model definitions (A, B, C, D) for Phase 1 agent."""

from __future__ import annotations

import numpy as np

from env.actions import CLIMB, CLIMB_OBJECT_1, CLIMB_OBJECT_2, DO_NOTHING
from env.observations import CAN_REACH_YES

HEIGHT_LOW = 0
HEIGHT_HIGH = 1
N_HEIGHT_STATES = 2

AFF_UNKNOWN = 0
AFF_CLIMBABLE = 1
AFF_NOT_CLIMBABLE = 2
N_AFF_STATES = 3


def _normalize(vec: np.ndarray) -> np.ndarray:
    s = vec.sum()
    if s <= 0:
        return np.ones_like(vec) / len(vec)
    return vec / s


def _normalize_along_next_state(arr: np.ndarray) -> np.ndarray:
    """Normalize transition probabilities along next-state axis.

    Transition tensors in this project always use the last two axes as:
    `(..., next_state, current_state)`.
    """

    denom = arr.sum(axis=-2, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return arr / denom


def _build_observation_model() -> np.ndarray:
    """Observation model shared by Phase 1 and 1.5.

    Observation stays outcome-only (`can_reach`), so object identity and
    affordance remain latent causes inside the agent.
    """

    return np.array(
        [
            [0.95, 0.05],  # can_reach - no
            [0.05, 0.95],  # can_reach - yes
        ],
        dtype=float,
    )


def _stay_matrix() -> np.ndarray:
    """Transition template for do-nothing action."""

    return np.array(
        [
            [0.95, 0.05],
            [0.05, 0.95],
        ],
        dtype=float,
    )


def _affordance_transition_templates() -> dict[int, np.ndarray]:
    """Reusable transition templates keyed by affordance state."""

    unknown_matrix = np.array(
        [
            [0.55, 0.05],
            [0.45, 0.95],
        ],
        dtype=float,
    )
    climbable_matrix = np.array(
        [
            [0.10, 0.05],
            [0.90, 0.95],
        ],
        dtype=float,
    )
    not_climbable_matrix = np.array(
        [
            [0.95, 0.05],
            [0.05, 0.95],
        ],
        dtype=float,
    )
    return {
        AFF_UNKNOWN: unknown_matrix,
        AFF_CLIMBABLE: climbable_matrix,
        AFF_NOT_CLIMBABLE: not_climbable_matrix,
    }


def build_phase1_model() -> dict:
    """Builds minimal model in PyMDP-style A/B/C/D containers."""

    # Only the controllable latent state (agent height) drives observations.
    a_matrix = _build_observation_model()

    # Height transition model: (action, affordance, next_height, current_height).
    b_height = np.zeros((2, N_AFF_STATES, N_HEIGHT_STATES, N_HEIGHT_STATES), dtype=float)
    stay_matrix = _stay_matrix()
    aff_templates = _affordance_transition_templates()

    # do_nothing: mostly keep current height
    for aff in range(N_AFF_STATES):
        b_height[DO_NOTHING, aff] = stay_matrix

    # climb transitions conditioned on affordance belief.
    b_height[CLIMB, AFF_UNKNOWN] = aff_templates[AFF_UNKNOWN]
    b_height[CLIMB, AFF_CLIMBABLE] = aff_templates[AFF_CLIMBABLE]
    b_height[CLIMB, AFF_NOT_CLIMBABLE] = aff_templates[AFF_NOT_CLIMBABLE]

    # Affordance is static in Phase 1: dynamics only update beliefs, not ground truth.
    b_aff = np.eye(N_AFF_STATES, dtype=float)

    # Preferences C over observations (strongly prefer can_reach=yes).
    c_pref = np.array([0.01, 0.99], dtype=float)
    c_pref = _normalize(c_pref)

    # Initial beliefs D over hidden states.
    d_height = np.array([0.9, 0.1], dtype=float)
    d_aff = np.array([0.80, 0.10, 0.10], dtype=float)

    return {
        "A": a_matrix,
        "B_height": b_height,
        "B_aff": b_aff,
        "C": c_pref,
        "D_height": _normalize(d_height),
        "D_aff": _normalize(d_aff),
        "obs_preferred_index": CAN_REACH_YES,
    }


def build_phase1_5_model() -> dict:
    """Build model for two-object affordance disambiguation.

    Modeling assumptions:
    - only agent height is action-controllable in this phase
    - object affordances are static latent causes inferred from outcomes
    - experiments instantiate exactly one climbable object in each world
    """

    a_matrix = _build_observation_model()

    # Height transitions: (action, aff1, aff2, next_height, current_height)
    # with exactly two object-indexed affordance factors in Phase 1.5.
    b_height = np.zeros((3, N_AFF_STATES, N_AFF_STATES, N_HEIGHT_STATES, N_HEIGHT_STATES), dtype=float)
    stay_matrix = _stay_matrix()
    aff_templates = _affordance_transition_templates()

    aff_to_matrix = {
        AFF_UNKNOWN: aff_templates[AFF_UNKNOWN],
        AFF_CLIMBABLE: aff_templates[AFF_CLIMBABLE],
        AFF_NOT_CLIMBABLE: aff_templates[AFF_NOT_CLIMBABLE],
    }

    for aff_1 in range(N_AFF_STATES):
        for aff_2 in range(N_AFF_STATES):
            b_height[DO_NOTHING, aff_1, aff_2] = stay_matrix
            b_height[CLIMB_OBJECT_1, aff_1, aff_2] = aff_to_matrix[aff_1]
            b_height[CLIMB_OBJECT_2, aff_1, aff_2] = aff_to_matrix[aff_2]

    b_aff_1 = np.eye(N_AFF_STATES, dtype=float)
    b_aff_2 = np.eye(N_AFF_STATES, dtype=float)

    c_pref = np.array([0.01, 0.99], dtype=float)
    c_pref = _normalize(c_pref)

    d_height = np.array([0.9, 0.1], dtype=float)
    d_aff_1 = np.array([1.0 / N_AFF_STATES] * N_AFF_STATES, dtype=float)
    d_aff_2 = np.array([1.0 / N_AFF_STATES] * N_AFF_STATES, dtype=float)

    assert b_height.shape == (3, N_AFF_STATES, N_AFF_STATES, N_HEIGHT_STATES, N_HEIGHT_STATES)

    return {
        "A": a_matrix,
        "B_height": b_height,
        "B_aff_1": b_aff_1,
        "B_aff_2": b_aff_2,
        "C": c_pref,
        "D_height": _normalize(d_height),
        "D_aff_1": _normalize(d_aff_1),
        "D_aff_2": _normalize(d_aff_2),
        "obs_preferred_index": CAN_REACH_YES,
    }


def init_dirichlet_from_b(
    b_height: np.ndarray,
    prior_strength: float = 8.0,
) -> np.ndarray:
    """Initialize Dirichlet concentration parameters from a transition tensor.

    This is Level 1 learning: concept structure stays fixed, while transition
    reliability is learned through concentration updates.
    """

    assert prior_strength > 0.0, "Dirichlet prior strength must be positive."
    b_norm = _normalize_along_next_state(np.clip(b_height, 1e-12, None))
    return prior_strength * b_norm


def expected_b_from_alpha(alpha_height: np.ndarray) -> np.ndarray:
    """Expected transition matrix under Dirichlet posterior."""

    return _normalize_along_next_state(np.clip(alpha_height, 1e-12, None))


def kl_b_tensors(p_height: np.ndarray, q_height: np.ndarray) -> float:
    """Mean KL divergence KL(P||Q) across all transition columns."""

    p = _normalize_along_next_state(np.clip(p_height, 1e-12, 1.0))
    q = _normalize_along_next_state(np.clip(q_height, 1e-12, 1.0))
    kl = p * (np.log(p) - np.log(q))
    # Reduce over next-state axis, then average across remaining indices.
    per_column = np.sum(kl, axis=-2)
    return float(np.mean(per_column))
