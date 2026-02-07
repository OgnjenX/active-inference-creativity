"""Generative model definitions (A, B, C, D) for Phase 1 agent."""

from __future__ import annotations

import numpy as np

from env.actions import CLIMB, DO_NOTHING
from env.observations import CAN_REACH_YES

HEIGHT_LOW = 0
HEIGHT_HIGH = 1

AFF_UNKNOWN = 0
AFF_CLIMBABLE = 1
AFF_NOT_CLIMBABLE = 2


def _normalize(vec: np.ndarray) -> np.ndarray:
    s = vec.sum()
    if s <= 0:
        return np.ones_like(vec) / len(vec)
    return vec / s


def build_phase1_model() -> dict:
    """Builds minimal model in PyMDP-style A/B/C/D containers."""

    # Observation model with shape (obs, height_state).
    a_matrix = np.array(
        [
            [0.95, 0.05],  # can_reach - no
            [0.05, 0.95],  # can_reach - yes
        ],
        dtype=float,
    )

    # Height transition model: (action, affordance, next_height, current_height).
    b_height = np.zeros((2, 3, 2, 2), dtype=float)

    # do_nothing: mostly keep current height
    for aff in range(3):
        b_height[DO_NOTHING, aff] = np.array(
            [
                [0.95, 0.05],
                [0.05, 0.95],
            ],
            dtype=float,
        )

    # climb transitions conditioned on affordance belief.
    b_height[CLIMB, AFF_UNKNOWN] = np.array(
        [
            [0.55, 0.05],
            [0.45, 0.95],
        ],
        dtype=float,
    )
    b_height[CLIMB, AFF_CLIMBABLE] = np.array(
        [
            [0.10, 0.05],
            [0.90, 0.95],
        ],
        dtype=float,
    )
    b_height[CLIMB, AFF_NOT_CLIMBABLE] = np.array(
        [
            [0.95, 0.05],
            [0.05, 0.95],
        ],
        dtype=float,
    )

    # B_aff[next_aff, current_aff] (static latent affordance)
    b_aff = np.eye(3, dtype=float)

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
