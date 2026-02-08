"""Phase 2 environment with hidden primitive object properties.

The world does not encode or expose explicit affordance labels. Outcomes are
computed from primitive hidden properties and intervention actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np

from env_phase2.actions import CLIMB_OBJECT_1, CLIMB_OBJECT_2, DO_NOTHING
from env_phase2.objects import PrimitiveObject
from env_phase2.observations import CAN_REACH_NO, CAN_REACH_YES, CLIMB_FAIL, CLIMB_SUCCESS


@dataclass
class Phase2WorldState:
    """Internal world state (hidden from the agent)."""

    agent_height: int


class PrimitivePropertyWorld:
    """Two-object world with probabilistic intervention outcomes.

    Each climb action selects an object and samples whether the intervention
    succeeds using that object's hidden stability. Successful interventions add
    the object's hidden height contribution to the agent state.
    """

    def __init__(
        self,
        objects: List[PrimitiveObject],
        factor_tables: Mapping[str, Mapping[str, float | int]],
        target_height: int,
        seed: int = 0,
        resample_assignment_each_episode: bool = False,
        resample_factors_independently_each_episode: bool = False,
        independent_observation_channels: bool = False,
    ):
        if len(objects) != 2:
            raise ValueError("Phase 2 world currently expects exactly two objects.")
        if target_height <= 0:
            raise ValueError("target_height must be positive.")

        self._objects = {obj.object_id: obj for obj in objects}
        expected_ids = {1, 2}
        if set(self._objects.keys()) != expected_ids:
            raise ValueError("Object ids must be exactly {1, 2}.")

        self._height_table = dict(factor_tables.get("height", {}))
        self._stability_table = dict(factor_tables.get("stability", {}))
        self._stackability_table = dict(factor_tables.get("stackability", {}))
        self._independent_observation_channels = bool(independent_observation_channels)
        if not self._height_table or not self._stability_table or not self._stackability_table:
            raise ValueError("factor_tables must include non-empty height, stability, and stackability maps.")

        for obj in objects:
            if obj.height_factor not in self._height_table:
                raise ValueError(f"Unknown height factor: {obj.height_factor}")
            if obj.stability_factor not in self._stability_table:
                raise ValueError(f"Unknown stability factor: {obj.stability_factor}")
            if obj.stackability_factor not in self._stackability_table:
                raise ValueError(f"Unknown stackability factor: {obj.stackability_factor}")

            stability = float(self._stability_table[obj.stability_factor])
            stackability = float(self._stackability_table[obj.stackability_factor])
            height_value = float(self._height_table[obj.height_factor])
            if not 0.0 <= stability <= 1.0:
                raise ValueError("Resolved object stability must be in [0, 1].")
            if not 0.0 <= stackability <= 1.0:
                raise ValueError("Resolved object stackability must be in [0, 1].")
            if self._independent_observation_channels:
                if not 0.0 <= height_value <= 1.0:
                    raise ValueError(
                        "In independent_observation_channels mode, height table values must be in [0, 1]."
                    )
            elif int(height_value) < 0:
                raise ValueError("Resolved object height gain must be non-negative.")

        self._target_height = target_height
        self._rng = np.random.default_rng(seed)
        self._resample_assignment_each_episode = bool(resample_assignment_each_episode)
        self._resample_factors_independently_each_episode = bool(
            resample_factors_independently_each_episode
        )
        self._object_pool = [self._objects[1], self._objects[2]]
        self._slot_assignment = {
            CLIMB_OBJECT_1: self._objects[1],
            CLIMB_OBJECT_2: self._objects[2],
        }
        self._state = Phase2WorldState(agent_height=0)

    def _resample_slots(self) -> None:
        if self._resample_factors_independently_each_episode:
            height_keys = list(self._height_table.keys())
            stability_keys = list(self._stability_table.keys())
            stackability_keys = list(self._stackability_table.keys())
            self._slot_assignment = {
                CLIMB_OBJECT_1: PrimitiveObject(
                    object_id=1,
                    name="episode_slot_1",
                    height_factor=str(self._rng.choice(height_keys)),
                    stability_factor=str(self._rng.choice(stability_keys)),
                    stackability_factor=str(self._rng.choice(stackability_keys)),
                ),
                CLIMB_OBJECT_2: PrimitiveObject(
                    object_id=2,
                    name="episode_slot_2",
                    height_factor=str(self._rng.choice(height_keys)),
                    stability_factor=str(self._rng.choice(stability_keys)),
                    stackability_factor=str(self._rng.choice(stackability_keys)),
                ),
            }
            return

        if not self._resample_assignment_each_episode:
            self._slot_assignment = {
                CLIMB_OBJECT_1: self._objects[1],
                CLIMB_OBJECT_2: self._objects[2],
            }
            return
        perm = self._rng.permutation(len(self._object_pool))
        self._slot_assignment = {
            CLIMB_OBJECT_1: self._object_pool[int(perm[0])],
            CLIMB_OBJECT_2: self._object_pool[int(perm[1])],
        }

    def reset(self) -> Dict[str, int]:
        """Reset world state and return outcome-only observations."""

        self._state = Phase2WorldState(agent_height=0)
        self._resample_slots()
        return {
            "can_reach": CAN_REACH_NO,
            "climb_result": CLIMB_FAIL,
        }

    def step(self, action: int) -> Tuple[Dict[str, int], bool, Dict[str, float]]:
        """Execute one intervention action and return observation tuple."""

        climb_result = CLIMB_FAIL
        can_reach = (
            CAN_REACH_YES
            if self._state.agent_height >= self._target_height
            else CAN_REACH_NO
        )
        if action in (CLIMB_OBJECT_1, CLIMB_OBJECT_2):
            target_obj = self._slot_assignment[action]
            stability = float(self._stability_table[target_obj.stability_factor])
            if self._independent_observation_channels:
                reach_prob = float(self._height_table[target_obj.height_factor])
                reach_prob = float(np.clip(reach_prob, 0.0, 1.0))
                climb_result = CLIMB_SUCCESS if bool(self._rng.random() < stability) else CLIMB_FAIL
                can_reach = CAN_REACH_YES if bool(self._rng.random() < reach_prob) else CAN_REACH_NO
            else:
                height_gain = int(self._height_table[target_obj.height_factor])
                success = bool(self._rng.random() < stability)
                if success:
                    climb_result = CLIMB_SUCCESS
                    self._state.agent_height = min(
                        self._target_height,
                        self._state.agent_height + height_gain,
                    )
                can_reach = (
                    CAN_REACH_YES
                    if self._state.agent_height >= self._target_height
                    else CAN_REACH_NO
                )
        done = can_reach == CAN_REACH_YES

        obs = {
            "can_reach": int(can_reach),
            "climb_result": int(climb_result),
        }
        info = {
            "agent_height_internal": float(self._state.agent_height),
            "target_height_internal": float(self._target_height),
            "slot_1_object_internal": float(self._slot_assignment[CLIMB_OBJECT_1].object_id),
            "slot_2_object_internal": float(self._slot_assignment[CLIMB_OBJECT_2].object_id),
        }
        return obs, done, info
