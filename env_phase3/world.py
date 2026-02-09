"""Phase 3 environment: hidden object causes with outcome-only observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from env_phase3.actions import DO_NOTHING
from env_phase3.objects import EmergentObject
from env_phase3.observations import CAN_REACH_NO, CAN_REACH_YES, CLIMB_FAIL, CLIMB_SUCCESS


@dataclass
class Phase3WorldState:
    """Internal state hidden from the agent."""

    agent_height: int


class EmergentAffordanceWorld:
    """Object interaction world without explicit affordance variables.

    Objects generate outcomes via hidden Bernoulli parameters. The agent only
    observes action outcomes and must infer reusable latent causes.
    """

    def __init__(
        self,
        objects: List[EmergentObject],
        target_height: int = 1,
        seed: int = 0,
        resample_assignment_each_episode: bool = True,
        randomize_outcomes_each_step: bool = False,
    ):
        if len(objects) < 2:
            raise ValueError("Phase 3 world expects at least two objects.")
        if target_height <= 0:
            raise ValueError("target_height must be positive")

        for obj in objects:
            if not 0.0 <= float(obj.success_prob) <= 1.0:
                raise ValueError("success_prob must be in [0, 1]")
            if not 0.0 <= float(obj.gain_prob) <= 1.0:
                raise ValueError("gain_prob must be in [0, 1]")

        self._objects = list(objects)
        self._target_height = int(target_height)
        self._rng = np.random.default_rng(seed)
        self._resample_assignment_each_episode = bool(resample_assignment_each_episode)
        self._randomize_outcomes_each_step = bool(randomize_outcomes_each_step)

        self._slot_assignment: List[EmergentObject] = list(self._objects)
        self._state = Phase3WorldState(agent_height=0)

    @property
    def num_objects(self) -> int:
        return len(self._slot_assignment)

    def _resample_slots(self) -> None:
        if self._resample_assignment_each_episode:
            perm = self._rng.permutation(len(self._objects))
            self._slot_assignment = [self._objects[int(i)] for i in perm]
        else:
            self._slot_assignment = list(self._objects)

    def reset(self) -> Dict[str, int]:
        """Reset world state and return initial observation."""
        self._state = Phase3WorldState(agent_height=0)
        self._resample_slots()
        return {
            "can_reach": CAN_REACH_NO,
            "climb_result": CLIMB_FAIL,
        }

    def _sample_outcome(self, obj: EmergentObject) -> tuple[int, int]:
        if self._randomize_outcomes_each_step:
            # No stable structure: object identity does not predict outcomes.
            success_prob = float(self._rng.uniform(0.15, 0.85))
            gain_prob = float(self._rng.uniform(0.15, 0.85))
        else:
            success_prob = float(obj.success_prob)
            gain_prob = float(obj.gain_prob)

        success = bool(self._rng.random() < success_prob)
        gain = bool(self._rng.random() < gain_prob) if success else False
        climb_result = CLIMB_SUCCESS if success else CLIMB_FAIL

        if gain:
            self._state.agent_height = min(self._target_height, self._state.agent_height + 1)

        can_reach = CAN_REACH_YES if self._state.agent_height >= self._target_height else CAN_REACH_NO
        return can_reach, climb_result

    def step(self, action: int) -> Tuple[Dict[str, int], bool, Dict[str, float]]:
        """Execute one action and return observation, done flag, and debug info."""
        if action == DO_NOTHING:
            can_reach = CAN_REACH_YES if self._state.agent_height >= self._target_height else CAN_REACH_NO
            climb_result = CLIMB_FAIL
            object_slot = -1
            object_id = -1
        else:
            object_slot = int(action - 1)
            if object_slot < 0 or object_slot >= len(self._slot_assignment):
                raise ValueError(f"Invalid climb action for num_objects={len(self._slot_assignment)}: {action}")
            target_obj = self._slot_assignment[object_slot]
            object_id = int(target_obj.object_id)
            can_reach, climb_result = self._sample_outcome(target_obj)

        done = can_reach == CAN_REACH_YES
        obs = {
            "can_reach": int(can_reach),
            "climb_result": int(climb_result),
        }
        info = {
            "agent_height_internal": float(self._state.agent_height),
            "target_height_internal": float(self._target_height),
            "acted_object_slot": float(object_slot),
            "acted_object_id_internal": float(object_id),
        }
        return obs, done, info
