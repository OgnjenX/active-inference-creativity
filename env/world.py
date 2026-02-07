"""Deterministic object-based environment for Phase 1.

The environment exposes only outcome-level observations and keeps all world
internals private to preserve separation of concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from env.actions import CLIMB, CLIMB_OBJECT_1, CLIMB_OBJECT_2
from env.objects import WorldObject
from env.observations import CAN_REACH_NO, CAN_REACH_YES


@dataclass
class WorldState:
    """Internal world state (never directly exposed to the agent)."""

    agent_height: int


class AffordanceWorld:
    """Single-agent, single-object, single-target world."""

    def __init__(self, obj: WorldObject, target_height: int = 1):
        self._object = obj
        self._target_height = target_height
        self._state = WorldState(agent_height=0)

    @property
    def object_name(self) -> str:
        """Return the object name."""
        return self._object.name

    def reset(self) -> Dict[str, int]:
        """Reset world to initial state and return initial observation."""
        self._state = WorldState(agent_height=0)
        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict[str, int], bool, Dict[str, int]]:
        """Execute action and return (observation, done, info)."""
        if action == CLIMB and self._object.height >= 1:
            self._state.agent_height = 1

        obs = self._get_observation()
        done = obs["can_reach"] == CAN_REACH_YES
        info = {
            "agent_height_internal": self._state.agent_height,
            "target_height_internal": self._target_height,
        }
        return obs, done, info

    def _get_observation(self) -> Dict[str, int]:
        can_reach = (
            CAN_REACH_YES
            if self._state.agent_height >= self._target_height
            else CAN_REACH_NO
        )
        return {"can_reach": can_reach}


class TwoObjectAffordanceWorld:
    """Two-object world where exactly one object can enable reaching the target."""

    def __init__(self, object_1: WorldObject, object_2: WorldObject, target_height: int = 1):
        self._object_1 = object_1
        self._object_2 = object_2
        self._target_height = target_height
        self._state = WorldState(agent_height=0)

    def reset(self) -> Dict[str, int]:
        """Reset world to initial state and return initial observation."""
        self._state = WorldState(agent_height=0)
        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict[str, int], bool, Dict[str, int]]:
        """Execute action and return (observation, done, info)."""
        if (action == CLIMB_OBJECT_1 and self._object_1.height >= 1) or (
            action == CLIMB_OBJECT_2 and self._object_2.height >= 1
        ):
            self._state.agent_height = 1

        obs = self._get_observation()
        done = obs["can_reach"] == CAN_REACH_YES
        info = {
            "agent_height_internal": self._state.agent_height,
            "target_height_internal": self._target_height,
        }
        return obs, done, info

    def _get_observation(self) -> Dict[str, int]:
        can_reach = (
            CAN_REACH_YES
            if self._state.agent_height >= self._target_height
            else CAN_REACH_NO
        )
        return {"can_reach": can_reach}
