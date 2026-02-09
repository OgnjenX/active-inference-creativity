"""World-definition loader utilities for Phase 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from env_phase3.objects import EmergentObject
from env_phase3.world import EmergentAffordanceWorld


def load_world_definition(path: str | Path) -> Dict[str, Any]:
    """Load a world definition from JSON and perform schema checks."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required_top_level = {"name", "target_height", "objects"}
    missing = required_top_level.difference(data.keys())
    if missing:
        raise ValueError(f"Missing required world fields: {sorted(missing)}")

    objects = data["objects"]
    if not isinstance(objects, list) or len(objects) < 2:
        raise ValueError("World definition must contain at least two objects.")

    required_object_fields = {"object_id", "name", "success_prob", "gain_prob"}
    for obj in objects:
        obj_missing = required_object_fields.difference(obj.keys())
        if obj_missing:
            raise ValueError(f"Object entry missing fields: {sorted(obj_missing)}")
    return data


def build_world_from_definition(world_def: Dict[str, Any], seed: int = 0) -> EmergentAffordanceWorld:
    """Instantiate a Phase 3 world from validated JSON definition."""
    objects: List[EmergentObject] = []
    for row in world_def["objects"]:
        objects.append(
            EmergentObject(
                object_id=int(row["object_id"]),
                name=str(row["name"]),
                success_prob=float(row["success_prob"]),
                gain_prob=float(row["gain_prob"]),
            )
        )

    return EmergentAffordanceWorld(
        objects=objects,
        target_height=int(world_def["target_height"]),
        seed=seed,
        resample_assignment_each_episode=bool(world_def.get("resample_assignment_each_episode", True)),
        randomize_outcomes_each_step=bool(world_def.get("randomize_outcomes_each_step", False)),
    )


def load_world(path: str | Path, seed: int = 0) -> EmergentAffordanceWorld:
    """Convenience wrapper for loading and instantiating a world from JSON."""
    definition = load_world_definition(path)
    return build_world_from_definition(definition, seed=seed)
