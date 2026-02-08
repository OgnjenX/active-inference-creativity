"""World-definition loader utilities for Phase 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from env_phase2.objects import PrimitiveObject
from env_phase2.world import PrimitivePropertyWorld


def load_world_definition(path: str | Path) -> Dict[str, Any]:
    """Load a world definition from JSON and perform minimal schema checks."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required_top_level = {"name", "target_height", "objects"}
    missing = required_top_level.difference(data.keys())
    if missing:
        raise ValueError(f"Missing required world fields: {sorted(missing)}")

    objects = data["objects"]
    if not isinstance(objects, list) or len(objects) != 2:
        raise ValueError("World definition must contain exactly two objects.")

    required_object_fields = {
        "object_id",
        "name",
        "height_factor",
        "stability_factor",
        "stackability_factor",
    }
    for obj in objects:
        if required_object_fields.issubset(obj.keys()):
            continue

        # Backward-compatible fallback for earlier numeric property schema.
        legacy_fields = {"height_delta", "stability", "stackability"}
        if legacy_fields.issubset(obj.keys()):
            continue

        obj_missing = required_object_fields.difference(obj.keys())
        raise ValueError(f"Object entry missing fields: {sorted(obj_missing)}")

    factor_tables = data.get("factor_tables")
    if factor_tables is not None:
        required_tables = {"height", "stability", "stackability"}
        table_missing = required_tables.difference(factor_tables.keys())
        if table_missing:
            raise ValueError(f"factor_tables missing entries: {sorted(table_missing)}")
    return data


def build_world_from_definition(world_def: Dict[str, Any], seed: int = 0) -> PrimitivePropertyWorld:
    """Instantiate a Phase 2 world from a validated JSON definition."""

    factor_tables = world_def.get("factor_tables", {})
    objects: List[PrimitiveObject] = []
    for idx, row in enumerate(world_def["objects"], start=1):
        if "height_factor" in row:
            height_factor = str(row["height_factor"])
            stability_factor = str(row["stability_factor"])
            stackability_factor = str(row["stackability_factor"])
        else:
            # Legacy schema adapter. We derive deterministic factor labels for
            # each object so older files still load under the new world class.
            height_factor = f"legacy_height_{idx}"
            stability_factor = f"legacy_stability_{idx}"
            stackability_factor = f"legacy_stackability_{idx}"
            factor_tables.setdefault("height", {})[height_factor] = int(row["height_delta"])
            factor_tables.setdefault("stability", {})[stability_factor] = float(row["stability"])
            factor_tables.setdefault("stackability", {})[stackability_factor] = float(row["stackability"])

        objects.append(
            PrimitiveObject(
                object_id=int(row["object_id"]),
                name=str(row["name"]),
                height_factor=height_factor,
                stability_factor=stability_factor,
                stackability_factor=stackability_factor,
            )
        )

    return PrimitivePropertyWorld(
        objects=objects,
        factor_tables=factor_tables,
        target_height=int(world_def["target_height"]),
        seed=seed,
        resample_assignment_each_episode=bool(world_def.get("resample_assignment_each_episode", False)),
        resample_factors_independently_each_episode=bool(
            world_def.get("resample_factors_independently_each_episode", False)
        ),
        independent_observation_channels=bool(world_def.get("independent_observation_channels", False)),
    )


def load_world(path: str | Path, seed: int = 0) -> PrimitivePropertyWorld:
    """Convenience wrapper for loading and instantiating a world from JSON."""

    definition = load_world_definition(path)
    return build_world_from_definition(definition, seed=seed)
