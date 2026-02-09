"""Action definitions for Phase 3 worlds with variable object counts."""

DO_NOTHING = 0


def climb_action(object_slot: int) -> int:
    """Return action index for climbing an object slot (0-based)."""
    if object_slot < 0:
        raise ValueError("object_slot must be non-negative")
    return object_slot + 1


def action_name(action: int, num_objects: int) -> str:
    """Human-readable action name."""
    if action == DO_NOTHING:
        return "do_nothing"
    idx = action - 1
    if 0 <= idx < num_objects:
        return f"climb_object_{idx + 1}"
    return f"unknown_action_{action}"
