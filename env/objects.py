"""Object model for the world.

The world uses object height internally. This value is never exposed directly
through observations.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldObject:
    """Single object with internal physical attributes."""

    name: str
    height: int
