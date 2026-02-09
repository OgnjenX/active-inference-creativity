"""Object schema for Phase 3 environments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmergentObject:
    """Hidden object parameters; no explicit affordance labels are exposed."""

    object_id: int
    name: str
    success_prob: float
    gain_prob: float
