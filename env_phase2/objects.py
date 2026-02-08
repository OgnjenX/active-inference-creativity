"""Hidden primitive factor assignments used by Phase 2 worlds."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrimitiveObject:
    """Internal object representation never directly observed by the agent.

    Objects reference reusable latent factors instead of storing direct
    outcome parameters. This makes causal structure independent of identity.
    """

    object_id: int
    name: str
    height_factor: str
    stability_factor: str
    stackability_factor: str
