"""Target 10min-height query construction interface."""

from __future__ import annotations


class TargetQueryBuilder:
    """Placeholder target query builder for HTQ decoding."""

    def __init__(self, target_steps: int = 6) -> None:
        self.target_steps = target_steps

    def build(self, batch_size: int, height_levels: int):
        """Return query metadata placeholder.

        Full learned query tensors are deferred to the HTQ implementation phase.
        """

        return {
            "batch_size": batch_size,
            "target_steps": self.target_steps,
            "height_levels": height_levels,
        }

