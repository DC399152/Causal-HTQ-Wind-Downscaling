"""Trainer placeholder.

Phase 0 intentionally does not implement model training.
"""

from __future__ import annotations


class Trainer:
    """Placeholder trainer that makes the deferred boundary explicit."""

    def fit(self, *_, **__):
        raise NotImplementedError("Training is deferred beyond Phase 0.")

