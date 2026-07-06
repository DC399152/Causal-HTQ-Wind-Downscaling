"""Height-time tokenization interface."""

from __future__ import annotations


class HeightTimeTokenizer:
    """Placeholder tokenizer for hourly profiles.

    Future implementation will map [B, L, H, C] profiles to Transformer tokens.
    """

    def __call__(self, x):
        return x

