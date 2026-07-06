"""Dataset interface for generated Causal HTQ arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SampleShapes:
    """Semantic sample shapes."""

    input_context: tuple[int, int, int]
    target_10min: tuple[int, int, int]


class WindDownscalingDataset:
    """Minimal dataset wrapper for generated artifacts.

    Expected sample semantics:
    - input: [L, H, C]
    - target: [T_out, H, C]
    """

    def __init__(self, dataset_dir: str | Path, split: str = "train") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self._samples: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._samples[index]

