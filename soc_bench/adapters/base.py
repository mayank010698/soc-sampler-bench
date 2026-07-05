"""The contract every method adapter implements.

Adding method #N means writing one file like `trust_region_soc.py` that
subclasses `SamplerAdapter` and calls `register(...)` on itself -- nothing
else in this package (targets, metrics, run_benchmark, make_leaderboard)
needs to change.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Budget:
    """Shared compute budget so methods are compared at matched cost."""
    nfe: int          # SDE/diffusion integration steps per sample
    train_steps: int  # total gradient steps (method-specific interpretation)


class SamplerAdapter(ABC):
    name: str

    @abstractmethod
    def fit(self, target, budget: Budget) -> None:
        """Train against `target` (from soc_bench.targets) within `budget`.

        Should call into the method's own training code as directly as
        possible -- this is glue, not a reimplementation.
        """

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """Return `n` terminal samples from the trained sampler."""

    def log_weight(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Optional: log importance weight of `x` under the sampler's path
        measure vs. the target, enabling ESS / log-Z estimation. Return None
        if the method doesn't expose a path likelihood.
        """
        return None
