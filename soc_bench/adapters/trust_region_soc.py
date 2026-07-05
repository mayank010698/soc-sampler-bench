"""Adapter for Trust Region SOC.

Blessing, Berner, Richter, Domingo-Enrich, Du, Vahdat & Neumann. "Trust Region
Constrained Measure Transport in Path Space for Stochastic Optimal Control and
Inference." NeurIPS 2025 (Spotlight). https://arxiv.org/abs/2508.12511

This file only translates our Budget/Target into DenisBless/TrustRegionSOC's
own Hydra config and calls its own learner classes directly in-process --
no reimplementation of the method itself.
"""
import sys

import torch
from hydra import compose, initialize_config_dir

from soc_bench.adapters import register
from soc_bench.adapters.base import Budget, SamplerAdapter
from soc_bench.vendor_paths import vendored_path

_METHOD = "trust_region_soc"


def _compose_cfg(overrides):
    repo = vendored_path(_METHOD)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        return compose(config_name="base_conf", overrides=overrides)


@register(_METHOD)
class TrustRegionSOCAdapter(SamplerAdapter):
    name = _METHOD

    def __init__(self, target_name: str, dim: int, loss: str = "buffer_tr_lv",
                 seed: int = 0, device: str = "cpu"):
        if target_name not in ("gmm", "gmm40"):
            raise ValueError(
                f"trust_region_soc only ships 'gmm'/'gmm40' targets, got '{target_name}'."
            )
        self.target_name = target_name
        self.dim = dim
        self.loss = loss
        self.seed = seed
        self.device = device
        self._learner = None

    def fit(self, target, budget: Budget) -> None:
        from trsoc.learner import SocLearner
        from trsoc.learner_buffer import SocLearnerBuffer
        from trsoc.utils.helper import set_seed

        set_seed(self.seed)
        is_buffer_loss = self.loss.startswith("buffer")
        cfg = _compose_cfg([
            f"soc_problem={self.target_name}",
            f"soc_problem.dim={self.dim}",
            f"soc_problem.seed={self.seed}",
            f"loss={self.loss}",
            f"num_steps={budget.nfe}",
            f"iters={budget.train_steps}",
            # Buffer learner's actual step count is outer_steps * inner_steps,
            # not `iters` -- tie it to the requested budget (coarse: rounds
            # down to a whole number of 400-step outer refills).
            f"sampling.outer_steps={max(budget.train_steps // 400, 1)}" if is_buffer_loss else "sampling.outer_steps=150",
            "use_wandb=False",
            "visualize_samples=False",
        ])
        # `target` was itself constructed via this same repo's `init_target`
        # (see soc_bench.targets.make_target), so it's already the native
        # type this learner expects -- passed straight through, unmodified.
        learner_cls = SocLearnerBuffer if is_buffer_loss else SocLearner
        self._learner = learner_cls(cfg, target, self.device)
        self._learner.train()

    def sample(self, n: int) -> torch.Tensor:
        if self._learner is None:
            raise RuntimeError("call fit() before sample()")
        x_t, *_ = self._learner.integrate(batch_size=n, detach=True)
        return x_t[:, -1]

    def log_weight(self, x: torch.Tensor):
        _, _, lpw_det, lpw_stoch, ltw, _ = self._learner.integrate(
            batch_size=x.shape[0], detach=True
        )
        return lpw_det + lpw_stoch + ltw
