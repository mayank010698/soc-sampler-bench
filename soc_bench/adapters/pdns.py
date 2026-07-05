"""Adapter for PDNS (Proximal Diffusion Neural Sampler).

Guo, Choi, Zhu, Tao & Chen. "Proximal Diffusion Neural Sampler." ICLR 2026.
https://arxiv.org/abs/2510.03824

PDNS's own `continuous/train.py` is a `@hydra.main`-decorated script with the whole
training orchestration (instantiate energy/source/sde/controller, then the epoch loop
calling PDNS's own `train_one_epoch`) inlined in one function -- there's no packaged
`Learner` class the way TrustRegionSOC has `SocLearner`/`SocLearnerBuffer`. This adapter
composes `cfg` with Hydra exactly the way `trust_region_soc.py` does, then calls that
`main` function's raw, undecorated body directly via its `__wrapped__` attribute (the
function `@hydra.main` wraps via `functools.wraps`). This bypasses only Hydra's
CLI/argv-parsing shell and its automatic output-directory chdir -- every actual
computation (proximal WDCE loss, buffer/matcher, scheduler) still runs as PDNS's own
code, unmodified.

Because `main` doesn't return the trained objects, `sample()` rebuilds the identical
controller/ref_sde/source (same cfg -> same architecture, freshly-random-initialized
weights) and immediately overwrites those weights from the checkpoint `main` itself
saved via `train_utils.save(...)` -- so the "rebuild" only recovers handles, not the
actual trained parameters.

Requires PDNS's own dependencies on top of this repo's `requirements.txt`: wandb,
termcolor, tqdm, torchmetrics, pot, ipdb, and mace-torch (a transitive import of
`src/eval_loop.py`, pulled in even for non-molecule targets because `train.py` imports
`eval_epoch` unconditionally at module load time).

`continuous/train.py` originally hardcoded `device = "cuda"` with no CPU/MPS fallback;
the vendored copy in `external/PDNS` has one line patched
(`device = "cuda" if torch.cuda.is_available() else "cpu"`) so it also runs on CPU-only
machines for smoke tests. This is the one intentional deviation from the pinned commit --
everything else in the vendored tree is untouched. `PDNSAdapter`'s own device handling
mirrors the same auto-detect logic so the adapter and PDNS's internal `main()` always
agree on which device is in use.
"""
import os
import sys
import tempfile
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from soc_bench.adapters import register
from soc_bench.adapters.base import Budget, SamplerAdapter
from soc_bench.vendor_paths import vendored_path

_METHOD = "pdns"

# Set by PDNSAdapter.fit()/`_rebuild_components()` right before calling into PDNS's own
# `hydra.utils.instantiate(cfg.energy, ...)`; read by HarnessEnergy.__init__ below. The
# Hydra config can't carry a live Python object, so the shared `target` is threaded
# through this module-level slot instead -- the standard way to inject an externally
# constructed object into a hydra-instantiate-based training script.
_ACTIVE_TARGET = None


def _repo_dir() -> Path:
    return vendored_path(_METHOD) / "continuous"


def _ensure_on_path() -> Path:
    repo = _repo_dir()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


class HarnessEnergy:
    """Wraps a `soc_bench.targets` Target as PDNS's own `BaseEnergy` interface.

    PDNS's `BaseEnergy` (src/energies/base_energy_function.py) exposes `eval(x)` (= E(x))
    and a `score(x)` that defaults to `-grad_E(x)` via autograd. The harness target
    already has an analytic `.score`, so it's used directly here instead of going through
    PDNS's autograd default -- the same shortcut PDNS's own `dist_energy.DistEnergy`
    takes for its built-in targets.
    """

    def __init__(self, device: str = "cuda", **_ignored):
        # `_ignored` absorbs any other keys Hydra forwards from the `energy` config node
        # (left over from the schema PDNS's own DistEnergy would have consumed) -- this
        # shim only needs the shared target itself.
        if _ACTIVE_TARGET is None:
            raise RuntimeError(
                "HarnessEnergy instantiated with no active target set. "
                "soc_bench.adapters.pdns._ACTIVE_TARGET must be set before calling into "
                "PDNS's training entrypoint (see PDNSAdapter.fit())."
            )
        self.target = _ACTIVE_TARGET
        self.name = "harness_target"
        self.dim = self.target.dim
        self.device = device

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        return -self.target.log_prob(x.to(self.device))

    def grad_E(self, x: torch.Tensor) -> torch.Tensor:
        return -self.score(x)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self.target.score(x.to(self.device))

    def sample(self, shape) -> torch.Tensor:
        # PDNS's own `eval_loop.toy_evaluation` gates its whole ground-truth-comparison
        # branch on `hasattr(energy, "sample")` -- without this, `eval_dict` never gets
        # assigned and `main()`'s periodic eval call raises UnboundLocalError. `shape` is
        # called as `energy.sample((B,))` by that code, matching PDNS's own targets'
        # `dist_utils` convention.
        n = shape[0] if isinstance(shape, (tuple, list)) else int(shape)
        return self.target.sample(n)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        # Also called unconditionally by `toy_evaluation`. The harness target has no
        # normalization concept (matches `BaseEnergyFunction.unnormalize`'s own
        # identity-passthrough default when normalization bounds aren't set).
        return x

    def __call__(self, x: torch.Tensor):
        return {"forces": self.grad_E(x)}


def _compose_cfg(dim, nfe, train_steps, seed):
    repo = _ensure_on_path()
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base="1.1"):
        return compose(config_name="train.yaml", overrides=[
            "experiment=gmm40",
            f"dim={dim}",
            f"nfe={nfe}",
            f"seed={seed}",
            f"num_epochs={train_steps}",
            "energy._target_=soc_bench.adapters.pdns.HarnessEnergy",
            "use_wandb=False",
            "dryrun=False",
            # PDNS's own defaults assume num_epochs in the thousands (eval/save every
            # num_epochs_per_stage); for small smoke-test budgets, force at least one
            # eval/save so a checkpoint exists for sample() to load.
            f"eval_freq={max(min(train_steps // 2, 500), 1)}",
            f"save_freq={max(min(train_steps // 2, 500), 1)}",
        ])


@register(_METHOD)
class PDNSAdapter(SamplerAdapter):
    name = _METHOD

    def __init__(self, target_name: str, dim: int, seed: int = 0, device: str = "cuda"):
        # If "cuda" was requested but isn't actually available, fall back to CPU rather
        # than failing later -- mirrors the patch applied to PDNS's own train.py so the
        # adapter and PDNS's internal `main()` always agree on which device is in use.
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        if target_name not in ("gmm", "gmm40"):
            raise ValueError(
                f"pdns adapter is only wired up for 'gmm'/'gmm40', got '{target_name}'."
            )
        self.target_name = target_name
        self.dim = dim
        self.seed = seed
        self.device = device
        self._cfg = None
        self._sde = None
        self._source = None

    def _rebuild_components(self, cfg):
        """Recreate the same controller/ref_sde/source `main()` built internally.

        Needed because `main` has no return value -- the architecture is fully
        determined by `cfg` (identical to what was actually trained), but the weights
        are freshly random here and get overwritten from the checkpoint right after.
        Must also match `main()`'s own `EMA` wrapping exactly (line
        `if cfg.ema_decay > 0: controller = EMA(controller, cfg.ema_decay)`), since
        that's what's actually inside the saved checkpoint's state dict.
        """
        import hydra
        from src.utils.ema import EMA

        energy = hydra.utils.instantiate(cfg.energy, device=self.device)
        source = hydra.utils.instantiate(cfg.source, device=self.device)
        ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(self.device)
        controller = hydra.utils.instantiate(
            cfg.controller, score=energy.score, prior_score=source.score
        ).to(self.device)
        if cfg.ema_decay > 0:
            controller = EMA(controller, cfg.ema_decay)
        return source, ref_sde, controller

    def fit(self, target, budget: Budget) -> None:
        global _ACTIVE_TARGET
        _ensure_on_path()
        cfg = _compose_cfg(self.dim, budget.nfe, budget.train_steps, self.seed)
        self._cfg = cfg

        # PDNS's `main` writes config.yaml/env.json/checkpoints/ relative to the process
        # CWD (Hydra's own automatic per-run output-dir chdir doesn't happen, since we
        # call the raw __wrapped__ function instead of going through Hydra's CLI runner).
        # Isolate each run in its own temp directory instead of scribbling into the repo.
        run_dir = tempfile.mkdtemp(prefix="pdns_run_")
        prev_cwd = os.getcwd()
        os.chdir(run_dir)
        try:
            import train as pdns_train  # PDNS's own continuous/train.py

            _ACTIVE_TARGET = target
            try:
                raw_main = getattr(pdns_train.main, "__wrapped__", pdns_train.main)
                raw_main(cfg)

                source, ref_sde, controller = self._rebuild_components(cfg)
                ckpt_path = Path(cfg.checkpoint or "checkpoints/checkpoint_latest.pt")
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                controller.load_state_dict(checkpoint["controller"])
            finally:
                _ACTIVE_TARGET = None
        finally:
            os.chdir(prev_cwd)

        from src.components.sdes import ControlledSDE
        self._source = source
        self._sde = ControlledSDE(ref_sde, controller, cfg.param_type).to(self.device)

    def sample(self, n: int) -> torch.Tensor:
        if self._sde is None:
            raise RuntimeError("call fit() before sample()")
        from src.components.sdes import sdeint
        from src.utils.common import get_timesteps

        with torch.no_grad():
            x0 = self._source.sample([n]).to(self.device)
            timesteps = get_timesteps(**self._cfg.timesteps).to(x0)
            _, x_term = sdeint(
                self._sde, x0, timesteps,
                zero_last_step_noise=self._cfg.zero_last_step_noise,
                only_boundary=True,
            )
        return x_term.detach()
