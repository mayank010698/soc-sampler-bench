"""Adapter for Fisher Adjoint Matching (FAM).

Shrivastava, Nagda, Deb & Banerjee. "Fisher Adjoint Matching: Natural Gradients for
Stochastic Optimal Control." https://mayank010698.github.io/fam.pdf

FAM's own `GMM/run_ngd.py` is a `@hydra.main` script with the whole training loop
inlined (sample trajectories -> lean-adjoint solve -> linearized loss/grad ->
`natural_gradient_step`) -- there's no packaged `Learner` class the way TrustRegionSOC
has `SocLearner`. This adapter reassembles that same loop directly from FAM's own
already-importable building blocks (`SOC.process.GMM`, `SOC.scheduler.NoiseScheduler`,
`SOC.controller.NeuralSDE`, `SOC.utils.adjoint_solve_terminal`,
`ngd.natural_gradient_step`) -- dropping only the wandb logging and plotting side
effects `run_ngd.py` also does. Every actual training computation is still FAM's own
code, called from this loop instead of theirs.

Unlike TrustRegionSOC's/PDNS's target abstractions, FAM's `SOC.process.GMM` takes
explicit Gaussian-mixture tensors (`gmm_means`, `gmm_covs`, `mixing_weights`) rather than
a generic score-function interface, because it computes an analytic score/adjoint
internally. The harness's shared `gmm`/`gmm40` target (sourced from TrustRegionSOC,
diagonal-covariance) is translated into that exact tensor format below -- not
regenerated or approximated.

FAM's GMM process doesn't track a Girsanov/importance-weight quantity the way
TrustRegionSOC's learner does, so `log_weight` returns `None`: ESS / log-Z metrics will
be blank for this adapter's leaderboard rows while Sinkhorn / mode-discrepancy still
populate.
"""
import sys

import torch

from soc_bench.adapters import register
from soc_bench.adapters.base import Budget, SamplerAdapter
from soc_bench.vendor_paths import vendored_path

_METHOD = "fam"


def _ensure_on_path():
    repo = vendored_path(_METHOD)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


@register(_METHOD)
class FAMAdapter(SamplerAdapter):
    name = _METHOD

    # FAM's own GMM/configs/gmm.yaml defaults, kept identical here so results are
    # comparable to the paper's own reported numbers wherever this adapter doesn't
    # override something via Budget.
    _DEFAULTS = {
        "init_mean": 0.0, "init_std": 0.5, "sigma": 1.0, "ld": 1.0,
        "hidden_dim": 256, "num_layers": 6, "ntrajs": 1000,
        "max_kl": 1e-2, "cg_iters": 20, "damping": 1e-3, "cg_tol": 1e-10, "cg_rel_tol": None,
        "backtracks": 12, "ratio_backtrack": 0.5, "accept_ratio": 1e-4,
    }

    def __init__(self, target_name: str, dim: int, seed: int = 0, device: str = "cpu",
                 **overrides):
        if target_name not in ("gmm", "gmm40"):
            raise ValueError(
                f"fam adapter is only wired up for 'gmm'/'gmm40', got '{target_name}'."
            )
        self.target_name = target_name
        self.dim = dim
        self.seed = seed
        self.device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        self.cfg = {**self._DEFAULTS, **overrides}
        self._process = None
        self._scheduler = None
        self._controller = None

    def _build_process(self, target):
        from SOC.process import GMM

        gmm_means = target.means.to(self.device)
        gmm_covs = torch.diag_embed(target.scales.to(self.device) ** 2)
        mixing_weights = target.weights.to(self.device)
        return GMM(
            gmm_means=gmm_means,
            gmm_covs=gmm_covs,
            mixing_weights=mixing_weights,
            init_mean=self.cfg["init_mean"],
            init_std=self.cfg["init_std"],
            sigma=self.cfg["sigma"],
            dim=target.dim,
            device=self.device,
        )

    def fit(self, target, budget: Budget) -> None:
        _ensure_on_path()
        torch.manual_seed(self.seed)

        from SOC.controller import NeuralSDE
        from SOC.scheduler import NoiseScheduler
        from SOC.utils import adjoint_solve_terminal
        from ngd import natural_gradient_step

        process = self._build_process(target)
        scheduler = NoiseScheduler(K=budget.nfe, device=self.device)
        controller = NeuralSDE(
            input_dim=target.dim,
            hidden_dim=self.cfg["hidden_dim"],
            num_layers=self.cfg["num_layers"],
            K=budget.nfe,
            device=self.device,
        )

        ntrajs = self.cfg["ntrajs"]
        sigma, ld = self.cfg["sigma"], self.cfg["ld"]

        for it in range(budget.train_steps):
            x_traj, _controls_it = process.stochastic_trajectories(
                n_trajs=ntrajs, ts=scheduler.ts, controller=controller
            )
            a_traj = adjoint_solve_terminal(process=process, x_traj=x_traj, scheduler=scheduler)

            B = x_traj.shape[1]
            x_batch = x_traj.reshape(-1, target.dim)
            a_batch = a_traj.reshape(-1, target.dim)
            t_batch = scheduler.ts.clone()[:, None].expand(budget.nfe + 1, B).reshape(-1, 1)

            controller.net.train()
            preds = controller.net(x_batch, t_batch)
            loss = (preds + a_batch * sigma / ld).square().mean()

            controller.net.zero_grad(set_to_none=True)
            loss.backward()
            grads = torch.cat(
                [p.grad.flatten().detach().clone() for p in controller.net.parameters()], dim=0
            )
            controller.net.zero_grad(set_to_none=True)

            natural_gradient_step(
                it=it, model=controller.net,
                x_batch=x_batch, t_batch=t_batch, a_batch=a_batch, grads=grads,
                max_kl=self.cfg["max_kl"], cg_iters=self.cfg["cg_iters"],
                damping=self.cfg["damping"], cg_tol=self.cfg["cg_tol"],
                cg_rel_tol=self.cfg["cg_rel_tol"], eps_step=self.cfg["max_kl"],
                sigma=sigma, ld=ld, accept_ratio=self.cfg["accept_ratio"],
                backtrack=self.cfg["ratio_backtrack"], max_backtracks=self.cfg["backtracks"],
                log_cg_residuals=False,
            )

        self._process = process
        self._scheduler = scheduler
        self._controller = controller

    def sample(self, n: int) -> torch.Tensor:
        if self._controller is None:
            raise RuntimeError("call fit() before sample()")
        with torch.no_grad():
            # `stochastic_trajectories` returns `(x_trajs, controls)` whenever a
            # controller is passed (see SOC/process.py) -- must unpack, not just index
            # the return value, or you silently get the controls tensor instead.
            x_traj, _controls = self._process.stochastic_trajectories(
                n_trajs=n, ts=self._scheduler.ts, controller=self._controller, mode="eval"
            )
        return x_traj[-1].detach()
