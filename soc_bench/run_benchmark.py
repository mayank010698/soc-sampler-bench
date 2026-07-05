"""Run one (method, target) benchmark cell and write a result artifact.

Example
-------
    python -m soc_bench.run_benchmark --method trust_region_soc \
        --target gmm --dim 50 --nfe 50 --train-steps 60000
"""
import argparse
import json
import time
from pathlib import Path

from soc_bench.adapters import get_adapter_cls
from soc_bench.adapters.base import Budget
from soc_bench.metrics import compute_metrics
from soc_bench.targets import make_target

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run(method: str, target_name: str, dim: int, nfe: int, train_steps: int,
        seed: int = 0, n_eval_samples: int = 2000, device: str = "cpu",
        **method_kwargs) -> dict:
    target = make_target(target_name, dim=dim, seed=seed, device=device)
    adapter_cls = get_adapter_cls(method)
    adapter = adapter_cls(target_name=target_name, dim=dim, seed=seed,
                          device=device, **method_kwargs)

    budget = Budget(nfe=nfe, train_steps=train_steps)
    t0 = time.time()
    adapter.fit(target, budget)
    train_wallclock = time.time() - t0

    samples = adapter.sample(n_eval_samples)
    log_w = adapter.log_weight(samples)
    metrics = compute_metrics(samples, target, log_w)

    return {
        "method": method, "target": target_name, "dim": dim,
        "nfe": nfe, "train_steps": train_steps, "seed": seed,
        "train_wallclock_s": train_wallclock, **metrics,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--nfe", type=int, default=50)
    p.add_argument("--train-steps", type=int, default=60000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval-samples", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--loss", default=None,
                    help="method-specific, e.g. trust_region_soc's loss variant")
    args = p.parse_args()

    method_kwargs = {}
    if args.loss:
        method_kwargs["loss"] = args.loss

    result = run(args.method, args.target, args.dim, args.nfe, args.train_steps,
                 args.seed, args.n_eval_samples, args.device, **method_kwargs)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.method}__{args.target}{args.dim}d__seed{args.seed}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
