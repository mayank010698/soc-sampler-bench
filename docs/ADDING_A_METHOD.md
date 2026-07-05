# Adding a new method

Adding method #N should only ever require the four steps below — nothing in
`soc_bench/run_benchmark.py`, `soc_bench/make_leaderboard.py`,
`soc_bench/targets/`, or `soc_bench/metrics/` should need to change.

## 1. Pin the upstream repo

Add an entry to `PINNED` in `soc_bench/vendor_paths.py`:

```python
"my_method": {
    "dirname": "MyMethodRepo",
    "url": "https://github.com/author/my-method-repo",
    "commit": "<exact commit hash you tested against>",
},
```

and a matching case in `scripts/fetch_method.sh`. Pin an exact commit, not a
branch — a method's reported numbers should never silently change because the
upstream repo changed underneath it.

## 2. Write one adapter file

`soc_bench/adapters/my_method.py`:

```python
from soc_bench.adapters import register
from soc_bench.adapters.base import Budget, SamplerAdapter
from soc_bench.vendor_paths import vendored_path

@register("my_method")
class MyMethodAdapter(SamplerAdapter):
    name = "my_method"

    def __init__(self, target_name: str, dim: int, seed: int = 0, device: str = "cpu"):
        ...  # store config; import the vendored repo's modules here (see
             # trust_region_soc.py for the sys.path-insert pattern)

    def fit(self, target, budget: Budget) -> None:
        ...  # call the upstream repo's own training entrypoint against `target`.
             # `target` already implements log_prob/score/sample/dim -- if the
             # method's own energy interface differs (e.g. PDNS's `BaseEnergy`
             # expects eval()=-log_prob and grad_E()=-score), write a small
             # shim class here, not a reimplementation of the target itself.

    def sample(self, n: int):
        ...  # return n terminal samples as a torch.Tensor

    def log_weight(self, x):
        ...  # optional: return log importance weight if the method exposes
             # a path likelihood (enables ESS / log-Z). Omit to skip those
             # metrics for this method -- the default returns None.
```

That's the entire contract: `fit`/`sample` required, `log_weight` optional.

## 3. (Only if needed) add a target

If the method's paper introduces or requires a target not yet in
`soc_bench/targets/` (e.g. Funnel, ManyWell, DW4, LJ13/55 — all already
implemented in PDNS's `continuous/src/energies/`), add one branch to
`make_target()` sourcing it from wherever it's best implemented, following the
same vendor-and-wrap pattern as `gmm`/`gmm40`. Don't reimplement it from the
paper's math — every target in this space traces back to an existing repo.

## 4. Run it and commit the result artifact

```bash
scripts/fetch_method.sh my_method
python -m soc_bench.run_benchmark --method my_method --target gmm --dim 50
python -m soc_bench.make_leaderboard
```

Commit the new `results/*.json` (and the regenerated `LEADERBOARD.md`) — never
hand-edit the leaderboard table.

## Known limitations to be aware of

- Two vendored repos that happen to share a top-level Python package name will
  collide once both are on `sys.path` in the same process. If that happens,
  either rename the vendored package on import (`importlib` tricks) or run
  that adapter's `fit()` out-of-process (subprocess call into the upstream
  repo's own CLI, writing samples to disk for `sample()` to read back).
- `Budget.train_steps` is interpreted per-method (e.g. TrustRegionSOC's buffer
  learner maps it onto `outer_steps * inner_steps`) since training loops
  aren't structurally identical across methods -- document the mapping in
  each adapter's docstring so budget comparisons stay honest.
