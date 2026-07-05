# soc-sampler-bench

A common benchmark for comparing **stochastic-optimal-control (SOC) based samplers** — methods that train a neural controller to steer a diffusion/SDE so its terminal law matches an unnormalized target. Companion project to [awesome-soc-samplers](https://github.com/mayank010698/awesome-adjoint-matching).

## Design

Every paper in this space has its *own* training algorithm (adjoint ODEs, proximal-point sequences, trust-region sub-problems, natural gradient...) — those are the actual research contributions and shouldn't be reimplemented here. What *is* shared across papers is the **target** (the density being sampled) and the **metrics** used to judge sample quality. So the benchmark is built like a Gym/bsuite environment, not like a model zoo:

- `soc_bench/targets/` — one shared `Target` interface (`log_prob`, `score`, `sample`, `dim`), currently sourced from the pinned TrustRegionSOC repo (`gmm`, `gmm40`).
- `soc_bench/metrics/` — Sinkhorn distance, mode discrepancy, ESS, log-Z estimate. Shared across all methods.
- `soc_bench/adapters/` — one thin file per method. Each adapter's `fit()` calls straight into that paper's own pinned upstream repo; `sample()` returns terminal samples. This is the only method-specific code, and it's small (~80 lines for `trust_region_soc.py`).
- `soc_bench/run_benchmark.py` / `make_leaderboard.py` — orchestration; never touched when a new method is added.

See [docs/ADDING_A_METHOD.md](docs/ADDING_A_METHOD.md) for the exact recipe to plug in method #2.

## Quickstart

```bash
pip install -r requirements.txt
scripts/fetch_method.sh trust_region_soc   # clones the pinned upstream commit into external/

python -m soc_bench.run_benchmark \
    --method trust_region_soc --target gmm --dim 50 --nfe 50 --train-steps 60000

python -m soc_bench.make_leaderboard   # regenerates LEADERBOARD.md from results/*.json
```

Each run writes one `results/<method>__<target><dim>d__seed<seed>.json`. Nothing is ever hand-edited into a table — `make_leaderboard.py` is the only thing that writes `LEADERBOARD.md`.

## Status

- [x] Harness (targets, metrics, adapter contract, registry, leaderboard) proven end-to-end.
- [x] `trust_region_soc` adapter (gmm / gmm40 targets).
- [ ] Additional targets (Funnel, ManyWell, DW4, LJ13/55 — vendor from PDNS).
- [ ] Additional method adapters (PDNS, Adjoint Sampling, Fisher Adjoint Matching, ...).
