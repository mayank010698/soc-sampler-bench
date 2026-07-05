"""Canonical target registry used by the harness.

`gmm`/`gmm40` are sourced directly from the pinned TrustRegionSOC repo, since
that's already the highest-fidelity implementation available (it's a faithful
port of the FAB-lineage targets that most SOC-sampler papers benchmark
against) and is vendored for the trust_region_soc adapter anyway.

Adding a new target family (e.g. Funnel/ManyWell/DW4 from PDNS) means adding
one branch here that sources it from wherever it's best implemented -- see
docs/ADDING_A_METHOD.md.
"""
import sys

from hydra import compose, initialize_config_dir

from soc_bench.vendor_paths import vendored_path

_SOURCE_METHOD = "trust_region_soc"  # repo that currently supplies gmm/gmm40
_KNOWN_TARGETS = ("gmm", "gmm40")


def make_target(name: str, dim: int, seed: int = 0, device: str = "cpu"):
    if name not in _KNOWN_TARGETS:
        raise KeyError(f"Unknown target '{name}'. Available: {_KNOWN_TARGETS}")

    repo = vendored_path(_SOURCE_METHOD)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from trsoc.targets import init_target

    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        cfg = compose(config_name="base_conf", overrides=[
            f"soc_problem={name}",
            f"soc_problem.dim={dim}",
            f"soc_problem.seed={seed}",
        ])
    return init_target(cfg, device)
