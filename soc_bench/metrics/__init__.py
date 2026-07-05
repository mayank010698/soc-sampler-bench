"""Sampling-quality metrics shared across all methods.

A method only has to supply terminal samples (`SamplerAdapter.sample`).
`log_weight` is optional and unlocks the importance-sampling metrics (ESS,
log Z estimate) on top of it -- methods that don't expose a path likelihood
just get the sample-based metrics (Sinkhorn, mode discrepancy).
"""
from typing import Optional

import torch


def effective_sample_size(log_weights: torch.Tensor) -> float:
    """Normalised reverse ESS in [0, 1] (same formula used by TrustRegionSOC's
    own evaluator: (sum w)^2 / (N * sum w^2), computed in a stable log-domain)."""
    stable = log_weights - log_weights.max()
    w = torch.exp(stable)
    ess = (w.sum() ** 2) / (log_weights.numel() * (w ** 2).sum())
    return float(ess.item())


def log_partition_estimate(log_weights: torch.Tensor) -> float:
    """Importance-weighted estimate of log Z (0 if the target is already normalised)."""
    n = log_weights.numel()
    return float((torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(float(n)))).item())


def compute_metrics(samples: torch.Tensor, target, log_weights: Optional[torch.Tensor] = None) -> dict:
    metrics = {}
    if hasattr(target, "compute_sinkhorn"):
        try:
            metrics["sinkhorn"] = target.compute_sinkhorn(samples)
        except Exception as exc:  # pragma: no cover
            metrics["sinkhorn_error"] = str(exc)
    if hasattr(target, "mode_discrepancy"):
        metrics["mode_discrepancy"] = float(target.mode_discrepancy(samples).item())
    if log_weights is not None:
        metrics["ess"] = effective_sample_size(log_weights)
        metrics["log_Z_estimate"] = log_partition_estimate(log_weights)
    return metrics
