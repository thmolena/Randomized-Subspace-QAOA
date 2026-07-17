"""Deterministic banks and streams of related weighted-MaxCut objectives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch

from ..circuits import RDTYPE


@dataclass(frozen=True)
class LowRankWeightModel:
    """Fixed edge-loading model shared by train and evaluation trajectories."""

    base: torch.Tensor
    loadings: torch.Tensor
    seed: int

    def __post_init__(self) -> None:
        base = torch.as_tensor(self.base, dtype=RDTYPE).detach().clone()
        loadings = torch.as_tensor(
            self.loadings, dtype=RDTYPE
        ).detach().clone()
        if base.ndim != 1 or loadings.ndim != 2:
            raise ValueError("base and loadings must be a vector and matrix")
        if loadings.shape[0] != base.numel() or loadings.shape[1] < 1:
            raise ValueError("loadings must have one row per edge")
        if not torch.isfinite(base).all() or not torch.isfinite(loadings).all():
            raise ValueError("weight-model values must be finite")
        if not torch.all(base > 0):
            raise ValueError("weight-model base must be positive")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "loadings", loadings)
        object.__setattr__(self, "seed", int(self.seed))

    @property
    def n_edges(self) -> int:
        return int(self.base.numel())

    @property
    def latent_rank(self) -> int:
        return int(self.loadings.shape[1])


@dataclass(frozen=True)
class WeightStream:
    """A recorded sequence of positive edge-weight vectors."""

    weights: torch.Tensor
    mode: str
    seed: int
    latent_rank: int
    drift_scale: float

    def __post_init__(self) -> None:
        weights = torch.as_tensor(self.weights, dtype=RDTYPE).detach().clone()
        if weights.ndim != 2 or min(weights.shape) < 1:
            raise ValueError("weights must have shape (n_tasks, n_edges)")
        if not torch.isfinite(weights).all() or not torch.all(weights > 0):
            raise ValueError("all task weights must be positive and finite")
        if int(self.seed) != self.seed:
            raise ValueError("seed must be an integer")
        if int(self.latent_rank) != self.latent_rank or self.latent_rank < 1:
            raise ValueError("latent_rank must be a positive integer")
        if not math.isfinite(self.drift_scale) or self.drift_scale < 0:
            raise ValueError("drift_scale must be finite and nonnegative")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "latent_rank", int(self.latent_rank))
        object.__setattr__(self, "drift_scale", float(self.drift_scale))

    @property
    def n_tasks(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.weights.shape[1])

    def second_moment(self, center: bool = False) -> torch.Tensor:
        values = self.weights
        if center:
            values = values - values.mean(dim=0, keepdim=True)
        return values.t() @ values / values.shape[0]

    def factor(self, center: bool = False) -> torch.Tensor:
        """Return ``L`` with ``L L^T`` equal to the empirical second moment."""
        values = self.weights
        if center:
            values = values - values.mean(dim=0, keepdim=True)
        return values.t().contiguous() / math.sqrt(values.shape[0])


def _normalize_rows(weights: torch.Tensor) -> torch.Tensor:
    return weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-12)


def make_low_rank_weight_model(n_edges: int, latent_rank: int = 3,
                               seed: int = 0) -> LowRankWeightModel:
    if int(n_edges) != n_edges or n_edges < 1:
        raise ValueError("n_edges must be a positive integer")
    if int(latent_rank) != latent_rank or latent_rank < 1:
        raise ValueError("latent_rank must be a positive integer")
    generator = torch.Generator().manual_seed(int(seed))
    q = min(int(latent_rank), int(n_edges))
    loadings = torch.randn(
        int(n_edges), q, generator=generator, dtype=RDTYPE
    )
    loadings, _ = torch.linalg.qr(loadings, mode="reduced")
    base = 0.85 + 0.30 * torch.rand(
        int(n_edges), generator=generator, dtype=RDTYPE
    )
    return LowRankWeightModel(base=base, loadings=loadings, seed=int(seed))


def sample_low_rank_drift_stream(
    model: LowRankWeightModel,
    n_edges: int,
    n_tasks: int = 8,
    drift_scale: float = 0.15,
    persistence: float = 0.85,
    seed: int = 0,
    changepoint: Optional[int] = None,
) -> WeightStream:
    """Generate a positive low-rank AR(1) edge-weight stream.

    Rows are normalized to mean one, so objective scale cannot masquerade as an
    optimization improvement. A changepoint, when supplied, reverses and
    perturbs the latent state at the indicated task.
    """
    for name, value in (("n_edges", n_edges), ("n_tasks", n_tasks)):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if int(n_edges) != model.n_edges:
        raise ValueError("n_edges must match the supplied weight model")
    if not 0 <= persistence < 1:
        raise ValueError("persistence must lie in [0, 1)")
    if not math.isfinite(drift_scale) or drift_scale < 0:
        raise ValueError("drift_scale must be finite and nonnegative")
    if changepoint is not None and not 1 <= changepoint < n_tasks:
        raise ValueError("changepoint must lie strictly inside the stream")

    generator = torch.Generator().manual_seed(int(seed))
    q = model.latent_rank
    latent = torch.zeros(q, dtype=RDTYPE)
    rows = []
    for task in range(int(n_tasks)):
        if changepoint is not None and task == changepoint:
            latent = -latent + 2.0 * drift_scale * torch.randn(
                q, generator=generator, dtype=RDTYPE
            )
        innovation = torch.randn(q, generator=generator, dtype=RDTYPE)
        latent = persistence * latent + drift_scale * innovation
        raw = model.base + 0.65 * (model.loadings @ latent)
        rows.append(raw.clamp_min(0.10))
    weights = _normalize_rows(torch.stack(rows))
    return WeightStream(
        weights=weights,
        mode="low_rank_drift" if changepoint is None else "changepoint",
        seed=int(seed),
        latent_rank=q,
        drift_scale=float(drift_scale),
    )


def low_rank_drift_stream(
    n_edges: int,
    n_tasks: int = 8,
    latent_rank: int = 3,
    drift_scale: float = 0.15,
    persistence: float = 0.85,
    seed: int = 0,
    changepoint: Optional[int] = None,
) -> WeightStream:
    """Convenience wrapper using one seed for a model and trajectory."""
    model = make_low_rank_weight_model(
        n_edges, latent_rank=latent_rank, seed=seed
    )
    return sample_low_rank_drift_stream(
        model, n_edges=n_edges, n_tasks=n_tasks,
        drift_scale=drift_scale, persistence=persistence,
        seed=seed + 1, changepoint=changepoint,
    )


def iid_weight_stream(n_edges: int, n_tasks: int = 8,
                      seed: int = 0) -> WeightStream:
    """Adverse full-rank control with independent positive weights per task."""
    if int(n_edges) != n_edges or n_edges < 1:
        raise ValueError("n_edges must be a positive integer")
    if int(n_tasks) != n_tasks or n_tasks < 1:
        raise ValueError("n_tasks must be a positive integer")
    generator = torch.Generator().manual_seed(int(seed))
    weights = torch.exp(0.45 * torch.randn(
        int(n_tasks), int(n_edges), generator=generator, dtype=RDTYPE
    ))
    return WeightStream(
        weights=_normalize_rows(weights), mode="iid", seed=int(seed),
        latent_rank=min(int(n_edges), int(n_tasks)), drift_scale=1.0,
    )
