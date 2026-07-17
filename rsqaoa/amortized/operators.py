"""Task-distribution-weighted sensitivity operators and residual probes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch

from ..circuits import MaxCutProblem, RDTYPE
from ..operator import QAOASensitivity
from ..randqb import QBResult, randqb
from .budget import ResourceLedger


@dataclass(frozen=True)
class TaskSubspace:
    Q: torch.Tensor
    B: torch.Tensor
    result: QBResult
    ledger: ResourceLedger

    @property
    def rank(self) -> int:
        return int(self.Q.shape[1])


@dataclass(frozen=True)
class ResidualEstimate:
    ratio: float
    numerator: float
    denominator: float
    ledger: ResourceLedger


def empirical_weight_factor(weights: torch.Tensor,
                            center: bool = False) -> torch.Tensor:
    values = torch.as_tensor(weights, dtype=RDTYPE)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("weights must have shape (n_tasks, n_edges)")
    if not torch.isfinite(values).all():
        raise ValueError("weights must be finite")
    if center:
        values = values - values.mean(dim=0, keepdim=True)
    return values.t().contiguous() / math.sqrt(values.shape[0])


def build_task_subspace(
    problem: MaxCutProblem,
    theta: torch.Tensor,
    weights: torch.Tensor,
    *,
    rank: int = 4,
    tol: float = 0.10,
    block: int = 2,
    indicator: str = "fro",
    center_weights: bool = False,
    fd_eps: float = 1e-4,
    jvp_mode: str = "fd",
    norm_probes: int = 8,
    residual_probes: int = 6,
    seed: int = 0,
    Q_init: Optional[torch.Tensor] = None,
) -> TaskSubspace:
    """Build the right singular subspace of ``M^(1/2) J`` matrix-free.

    If ``L L^T`` is the empirical weight second moment, the factorized
    operator is ``A = J^T L``. Its leading left singular vectors minimize the
    empirical expected squared discarded objective-gradient norm.
    """
    values = torch.as_tensor(weights, dtype=RDTYPE)
    if values.ndim != 2 or values.shape[1] != problem.m:
        raise ValueError(
            f"weights must have shape (n_tasks, {problem.m})"
        )
    if int(rank) != rank or rank < 1:
        raise ValueError("rank must be a positive integer")
    factor = empirical_weight_factor(values, center=center_weights)
    if center_weights and float(factor.norm()) <= 1e-14:
        raise ValueError("centered task weights have zero variation")
    operator = QAOASensitivity(problem, theta, fd_eps=fd_eps)

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        return operator.vjp(factor @ vector)

    def rmatvec(vector: torch.Tensor) -> torch.Tensor:
        return factor.t() @ operator.jvp(vector, mode=jvp_mode)

    result = randqb(
        matvec, rmatvec, dout=problem.dim, din=factor.shape[1],
        tol=tol, block=block,
        maxrank=min(int(rank), problem.dim, factor.shape[1]),
        indicator=indicator, n_norm=norm_probes, n_res=residual_probes,
        Q_init=Q_init,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    counts = operator.counts
    ledger = ResourceLedger(
        # FD/autograd JVP forward calls are hardware-style parameter points;
        # VJP forward passes stay in the separate simulator-only category.
        observable_evaluations=max(0, counts.forward_F - counts.vjp),
        sensitivity_jvps=counts.jvp,
        simulator_vjps=counts.vjp,
        subspace_builds=1,
    )
    return TaskSubspace(result.Q, result.B, result, ledger)


def objective_conditioned_residual(
    evaluator,
    theta: torch.Tensor,
    Q: torch.Tensor,
    weights: torch.Tensor,
    *,
    n_probe: int = 4,
    fd_eps: float = 1e-2,
    seed: int = 0,
) -> ResidualEstimate:
    """Forward-only estimate of omitted objective-gradient energy.

    Gaussian directions estimate ``||(I-QQ^T) J^T w||`` relative to
    ``||J^T w||``. Each probe uses four scalar objective evaluations: two for
    the full direction and two for its component orthogonal to ``Q``.
    """
    if int(n_probe) != n_probe or n_probe < 1:
        raise ValueError("n_probe must be a positive integer")
    if fd_eps <= 0:
        raise ValueError("fd_eps must be positive")
    if Q.ndim != 2 or Q.shape[0] != theta.numel():
        raise ValueError("Q must be a matrix with one row per parameter")
    before = evaluator.ledger.copy()
    generator = torch.Generator().manual_seed(int(seed))
    numerator = 0.0
    denominator = 0.0
    for _ in range(int(n_probe)):
        direction = torch.randn(
            theta.numel(), generator=generator, dtype=RDTYPE
        )
        residual_direction = direction - Q @ (Q.t() @ direction)
        full = (
            evaluator.objective(theta + fd_eps * direction, weights)
            - evaluator.objective(theta - fd_eps * direction, weights)
        ) / (2.0 * fd_eps)
        omitted = (
            evaluator.objective(
                theta + fd_eps * residual_direction, weights
            )
            - evaluator.objective(
                theta - fd_eps * residual_direction, weights
            )
        ) / (2.0 * fd_eps)
        denominator += full * full
        numerator += omitted * omitted
    ratio = math.sqrt(numerator / max(denominator, 1e-30))
    evaluator.ledger.residual_checks += 1
    return ResidualEstimate(
        ratio=ratio, numerator=math.sqrt(numerator / n_probe),
        denominator=math.sqrt(denominator / n_probe),
        ledger=evaluator.ledger.difference(before),
    )
