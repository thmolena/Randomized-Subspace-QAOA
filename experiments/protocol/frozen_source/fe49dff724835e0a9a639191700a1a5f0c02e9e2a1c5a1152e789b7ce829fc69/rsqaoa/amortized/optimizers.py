"""Matched-evaluation full- and reduced-space SPSA optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional

import torch

from ..circuits import MaxCutProblem, RDTYPE
from .budget import ResourceLedger


@dataclass(frozen=True)
class SPSAConfig:
    steps: int = 40
    learning_rate: float = 0.08
    perturbation: float = 0.08
    gain_exponent: float = 0.602
    perturbation_exponent: float = 0.101
    stability: float = 5.0
    averages: int = 1
    max_step_norm: Optional[float] = 0.35

    def __post_init__(self) -> None:
        if int(self.steps) != self.steps or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if self.learning_rate <= 0 or self.perturbation <= 0:
            raise ValueError("learning_rate and perturbation must be positive")
        if self.gain_exponent <= 0 or self.perturbation_exponent < 0:
            raise ValueError("gain exponents must be nonnegative")
        if self.stability < 0:
            raise ValueError("stability must be nonnegative")
        if int(self.averages) != self.averages or self.averages < 1:
            raise ValueError("averages must be a positive integer")
        if self.max_step_norm is not None and self.max_step_norm <= 0:
            raise ValueError("max_step_norm must be positive when supplied")

    @property
    def objective_evaluations(self) -> int:
        return 2 * int(self.steps) * int(self.averages) + 1


@dataclass
class SPSAResult:
    theta: torch.Tensor
    best_theta: torch.Tensor
    cut: float
    best_cut: float
    best_step: int
    dimension: int
    best_history: List[float] = field(default_factory=list)
    ledger: ResourceLedger = field(default_factory=ResourceLedger)


def _validate_basis(theta0: torch.Tensor,
                    basis: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if basis is None:
        return None
    Q = torch.as_tensor(basis, dtype=RDTYPE)
    if Q.ndim != 2 or Q.shape[0] != theta0.numel() or Q.shape[1] < 1:
        raise ValueError("basis must have shape (n_parameters, positive_rank)")
    if not torch.isfinite(Q).all():
        raise ValueError("basis must be finite")
    gram = Q.t() @ Q
    if not torch.allclose(
            gram, torch.eye(Q.shape[1], dtype=RDTYPE),
            atol=1e-7, rtol=1e-7):
        raise ValueError("basis columns must be orthonormal")
    return Q


def optimize_spsa(
    problem: MaxCutProblem,
    weights: torch.Tensor,
    theta0: torch.Tensor,
    *,
    evaluator,
    config: SPSAConfig,
    seed: int = 0,
    basis: Optional[torch.Tensor] = None,
) -> SPSAResult:
    """Maximize a weighted objective with matched two-evaluation SPSA steps.

    When ``basis`` is absent, SPSA acts in all native parameters. Otherwise it
    acts on reduced coordinates ``z`` with ``theta = theta0 + Q z``. Full and
    reduced modes therefore use exactly the same number of objective calls.
    """
    theta0 = torch.as_tensor(theta0, dtype=RDTYPE).detach().clone()
    if theta0.ndim != 1 or theta0.numel() != problem.dim:
        raise ValueError(f"theta0 must have shape ({problem.dim},)")
    weights = torch.as_tensor(weights, dtype=RDTYPE)
    if weights.ndim != 1 or weights.numel() != problem.m:
        raise ValueError(f"weights must have shape ({problem.m},)")
    Q = _validate_basis(theta0, basis)
    reduced = Q is not None
    dimension = Q.shape[1] if reduced else theta0.numel()
    coordinates = (
        torch.zeros(dimension, dtype=RDTYPE)
        if reduced else theta0.clone()
    )
    anchor = theta0.clone()
    generator = torch.Generator().manual_seed(int(seed))
    before = evaluator.ledger.copy()
    best_cut = float("-inf")
    best_theta = theta0.clone()
    best_step = -1
    history: List[float] = []

    def native(value: torch.Tensor) -> torch.Tensor:
        return anchor + Q @ value if reduced else value

    for step in range(config.steps):
        gradient = torch.zeros(dimension, dtype=RDTYPE)
        step_best = float("-inf")
        for _ in range(config.averages):
            delta = (2 * torch.randint(
                0, 2, (dimension,), generator=generator, dtype=torch.int64
            ) - 1).to(RDTYPE)
            ck = config.perturbation / (
                (step + 1) ** config.perturbation_exponent
            )
            plus_coordinates = coordinates + ck * delta
            minus_coordinates = coordinates - ck * delta
            plus_theta = native(plus_coordinates)
            minus_theta = native(minus_coordinates)
            plus = evaluator.objective(plus_theta, weights)
            minus = evaluator.objective(minus_theta, weights)
            gradient += ((plus - minus) / (2.0 * ck)) * delta
            if plus >= minus:
                observed_cut, observed_theta = plus, plus_theta
            else:
                observed_cut, observed_theta = minus, minus_theta
            step_best = max(step_best, observed_cut)
            if observed_cut > best_cut:
                best_cut = float(observed_cut)
                best_theta = observed_theta.detach().clone()
                best_step = step
        gradient /= config.averages
        ak = config.learning_rate / (
            (step + 1 + config.stability) ** config.gain_exponent
        )
        update = ak * gradient
        if config.max_step_norm is not None:
            norm = update.norm()
            if float(norm) > config.max_step_norm:
                update *= config.max_step_norm / norm.clamp_min(1e-30)
        coordinates = coordinates + update
        history.append(max(best_cut, step_best))

    theta = native(coordinates).detach()
    final_cut = evaluator.objective(theta, weights)
    if final_cut > best_cut:
        best_cut = float(final_cut)
        best_theta = theta.clone()
        best_step = config.steps
    ledger = evaluator.ledger.difference(before)
    if ledger.objective_evaluations != config.objective_evaluations:
        raise RuntimeError(
            "SPSA objective-evaluation ledger does not match its configuration"
        )
    return SPSAResult(
        theta=theta, best_theta=best_theta, cut=float(final_cut),
        best_cut=float(best_cut), best_step=best_step,
        dimension=int(dimension), best_history=history, ledger=ledger,
    )


def full_space_spsa(problem: MaxCutProblem, weights: torch.Tensor,
                    theta0: torch.Tensor, *, evaluator, config: SPSAConfig,
                    seed: int = 0) -> SPSAResult:
    return optimize_spsa(
        problem, weights, theta0, evaluator=evaluator, config=config,
        seed=seed, basis=None,
    )


def reduced_space_spsa(problem: MaxCutProblem, weights: torch.Tensor,
                       theta0: torch.Tensor, basis: torch.Tensor, *, evaluator,
                       config: SPSAConfig, seed: int = 0) -> SPSAResult:
    return optimize_spsa(
        problem, weights, theta0, evaluator=evaluator, config=config,
        seed=seed, basis=basis,
    )
