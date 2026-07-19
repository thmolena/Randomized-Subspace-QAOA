"""Exact and shot-based observable evaluators for amortized experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..circuits import MaxCutProblem, RDTYPE, statevector
from .budget import ResourceLedger


@dataclass(frozen=True)
class ObservableEstimate:
    mean: torch.Tensor
    covariance: torch.Tensor
    shots: int


class ExactEvaluator:
    """Exact-statevector evaluator with explicit forward-resource accounting."""

    def __init__(self, problem: MaxCutProblem):
        self.problem = problem
        self.ledger = ResourceLedger()

    def objective(self, theta: torch.Tensor, weights: torch.Tensor) -> float:
        weights = _checked_weights(weights, self.problem.m)
        value = torch.dot(weights, self.problem.F(theta))
        self.ledger.objective_evaluations += 1
        return float(value.detach())

    def observables(self, theta: torch.Tensor) -> ObservableEstimate:
        values = self.problem.F(theta).detach()
        self.ledger.observable_evaluations += 1
        covariance = torch.zeros(
            self.problem.m, self.problem.m, dtype=RDTYPE
        )
        return ObservableEstimate(values, covariance, 0)


class ShotEvaluator:
    """CPU shot sampler for diagonal MaxCut observables.

    This is a sampling model, not a hardware or calibrated-device model. All
    edge observables are extracted from the same computational-basis samples.
    Optional readout noise independently flips measured bits.
    """

    def __init__(self, problem: MaxCutProblem, shots: int,
                 seed: int = 0, readout_error: float = 0.0):
        if int(shots) != shots or shots < 1:
            raise ValueError("shots must be a positive integer")
        if not 0.0 <= readout_error < 0.5:
            raise ValueError("readout_error must lie in [0, 0.5)")
        self.problem = problem
        self.shots = int(shots)
        self.readout_error = float(readout_error)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.ledger = ResourceLedger()

    def _sample_edge_values(self, theta: torch.Tensor) -> torch.Tensor:
        psi = statevector(
            theta, self.problem.n, self.problem.edges, self.problem.p,
            self.problem.C,
        )
        probabilities = (psi.conj() * psi).real.clamp_min(0)
        probabilities = probabilities / probabilities.sum()
        states = torch.multinomial(
            probabilities, self.shots, replacement=True,
            generator=self.generator,
        )
        if self.readout_error == 0.0:
            return self.problem.C[:, states].t().contiguous()

        shifts = torch.arange(self.problem.n, dtype=torch.int64)
        bits = ((states[:, None] >> shifts[None, :]) & 1).to(torch.int64)
        flips = torch.rand(
            bits.shape, generator=self.generator, dtype=RDTYPE
        ) < self.readout_error
        bits = torch.logical_xor(bits.bool(), flips).to(RDTYPE)
        columns = [
            torch.logical_xor(bits[:, i].bool(), bits[:, j].bool()).to(RDTYPE)
            for i, j in self.problem.edges
        ]
        return torch.stack(columns, dim=1)

    def _estimate(self, theta: torch.Tensor) -> ObservableEstimate:
        samples = self._sample_edge_values(theta)
        mean = samples.mean(dim=0)
        if self.shots > 1:
            centered = samples - mean
            covariance = centered.t() @ centered / (self.shots - 1)
        else:
            covariance = torch.zeros(
                self.problem.m, self.problem.m, dtype=RDTYPE
            )
        return ObservableEstimate(mean, covariance, self.shots)

    def objective(self, theta: torch.Tensor, weights: torch.Tensor) -> float:
        weights = _checked_weights(weights, self.problem.m)
        estimate = self._estimate(theta)
        self.ledger.objective_evaluations += 1
        self.ledger.shots += self.shots
        return float(torch.dot(weights, estimate.mean))

    def observables(self, theta: torch.Tensor) -> ObservableEstimate:
        estimate = self._estimate(theta)
        self.ledger.observable_evaluations += 1
        self.ledger.shots += self.shots
        return estimate


def _checked_weights(weights: torch.Tensor, n_edges: int) -> torch.Tensor:
    values = torch.as_tensor(weights, dtype=RDTYPE)
    if values.ndim != 1 or values.numel() != n_edges:
        raise ValueError(f"weights must have shape ({n_edges},)")
    if not torch.isfinite(values).all():
        raise ValueError("weights must be finite")
    return values


def make_evaluator(problem: MaxCutProblem, *, shots: Optional[int] = None,
                   seed: int = 0, readout_error: float = 0.0):
    if shots is None:
        if readout_error != 0.0:
            raise ValueError("readout_error requires a shot evaluator")
        return ExactEvaluator(problem)
    return ShotEvaluator(
        problem, shots=shots, seed=seed, readout_error=readout_error
    )
