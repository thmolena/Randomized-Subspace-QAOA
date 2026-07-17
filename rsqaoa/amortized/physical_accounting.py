"""Physical measurement accounting for repeated weighted-MaxCut objectives.

This module is intentionally separate from the frozen development runner.  It
contains no device-specific cost conversion: one call to ``observables`` is
one circuit parameter point and its evaluator ledger records the actual shots.
Because all MaxCut edge terms are diagonal, one bitstring batch can be
scalarized against every task weight vector without another circuit call.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Optional

import torch

from ..circuits import MaxCutProblem, RDTYPE
from .budget import ResourceLedger
from .operators import empirical_weight_factor


@dataclass(frozen=True)
class SharedObjectiveBatch:
    """All task objectives estimated from one observable measurement batch."""

    values: torch.Tensor
    covariance_of_means: torch.Tensor
    observable_mean: torch.Tensor
    circuit_points: int
    shots: int


@dataclass(frozen=True)
class ForwardTaskSubspace:
    """Dense forward-difference reference used to audit physical cost."""

    Q: torch.Tensor
    singular_values: torch.Tensor
    ledger: ResourceLedger
    finite_difference_step: float


def shared_task_objectives(
    evaluator,
    theta: torch.Tensor,
    task_weights: torch.Tensor,
) -> SharedObjectiveBatch:
    """Estimate any number of scalarizations from one bitstring batch.

    The returned covariance is the covariance of the estimated task means,
    not the single-shot covariance.  Exact evaluators report a zero covariance
    and zero shots.  Increasing the number of task rows does not change the
    evaluator ledger.
    """
    weights = torch.as_tensor(task_weights, dtype=RDTYPE)
    if weights.ndim != 2 or weights.shape[1] != evaluator.problem.m:
        raise ValueError(
            f"task_weights must have shape (n_tasks, {evaluator.problem.m})"
        )
    if not torch.isfinite(weights).all():
        raise ValueError("task_weights must be finite")
    before = evaluator.ledger.copy()
    estimate = evaluator.observables(theta)
    delta = evaluator.ledger.difference(before)
    if delta.observable_evaluations != 1:
        raise RuntimeError("one shared batch must use one observable evaluation")
    values = weights @ estimate.mean
    if estimate.shots:
        covariance = (
            weights @ estimate.covariance @ weights.t() / estimate.shots
        )
    else:
        covariance = torch.zeros(
            weights.shape[0], weights.shape[0], dtype=RDTYPE
        )
    return SharedObjectiveBatch(
        values=values,
        covariance_of_means=covariance,
        observable_mean=estimate.mean,
        circuit_points=delta.forward_circuit_evaluations,
        shots=delta.shots,
    )


def dense_forward_task_subspace(
    problem: MaxCutProblem,
    theta: torch.Tensor,
    basis_weights: torch.Tensor,
    evaluator,
    *,
    rank: int,
    finite_difference_step: float = 1e-2,
) -> ForwardTaskSubspace:
    """Construct a task-weighted reference basis using only forward batches.

    This routine forms a dense finite-difference Jacobian and is therefore a
    physical-accounting reference, not the proposed scalable algorithm.  It
    costs exactly ``2 * d`` observable circuit points and uses no VJPs.  Its
    purpose is to prevent simulator reverse-mode actions from being mislabeled
    as hardware queries in a confirmatory audit.
    """
    theta = torch.as_tensor(theta, dtype=RDTYPE)
    weights = torch.as_tensor(basis_weights, dtype=RDTYPE)
    if theta.ndim != 1 or theta.numel() != problem.dim:
        raise ValueError(f"theta must have shape ({problem.dim},)")
    if weights.ndim != 2 or weights.shape[1] != problem.m:
        raise ValueError(
            f"basis_weights must have shape (n_tasks, {problem.m})"
        )
    if int(rank) != rank or not 1 <= rank <= min(
            problem.dim, weights.shape[0]):
        raise ValueError("rank exceeds the available task or parameter rank")
    if not math.isfinite(finite_difference_step) or finite_difference_step <= 0:
        raise ValueError("finite_difference_step must be positive and finite")

    before = evaluator.ledger.copy()
    columns = []
    for coordinate in range(problem.dim):
        direction = torch.zeros(problem.dim, dtype=RDTYPE)
        direction[coordinate] = finite_difference_step
        plus = evaluator.observables(theta + direction).mean
        minus = evaluator.observables(theta - direction).mean
        columns.append((plus - minus) / (2.0 * finite_difference_step))
    jacobian = torch.stack(columns, dim=1)
    factor = empirical_weight_factor(weights)
    operator = jacobian.t() @ factor
    U, singular_values, _ = torch.linalg.svd(operator, full_matrices=False)
    ledger = evaluator.ledger.difference(before)
    expected_points = 2 * problem.dim
    if ledger.observable_evaluations != expected_points:
        raise RuntimeError("dense forward construction ledger is inconsistent")
    if ledger.simulator_vjps != 0:
        raise RuntimeError("forward construction must not use simulator VJPs")
    return ForwardTaskSubspace(
        Q=U[:, : int(rank)].contiguous(),
        singular_values=singular_values,
        ledger=ledger,
        finite_difference_step=float(finite_difference_step),
    )


def amortization_break_even_tasks(
    *,
    ambient_dimension: int,
    reduced_rank: int,
    steps_per_task: int,
    basis_circuit_points: int,
    refresh_circuit_points: int = 0,
) -> int:
    """Smallest integer task count giving lower coordinate-FD circuit cost.

    Full and reduced central-coordinate updates cost ``2*d`` and ``2*r``
    circuit points per step.  This calculation does not apply to SPSA, whose
    two-call update cost is independent of dimension.
    """
    values = (
        ambient_dimension, reduced_rank, steps_per_task,
        basis_circuit_points, refresh_circuit_points,
    )
    if any(int(value) != value for value in values):
        raise ValueError("all break-even inputs must be integers")
    d, r, steps, build, refresh = (int(value) for value in values)
    if d < 1 or not 1 <= r < d or steps < 1 or build < 0 or refresh < 0:
        raise ValueError("invalid break-even dimensions or resource counts")
    overhead = build + refresh
    saving_per_task = 2 * steps * (d - r)
    return overhead // saving_per_task + 1


def first_target_hitting_cost(
    records: Iterable[Mapping[str, float]],
    *,
    target: float,
    value_key: str = "approximation_ratio",
    circuit_key: str = "cumulative_forward_circuit_evaluations",
    shot_key: str = "cumulative_shots",
) -> Optional[dict]:
    """Return the first prespecified target hit, or ``None`` if right-censored."""
    if not math.isfinite(target):
        raise ValueError("target must be finite")
    ordered = list(records)
    previous_circuits = -1
    previous_shots = -1
    for index, row in enumerate(ordered):
        value = float(row[value_key])
        circuits = int(row[circuit_key])
        shots = int(row[shot_key])
        if circuits < previous_circuits or shots < previous_shots:
            raise ValueError("resource trajectories must be cumulative")
        previous_circuits, previous_shots = circuits, shots
        if value >= target:
            return {
                "record_index": index,
                "value": value,
                "circuit_points": circuits,
                "shots": shots,
            }
    return None
