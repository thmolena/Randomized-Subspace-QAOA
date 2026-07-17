"""Amortized optimization over a bank or stream of weighted objectives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from ..circuits import MaxCutProblem, RDTYPE
from ..randqb import QBResult
from .budget import ResourceLedger
from .evaluators import make_evaluator
from .operators import (TaskSubspace, build_task_subspace,
                        objective_conditioned_residual)
from .optimizers import SPSAConfig, full_space_spsa, reduced_space_spsa
from .task_streams import WeightStream


VALID_REFRESH_MODES = {
    "none", "fixed", "gated", "per_task", "random", "random_basis"
}


@dataclass
class TaskRecord:
    task_index: int
    exact_cut: float
    reported_best_cut: float
    final_cut: float
    optimized_dimension: int
    basis_rank: int
    residual: Optional[float]
    refreshed: bool
    ledger: ResourceLedger


@dataclass
class StreamResult:
    method: str
    theta: torch.Tensor
    records: List[TaskRecord] = field(default_factory=list)
    ledger: ResourceLedger = field(default_factory=ResourceLedger)
    build_status: List[dict] = field(default_factory=list)


def _exact_cut(problem: MaxCutProblem, theta: torch.Tensor,
               weights: torch.Tensor) -> float:
    with torch.no_grad():
        return float(torch.dot(weights, problem.F(theta)))


def optimize_full_stream(
    problem: MaxCutProblem,
    stream: WeightStream,
    theta0: torch.Tensor,
    *,
    spsa: SPSAConfig,
    seed: int = 0,
    measurement_seed: Optional[int] = None,
    shots: Optional[int] = None,
    readout_error: float = 0.0,
) -> StreamResult:
    """Warm-started full-space SPSA baseline over all tasks."""
    theta = torch.as_tensor(theta0, dtype=RDTYPE).detach().clone()
    total = ResourceLedger()
    records: List[TaskRecord] = []
    measurement_seed = seed if measurement_seed is None else int(measurement_seed)
    for task, weights in enumerate(stream.weights):
        before = total.copy()
        evaluator = make_evaluator(
            problem, shots=shots, seed=measurement_seed + 10_000 + task,
            readout_error=readout_error,
        )
        result = full_space_spsa(
            problem, weights, theta, evaluator=evaluator, config=spsa,
            seed=seed + task,
        )
        total.add(evaluator.ledger)
        theta = result.best_theta.clone()
        records.append(TaskRecord(
            task_index=task,
            exact_cut=_exact_cut(problem, theta, weights),
            reported_best_cut=result.best_cut,
            final_cut=result.cut,
            optimized_dimension=problem.dim,
            basis_rank=problem.dim,
            residual=None,
            refreshed=False,
            ledger=total.difference(before),
        ))
    return StreamResult(
        method="full_spsa", theta=theta, records=records, ledger=total
    )


def optimize_amortized_stream(
    problem: MaxCutProblem,
    stream: WeightStream,
    theta0: torch.Tensor,
    *,
    mode: str = "gated",
    rank: int = 4,
    subspace_tol: float = 0.10,
    subspace_block: int = 2,
    subspace_norm_probes: int = 8,
    subspace_residual_probes: int = 6,
    refresh_every_tasks: int = 4,
    refresh_threshold: float = 0.30,
    gate_probes: int = 4,
    gate_fd_eps: float = 1e-2,
    random_refresh_probability: float = 0.25,
    spsa: SPSAConfig = SPSAConfig(),
    seed: int = 0,
    measurement_seed: Optional[int] = None,
    shots: Optional[int] = None,
    readout_error: float = 0.0,
    basis_weights: Optional[torch.Tensor] = None,
) -> StreamResult:
    """Optimize a related task bank with a reused sensitivity subspace.

    The bank weights are assumed known when constructing their empirical task
    second moment. For a genuinely online protocol, callers must pass a
    training-only ``basis_weights`` matrix and evaluate on a separate stream.
    """
    if mode not in VALID_REFRESH_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_REFRESH_MODES)}")
    if int(refresh_every_tasks) != refresh_every_tasks or refresh_every_tasks < 1:
        raise ValueError("refresh_every_tasks must be a positive integer")
    if refresh_threshold < 0:
        raise ValueError("refresh_threshold must be nonnegative")
    if not 0.0 <= random_refresh_probability <= 1.0:
        raise ValueError("random_refresh_probability must lie in [0, 1]")
    theta = torch.as_tensor(theta0, dtype=RDTYPE).detach().clone()
    if theta.ndim != 1 or theta.numel() != problem.dim:
        raise ValueError(f"theta0 must have shape ({problem.dim},)")
    bank = stream.weights if basis_weights is None else torch.as_tensor(
        basis_weights, dtype=RDTYPE
    )
    if bank.ndim != 2 or bank.shape[1] != problem.m:
        raise ValueError(f"basis_weights must have {problem.m} columns")
    total = ResourceLedger()
    records: List[TaskRecord] = []
    statuses: List[dict] = []
    random_generator = torch.Generator().manual_seed(seed + 900_000)
    measurement_seed = seed if measurement_seed is None else int(measurement_seed)
    current: Optional[TaskSubspace] = None

    def rebuild(task: int, *, refresh: bool) -> None:
        nonlocal current
        current = build_task_subspace(
            problem, theta, bank, rank=rank, tol=subspace_tol,
            block=subspace_block, norm_probes=subspace_norm_probes,
            residual_probes=subspace_residual_probes,
            seed=seed + 100_000 + task,
            Q_init=(current.Q if current is not None else None),
        )
        ledger = current.ledger.copy()
        if refresh:
            ledger.refreshes += 1
        total.add(ledger)
        statuses.append({
            "task": int(task), "rank": current.rank,
            "relative_residual": float(current.result.rel_residual),
            "converged": bool(current.result.converged),
            "stop_reason": current.result.stop_reason,
            "refresh": bool(refresh),
        })

    def build_random_control() -> None:
        nonlocal current
        generator = torch.Generator().manual_seed(seed + 700_000)
        width = min(int(rank), problem.dim)
        trial = torch.randn(
            problem.dim, width, generator=generator, dtype=RDTYPE
        )
        Q, _ = torch.linalg.qr(trial, mode="reduced")
        B = torch.zeros(width, bank.shape[0], dtype=RDTYPE)
        result = QBResult(
            Q=Q, B=B, rank=width, rel_residual=float("nan"),
            indicator="random_control", matvecs=0, rmatvecs=0,
            converged=False, stop_reason="random_control",
        )
        current = TaskSubspace(
            Q=Q, B=B, result=result,
            ledger=ResourceLedger(subspace_builds=1),
        )
        total.add(current.ledger)
        statuses.append({
            "task": 0, "rank": width, "relative_residual": None,
            "converged": None, "stop_reason": "random_control",
            "refresh": False,
        })

    if mode == "random_basis":
        build_random_control()
    elif mode != "per_task":
        rebuild(0, refresh=False)

    for task, weights in enumerate(stream.weights):
        # Attribute the one-time initial basis construction to task zero so a
        # row-wise sum exactly reproduces the stream ledger and break-even
        # analyses cannot accidentally drop setup cost.
        before = ResourceLedger() if task == 0 else total.copy()
        evaluator = make_evaluator(
            problem, shots=shots, seed=measurement_seed + 20_000 + task,
            readout_error=readout_error,
        )
        refreshed = False
        residual = None
        if mode == "per_task":
            rebuild(task, refresh=(task > 0))
            refreshed = task > 0
        elif task > 0 and mode == "fixed" and task % refresh_every_tasks == 0:
            rebuild(task, refresh=True)
            refreshed = True
        elif task > 0 and mode == "gated":
            estimate = objective_conditioned_residual(
                evaluator, theta, current.Q, weights,
                n_probe=gate_probes, fd_eps=gate_fd_eps,
                seed=seed + 300_000 + task,
            )
            residual = estimate.ratio
            if residual > refresh_threshold:
                rebuild(task, refresh=True)
                refreshed = True
        elif task > 0 and mode == "random":
            draw = float(torch.rand((), generator=random_generator))
            if draw < random_refresh_probability:
                rebuild(task, refresh=True)
                refreshed = True

        result = reduced_space_spsa(
            problem, weights, theta, current.Q, evaluator=evaluator,
            config=spsa, seed=seed + task,
        )
        total.add(evaluator.ledger)
        theta = result.best_theta.clone()
        records.append(TaskRecord(
            task_index=task,
            exact_cut=_exact_cut(problem, theta, weights),
            reported_best_cut=result.best_cut,
            final_cut=result.cut,
            optimized_dimension=result.dimension,
            basis_rank=current.rank,
            residual=residual,
            refreshed=refreshed,
            ledger=total.difference(before),
        ))
    return StreamResult(
        method=f"amortized_{mode}", theta=theta, records=records,
        ledger=total, build_status=statuses,
    )
