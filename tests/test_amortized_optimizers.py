"""Matched SPSA and amortized refresh-control tests."""

import numpy as np  # load before torch on macOS
import torch

from rsqaoa import graphs
from rsqaoa.circuits import MaxCutProblem
from rsqaoa.amortized.evaluators import ExactEvaluator
from rsqaoa.amortized.operators import build_task_subspace
from rsqaoa.amortized.optimizers import (SPSAConfig, full_space_spsa,
                                         reduced_space_spsa)
from rsqaoa.amortized.stream_opt import (optimize_amortized_stream,
                                         optimize_full_stream)
from rsqaoa.amortized.task_streams import low_rank_drift_stream


def _fixture():
    problem = MaxCutProblem(5, graphs.ring(5), 1)
    theta = problem.random_theta(generator=torch.Generator().manual_seed(10))
    stream = low_rank_drift_stream(problem.m, n_tasks=4, latent_rank=2, seed=11)
    return problem, theta, stream


def test_full_and_reduced_spsa_use_matched_objective_evaluations():
    problem, theta, stream = _fixture()
    config = SPSAConfig(steps=6, averages=2)
    built = build_task_subspace(
        problem, theta, stream.weights, rank=2,
        norm_probes=2, residual_probes=2, seed=12,
    )
    full = full_space_spsa(
        problem, stream.weights[0], theta,
        evaluator=ExactEvaluator(problem), config=config, seed=13,
    )
    reduced = reduced_space_spsa(
        problem, stream.weights[0], theta, built.Q,
        evaluator=ExactEvaluator(problem), config=config, seed=13,
    )
    assert full.ledger.objective_evaluations == config.objective_evaluations
    assert reduced.ledger.objective_evaluations == config.objective_evaluations
    assert full.dimension == problem.dim and reduced.dimension == built.rank
    assert torch.isfinite(torch.tensor([full.best_cut, reduced.best_cut])).all()


def test_stream_controls_have_auditable_build_and_check_counts():
    problem, theta, stream = _fixture()
    config = SPSAConfig(steps=3)
    full = optimize_full_stream(problem, stream, theta, spsa=config, seed=20)
    none = optimize_amortized_stream(
        problem, stream, theta, mode="none", rank=2, spsa=config,
        subspace_norm_probes=2, subspace_residual_probes=2, seed=20,
    )
    fixed = optimize_amortized_stream(
        problem, stream, theta, mode="fixed", rank=2,
        refresh_every_tasks=2, spsa=config,
        subspace_norm_probes=2, subspace_residual_probes=2, seed=20,
    )
    gated = optimize_amortized_stream(
        problem, stream, theta, mode="gated", rank=2,
        refresh_threshold=0.0, gate_probes=1, spsa=config,
        subspace_norm_probes=2, subspace_residual_probes=2, seed=20,
    )
    per_task = optimize_amortized_stream(
        problem, stream, theta, mode="per_task", rank=2, spsa=config,
        subspace_norm_probes=2, subspace_residual_probes=2, seed=20,
    )
    random_basis = optimize_amortized_stream(
        problem, stream, theta, mode="random_basis", rank=2, spsa=config,
        seed=20,
    )
    assert len(full.records) == len(none.records) == stream.n_tasks
    assert none.ledger.subspace_builds == 1 and none.ledger.refreshes == 0
    assert fixed.ledger.subspace_builds == 2 and fixed.ledger.refreshes == 1
    assert gated.ledger.residual_checks == stream.n_tasks - 1
    assert gated.ledger.refreshes == stream.n_tasks - 1
    assert per_task.ledger.subspace_builds == stream.n_tasks
    assert random_basis.ledger.simulator_vjps == 0
    for result in (full, none, fixed, gated, per_task, random_basis):
        assert all(torch.isfinite(torch.tensor(row.exact_cut))
                   for row in result.records)
        accumulated = result.ledger.__class__()
        for row in result.records:
            accumulated.add(row.ledger)
        assert accumulated.as_dict() == result.ledger.as_dict()
