"""Task-weighted QB and objective-conditioned residual tests."""

import numpy as np  # load before torch on macOS
import torch

from rsqaoa import graphs
from rsqaoa.circuits import MaxCutProblem
from rsqaoa.operator import QAOASensitivity
from rsqaoa.amortized.evaluators import ExactEvaluator
from rsqaoa.amortized.operators import (build_task_subspace,
                                        empirical_weight_factor,
                                        objective_conditioned_residual)
from rsqaoa.amortized.task_streams import low_rank_drift_stream


def _problem():
    return MaxCutProblem(5, graphs.ring(5), 2)


def test_task_weighted_qb_matches_dense_operator():
    problem = _problem()
    theta = problem.random_theta(generator=torch.Generator().manual_seed(1))
    stream = low_rank_drift_stream(
        problem.m, n_tasks=6, latent_rank=2, drift_scale=0.1, seed=2
    )
    built = build_task_subspace(
        problem, theta, stream.weights, rank=3, tol=0.05,
        norm_probes=8, residual_probes=6, seed=3,
    )
    jacobian = QAOASensitivity(problem, theta).dense_jacobian()
    factor = empirical_weight_factor(stream.weights)
    dense = jacobian.t() @ factor
    relative = torch.norm(dense - built.Q @ (built.Q.t() @ dense)) / torch.norm(dense)
    assert relative < 0.10
    assert built.ledger.subspace_builds == 1
    assert built.ledger.simulator_vjps > 0


def test_objective_conditioned_probe_is_small_for_exact_gradient_span():
    problem = _problem()
    theta = problem.random_theta(generator=torch.Generator().manual_seed(4))
    weights = torch.linspace(0.5, 1.5, problem.m, dtype=torch.float64)
    jacobian = QAOASensitivity(problem, theta).dense_jacobian()
    gradient = jacobian.t() @ weights
    Q = (gradient / gradient.norm()).reshape(-1, 1)
    evaluator = ExactEvaluator(problem)
    estimate = objective_conditioned_residual(
        evaluator, theta, Q, weights, n_probe=8, fd_eps=1e-4, seed=5
    )
    assert estimate.ratio < 1e-4
    assert estimate.ledger.objective_evaluations == 32
    assert estimate.ledger.residual_checks == 1
