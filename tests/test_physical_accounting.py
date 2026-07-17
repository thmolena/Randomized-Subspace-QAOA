"""Tests for shared-shot and target-hitting physical resource accounting."""

import numpy as np  # load before torch on macOS
import torch

from rsqaoa.amortized.physical_accounting import (
    amortization_break_even_tasks,
    dense_forward_task_subspace,
    first_target_hitting_cost,
    shared_task_objectives,
)
from rsqaoa import graphs
from rsqaoa.amortized.evaluators import ExactEvaluator, ShotEvaluator
from rsqaoa.circuits import MaxCutProblem


def _problem():
    return MaxCutProblem(4, graphs.ring(4), 1)


def test_one_bitstring_batch_scalarizes_any_number_of_tasks():
    problem = _problem()
    theta = problem.random_theta(generator=torch.Generator().manual_seed(7))
    weights = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [0.5, 1.0, 1.5, 1.0]],
        dtype=torch.float64,
    )
    evaluator = ShotEvaluator(problem, shots=512, seed=8)
    batch = shared_task_objectives(evaluator, theta, weights)
    assert batch.values.shape == (2,)
    assert batch.covariance_of_means.shape == (2, 2)
    assert batch.circuit_points == 1 and batch.shots == 512
    assert evaluator.ledger.observable_evaluations == 1


def test_forward_reference_build_has_exact_physical_ledger():
    problem = _problem()
    theta = problem.random_theta(generator=torch.Generator().manual_seed(9))
    weights = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [0.8, 1.2, 1.1, 0.9]],
        dtype=torch.float64,
    )
    result = dense_forward_task_subspace(
        problem, theta, weights, ExactEvaluator(problem), rank=2,
    )
    assert result.Q.shape == (problem.dim, 2)
    assert torch.allclose(result.Q.t() @ result.Q, torch.eye(2, dtype=torch.float64))
    assert result.ledger.observable_evaluations == 2 * problem.dim
    assert result.ledger.simulator_vjps == 0


def test_break_even_and_target_hitting_are_explicit():
    # d=40, r=8, S=40 and a 2d-point build: one task already amortizes
    # coordinate finite differences. This does not apply to SPSA.
    assert amortization_break_even_tasks(
        ambient_dimension=40,
        reduced_rank=8,
        steps_per_task=40,
        basis_circuit_points=80,
    ) == 1
    trajectory = [
        {
            "approximation_ratio": 0.72,
            "cumulative_forward_circuit_evaluations": 10,
            "cumulative_shots": 1000,
        },
        {
            "approximation_ratio": 0.81,
            "cumulative_forward_circuit_evaluations": 20,
            "cumulative_shots": 2000,
        },
    ]
    hit = first_target_hitting_cost(trajectory, target=0.80)
    assert hit == {
        "record_index": 1,
        "value": 0.81,
        "circuit_points": 20,
        "shots": 2000,
    }
    assert first_target_hitting_cost(trajectory, target=0.90) is None
