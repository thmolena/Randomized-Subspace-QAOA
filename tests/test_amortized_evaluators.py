"""Exact and finite-shot evaluator behavior."""

import numpy as np  # load before torch on macOS
import torch

from rsqaoa import graphs
from rsqaoa.circuits import MaxCutProblem
from rsqaoa.amortized.evaluators import ExactEvaluator, ShotEvaluator


def test_shot_observables_are_reproducible_and_track_exact_means():
    problem = MaxCutProblem(4, graphs.ring(4), 1)
    theta = problem.random_theta(generator=torch.Generator().manual_seed(1))
    exact = ExactEvaluator(problem).observables(theta).mean
    left = ShotEvaluator(problem, shots=20_000, seed=2).observables(theta)
    right = ShotEvaluator(problem, shots=20_000, seed=2).observables(theta)
    assert torch.equal(left.mean, right.mean)
    assert left.covariance.shape == (problem.m, problem.m)
    assert torch.max(torch.abs(left.mean - exact)) < 0.025


def test_readout_noise_and_shot_budget_are_explicit():
    problem = MaxCutProblem(4, graphs.ring(4), 1)
    theta = problem.random_theta(generator=torch.Generator().manual_seed(3))
    evaluator = ShotEvaluator(
        problem, shots=64, seed=4, readout_error=0.02
    )
    value = evaluator.objective(theta, torch.ones(problem.m))
    assert 0 <= value <= problem.m
    assert evaluator.ledger.objective_evaluations == 1
    assert evaluator.ledger.shots == 64
