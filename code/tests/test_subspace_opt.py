"""Smoke + sanity tests for the RSQ optimizer and baselines."""

import numpy as np
import torch

from rsqaoa.circuits import MaxCutProblem
from rsqaoa import graphs
from rsqaoa.subspace_opt import optimize_rsq
from rsqaoa.baselines import full_maqaoa, fixed_rank_subspace, symmetry_reduced


def _ring(n=6, p=1):
    return MaxCutProblem(n=n, edges=graphs.ring(n), p=p)


def test_rsq_improves_objective():
    prob = _ring(6, 1)
    theta0 = prob.random_theta(generator=torch.Generator().manual_seed(0))
    init_cut = float(prob.cut(theta0))
    res = optimize_rsq(prob, theta0=theta0.clone(), tol=1e-2, steps=150,
                       inner_lr=0.08, refresh_every=30, seed=0)
    assert max(res.cut_history) >= init_cut + 1e-6      # subspace search made progress
    assert 0 <= res.cut <= prob.m + 1e-9
    assert res.final_rank <= prob.dim
    assert set(res.counts) == {"forward_F", "jvp", "vjp"}


def test_rsq_adjoint_free_runs():
    prob = _ring(6, 1)
    res = optimize_rsq(prob, tol=1e-2, steps=60, adjoint_free=True, af_rank=4, seed=1)
    assert res.counts["vjp"] == 0   # build and refresh certificate are forward-only
    assert res.counts["jvp"] > 0
    assert 0 <= res.cut <= prob.m + 1e-9


def test_baselines_run():
    prob = _ring(6, 1)
    theta0 = prob.random_theta(generator=torch.Generator().manual_seed(2))
    full = full_maqaoa(prob, theta0=theta0.clone(), steps=80)
    fr = fixed_rank_subspace(prob, rank=3, theta0=theta0.clone(), steps=80)
    sym = symmetry_reduced(prob, steps=80)
    for r in (full, fr, sym):
        assert 0 <= r.cut <= prob.m + 1e-9
    # ring has a large automorphism group -> symmetry should compress parameters
    assert sym.n_params_opt <= prob.dim
